"""Data ingestion utilities for the Makoding package.

Makoding supports:

- CSV files
- TSV files
- Excel files (.xlsx, .xls)
- Remote HTTP(S) URLs

This module intentionally has no Streamlit dependency so it can be reused
by Streamlit applications, Jupyter notebooks, automated tests, command-line
applications, APIs, and other Python applications.

All user-facing ingestion failures are exposed through DataLoadError.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from makoding.config import LIMITS
from makoding.logging_config import setup_logging


logger = setup_logging()


# ============================================================================
# Exceptions and result objects
# ============================================================================


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded or validated."""


@dataclass(frozen=True)
class LoadedDataset:
    """Container representing a successfully loaded dataset.

    Attributes:
        frame:
            The loaded pandas DataFrame.

        source_name:
            Original filename or source URL.
    """

    frame: pd.DataFrame
    source_name: str


# ============================================================================
# File validation
# ============================================================================


def validate_file_extension(filename: str) -> str:
    """Validate and return a normalized file extension.

    Args:
        filename:
            Original filename.

    Returns:
        Lowercase extension such as ``.csv`` or ``.xlsx``.

    Raises:
        DataLoadError:
            If the filename is empty or has an unsupported extension.
    """

    if not filename or not filename.strip():
        raise DataLoadError(
            "The filename cannot be empty."
        )

    extension = Path(filename).suffix.lower()

    if extension not in LIMITS.allowed_upload_extensions:
        supported = ", ".join(
            LIMITS.allowed_upload_extensions
        )

        raise DataLoadError(
            f"Unsupported file type "
            f"'{extension or '[none]'}'. "
            f"Supported types: {supported}"
        )

    return extension


def validate_upload_size(size_bytes: int) -> None:
    """Validate a dataset against the configured upload-size limit.

    The maximum size is inclusive.

    For example, if the limit is 200 MB:

    - 200 MB -> accepted
    - 200 MB + 1 byte -> rejected

    Args:
        size_bytes:
            File size in bytes.

    Raises:
        DataLoadError:
            If the size is negative or exceeds the configured limit.
    """

    if size_bytes < 0:
        raise DataLoadError(
            "Invalid file size."
        )

    max_bytes = (
        LIMITS.max_upload_mb
        * 1024
        * 1024
    )

    if size_bytes > max_bytes:
        actual_mb = size_bytes / 1_048_576

        raise DataLoadError(
            f"File is {actual_mb:.1f} MB, which exceeds "
            f"the {LIMITS.max_upload_mb} MB upload limit."
        )


def _validate_row_count(
    frame: pd.DataFrame,
    source_name: str,
) -> None:
    """Validate the number of rows against the configured limit.

    Args:
        frame:
            Parsed DataFrame.

        source_name:
            Filename or URL used in the error message.

    Raises:
        DataLoadError:
            If the dataset exceeds ``LIMITS.max_rows``.
    """

    max_rows = LIMITS.max_rows

    if max_rows <= 0:
        raise DataLoadError(
            "The configured maximum row limit must be greater than zero."
        )

    row_count = len(frame)

    if row_count > max_rows:
        raise DataLoadError(
            f"The dataset '{source_name}' contains "
            f"{row_count:,} rows, which exceeds the configured "
            f"limit of {max_rows:,} rows."
        )


# ============================================================================
# URL validation
# ============================================================================


def validate_url(url: str) -> None:
    """Validate that a URL uses an allowed HTTP(S) scheme.

    Args:
        url:
            URL to validate.

    Raises:
        DataLoadError:
            If the URL is empty, malformed, or uses an unsupported scheme.
    """

    if not url or not url.strip():
        raise DataLoadError(
            "The URL cannot be empty."
        )

    parsed = urlparse(
        url.strip()
    )

    allowed_schemes = tuple(
        scheme.lower()
        for scheme in LIMITS.allowed_url_schemes
    )

    if (
        parsed.scheme.lower() not in allowed_schemes
        or not parsed.netloc
    ):
        raise DataLoadError(
            "That doesn't look like a valid HTTP(S) URL. "
            "Example: https://example.com/data.csv"
        )


def _get_url_filename(url: str) -> str:
    """Extract a filename from a URL.

    The URL path is used first. If the URL has no filename component,
    a generic CSV filename is returned.

    This allows endpoints such as:

        https://example.com/data

    to be treated consistently while still requiring a supported
    extension when one is explicitly supplied.
    """

    parsed = urlparse(url)

    filename = Path(
        parsed.path
    ).name

    if filename:
        return filename

    return "remote_dataset.csv"


# ============================================================================
# Data parsing
# ============================================================================


def _read_bytes(
    name: str,
    data: bytes,
) -> pd.DataFrame:
    """Parse raw bytes into a pandas DataFrame.

    The parser is selected from the filename extension.

    CSV and TSV files are first attempted using UTF-8. If decoding fails,
    Latin-1 is attempted as a compatibility fallback.

    Args:
        name:
            Filename used to determine the file format.

        data:
            Raw file contents.

    Returns:
        Parsed pandas DataFrame.

    Raises:
        DataLoadError:
            If the file cannot be parsed.
    """

    extension = validate_file_extension(
        name
    )

    # ------------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------------

    if extension in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(
                io.BytesIO(data)
            )

        except ImportError as exc:
            raise DataLoadError(
                "Excel support is not fully installed. "
                "Please install the required Excel package."
            ) from exc

        except Exception as exc:
            raise DataLoadError(
                f"Could not parse Excel file: {exc}"
            ) from exc

    # ------------------------------------------------------------------------
    # CSV / TSV
    # ------------------------------------------------------------------------

    separator = (
        "\t"
        if extension == ".tsv"
        else ","
    )

    try:
        return pd.read_csv(
            io.BytesIO(data),
            sep=separator,
            encoding="utf-8",
        )

    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decoding failed for '%s'. "
            "Retrying with Latin-1.",
            name,
        )

        try:
            return pd.read_csv(
                io.BytesIO(data),
                sep=separator,
                encoding="latin-1",
            )

        except Exception as exc:
            raise DataLoadError(
                "Could not decode the text file using "
                "UTF-8 or Latin-1."
            ) from exc

    except Exception as exc:
        raise DataLoadError(
            f"Could not parse '{name}' as a delimited "
            f"text file: {exc}"
        ) from exc


# ============================================================================
# Local / uploaded files
# ============================================================================


def load_dataframe(
    file,
    filename: str,
) -> pd.DataFrame:
    """Load an uploaded file into a pandas DataFrame.

    The function is framework-independent and accepts:

    - raw bytes
    - binary file-like objects
    - Streamlit UploadedFile objects

    Validation occurs at multiple levels:

    1. File extension
    2. File size
    3. Binary data availability
    4. Parsing
    5. Empty dataset detection
    6. Maximum row count

    Args:
        file:
            File bytes or a binary file-like object.

        filename:
            Original filename.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        DataLoadError:
            If validation, reading, or parsing fails.
    """

    # ------------------------------------------------------------------------
    # Validate extension
    # ------------------------------------------------------------------------

    validate_file_extension(
        filename
    )

    # ------------------------------------------------------------------------
    # Convert input to bytes
    # ------------------------------------------------------------------------

    if isinstance(file, bytes):
        data = file

    else:
        try:
            data = file.read()

        except AttributeError as exc:
            raise DataLoadError(
                "The supplied file object does not provide "
                "a readable interface."
            ) from exc

        except Exception as exc:
            raise DataLoadError(
                f"Could not read uploaded file: {exc}"
            ) from exc

    # ------------------------------------------------------------------------
    # Validate binary data
    # ------------------------------------------------------------------------

    if not isinstance(data, bytes):
        raise DataLoadError(
            "The uploaded file did not return binary data."
        )

    # ------------------------------------------------------------------------
    # Validate file size
    # ------------------------------------------------------------------------

    validate_upload_size(
        len(data)
    )

    # ------------------------------------------------------------------------
    # Reject zero-byte files
    # ------------------------------------------------------------------------

    if not data:
        raise DataLoadError(
            "The uploaded file is empty."
        )

    # ------------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------------

    frame = _read_bytes(
        filename,
        data,
    )

    # ------------------------------------------------------------------------
    # Reject datasets with no rows
    # ------------------------------------------------------------------------

    if frame.empty:
        raise DataLoadError(
            "The file was parsed successfully but "
            "contains no rows."
        )

    # ------------------------------------------------------------------------
    # IMPORTANT: enforce maximum row count
    # ------------------------------------------------------------------------

    _validate_row_count(
        frame,
        filename,
    )

    # ------------------------------------------------------------------------
    # Log successful load
    # ------------------------------------------------------------------------

    logger.info(
        "Loaded dataset '%s': %s rows × %s columns.",
        filename,
        len(frame),
        len(frame.columns),
    )

    return frame


# ============================================================================
# Remote datasets
# ============================================================================


def load_csv_url(
    url: str,
) -> pd.DataFrame:
    """Load a supported dataset from a remote HTTP(S) URL.

    Despite the historical function name, the loader supports:

    - CSV
    - TSV
    - XLSX
    - XLS

    when the URL contains an appropriate filename extension.

    Args:
        url:
            Remote HTTP(S) dataset URL.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        DataLoadError:
            If the URL is invalid, inaccessible, oversized, malformed,
            empty, or exceeds the row limit.
    """

    # ------------------------------------------------------------------------
    # Validate URL
    # ------------------------------------------------------------------------

    url = url.strip()

    validate_url(
        url
    )

    # ------------------------------------------------------------------------
    # Determine file type
    # ------------------------------------------------------------------------

    filename = _get_url_filename(
        url
    )

    validate_file_extension(
        filename
    )

    logger.info(
        "Loading dataset from URL: %s",
        url,
    )

    # ------------------------------------------------------------------------
    # Request remote resource
    # ------------------------------------------------------------------------

    try:
        response = requests.get(
            url,
            timeout=LIMITS.url_timeout_seconds,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout as exc:
        raise DataLoadError(
            f"Request timed out after "
            f"{LIMITS.url_timeout_seconds} seconds."
        ) from exc

    except requests.exceptions.TooManyRedirects as exc:
        raise DataLoadError(
            "The URL redirected too many times."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        status_code = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )

        raise DataLoadError(
            f"The server returned HTTP status "
            f"{status_code}."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise DataLoadError(
            "Could not connect to the data source."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise DataLoadError(
            f"Could not fetch data from URL: {exc}"
        ) from exc

    # ------------------------------------------------------------------------
    # Validate response size
    # ------------------------------------------------------------------------

    data = response.content

    validate_upload_size(
        len(data)
    )

    # ------------------------------------------------------------------------
    # Reject empty response
    # ------------------------------------------------------------------------

    if not data:
        raise DataLoadError(
            "The URL returned an empty response."
        )

    # ------------------------------------------------------------------------
    # Parse response
    # ------------------------------------------------------------------------

    frame = _read_bytes(
        filename,
        data,
    )

    # ------------------------------------------------------------------------
    # Reject empty dataset
    # ------------------------------------------------------------------------

    if frame.empty:
        raise DataLoadError(
            "The URL returned data but it contains "
            "no rows."
        )

    # ------------------------------------------------------------------------
    # IMPORTANT: enforce maximum row count
    # ------------------------------------------------------------------------

    _validate_row_count(
        frame,
        url,
    )

    # ------------------------------------------------------------------------
    # Log successful load
    # ------------------------------------------------------------------------

    logger.info(
        "Loaded remote dataset: %s rows × %s columns.",
        len(frame),
        len(frame.columns),
    )

    return frame


# ============================================================================
# Unified ingestion interface
# ============================================================================


def load_data(
    uploaded_file=None,
    url: str = "",
) -> LoadedDataset:
    """Load a dataset from either an uploaded file or a URL.

    Exactly one source must be provided.

    Args:
        uploaded_file:
            A Streamlit UploadedFile or compatible binary
            file-like object.

        url:
            Remote HTTP(S) dataset URL.

    Returns:
        LoadedDataset containing the DataFrame and source name.

    Raises:
        DataLoadError:
            If both sources or neither source are provided.
    """

    has_upload = (
        uploaded_file is not None
    )

    has_url = bool(
        url and url.strip()
    )

    # ------------------------------------------------------------------------
    # Prevent ambiguous input
    # ------------------------------------------------------------------------

    if has_upload and has_url:
        raise DataLoadError(
            "Provide either a file upload or a URL, "
            "not both."
        )

    # ------------------------------------------------------------------------
    # Uploaded file
    # ------------------------------------------------------------------------

    if has_upload:

        filename = getattr(
            uploaded_file,
            "name",
            None,
        )

        if not filename:
            raise DataLoadError(
                "The uploaded file does not have "
                "a valid filename."
            )

        logger.info(
            "Loading uploaded file: %s",
            filename,
        )

        frame = load_dataframe(
            uploaded_file,
            filename,
        )

        return LoadedDataset(
            frame=frame,
            source_name=filename,
        )

    # ------------------------------------------------------------------------
    # Remote URL
    # ------------------------------------------------------------------------

    if has_url:

        clean_url = url.strip()

        frame = load_csv_url(
            clean_url
        )

        return LoadedDataset(
            frame=frame,
            source_name=clean_url,
        )

    # ------------------------------------------------------------------------
    # Nothing supplied
    # ------------------------------------------------------------------------

    raise DataLoadError(
        "Provide a file upload or a URL to continue."
    )
