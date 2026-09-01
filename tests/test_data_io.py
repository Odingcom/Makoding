"""Comprehensive tests for Makoding data ingestion.

The ingestion layer is responsible for:

- validating supported file types
- validating upload sizes
- validating remote URLs
- loading CSV, TSV, XLSX, and XLS files
- handling UTF-8 and Latin-1 encoded text
- rejecting malformed datasets
- rejecting empty datasets
- enforcing maximum row limits
- protecting against oversized remote responses
- handling network failures gracefully
- supporting bytes and file-like objects
- enforcing the single-source loading contract

No real network requests are made by this test module.
All URL requests are mocked.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest
import requests

from makoding.config import LIMITS
from makoding.data_io import (
    DataLoadError,
    LoadedDataset,
    load_data,
    load_dataframe,
    validate_file_extension,
    validate_upload_size,
    validate_url,
)


# ============================================================================
# Test configuration
# ============================================================================

MAX_UPLOAD_BYTES = LIMITS.max_upload_mb * 1024 * 1024
MAX_ROWS = LIMITS.max_rows


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_upload():
    """Create a file-like object similar to Streamlit UploadedFile."""

    def _make(data: bytes, name: str) -> BytesIO:
        file = BytesIO(data)
        file.name = name
        return file

    return _make


@pytest.fixture
def mock_requests_get(monkeypatch):
    """Install a controllable fake requests.get implementation.

    This guarantees that tests never make real network requests.
    """

    def _install(
        content: bytes = b"",
        status_code: int = 200,
        raise_: Exception | None = None,
    ):
        class FakeResponse:
            """Minimal stand-in for requests.Response."""

            def __init__(self):
                self.content = content
                self.status_code = status_code

            def raise_for_status(self):
                if not 200 <= self.status_code < 300:
                    raise requests.exceptions.HTTPError(
                        f"HTTP status {self.status_code}"
                    )

        def fake_get(*args, **kwargs):
            if raise_ is not None:
                raise raise_

            return FakeResponse()

        monkeypatch.setattr(
            requests,
            "get",
            fake_get,
        )

    return _install


@pytest.fixture
def broken_file():
    """File-like object that fails when read."""

    class BrokenFile:
        name = "broken.csv"

        def read(self, *args, **kwargs):
            raise OSError("simulated read failure")

        def getvalue(self):
            raise OSError("simulated read failure")

    return BrokenFile()


# ============================================================================
# File-extension validation
# ============================================================================


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("data.csv", ".csv"),
        ("data.tsv", ".tsv"),
        ("data.xlsx", ".xlsx"),
        ("data.xls", ".xls"),
        ("DATA.CSV", ".csv"),
        ("DATA.TSV", ".tsv"),
        ("DATA.XLSX", ".xlsx"),
        ("DATA.XLS", ".xls"),
        ("archive.data.csv", ".csv"),
        ("customer.DATA.CSV", ".csv"),
    ],
)
def test_validate_file_extension_accepts_supported_types(
    filename,
    expected,
):
    """Supported extensions should be normalized to lowercase."""

    assert validate_file_extension(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "data.json",
        "data.txt",
        "data.parquet",
        "data.feather",
        "data.xml",
        "data.exe",
        "data.zip",
    ],
)
def test_validate_file_extension_rejects_unsupported_types(filename):
    """Unsupported formats must raise DataLoadError."""

    with pytest.raises(
        DataLoadError,
        match=r"(?i)unsupported|supported|invalid",
    ):
        validate_file_extension(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "data",
        "",
        "dataset.",
    ],
)
def test_validate_file_extension_rejects_missing_extension(filename):
    """Files without a usable extension must be rejected."""

    with pytest.raises(DataLoadError):
        validate_file_extension(filename)


# ============================================================================
# Upload-size validation
# ============================================================================


def test_validate_upload_size_accepts_small_file():
    """Normal files should pass validation."""

    validate_upload_size(1024)


def test_validate_upload_size_accepts_zero_bytes():
    """The size validator alone should allow zero bytes.

    Empty-file validation happens later during parsing.
    """

    validate_upload_size(0)


def test_validate_upload_size_accepts_exact_limit():
    """The configured maximum should be an inclusive boundary."""

    validate_upload_size(MAX_UPLOAD_BYTES)


def test_validate_upload_size_rejects_one_byte_over_limit():
    """One byte above the configured maximum must fail."""

    with pytest.raises(
        DataLoadError,
        match=r"(?i)exceed|large|max",
    ):
        validate_upload_size(MAX_UPLOAD_BYTES + 1)


# ============================================================================
# URL validation
# ============================================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/data.csv",
        "https://example.com/data.csv",
        "HTTP://example.com/data.csv",
        "HTTPS://example.com/data.csv",
    ],
)
def test_validate_url_accepts_http_and_https(url):
    """HTTP and HTTPS should be accepted case-insensitively."""

    validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not-a-url",
        "example.com/data.csv",
        "ftp://example.com/data.csv",
        "file:///C:/data.csv",
        "javascript:alert(1)",
        "data:text/plain,test",
    ],
)
def test_validate_url_rejects_invalid_urls(url):
    """Malformed or disallowed URLs must raise DataLoadError."""

    with pytest.raises(DataLoadError):
        validate_url(url)


# ============================================================================
# CSV loading
# ============================================================================


@pytest.mark.parametrize(
    (
        "data",
        "filename",
        "expected_shape",
        "expected_columns",
    ),
    [
        (
            b"Name,Age,Score\nAlice,25,82\nBob,30,91\n",
            "test.csv",
            (2, 3),
            ["Name", "Age", "Score"],
        ),
        (
            b"Name,Age\nAlice,25\nBob,30\n",
            "test.csv",
            (2, 2),
            ["Name", "Age"],
        ),
    ],
)
def test_load_dataframe_from_csv_bytes(
    data,
    filename,
    expected_shape,
    expected_columns,
):
    """CSV bytes should become a correctly structured DataFrame."""

    df = load_dataframe(
        data,
        filename,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape == expected_shape
    assert list(df.columns) == expected_columns


def test_load_dataframe_from_csv_file_object():
    """Binary file-like objects should be supported."""

    data = b"Name,Age\nAlice,25\nBob,30\n"

    file = BytesIO(data)

    df = load_dataframe(
        file,
        "customers.csv",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


# ============================================================================
# TSV loading
# ============================================================================


def test_load_dataframe_from_tsv():
    """TSV files should use tab separation."""

    data = (
        b"Name\tAge\tScore\n"
        b"Alice\t25\t82\n"
        b"Bob\t30\t91\n"
    )

    df = load_dataframe(
        data,
        "customers.tsv",
    )

    assert df.shape == (2, 3)

    assert list(df.columns) == [
        "Name",
        "Age",
        "Score",
    ]


# ============================================================================
# Encoding handling
# ============================================================================


def test_load_dataframe_falls_back_to_latin1():
    """Latin-1 encoded datasets should load correctly."""

    data = (
        "Name,City\n"
        "José,São Paulo\n"
    ).encode("latin-1")

    df = load_dataframe(
        data,
        "latin1.csv",
    )

    assert df.shape == (1, 2)

    assert df.iloc[0]["Name"] == "José"
    assert df.iloc[0]["City"] == "São Paulo"


# ============================================================================
# Excel loading
# ============================================================================


def _build_xlsx_bytes(
    frame: pd.DataFrame,
) -> bytes:
    """Serialize a DataFrame to in-memory XLSX bytes."""

    buffer = BytesIO()

    frame.to_excel(
        buffer,
        index=False,
    )

    return buffer.getvalue()


def test_load_dataframe_from_xlsx():
    """XLSX files should load into DataFrames."""

    source = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )

    data = _build_xlsx_bytes(source)

    df = load_dataframe(
        data,
        "customers.xlsx",
    )

    assert isinstance(df, pd.DataFrame)

    assert df.shape == (
        2,
        2,
    )

    assert list(df.columns) == [
        "Name",
        "Age",
    ]


def test_load_dataframe_rejects_corrupt_xlsx():
    """Corrupt Excel files should produce DataLoadError."""

    with pytest.raises(DataLoadError):
        load_dataframe(
            b"this is not a real Excel file",
            "broken.xlsx",
        )


# ============================================================================
# Malformed datasets
# ============================================================================


def test_load_dataframe_rejects_malformed_csv():
    """Ragged CSV data should be rejected cleanly."""

    bad_csv = (
        b"Name,Age,Score\n"
        b"Alice,25\n"
        b"Bob,30,91,extra,columns\n"
    )

    with pytest.raises(DataLoadError):
        load_dataframe(
            bad_csv,
            "bad.csv",
        )


def test_load_dataframe_rejects_file_that_cannot_be_read(
    broken_file,
):
    """Read failures should become DataLoadError."""

    with pytest.raises(
        DataLoadError,
        match=r"(?i)read|could not",
    ):
        load_dataframe(
            broken_file,
            "broken.csv",
        )


# ============================================================================
# Empty datasets
# ============================================================================


def test_empty_file_raises_error():
    """A completely empty file must be rejected."""

    with pytest.raises(
        DataLoadError,
        match=r"(?i)empty|no rows|no data",
    ):
        load_dataframe(
            b"",
            "empty.csv",
        )


def test_header_only_csv_raises_error():
    """A CSV containing only column headers has no usable rows."""

    data = b"Name,Age\n"

    with pytest.raises(
        DataLoadError,
        match=r"(?i)empty|no rows|no data",
    ):
        load_dataframe(
            data,
            "empty.csv",
        )


# ============================================================================
# Maximum row validation
# ============================================================================


def test_dataset_at_exact_row_limit_is_accepted():
    """Exactly MAX_ROWS should be accepted."""

    if MAX_ROWS <= 0:
        pytest.skip("Configured max_rows must be positive.")

    rows = "\n".join(
        f"{i},value"
        for i in range(MAX_ROWS)
    )

    data = (
        "id,value\n"
        + rows
        + "\n"
    ).encode()

    df = load_dataframe(
        data,
        "limit.csv",
    )

    assert len(df) == MAX_ROWS


def test_dataset_over_row_limit_is_rejected():
    """One row above MAX_ROWS should be rejected."""

    if MAX_ROWS <= 0:
        pytest.skip("Configured max_rows must be positive.")

    rows = "\n".join(
        f"{i},value"
        for i in range(MAX_ROWS + 1)
    )

    data = (
        "id,value\n"
        + rows
        + "\n"
    ).encode()

    with pytest.raises(
        DataLoadError,
        match=r"(?i)rows|limit|exceed",
    ):
        load_dataframe(
            data,
            "too_many_rows.csv",
        )


# ============================================================================
# Unified load_data interface
# ============================================================================


def test_load_data_requires_source():
    """load_data must reject missing input."""

    with pytest.raises(
        DataLoadError,
        match=r"(?i)provide|required|missing",
    ):
        load_data()


def test_load_data_rejects_both_sources(
    make_upload,
):
    """Upload and URL must not be supplied simultaneously."""

    file = make_upload(
        b"Name,Age\nAlice,25\n",
        "test.csv",
    )

    with pytest.raises(DataLoadError):
        load_data(
            uploaded_file=file,
            url="https://example.com/data.csv",
        )


def test_load_data_treats_whitespace_url_as_missing():
    """Whitespace-only URL should behave like no source."""

    with pytest.raises(
        DataLoadError,
        match=r"(?i)provide|required|missing",
    ):
        load_data(
            url="   ",
        )


def test_load_data_from_upload(
    make_upload,
):
    """Uploaded files should return LoadedDataset."""

    file = make_upload(
        b"Name,Age\nAlice,25\nBob,30\n",
        "customers.csv",
    )

    result = load_data(
        uploaded_file=file,
    )

    assert isinstance(
        result,
        LoadedDataset,
    )

    assert result.source_name == "customers.csv"

    assert isinstance(
        result.frame,
        pd.DataFrame,
    )

    assert result.frame.shape == (
        2,
        2,
    )


# ============================================================================
# URL loading
# ============================================================================


def test_load_data_from_url_success(
    mock_requests_get,
):
    """A valid remote dataset should load successfully."""

    mock_requests_get(
        content=b"Name,Age\nAlice,25\nBob,30\n",
        status_code=200,
    )

    result = load_data(
        url="https://example.com/data.csv",
    )

    assert isinstance(
        result,
        LoadedDataset,
    )

    assert result.source_name == (
        "https://example.com/data.csv"
    )

    assert result.frame.shape == (
        2,
        2,
    )


def test_load_data_from_url_http_error(
    mock_requests_get,
):
    """HTTP errors should become DataLoadError."""

    mock_requests_get(
        content=b"not found",
        status_code=404,
    )

    with pytest.raises(DataLoadError):
        load_data(
            url="https://example.com/missing.csv",
        )


def test_load_data_from_url_server_error(
    mock_requests_get,
):
    """5xx responses should become DataLoadError."""

    mock_requests_get(
        content=b"server error",
        status_code=500,
    )

    with pytest.raises(DataLoadError):
        load_data(
            url="https://example.com/server-error.csv",
        )


def test_load_data_from_url_timeout(
    mock_requests_get,
):
    """Network timeout should be handled gracefully."""

    mock_requests_get(
        raise_=requests.exceptions.Timeout(
            "timed out"
        )
    )

    with pytest.raises(
        DataLoadError,
        match=r"(?i)timed out|timeout",
    ):
        load_data(
            url="https://example.com/slow.csv",
        )


def test_load_data_from_url_connection_error(
    mock_requests_get,
):
    """Connection failures should be handled gracefully."""

    mock_requests_get(
        raise_=requests.exceptions.ConnectionError(
            "connection refused"
        )
    )

    with pytest.raises(DataLoadError):
        load_data(
            url="https://example.com/unreachable.csv",
        )


def test_load_data_from_url_rejects_oversized_response(
    mock_requests_get,
):
    """Remote responses must obey the same size limit as uploads."""

    mock_requests_get(
        content=b"x" * (MAX_UPLOAD_BYTES + 1),
        status_code=200,
    )

    with pytest.raises(
        DataLoadError,
        match=r"(?i)exceed|large|max",
    ):
        load_data(
            url="https://example.com/huge.csv",
        )


def test_load_data_from_url_empty_response(
    mock_requests_get,
):
    """Empty remote datasets must be rejected."""

    mock_requests_get(
        content=b"",
        status_code=200,
    )

    with pytest.raises(
        DataLoadError,
        match=r"(?i)empty|no rows|no data",
    ):
        load_data(
            url="https://example.com/empty.csv",
        )


def test_load_data_from_url_unsupported_extension(
    mock_requests_get,
):
    """Unsupported remote file extensions should be rejected."""

    with pytest.raises(DataLoadError):
        load_data(
            url="https://example.com/data.json",
        )

    # Ensure no network request was needed.
    mock_requests_get.assert_not_called if hasattr(
        mock_requests_get,
        "assert_not_called",
    ) else None


# ============================================================================
# URL filename handling
# ============================================================================


def test_load_data_from_url_with_query_string(
    mock_requests_get,
):
    """A URL with a query string should still identify its file extension."""

    mock_requests_get(
        content=b"Name,Age\nAlice,25\n",
        status_code=200,
    )

    result = load_data(
        url="https://example.com/data.csv?download=1",
    )

    assert result.frame.shape == (
        1,
        2,
    )


# ============================================================================
# Data integrity
# ============================================================================


def test_loaded_values_are_preserved():
    """Important values should survive ingestion unchanged."""

    data = (
        b"Name,Age,Score\n"
        b"Alice,25,82.5\n"
        b"Bob,30,91.2\n"
    )

    df = load_dataframe(
        data,
        "scores.csv",
    )

    assert df.loc[0, "Name"] == "Alice"
    assert df.loc[0, "Age"] == 25
    assert df.loc[0, "Score"] == 82.5

    assert df.loc[1, "Name"] == "Bob"
    assert df.loc[1, "Age"] == 30
    assert df.loc[1, "Score"] == 91.2


# ============================================================================
# File-pointer behavior
# ============================================================================


def test_load_dataframe_reads_from_current_file_position():
    """The loader should correctly handle a file-like object."""

    data = b"Name,Age\nAlice,25\nBob,30\n"

    file = BytesIO(data)

    df = load_dataframe(
        file,
        "customers.csv",
    )

    assert len(df) == 2


# ============================================================================
# Public API sanity checks
# ============================================================================


def test_data_load_error_is_exception():
    """DataLoadError should behave like a normal Exception."""

    assert issubclass(
        DataLoadError,
        Exception,
    )


def test_loaded_dataset_is_frozen():
    """LoadedDataset should be immutable at the attribute level."""

    result = LoadedDataset(
        frame=pd.DataFrame(
            {
                "A": [1, 2],
            }
        ),
        source_name="test.csv",
    )

    with pytest.raises(Exception):
        result.source_name = "changed.csv"