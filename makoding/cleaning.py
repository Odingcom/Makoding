"""Data cleaning utilities for Makoding.

Provides safe, reusable DataFrame cleaning operations including:

- whitespace trimming (values and column names)
- duplicate-row removal
- missing-value handling (numeric median, categorical/boolean mode)
- cleaning statistics and reporting

The module contains no Streamlit dependencies and can therefore be
used by the application, tests, notebooks, APIs, and CLI tools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MISSING_STRATEGIES",
    "CleaningReport",
    "clean_frame",
]


# ============================================================================
# Configuration
# ============================================================================

MISSING_STRATEGIES = (
    "Keep",
    "Drop rows",
    "Fill numeric median",
    "Fill numeric mean",
    "Fill categorical mode",
    "Fill all with mode",
    "Fill missing values",
)


# ============================================================================
# Report
# ============================================================================

@dataclass(frozen=True)
class CleaningReport:
    """Summary of changes made during data cleaning."""

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duplicates_removed: int
    missing_before: int
    missing_after: int
    whitespace_trimmed: int

    @property
    def rows_removed(self) -> int:
        """Number of rows removed during cleaning."""
        return self.rows_before - self.rows_after

    @property
    def missing_removed(self) -> int:
        """Number of missing values resolved."""
        return self.missing_before - self.missing_after

    def as_markdown(self) -> str:
        """Render the report as a short Markdown summary for display."""
        lines = [
            f"- Rows: {self.rows_before:,} -> {self.rows_after:,} "
            f"({self.rows_removed:,} removed)",
            f"- Duplicate rows removed: {self.duplicates_removed:,}",
            f"- Missing values: {self.missing_before:,} -> "
            f"{self.missing_after:,} ({self.missing_removed:,} resolved)",
            f"- Whitespace-trimmed values: {self.whitespace_trimmed:,}",
        ]
        return "\n".join(lines)


# ============================================================================
# Validation
# ============================================================================

def _validate_strategy(strategy: str) -> None:
    """Validate the requested missing-value strategy."""
    if strategy not in MISSING_STRATEGIES:
        supported = ", ".join(MISSING_STRATEGIES)
        raise ValueError(
            f"Invalid missing-value strategy '{strategy}'. "
            f"Supported strategies: {supported}"
        )


def _validate_frame(frame: pd.DataFrame) -> None:
    """Validate the input DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")


# ============================================================================
# Whitespace cleaning
# ============================================================================

def _trim_whitespace(frame: pd.DataFrame) -> int:
    """Trim surrounding whitespace from string values, in place.

    Mutates ``frame`` directly (columns are reassigned on the object
    passed in) -- callers must pass a copy they're willing to modify.

    Uses a per-cell Python loop (via ``.map``) rather than a vectorized
    ``.str.strip()`` call, because object/string columns can contain a
    mix of strings and non-string values (numbers, None, NaN); a
    vectorized ``.str`` accessor would silently turn every non-string
    value into NaN instead of leaving it untouched.

    Returns:
        Number of individual string values whose content actually
        changed (i.e. had leading/trailing whitespace to remove).
        Missing values (NaN/None) are never counted as changed.
    """
    changed = 0

    for column in frame.columns:
        series = frame[column]

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            local_changed = 0

            def _strip(value: object) -> object:
                nonlocal local_changed
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped != value:
                        local_changed += 1
                    return stripped
                return value

            frame[column] = series.map(_strip)
            changed += local_changed

    return changed


def _trim_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with whitespace stripped from column
    names (headers), e.g. ``" name "`` -> ``"name"``.

    Only string column labels are affected; non-string labels (e.g. an
    integer column index) are left untouched.
    """
    renamed = {
        col: col.strip()
        for col in frame.columns
        if isinstance(col, str) and col.strip() != col
    }
    if renamed:
        frame = frame.rename(columns=renamed)
    return frame


# ============================================================================
# Missing-value handling
# ============================================================================

def _fill_numeric(frame: pd.DataFrame, *, use_mean: bool = False) -> None:
    """Fill missing numeric values with the column median (default) or
    mean, in place.

    Mutates ``frame`` directly. Boolean columns are intentionally
    excluded: pandas classifies ``bool`` as a numeric dtype, but a
    boolean column filled with a median/mean (e.g. 1.0) would silently
    lose its boolean type. Boolean columns are handled by
    ``_fill_categorical_mode`` instead.

    Columns that are entirely missing are left unchanged, since their
    median/mean is NaN and there is no non-arbitrary value to fill with.
    """
    numeric_columns = frame.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            continue
        if frame[column].isna().any():
            fill_value = frame[column].mean() if use_mean else frame[column].median()
            if not pd.isna(fill_value):
                frame[column] = frame[column].fillna(fill_value)


def _fill_categorical_mode(frame: pd.DataFrame) -> None:
    """Fill missing categorical/boolean/object values with column modes,
    in place.

    Mutates ``frame`` directly. Numeric (non-boolean) columns are
    skipped -- use ``_fill_numeric`` for those. Columns with no usable
    mode (e.g. entirely missing) are left unchanged.
    """
    for column in frame.columns:
        series = frame[column]

        if not series.isna().any():
            continue

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue

        modes = series.mode(dropna=True)
        if not modes.empty:
            frame[column] = series.fillna(modes.iloc[0])


def _fill_all_with_mode(frame: pd.DataFrame) -> None:
    """Fill every column's missing values with that column's mode,
    in place -- including numeric columns.

    Unlike ``_fill_missing_values``, numeric columns are NOT given a
    median/mean here; every column (numeric or not) is filled with its
    own most frequent value.
    """
    for column in frame.columns:
        series = frame[column]
        if not series.isna().any():
            continue
        modes = series.mode(dropna=True)
        if not modes.empty:
            frame[column] = series.fillna(modes.iloc[0])


def _fill_missing_values(frame: pd.DataFrame) -> None:
    """Fill missing values using sensible column-type defaults, in place.

    Numeric (non-boolean) columns:
        Median

    Boolean, categorical, object, and string columns:
        Mode

    Columns with no usable fill value remain unchanged.
    """
    _fill_numeric(frame, use_mean=False)
    _fill_categorical_mode(frame)


# ============================================================================
# Main cleaning function
# ============================================================================

def clean_frame(
    frame: pd.DataFrame,
    *,
    missing: str = "Keep",
    remove_duplicates: bool = True,
    trim_whitespace: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a DataFrame using configurable operations.

    Args:
        frame:
            Input pandas DataFrame. Never mutated -- a deep copy is
            cleaned and returned.

        missing:
            Missing-value strategy. Supported values (see
            ``MISSING_STRATEGIES``):

            - ``"Keep"`` -- do nothing.
            - ``"Drop rows"`` -- drop any row containing a missing value.
            - ``"Fill numeric median"`` -- fill numeric (non-boolean)
              columns with their median; other columns untouched.
            - ``"Fill numeric mean"`` -- fill numeric (non-boolean)
              columns with their mean; other columns untouched.
            - ``"Fill categorical mode"`` -- fill boolean/categorical/
              object/string columns with their mode; numeric columns
              untouched.
            - ``"Fill all with mode"`` -- fill every column (including
              numeric ones) with that column's mode.
            - ``"Fill missing values"`` -- numeric columns get median,
              everything else gets mode.

        remove_duplicates:
            Remove duplicate rows when True.

        trim_whitespace:
            Trim surrounding whitespace from both string values and
            column names when True.

    Returns:
        A tuple containing:

        ``(cleaned_dataframe, cleaning_report)``

    Raises:
        TypeError:
            If frame is not a pandas DataFrame.

        ValueError:
            If an unsupported missing-value strategy is supplied.
    """
    _validate_frame(frame)
    _validate_strategy(missing)

    # Work on a copy so the caller's DataFrame is never modified.
    cleaned = frame.copy(deep=True)

    rows_before = len(cleaned)
    columns_before = len(cleaned.columns)
    missing_before = int(cleaned.isna().sum().sum())

    duplicates_removed = 0
    whitespace_trimmed = 0

    # ------------------------------------------------------------------------
    # Trim whitespace (values + column names)
    # ------------------------------------------------------------------------
    if trim_whitespace:
        cleaned = _trim_column_names(cleaned)
        if not cleaned.empty:
            whitespace_trimmed = _trim_whitespace(cleaned)

    # ------------------------------------------------------------------------
    # Remove duplicates
    # ------------------------------------------------------------------------
    if remove_duplicates and not cleaned.empty:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
        duplicates_removed = before - len(cleaned)

    # ------------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------------
    if missing == "Drop rows":
        cleaned = cleaned.dropna().reset_index(drop=True)
    elif missing == "Fill numeric median":
        _fill_numeric(cleaned, use_mean=False)
    elif missing == "Fill numeric mean":
        _fill_numeric(cleaned, use_mean=True)
    elif missing == "Fill categorical mode":
        _fill_categorical_mode(cleaned)
    elif missing == "Fill all with mode":
        _fill_all_with_mode(cleaned)
    elif missing == "Fill missing values":
        _fill_missing_values(cleaned)
    # "Keep" deliberately performs no missing-value operation.

    # ------------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------------
    missing_after = int(cleaned.isna().sum().sum())

    return cleaned, CleaningReport(
        rows_before=rows_before,
        rows_after=len(cleaned),
        columns_before=columns_before,
        columns_after=len(cleaned.columns),
        duplicates_removed=duplicates_removed,
        missing_before=missing_before,
        missing_after=missing_after,
        whitespace_trimmed=whitespace_trimmed,
    )