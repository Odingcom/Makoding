"""Explicit, non-mutating DataLab Pro data preparation operations.

This module has no Streamlit dependencies and can be tested independently.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


MISSING_TOKENS = frozenset({"", "na", "n/a", "null", "none", "missing", "nan", "?"})


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate column names must be resolved before cleaning.")


def summarize_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Return data-quality measurements without modifying the dataset."""
    _validate_frame(frame)
    rows, cols = frame.shape
    missing = int(frame.isna().to_numpy().sum())
    try:
        duplicates = int(frame.duplicated().sum())
    except TypeError:
        duplicates = None  # Some object-valued cells cannot be hashed.
    cells = rows * cols
    return {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing,
        "missing_pct": round((100 * missing / cells) if cells else 0.0, 2),
        "duplicate_rows": duplicates,
    }


def profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Produce a column-level audit table for the UI and exports."""
    _validate_frame(frame)
    rows = max(len(frame), 1)
    return pd.DataFrame(
        [
            {
                "Column": str(column),
                "Data type": str(frame[column].dtype),
                "Missing": int(frame[column].isna().sum()),
                "Missing %": round(100 * frame[column].isna().sum() / rows, 2),
                "Distinct (non-null)": int(frame[column].nunique(dropna=True)),
            }
            for column in frame.columns
        ]
    )


def apply_clean_step(
    frame: pd.DataFrame,
    operation: str,
    *,
    columns: Sequence[str] | None = None,
    strategy: str | None = None,
    constant: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a *new* frame and describe the proposed cleaning operation.

    No operation edits ``frame`` in place. Dropped observations retain their
    original index labels so downstream users can identify their provenance.
    """
    _validate_frame(frame)
    selected = list(columns) if columns is not None else []
    if len(selected) != len(set(selected)):
        raise ValueError("Select each column only once.")
    absent = [column for column in selected if column not in frame.columns]
    if absent:
        raise ValueError(f"Unknown columns: {absent!r}")
    before = summarize_quality(frame)
    result = frame.copy(deep=True)

    if operation == "remove_duplicates":
        result = result.drop_duplicates(keep="first").copy()

    elif operation == "trim_whitespace":
        if not selected:
            raise ValueError("Choose at least one text column.")
        for column in selected:
            if not (pd.api.types.is_object_dtype(result[column]) or
                    pd.api.types.is_string_dtype(result[column])):
                raise ValueError(f"{column!r} must be a text column.")
            result[column] = result[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    elif operation == "standardize_missing":
        if not selected:
            raise ValueError("Choose at least one text column.")
        for column in selected:
            if not (pd.api.types.is_object_dtype(result[column]) or
                    pd.api.types.is_string_dtype(result[column])):
                raise ValueError(f"{column!r} must be a text column.")
            result[column] = result[column].map(
                lambda value: pd.NA
                if isinstance(value, str) and value.strip().lower() in MISSING_TOKENS
                else value
            )

    elif operation == "fill_missing":
        if not selected:
            raise ValueError("Choose at least one column to fill.")
        if strategy not in {"mean", "median", "mode", "constant"}:
            raise ValueError("Choose a supported filling strategy.")
        for column in selected:
            series = result[column]
            if not series.isna().any():
                continue
            if strategy in {"mean", "median"}:
                if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                    raise ValueError(f"{column!r} requires a numeric dtype for {strategy}.")
                non_null = series.dropna()
                if non_null.empty:
                    raise ValueError(f"{column!r} contains no values for {strategy} imputation.")
                value = non_null.mean() if strategy == "mean" else non_null.median()
                if not np.isfinite(value):
                    raise ValueError(f"{column!r} has a non-finite {strategy}.")
            elif strategy == "mode":
                values = series.mode(dropna=True)
                if values.empty:
                    raise ValueError(f"{column!r} has no observed mode.")
                value = values.iloc[0]
            else:
                if constant is None or constant == "":
                    raise ValueError("Provide a non-empty replacement value.")
                if not (pd.api.types.is_object_dtype(series) or
                        pd.api.types.is_string_dtype(series)):
                    raise ValueError("Constant text replacement supports text columns only.")
                value = constant
            result[column] = series.fillna(value)

    elif operation == "drop_missing_rows":
        if not selected:
            raise ValueError("Choose columns to check for missing values.")
        result = result.dropna(subset=selected).copy()

    elif operation == "drop_columns":
        if not selected:
            raise ValueError("Choose at least one column to remove.")
        result = result.drop(columns=selected).copy()

    else:
        raise ValueError(f"Unsupported cleaning operation: {operation!r}")

    if result.empty or result.shape[1] == 0:
        raise ValueError("Operation would leave no usable rows or columns. Nothing was changed.")
    after = summarize_quality(result)
    return result, {
        "operation": operation,
        "columns": selected,
        "strategy": strategy,
        "constant": constant if strategy == "constant" else None,
        "before": before,
        "after": after,
        "removed_rows": before["rows"] - after["rows"],
        "removed_columns": before["columns"] - after["columns"],
        "missing_reduced": before["missing_cells"] - after["missing_cells"],
    }
