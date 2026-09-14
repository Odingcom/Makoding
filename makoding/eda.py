"""Exploratory data analysis utilities for Makoding.

This module provides framework-independent, reusable utilities for
profiling pandas DataFrames. Functions are designed to be deterministic,
non-mutating, dependency-light, and suitable for notebooks, pipelines,
APIs, and Streamlit applications.

The public API covers:

- dataset overview
- missing-value analysis
- numeric statistics
- categorical statistics
- correlation analysis
- data-type inspection
- duplicate-row analysis
- cardinality analysis
- IQR-based outlier detection
- memory usage
- consolidated EDA reporting
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "dataset_overview",
    "missing_values_summary",
    "numeric_summary",
    "categorical_summary",
    "correlation_matrix",
    "data_types_summary",
    "duplicate_rows_summary",
    "unique_values_summary",
    "outlier_summary",
    "memory_usage_summary",
    "generate_eda_report",
]


CorrelationMethod = Literal["pearson", "kendall", "spearman"]
OutlierMethod = Literal["iqr"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_dataframe(frame: Any) -> None:
    """Validate that ``frame`` is a pandas DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be a pandas DataFrame, got {type(frame).__name__!r}"
        )


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with predictable columns."""
    return pd.DataFrame(columns=columns)


def _numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric columns from a DataFrame."""
    return frame.select_dtypes(include="number")


def _categorical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return object, category, and string columns."""
    return frame.select_dtypes(
        include=["object", "category", "string"]
    )


# ---------------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------------


def dataset_overview(frame: pd.DataFrame) -> dict[str, int]:
    """Return high-level structural information about a dataset.

    Returns
    -------
    dict[str, int]
        Contains:

        - ``rows``
        - ``columns``
        - ``duplicate_rows``
        - ``total_missing_cells``
    """
    _validate_dataframe(frame)

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "total_missing_cells": int(frame.isna().sum().sum()),
    }


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def missing_values_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages by column.

    Results are sorted by missing percentage in descending order.
    """
    _validate_dataframe(frame)

    if len(frame.columns) == 0:
        return _empty_frame(
            ["missing_count", "missing_percentage"]
        )

    total_rows = len(frame)
    missing_count = frame.isna().sum().astype(int)

    if total_rows == 0:
        missing_percentage = pd.Series(
            0.0,
            index=frame.columns,
            dtype=float,
        )
    else:
        missing_percentage = (
            missing_count / total_rows * 100
        ).astype(float)

    result = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        },
        index=frame.columns,
    )

    return result.sort_values(
        "missing_percentage",
        ascending=False,
    )


# ---------------------------------------------------------------------------
# Numeric analysis
# ---------------------------------------------------------------------------


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric variables.

    Includes standard descriptive statistics plus skewness and kurtosis.
    """
    _validate_dataframe(frame)

    numeric = _numeric_columns(frame)

    if numeric.shape[1] == 0:
        return pd.DataFrame()

    summary = numeric.describe()

    summary.loc["skewness"] = numeric.skew()
    summary.loc["kurtosis"] = numeric.kurt()

    return summary


# ---------------------------------------------------------------------------
# Categorical analysis
# ---------------------------------------------------------------------------


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for categorical variables.

    The returned DataFrame is indexed by variable name and contains:

    - ``count``
    - ``unique``
    - ``top``
    - ``freq``
    """
    _validate_dataframe(frame)

    categorical = _categorical_columns(frame)

    if categorical.shape[1] == 0:
        return _empty_frame(
            ["count", "unique", "top", "freq"]
        )

    summary = categorical.describe().T

    expected_columns = [
        "count",
        "unique",
        "top",
        "freq",
    ]

    for column in expected_columns:
        if column not in summary.columns:
            summary[column] = pd.NA

    return summary[expected_columns]


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def correlation_matrix(
    frame: pd.DataFrame,
    method: CorrelationMethod = "pearson",
) -> pd.DataFrame:
    """Return a correlation matrix for numeric variables.

    Parameters
    ----------
    method:
        ``"pearson"``, ``"spearman"``, or ``"kendall"``.

    Returns
    -------
    pandas.DataFrame
        Empty when fewer than two numeric variables are available.
    """
    _validate_dataframe(frame)

    valid_methods = {
        "pearson",
        "spearman",
        "kendall",
    }

    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {sorted(valid_methods)}, "
            f"got {method!r}"
        )

    numeric = _numeric_columns(frame)

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    return numeric.corr(method=method)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


def data_types_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return pandas and Python type information for every column."""
    _validate_dataframe(frame)

    def first_non_null_type(series: pd.Series) -> str:
        non_null = series.dropna()

        if non_null.empty:
            return "unknown"

        return type(non_null.iloc[0]).__name__

    return pd.DataFrame(
        {
            "data_type": frame.dtypes.astype(str),
            "python_type": [
                first_non_null_type(frame[column])
                for column in frame.columns
            ],
        },
        index=frame.columns,
    )


# ---------------------------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------------------------


def duplicate_rows_summary(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Return duplicate-row count and percentage."""
    _validate_dataframe(frame)

    total_rows = len(frame)
    duplicate_count = int(frame.duplicated().sum())

    duplicate_percentage = (
        duplicate_count / total_rows * 100
        if total_rows > 0
        else 0.0
    )

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": float(
            duplicate_percentage
        ),
    }


# ---------------------------------------------------------------------------
# Cardinality / uniqueness
# ---------------------------------------------------------------------------


def unique_values_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return unique-value counts and percentages by column.

    Missing values are excluded from the unique-value count.
    """
    _validate_dataframe(frame)

    if len(frame.columns) == 0:
        return _empty_frame(
            ["unique_count", "unique_percentage"]
        )

    total_rows = len(frame)

    unique_count = frame.nunique(
        dropna=True
    ).astype(int)

    if total_rows == 0:
        unique_percentage = pd.Series(
            0.0,
            index=frame.columns,
            dtype=float,
        )
    else:
        unique_percentage = (
            unique_count / total_rows * 100
        ).astype(float)

    return pd.DataFrame(
        {
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
        },
        index=frame.columns,
    )


# ---------------------------------------------------------------------------
# Outlier analysis
# ---------------------------------------------------------------------------


def outlier_summary(
    frame: pd.DataFrame,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect numeric outliers using the IQR method.

    Parameters
    ----------
    method:
        Currently supports ``"iqr"``.

    threshold:
        IQR multiplier. The conventional value is ``1.5``.

    Returns
    -------
    pandas.DataFrame
        Indexed by numeric variable with:

        - ``outlier_count``
        - ``outlier_percentage``
        - ``lower_bound``
        - ``upper_bound``
    """
    _validate_dataframe(frame)

    if method != "iqr":
        raise ValueError(
            f"Unsupported method {method!r}; "
            "only 'iqr' is supported"
        )

    if not isinstance(threshold, (int, float)):
        raise TypeError(
            "threshold must be a numeric value"
        )

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(
            "threshold must be a finite value >= 0"
        )

    numeric = _numeric_columns(frame)

    columns = [
        "outlier_count",
        "outlier_percentage",
        "lower_bound",
        "upper_bound",
    ]

    if numeric.shape[1] == 0:
        return _empty_frame(columns)

    total_rows = len(frame)
    records: dict[str, dict[str, Any]] = {}

    for column in numeric.columns:
        series = numeric[column].dropna()

        if series.empty:
            records[column] = {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "lower_bound": np.nan,
                "upper_bound": np.nan,
            }
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        outlier_percentage = (
            outlier_count / total_rows * 100
            if total_rows > 0
            else 0.0
        )

        records[column] = {
            "outlier_count": outlier_count,
            "outlier_percentage": float(
                outlier_percentage
            ),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return pd.DataFrame.from_dict(
        records,
        orient="index",
    )[columns]


# ---------------------------------------------------------------------------
# Memory analysis
# ---------------------------------------------------------------------------


def memory_usage_summary(
    frame: pd.DataFrame,
    deep: bool = True,
) -> pd.DataFrame:
    """Return per-column memory consumption.

    Parameters
    ----------
    deep:
        Whether pandas should inspect object/string values
        for more accurate memory reporting.
    """
    _validate_dataframe(frame)

    usage_bytes = frame.memory_usage(
        index=False,
        deep=deep,
    )

    return pd.DataFrame(
        {
            "memory_bytes": usage_bytes.astype(int),
            "memory_megabytes": (
                usage_bytes / (1024**2)
            ).astype(float),
        },
        index=frame.columns,
    )


# ---------------------------------------------------------------------------
# Complete report
# ---------------------------------------------------------------------------


def generate_eda_report(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a complete structured EDA report.

    The returned dictionary is suitable for notebooks, dashboards,
    logging, serialization, or downstream analytical workflows.
    """
    _validate_dataframe(frame)

    logger.info(
        "Generating EDA report for DataFrame with shape %s",
        frame.shape,
    )

    return {
        "overview": dataset_overview(frame),
        "data_types": data_types_summary(frame),
        "missing_values": missing_values_summary(frame),
        "duplicates": duplicate_rows_summary(frame),
        "unique_values": unique_values_summary(frame),
        "numeric_summary": numeric_summary(frame),
        "categorical_summary": categorical_summary(frame),
        "correlation_matrix": correlation_matrix(frame),
        "outliers": outlier_summary(frame),
        "memory_usage": memory_usage_summary(frame),
    }