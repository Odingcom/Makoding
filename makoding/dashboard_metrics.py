"""Dataset quality metrics for Makoding DataLab Pro."""

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DashboardMetrics:
    rows: int
    columns: int
    missing_cells: int
    total_cells: int
    completeness_pct: float
    duplicate_rows: int
    numeric_columns: int
    categorical_columns: int
    other_columns: int
    columns_with_missing: int


def calculate_dashboard_metrics(frame: pd.DataFrame) -> DashboardMetrics:
    """Calculate dataset statistics without changing the input."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    rows, columns = frame.shape
    total_cells = rows * columns
    missing_cells = int(frame.isna().sum().sum())

    numeric_count = len(
        frame.select_dtypes(include="number").columns
    )

    categorical_count = len(
        frame.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns
    )

    completeness = (
        round(
            100 * (total_cells - missing_cells) / total_cells,
            2,
        )
        if total_cells
        else 100.0
    )

    return DashboardMetrics(
        rows=rows,
        columns=columns,
        missing_cells=missing_cells,
        total_cells=total_cells,
        completeness_pct=completeness,
        duplicate_rows=int(frame.duplicated().sum()) if rows else 0,
        numeric_columns=numeric_count,
        categorical_columns=categorical_count,
        other_columns=columns - numeric_count - categorical_count,
        columns_with_missing=int(frame.isna().any().sum()),
    )


def missingness_by_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing values by column, sorted highest first."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    missing = frame.isna().sum()

    result = pd.DataFrame({
        "Column": [str(c) for c in frame.columns],
        "Missing cells": missing.to_numpy(dtype=int),
        "Missing %": (
            (missing / len(frame) * 100).round(2)
            if len(frame)
            else missing.astype(float)
        ).to_numpy(),
    })

    return result.sort_values(
        "Missing cells",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
