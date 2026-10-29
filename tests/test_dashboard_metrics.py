"""Tests for the Makoding dashboard metrics."""

import pandas as pd
import pytest

from makoding.dashboard_metrics import (
    calculate_dashboard_metrics,
    missingness_by_column,
)


def test_quality_metrics_are_correct():
    frame = pd.DataFrame({
        "age": [20, 20, None],
        "group": ["A", "A", "B"],
    })

    result = calculate_dashboard_metrics(frame)

    assert result.rows == 3
    assert result.columns == 2
    assert result.missing_cells == 1
    assert result.completeness_pct == 83.33
    assert result.duplicate_rows == 1
    assert result.numeric_columns == 1
    assert result.categorical_columns == 1
    assert result.columns_with_missing == 1


def test_empty_dataframe_is_safe():
    result = calculate_dashboard_metrics(pd.DataFrame())

    assert result.rows == 0
    assert result.completeness_pct == 100.0
    assert missingness_by_column(pd.DataFrame()).empty


def test_missingness_order_and_rates():
    frame = pd.DataFrame({
        "a": [None, None, 3],
        "b": [1, None, 3],
    })

    result = missingness_by_column(frame)

    assert result["Column"].tolist() == ["a", "b"]
    assert result["Missing %"].tolist() == [66.67, 33.33]


def test_does_not_mutate_input():
    frame = pd.DataFrame({"a": [1, None]})
    original = frame.copy(deep=True)

    calculate_dashboard_metrics(frame)
    missingness_by_column(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_non_dataframe_rejected():
    with pytest.raises(TypeError):
        calculate_dashboard_metrics([1, 2])
