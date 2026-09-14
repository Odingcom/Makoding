"""Tests for the Makoding exploratory data analysis utilities."""

import numpy as np
import pandas as pd
import pytest

from makoding.eda import (
    categorical_summary,
    correlation_matrix,
    data_types_summary,
    dataset_overview,
    duplicate_rows_summary,
    generate_eda_report,
    memory_usage_summary,
    missing_values_summary,
    numeric_summary,
    outlier_summary,
    unique_values_summary,
)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Return a representative DataFrame for EDA tests."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice", None],
            "age": [25, 30, 25, 40],
            "salary": [50000, 60000, np.nan, 70000],
            "department": ["IT", "Finance", "IT", "HR"],
        }
    )


@pytest.fixture
def frame_with_duplicates() -> pd.DataFrame:
    """Return a DataFrame containing exact duplicate rows."""
    return pd.DataFrame(
        {
            "a": [1, 1, 2, 3],
            "b": ["x", "x", "y", "z"],
        }
    )


@pytest.fixture
def frame_with_outliers() -> pd.DataFrame:
    """Return a DataFrame with a single obvious numeric outlier."""
    return pd.DataFrame({"value": [10, 11, 12, 13, 14, 1000]})


# ---------------------------------------------------------------------------
# dataset_overview
# ---------------------------------------------------------------------------

def test_dataset_overview_requires_dataframe():
    """dataset_overview should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        dataset_overview([1, 2, 3])


def test_dataset_overview_returns_expected_information(sample_frame):
    """dataset_overview should report basic dataset dimensions."""
    result = dataset_overview(sample_frame)

    assert isinstance(result, dict)
    assert result["rows"] == 4
    assert result["columns"] == 4


def test_dataset_overview_reports_duplicate_rows(frame_with_duplicates):
    """dataset_overview should count fully duplicated rows."""
    result = dataset_overview(frame_with_duplicates)

    assert result["duplicate_rows"] == 1


def test_dataset_overview_reports_total_missing_cells(sample_frame):
    """dataset_overview should count all missing cells across the frame."""
    result = dataset_overview(sample_frame)

    assert result["total_missing_cells"] == 2


# ---------------------------------------------------------------------------
# missing_values_summary
# ---------------------------------------------------------------------------

def test_missing_values_summary_requires_dataframe():
    """missing_values_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        missing_values_summary("not a dataframe")


def test_missing_values_summary(sample_frame):
    """Missing-value summary should correctly identify missing values."""
    result = missing_values_summary(sample_frame)

    assert isinstance(result, pd.DataFrame)
    assert result.loc["salary", "missing_count"] == 1
    assert result.loc["name", "missing_count"] == 1


def test_missing_values_summary_contains_percentages(sample_frame):
    """Missing-value summary should include missing percentages."""
    result = missing_values_summary(sample_frame)

    assert "missing_percentage" in result.columns
    assert result.loc["salary", "missing_percentage"] == pytest.approx(25.0)


def test_missing_values_summary_is_sorted_descending(sample_frame):
    """Missing-value summary should be sorted by missing percentage, descending."""
    result = missing_values_summary(sample_frame)

    percentages = result["missing_percentage"].tolist()
    assert percentages == sorted(percentages, reverse=True)


# ---------------------------------------------------------------------------
# numeric_summary
# ---------------------------------------------------------------------------

def test_numeric_summary_requires_dataframe():
    """numeric_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        numeric_summary(42)


def test_numeric_summary(sample_frame):
    """Numeric summary should contain numeric columns only."""
    result = numeric_summary(sample_frame)

    assert isinstance(result, pd.DataFrame)
    assert "age" in result.columns
    assert "salary" in result.columns
    assert "name" not in result.columns
    assert "department" not in result.columns


def test_numeric_summary_contains_standard_statistics(sample_frame):
    """Numeric summary should expose standard descriptive statistics."""
    result = numeric_summary(sample_frame)

    expected_statistics = {
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    }

    assert expected_statistics.issubset(result.index)


def test_numeric_summary_contains_skewness_and_kurtosis(sample_frame):
    """Numeric summary should include skewness and kurtosis rows."""
    result = numeric_summary(sample_frame)

    assert "skewness" in result.index
    assert "kurtosis" in result.index


def test_numeric_summary_empty_when_no_numeric_columns():
    """Numeric summary should be empty when there are no numeric columns."""
    frame = pd.DataFrame({"name": ["Alice", "Bob"]})
    result = numeric_summary(frame)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# categorical_summary
# ---------------------------------------------------------------------------

def test_categorical_summary_requires_dataframe():
    """categorical_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        categorical_summary(None)


def test_categorical_summary(sample_frame):
    """Categorical summary should identify categorical columns."""
    result = categorical_summary(sample_frame)

    assert isinstance(result, pd.DataFrame)
    assert "name" in result.index
    assert "department" in result.index
    assert "age" not in result.index


def test_categorical_summary_counts_categories(sample_frame):
    """Categorical summary should report unique values and frequencies."""
    result = categorical_summary(sample_frame)

    assert result.loc["department", "unique"] == 3
    assert result.loc["department", "top"] == "IT"
    assert result.loc["department", "freq"] == 2


def test_categorical_summary_empty_when_no_categorical_columns():
    """Categorical summary should return the expected empty shape."""
    frame = pd.DataFrame({"age": [1, 2, 3]})
    result = categorical_summary(frame)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["count", "unique", "top", "freq"]
    assert result.empty


# ---------------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------------

def test_correlation_matrix_requires_dataframe():
    """correlation_matrix should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        correlation_matrix(object())


def test_correlation_matrix(sample_frame):
    """Correlation matrix should contain numeric variables."""
    result = correlation_matrix(sample_frame)

    assert isinstance(result, pd.DataFrame)
    assert "age" in result.columns
    assert "salary" in result.columns
    assert "age" in result.index
    assert "salary" in result.index


def test_correlation_matrix_is_square(sample_frame):
    """Correlation matrix should have matching rows and columns."""
    result = correlation_matrix(sample_frame)

    assert result.shape[0] == result.shape[1]
    assert list(result.index) == list(result.columns)


def test_correlation_matrix_accepts_alternate_methods(sample_frame):
    """Correlation matrix should support spearman and kendall methods."""
    spearman_result = correlation_matrix(sample_frame, method="spearman")
    kendall_result = correlation_matrix(sample_frame, method="kendall")

    assert isinstance(spearman_result, pd.DataFrame)
    assert isinstance(kendall_result, pd.DataFrame)


def test_correlation_matrix_rejects_invalid_method(sample_frame):
    """Correlation matrix should reject unsupported correlation methods."""
    with pytest.raises(ValueError):
        correlation_matrix(sample_frame, method="not-a-method")


def test_correlation_matrix_empty_with_fewer_than_two_numeric_columns():
    """Correlation matrix should be empty with fewer than two numeric columns."""
    frame = pd.DataFrame({"age": [1, 2, 3], "name": ["a", "b", "c"]})
    result = correlation_matrix(frame)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# data_types_summary
# ---------------------------------------------------------------------------

def test_data_types_summary_requires_dataframe():
    """data_types_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        data_types_summary({"a": [1, 2, 3]})


def test_data_types_summary(sample_frame):
    """Data-type summary should report every column."""
    result = data_types_summary(sample_frame)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_frame.columns)
    assert "name" in result.index
    assert "age" in result.index
    assert "department" in result.index


def test_data_types_summary_includes_python_type(sample_frame):
    """Data-type summary should report the underlying Python type."""
    result = data_types_summary(sample_frame)

    assert "python_type" in result.columns
    assert result.loc["age", "python_type"] in {"int", "int64"}


def test_data_types_summary_handles_all_null_column():
    """Data-type summary should report 'unknown' for all-null columns."""
    frame = pd.DataFrame({"empty": [None, None, None]})
    result = data_types_summary(frame)

    assert result.loc["empty", "python_type"] == "unknown"


# ---------------------------------------------------------------------------
# duplicate_rows_summary
# ---------------------------------------------------------------------------

def test_duplicate_rows_summary_requires_dataframe():
    """duplicate_rows_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        duplicate_rows_summary([1, 2, 3])


def test_duplicate_rows_summary_counts_duplicates(frame_with_duplicates):
    """duplicate_rows_summary should count and calculate the percentage of duplicates."""
    result = duplicate_rows_summary(frame_with_duplicates)

    assert result["duplicate_count"] == 1
    assert result["duplicate_percentage"] == pytest.approx(25.0)


def test_duplicate_rows_summary_handles_empty_dataframe():
    """duplicate_rows_summary should handle an empty DataFrame without error."""
    result = duplicate_rows_summary(pd.DataFrame())

    assert result["duplicate_count"] == 0
    assert result["duplicate_percentage"] == 0.0


# ---------------------------------------------------------------------------
# unique_values_summary
# ---------------------------------------------------------------------------

def test_unique_values_summary_requires_dataframe():
    """unique_values_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        unique_values_summary("nope")


def test_unique_values_summary(sample_frame):
    """unique_values_summary should report unique counts and percentages."""
    result = unique_values_summary(sample_frame)

    assert isinstance(result, pd.DataFrame)
    assert result.loc["department", "unique_count"] == 3
    assert result.loc["department", "unique_percentage"] == pytest.approx(75.0)


def test_unique_values_summary_empty_dataframe():
    """unique_values_summary should handle a DataFrame with no columns."""
    result = unique_values_summary(pd.DataFrame())

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# outlier_summary
# ---------------------------------------------------------------------------

def test_outlier_summary_requires_dataframe():
    """outlier_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        outlier_summary(123)


def test_outlier_summary_detects_outlier(frame_with_outliers):
    """outlier_summary should flag an obvious outlier using the IQR method."""
    result = outlier_summary(frame_with_outliers)

    assert result.loc["value", "outlier_count"] == 1


def test_outlier_summary_rejects_unsupported_method(sample_frame):
    """outlier_summary should reject unsupported detection methods."""
    with pytest.raises(ValueError):
        outlier_summary(sample_frame, method="zscore")


def test_outlier_summary_empty_when_no_numeric_columns():
    """outlier_summary should return the expected empty shape with no numeric data."""
    frame = pd.DataFrame({"name": ["a", "b", "c"]})
    result = outlier_summary(frame)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "outlier_count",
        "outlier_percentage",
        "lower_bound",
        "upper_bound",
    ]
    assert result.empty


def test_outlier_summary_respects_custom_threshold(frame_with_outliers):
    """A larger threshold should be at least as conservative in flagging outliers."""
    strict_result = outlier_summary(frame_with_outliers, threshold=1.5)
    lenient_result = outlier_summary(frame_with_outliers, threshold=3.0)

    assert (
        lenient_result.loc["value", "outlier_count"]
        <= strict_result.loc["value", "outlier_count"]
    )


# ---------------------------------------------------------------------------
# memory_usage_summary
# ---------------------------------------------------------------------------

def test_memory_usage_summary_requires_dataframe():
    """memory_usage_summary should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        memory_usage_summary([1, 2, 3])


def test_memory_usage_summary_reports_all_columns(sample_frame):
    """memory_usage_summary should report a row for every column."""
    result = memory_usage_summary(sample_frame)

    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == set(sample_frame.columns)
    assert "memory_bytes" in result.columns
    assert "memory_megabytes" in result.columns


def test_memory_usage_summary_values_are_non_negative(sample_frame):
    """Reported memory usage should never be negative."""
    result = memory_usage_summary(sample_frame)

    assert (result["memory_bytes"] >= 0).all()
    assert (result["memory_megabytes"] >= 0).all()


# ---------------------------------------------------------------------------
# generate_eda_report
# ---------------------------------------------------------------------------

def test_generate_eda_report_requires_dataframe():
    """generate_eda_report should reject non-DataFrame inputs."""
    with pytest.raises(TypeError):
        generate_eda_report("not a frame")


def test_generate_eda_report_contains_all_sections(sample_frame):
    """generate_eda_report should aggregate every individual summary."""
    report = generate_eda_report(sample_frame)

    expected_keys = {
        "overview",
        "data_types",
        "missing_values",
        "duplicates",
        "unique_values",
        "numeric_summary",
        "categorical_summary",
        "correlation_matrix",
        "outliers",
        "memory_usage",
    }

    assert expected_keys.issubset(report.keys())


def test_generate_eda_report_matches_individual_functions(sample_frame):
    """generate_eda_report values should match calling each function directly."""
    report = generate_eda_report(sample_frame)

    assert report["overview"] == dataset_overview(sample_frame)
    pd.testing.assert_frame_equal(report["missing_values"], missing_values_summary(sample_frame))
    pd.testing.assert_frame_equal(report["numeric_summary"], numeric_summary(sample_frame))


# ---------------------------------------------------------------------------
# Empty-DataFrame and non-mutation guarantees across the whole module
# ---------------------------------------------------------------------------

def test_empty_dataframe_is_supported():
    """EDA functions should handle an empty DataFrame gracefully."""
    frame = pd.DataFrame()

    overview = dataset_overview(frame)
    missing = missing_values_summary(frame)
    numeric = numeric_summary(frame)
    categorical = categorical_summary(frame)
    correlation = correlation_matrix(frame)
    data_types = data_types_summary(frame)
    duplicates = duplicate_rows_summary(frame)
    unique_values = unique_values_summary(frame)
    outliers = outlier_summary(frame)
    memory_usage = memory_usage_summary(frame)

    assert overview["rows"] == 0
    assert overview["columns"] == 0
    assert isinstance(missing, pd.DataFrame)
    assert isinstance(numeric, pd.DataFrame)
    assert isinstance(categorical, pd.DataFrame)
    assert isinstance(correlation, pd.DataFrame)
    assert isinstance(data_types, pd.DataFrame)
    assert isinstance(duplicates, dict)
    assert isinstance(unique_values, pd.DataFrame)
    assert isinstance(outliers, pd.DataFrame)
    assert isinstance(memory_usage, pd.DataFrame)


def test_eda_does_not_modify_original_dataframe(sample_frame):
    """EDA functions should not mutate the input DataFrame."""
    original = sample_frame.copy(deep=True)

    dataset_overview(sample_frame)
    missing_values_summary(sample_frame)
    numeric_summary(sample_frame)
    categorical_summary(sample_frame)
    correlation_matrix(sample_frame)
    data_types_summary(sample_frame)
    duplicate_rows_summary(sample_frame)
    unique_values_summary(sample_frame)
    outlier_summary(sample_frame)
    memory_usage_summary(sample_frame)
    generate_eda_report(sample_frame)

    pd.testing.assert_frame_equal(sample_frame, original)


@pytest.mark.parametrize(
    "func",
    [
        dataset_overview,
        missing_values_summary,
        numeric_summary,
        categorical_summary,
        correlation_matrix,
        data_types_summary,
        duplicate_rows_summary,
        unique_values_summary,
        outlier_summary,
        memory_usage_summary,
        generate_eda_report,
    ],
)
def test_all_functions_reject_non_dataframe_input(func):
    """Every public function should raise TypeError for non-DataFrame input."""
    with pytest.raises(TypeError):
        func("definitely not a dataframe")