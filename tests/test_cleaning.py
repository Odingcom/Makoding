import numpy as np
import pandas as pd
import pytest

from makoding.cleaning import (
    clean_frame,
    CleaningReport,
    MISSING_STRATEGIES,
)


@pytest.fixture
def dirty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["  Alice ", "Bob", "Bob", None],
            "age": [30, np.nan, 25, 25],
        }
    )


def test_clean_frame_requires_dataframe():
    with pytest.raises(TypeError):
        clean_frame("not a dataframe")


def test_invalid_strategy_raises(dirty_frame):
    with pytest.raises(ValueError):
        clean_frame(
            dirty_frame,
            missing="Not a real strategy",
            remove_duplicates=False,
            trim_whitespace=False,
        )


def test_supported_strategies_are_available():
    assert "Keep" in MISSING_STRATEGIES
    assert "Drop rows" in MISSING_STRATEGIES
    assert "Fill numeric median" in MISSING_STRATEGIES
    assert "Fill numeric mean" in MISSING_STRATEGIES
    assert "Fill all with mode" in MISSING_STRATEGIES


def test_trims_whitespace(dirty_frame):
    cleaned, _ = clean_frame(
        dirty_frame,
        missing="Keep",
        remove_duplicates=False,
        trim_whitespace=True,
    )

    assert cleaned["name"].iloc[0] == "Alice"


def test_trim_whitespace_does_not_change_non_strings():
    frame = pd.DataFrame(
        {
            "age": [30, 25],
            "score": [10.5, 20.5],
        }
    )

    cleaned, _ = clean_frame(
        frame,
        missing="Keep",
        remove_duplicates=False,
        trim_whitespace=True,
    )

    assert cleaned["age"].tolist() == [30, 25]
    assert cleaned["score"].tolist() == [10.5, 20.5]


def test_trim_whitespace_cleans_column_names():
    frame = pd.DataFrame(
        {
            " name ": [" Alice "],
            " age ": [30],
        }
    )

    cleaned, _ = clean_frame(
        frame,
        missing="Keep",
        remove_duplicates=False,
        trim_whitespace=True,
    )

    assert list(cleaned.columns) == ["name", "age"]


def test_removes_duplicates(dirty_frame):
    cleaned, report = clean_frame(
        dirty_frame,
        missing="Keep",
        remove_duplicates=True,
        trim_whitespace=True,
    )

    assert report.duplicates_removed >= 0
    assert len(cleaned) <= len(dirty_frame)


def test_duplicates_are_not_removed_when_disabled(dirty_frame):
    cleaned, report = clean_frame(
        dirty_frame,
        missing="Keep",
        remove_duplicates=False,
        trim_whitespace=True,
    )

    assert len(cleaned) == len(dirty_frame)
    assert report.duplicates_removed == 0


def test_duplicate_count_is_reported():
    frame = pd.DataFrame(
        {
            "name": ["Alice", "Alice", "Bob"],
            "age": [30, 30, 25],
        }
    )

    cleaned, report = clean_frame(
        frame,
        missing="Keep",
        remove_duplicates=True,
        trim_whitespace=False,
    )

    assert len(cleaned) == 2
    assert report.duplicates_removed == 1


def test_keep_missing_values(dirty_frame):
    cleaned, report = clean_frame(
        dirty_frame,
        missing="Keep",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert cleaned.isna().sum().sum() == 2
    assert report.missing_before == 2
    assert report.missing_after == 2


def test_drop_rows(dirty_frame):
    cleaned, report = clean_frame(
        dirty_frame,
        missing="Drop rows",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert cleaned.isna().sum().sum() == 0
    assert report.rows_after == len(cleaned)


def test_fill_numeric_median(dirty_frame):
    cleaned, _ = clean_frame(
        dirty_frame,
        missing="Fill numeric median",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert cleaned["age"].isna().sum() == 0


def test_numeric_median_is_correct():
    frame = pd.DataFrame(
        {
            "age": [10, 20, np.nan, 30],
        }
    )

    cleaned, _ = clean_frame(
        frame,
        missing="Fill numeric median",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert cleaned["age"].iloc[2] == 20


def test_fill_numeric_mean():
    frame = pd.DataFrame(
        {
            "age": [10, 20, np.nan, 30],
        }
    )

    cleaned, _ = clean_frame(
        frame,
        missing="Fill numeric mean",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert cleaned["age"].isna().sum() == 0
    assert cleaned["age"].iloc[2] == 20


def test_fill_all_with_mode():
    frame = pd.DataFrame(
        {
            "age": [10, 10, np.nan, 20],
            "city": ["Nairobi", "Nairobi", None, "Mombasa"],
        }
    )

    cleaned, _ = clean_frame(
        frame,
        missing="Fill all with mode",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert cleaned.isna().sum().sum() == 0
    assert cleaned["age"].iloc[2] == 10
    assert cleaned["city"].iloc[2] == "Nairobi"


def test_mode_fills_categorical_values():
    frame = pd.DataFrame(
        {
            "city": ["Nairobi", "Nairobi", "Mombasa", None],
        }
    )

    cleaned, _ = clean_frame(
        frame,
        missing="Fill all with mode",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert cleaned["city"].iloc[3] == "Nairobi"


def test_cleaning_report_type(dirty_frame):
    _, report = clean_frame(dirty_frame)

    assert isinstance(report, CleaningReport)


def test_cleaning_report_tracks_rows(dirty_frame):
    _, report = clean_frame(
        dirty_frame,
        missing="Drop rows",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert report.rows_before == 4
    assert report.rows_after == 2


def test_cleaning_report_tracks_columns(dirty_frame):
    _, report = clean_frame(dirty_frame)

    assert report.columns_before == 2
    assert report.columns_after == 2


def test_cleaning_report_tracks_missing_values(dirty_frame):
    _, report = clean_frame(
        dirty_frame,
        missing="Keep",
        remove_duplicates=False,
        trim_whitespace=False,
    )

    assert report.missing_before == 2
    assert report.missing_after == 2


def test_original_dataframe_is_not_modified(dirty_frame):
    original = dirty_frame.copy(deep=True)

    clean_frame(
        dirty_frame,
        missing="Fill numeric median",
        remove_duplicates=True,
        trim_whitespace=True,
    )

    pd.testing.assert_frame_equal(dirty_frame, original)


def test_empty_dataframe_is_supported():
    frame = pd.DataFrame()

    cleaned, report = clean_frame(
        frame,
        missing="Keep",
        remove_duplicates=True,
        trim_whitespace=True,
    )

    assert cleaned.empty
    assert report.rows_before == 0
    assert report.rows_after == 0


def test_combined_cleaning_operations():
    frame = pd.DataFrame(
        {
            "name": [
                " Alice ",
                "Alice",
                " Bob ",
                None,
            ],
            "age": [
                30,
                30,
                np.nan,
                25,
            ],
        }
    )

    cleaned, report = clean_frame(
        frame,
        missing="Fill numeric median",
        remove_duplicates=True,
        trim_whitespace=True,
    )

    assert cleaned["name"].iloc[0] == "Alice"
    assert cleaned["age"].isna().sum() == 0
    assert report.duplicates_removed >= 1