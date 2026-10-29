"""Unit tests for the cleaning engine. No UI or Streamlit needed."""
import pandas as pd
import pytest

from makoding.data_processing import apply_clean_step, profile_columns, summarize_quality


def test_profile():
    frame = pd.DataFrame({"a": [1, None], "b": ["x", "x"]})
    quality = summarize_quality(frame)
    assert (quality["rows"], quality["columns"], quality["missing_cells"]) == (2, 2, 1)
    assert profile_columns(frame).loc[0, "Missing"] == 1


def test_remove_duplicates_does_not_mutate_original():
    original = pd.DataFrame({"a": [1, 1, 2]})
    modified, detail = apply_clean_step(original, "remove_duplicates")
    assert len(modified) == 2
    assert detail["removed_rows"] == 1
    assert len(original) == 3


def test_trim_preserves_missing():
    original = pd.DataFrame({"name": [" A ", None, "B"]})
    result, _ = apply_clean_step(original, "trim_whitespace", columns=["name"])
    assert result.loc[0, "name"] == "A"
    assert pd.isna(result.loc[1, "name"])
    assert original.loc[0, "name"] == " A "


def test_tokens_become_missing_only_when_requested():
    frame = pd.DataFrame({"name": [" NA ", "Kenya", "?"]})
    result, _ = apply_clean_step(frame, "standardize_missing", columns=["name"])
    assert result["name"].isna().sum() == 2
    assert frame["name"].isna().sum() == 0


def test_numeric_median_only_selected_column():
    frame = pd.DataFrame({"x": [1.0, 5.0, None], "y": [None, 6.0, 4.0]})
    result, _ = apply_clean_step(frame, "fill_missing", columns=["x"], strategy="median")
    assert result.loc[2, "x"] == 3
    assert pd.isna(result.loc[0, "y"])


def test_mean_invalid_for_text():
    with pytest.raises(ValueError, match="numeric"):
        apply_clean_step(pd.DataFrame({"x": ["a", None]}), "fill_missing",
                         columns=["x"], strategy="mean")


def test_mode():
    result, _ = apply_clean_step(pd.DataFrame({"x": ["a", "a", None]}),
                                 "fill_missing", columns=["x"], strategy="mode")
    assert result.loc[2, "x"] == "a"


def test_all_missing_cannot_impute_median():
    with pytest.raises(ValueError, match="no values"):
        apply_clean_step(pd.DataFrame({"x": [float("nan"), float("nan")]}),
                         "fill_missing", columns=["x"], strategy="median")


def test_drop_missing_rows_preserves_original_index():
    frame = pd.DataFrame({"x": [1, None, 2], "y": [None, 2, 3]}, index=[5, 7, 9])
    result, _ = apply_clean_step(frame, "drop_missing_rows", columns=["x"])
    assert result.index.tolist() == [5, 9]


def test_cannot_drop_every_column():
    with pytest.raises(ValueError, match="no usable"):
        apply_clean_step(pd.DataFrame({"x": [1]}), "drop_columns", columns=["x"])


def test_reject_unknown_column():
    with pytest.raises(ValueError, match="Unknown"):
        apply_clean_step(pd.DataFrame({"x": [1]}), "trim_whitespace", columns=["bad"])


def test_reject_duplicate_column_names():
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(ValueError, match="Duplicate column"):
        summarize_quality(frame)
