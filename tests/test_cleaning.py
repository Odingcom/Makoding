"""
Tests for makoding.cleaning.

Organized to mirror the module's own sections:
    - CleaningPolicy validation
    - Cleaner: fit/transform/fit_transform, per-strategy behavior,
      leakage-safety, dtype drift, schema mismatch, persistence
    - clean_frame (legacy/convenience API)
    - stats_readiness
    - profile_frame
    - Data contracts (ColumnContract / DatasetContract / validate_contract)
    - Provenance (dataset_fingerprint / RunManifest)
    - AnalyticalPreprocessor

Run with:
    pytest tests/test_cleaning.py -v
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from makoding.cleaning import (
    AnalyticalPreprocessor,
    Cleaner,
    CleaningAudit,
    CleaningPolicy,
    ColumnContract,
    ColumnQuality,
    DatasetContract,
    LearnedColumnState,
    MISSING_STRATEGIES,
    NotFittedError,
    QualityReport,
    RunManifest,
    SUPPORTED_MISSING_STRATEGIES,
    SchemaMismatchError,
    ValidationReport,
    clean_frame,
    dataset_fingerprint,
    profile_frame,
    stats_readiness,
    validate_contract,
)


# ============================================================================
# Helpers (plain factory functions rather than fixtures, so every test
# stays a simple, readable, independent unit)
# ============================================================================


def make_basic_frame() -> pd.DataFrame:
    """A small, deliberately messy frame: whitespace, duplicates, a
    missing value in each column, one outlier."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 30.0, np.nan, 1000.0],
            "city": [" NYC", "LA ", "LA", None, "SF"],
        }
    )


def make_numeric_frame_with_inf() -> pd.DataFrame:
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, np.inf, -np.inf, np.nan]})


# ============================================================================
# CleaningPolicy
# ============================================================================


def test_policy_defaults_are_safe_noops():
    policy = CleaningPolicy()
    assert policy.missing == "keep"
    assert policy.remove_duplicates is False
    assert policy.cap_outliers is False
    assert policy.drop_all_missing_columns is False
    assert policy.drop_constant_columns is False


def test_policy_rejects_unknown_missing_strategy():
    with pytest.raises(ValueError):
        CleaningPolicy(missing="not_a_real_strategy")


def test_policy_rejects_non_finite_outlier_multiplier():
    with pytest.raises(ValueError):
        CleaningPolicy(outlier_multiplier=float("inf"))


def test_policy_rejects_non_positive_outlier_multiplier():
    with pytest.raises(ValueError):
        CleaningPolicy(outlier_multiplier=0)
    with pytest.raises(ValueError):
        CleaningPolicy(outlier_multiplier=-1.5)


def test_policy_rejects_missing_indicators_with_drop_rows():
    with pytest.raises(ValueError):
        CleaningPolicy(missing="drop_rows", add_missing_indicators=True)


def test_policy_is_frozen():
    policy = CleaningPolicy()
    with pytest.raises(Exception):
        policy.missing = "median"  # dataclasses raise FrozenInstanceError (a subclass of AttributeError)


@pytest.mark.parametrize("strategy", list(SUPPORTED_MISSING_STRATEGIES))
def test_every_supported_strategy_constructs_a_valid_policy(strategy):
    if strategy == "drop_rows":
        policy = CleaningPolicy(missing=strategy, add_missing_indicators=False)
    else:
        policy = CleaningPolicy(missing=strategy)
    assert policy.missing == strategy


# ============================================================================
# Cleaner: not-fitted guard
# ============================================================================


def test_transform_before_fit_raises_not_fitted_error():
    cleaner = Cleaner()
    with pytest.raises(NotFittedError):
        cleaner.transform(make_basic_frame())


def test_to_dict_before_fit_raises_not_fitted_error():
    cleaner = Cleaner()
    with pytest.raises(NotFittedError):
        cleaner.to_dict()


def test_save_before_fit_raises_not_fitted_error(tmp_path):
    cleaner = Cleaner()
    with pytest.raises(NotFittedError):
        cleaner.save(tmp_path / "cleaner.pkl")


def test_not_fitted_error_is_a_runtime_error():
    # Backward compatibility: old `except RuntimeError` call sites must
    # still catch this.
    assert issubclass(NotFittedError, RuntimeError)


# ============================================================================
# Cleaner: basic fit/transform mechanics
# ============================================================================


def test_fit_returns_self_for_chaining():
    cleaner = Cleaner()
    result = cleaner.fit(make_basic_frame())
    assert result is cleaner
    assert cleaner.fitted is True


def test_fit_does_not_mutate_the_input_frame():
    frame = make_basic_frame()
    original = frame.copy(deep=True)
    Cleaner(CleaningPolicy(missing="median")).fit(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_transform_does_not_mutate_the_input_frame():
    frame = make_basic_frame()
    cleaner = Cleaner(CleaningPolicy(missing="median")).fit(frame)
    original = frame.copy(deep=True)
    cleaner.transform(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_fit_transform_matches_fit_then_transform():
    frame = make_basic_frame()
    a_cleaner = Cleaner(CleaningPolicy(missing="mode", remove_duplicates=True))
    a_out, a_audit = a_cleaner.fit_transform(frame.copy())

    b_cleaner = Cleaner(CleaningPolicy(missing="mode", remove_duplicates=True))
    b_cleaner.fit(frame.copy())
    b_out, b_audit = b_cleaner.transform(frame.copy())

    pd.testing.assert_frame_equal(a_out, b_out)
    assert a_audit == b_audit


def test_input_and_output_columns_are_tracked():
    cleaner = Cleaner()
    cleaner.fit(make_basic_frame())
    assert cleaner.input_columns == ("age", "city")
    assert cleaner.output_columns == ("age", "city")


def test_repr_reports_fitted_state():
    cleaner = Cleaner()
    assert "fitted=False" in repr(cleaner)
    cleaner.fit(make_basic_frame())
    assert "fitted=True" in repr(cleaner)


# ============================================================================
# Cleaner: whitespace + column-name trimming
# ============================================================================


def test_column_names_are_trimmed():
    frame = pd.DataFrame({" age ": [1, 2, 3]})
    out, _ = clean_frame(frame)
    assert list(out.columns) == ["age"]


def test_string_values_are_trimmed_and_counted():
    frame = pd.DataFrame({"city": [" NYC", "LA ", " SF "]})
    out, audit = clean_frame(frame)
    assert out["city"].tolist() == ["NYC", "LA", "SF"]
    assert audit.whitespace_trimmed == 3


def test_trimming_can_be_disabled():
    frame = pd.DataFrame({"city": [" NYC"]})
    out, audit = clean_frame(frame, trim_whitespace=False)
    assert out["city"].iloc[0] == " NYC"
    assert audit.whitespace_trimmed == 0


def test_duplicate_column_names_after_trim_raise():
    frame = pd.DataFrame([[1, 2]], columns=["age", "age "])
    with pytest.raises(ValueError):
        Cleaner().fit(frame)


# ============================================================================
# Cleaner: infinite-value handling (and the statistic-corruption bug fix)
# ============================================================================


def test_infinite_values_are_replaced_with_missing_by_default():
    frame = make_numeric_frame_with_inf()
    out, audit = clean_frame(frame)
    assert not np.isinf(out["value"].to_numpy(dtype=float)).any()
    assert audit.infinite_replaced == 2


def test_infinite_replacement_can_be_disabled():
    frame = make_numeric_frame_with_inf()
    out, audit = clean_frame(frame, treat_infinite_as_missing=False)
    assert np.isinf(out["value"].to_numpy(dtype=float)).any()
    assert audit.infinite_replaced == 0


@pytest.mark.parametrize("strategy", ["mean", "median", "mode", "auto"])
def test_learned_fill_value_is_never_corrupted_by_infinity(strategy):
    # Regression test: Inf/-Inf must never leak into a learned median,
    # mean, or mode -- even when replace_infinite governs the *output*
    # data rather than the statistic itself.
    frame = pd.DataFrame({"v": [np.inf, np.inf, 10.0, 20.0, 30.0, np.nan]})
    cleaner = Cleaner(CleaningPolicy(missing=strategy))
    cleaner.fit(frame)
    fill_value = cleaner.learned_state["v"].fill_value
    assert fill_value is not None
    assert np.isfinite(fill_value)


def test_mode_strategy_does_not_learn_infinity_as_the_mode():
    # The most direct form of the bug: two `inf`s outnumber any single
    # finite value, so a naive `.mode()` call would pick `inf`.
    frame = pd.DataFrame({"v": [np.inf, np.inf, 5.0, 5.0, 5.0, np.nan]})
    cleaner = Cleaner(CleaningPolicy(missing="mode"))
    cleaner.fit(frame)
    assert cleaner.learned_state["v"].fill_value == 5.0


# ============================================================================
# Cleaner: missing-value strategies
# ============================================================================


def test_keep_strategy_leaves_missing_values_untouched():
    out, audit = clean_frame(make_basic_frame(), missing="keep")
    assert audit.missing_after == audit.missing_before
    assert out["age"].isna().any()


def test_drop_rows_strategy_removes_every_row_with_a_missing_value():
    out, audit = clean_frame(make_basic_frame(), missing="drop_rows")
    assert audit.missing_after == 0
    assert not out.isna().any().any()


def test_median_strategy_only_fills_numeric_columns():
    out, audit = clean_frame(make_basic_frame(), missing="median")
    assert not out["age"].isna().any()
    assert out["city"].isna().any()  # untouched: not numeric


def test_mean_strategy_only_fills_numeric_columns():
    out, audit = clean_frame(make_basic_frame(), missing="mean")
    assert not out["age"].isna().any()
    assert out["city"].isna().any()


def test_mode_strategy_fills_every_column_including_numeric():
    out, audit = clean_frame(make_basic_frame(), missing="mode")
    assert not out["age"].isna().any()
    assert not out["city"].isna().any()


def test_auto_strategy_blends_median_for_numeric_and_mode_for_categorical():
    frame = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0, np.nan],
            "city": ["NYC", "NYC", "LA", None],
        }
    )
    out, _ = clean_frame(frame, missing="auto")
    assert out["age"].iloc[3] == 20.0  # median of 10,20,30
    assert out["city"].iloc[3] == "NYC"  # mode


def test_auto_differs_from_mode_for_numeric_columns():
    # "mode" fills numeric columns with their most frequent value;
    # "auto" fills them with the median. With this data they diverge.
    frame = pd.DataFrame({"v": [1.0, 1.0, 1.0, 100.0, np.nan]})
    mode_cleaner = Cleaner(CleaningPolicy(missing="mode")).fit(frame)
    auto_cleaner = Cleaner(CleaningPolicy(missing="auto")).fit(frame)
    assert mode_cleaner.learned_state["v"].fill_value == 1.0
    assert auto_cleaner.learned_state["v"].fill_value == 1.0  # median of 1,1,1,100 is 1.0 too here

    frame2 = pd.DataFrame({"v": [1.0, 1.0, 50.0, 100.0, np.nan]})
    mode_cleaner2 = Cleaner(CleaningPolicy(missing="mode")).fit(frame2)
    auto_cleaner2 = Cleaner(CleaningPolicy(missing="auto")).fit(frame2)
    assert mode_cleaner2.learned_state["v"].fill_value == 1.0
    assert auto_cleaner2.learned_state["v"].fill_value == pytest.approx(25.5)


def test_values_filled_by_column_breakdown_is_accurate():
    out, audit = clean_frame(make_basic_frame(), missing="mode")
    by_column = dict(audit.values_filled_by_column)
    assert by_column["age"] == 1
    assert by_column["city"] == 1
    assert audit.values_filled == 2


def test_entirely_missing_column_cannot_be_filled_and_warns():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
    with pytest.warns(UserWarning):
        cleaner = Cleaner(CleaningPolicy(missing="median")).fit(frame)
    out, _ = cleaner.transform(frame)
    assert out["b"].isna().all()


# ============================================================================
# Cleaner: outlier capping (fit-time learning vs. transform-time application)
# ============================================================================


def test_outlier_capping_clips_values_beyond_the_learned_fence():
    frame = pd.DataFrame({"v": [10.0, 12.0, 11.0, 13.0, 500.0]})
    cleaner = Cleaner(CleaningPolicy(cap_outliers=True, outlier_multiplier=1.5))
    out, audit = cleaner.fit_transform(frame)
    assert out["v"].max() < 500.0
    assert audit.values_capped == 1


def test_outlier_fences_are_learned_once_and_reused_not_recomputed():
    # The core leakage-safety claim: transform() must use the exact
    # fences learned from the training data, even on a frame whose own
    # quartiles would produce different fences.
    train = pd.DataFrame({"v": [10.0, 11.0, 12.0, 13.0, 14.0, 200.0]})
    cleaner = Cleaner(CleaningPolicy(cap_outliers=True)).fit(train)
    learned_upper = cleaner.learned_state["v"].upper_fence

    other = pd.DataFrame({"v": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]})
    cleaned_other, audit = cleaner.transform(other)
    # Every value in `other` should be capped down to the fence learned
    # from `train`, not to any statistic derived from `other` itself.
    assert (cleaned_other["v"] == learned_upper).all()
    assert audit.values_capped == 6


def test_capped_values_by_column_breakdown():
    frame = pd.DataFrame({"v": [10.0, 12.0, 11.0, 13.0, 500.0]})
    _, audit = clean_frame(frame, cap_outliers=True)
    assert dict(audit.values_capped_by_column) == {"v": 1}


# ============================================================================
# Cleaner: degenerate-column dropping
# ============================================================================


def test_all_missing_columns_are_dropped_when_requested():
    frame = pd.DataFrame({"a": [1, 2, 3], "gone": [np.nan, np.nan, np.nan]})
    cleaner = Cleaner(CleaningPolicy(drop_all_missing_columns=True)).fit(frame)
    assert cleaner.dropped_columns == ("gone",)
    out, _ = cleaner.transform(frame)
    assert "gone" not in out.columns


def test_constant_columns_are_dropped_when_requested():
    frame = pd.DataFrame({"a": [1, 2, 3], "const": [5, 5, 5]})
    cleaner = Cleaner(CleaningPolicy(drop_constant_columns=True)).fit(frame)
    assert cleaner.dropped_columns == ("const",)


def test_columns_not_dropped_by_default():
    frame = pd.DataFrame({"a": [1, 2, 3], "const": [5, 5, 5], "gone": [np.nan] * 3})
    cleaner = Cleaner().fit(frame)
    assert cleaner.dropped_columns == ()


# ============================================================================
# Cleaner: missing-value indicator columns
# ============================================================================


def test_missing_indicators_are_added_before_filling():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out, audit = clean_frame(frame, missing="median", add_missing_indicators=True)
    assert "a__missing" in out.columns
    assert out["a__missing"].tolist() == [0, 1, 0]
    assert not out["a"].isna().any()  # still filled


# ============================================================================
# Cleaner: duplicate removal
# ============================================================================


def test_duplicate_rows_removed_when_requested():
    frame = pd.DataFrame({"a": [1, 1, 2]})
    out, audit = clean_frame(frame, remove_duplicates=True)
    assert len(out) == 2
    assert audit.duplicates_removed == 1


def test_duplicates_kept_when_not_requested():
    frame = pd.DataFrame({"a": [1, 1, 2]})
    out, audit = clean_frame(frame, remove_duplicates=False)
    assert len(out) == 3
    assert audit.duplicates_removed == 0


# ============================================================================
# Cleaner: schema mismatch + dtype drift
# ============================================================================


def test_schema_mismatch_raises_with_missing_and_extra_columns_named():
    cleaner = Cleaner().fit(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(SchemaMismatchError) as exc_info:
        cleaner.transform(pd.DataFrame({"a": [1], "c": [2]}))
    message = str(exc_info.value)
    assert "'b'" in message
    assert "'c'" in message


def test_schema_mismatch_error_is_a_value_error():
    assert issubclass(SchemaMismatchError, ValueError)


def test_column_order_is_realigned_not_rejected():
    cleaner = Cleaner().fit(pd.DataFrame({"a": [1], "b": [2]}))
    out, _ = cleaner.transform(pd.DataFrame({"b": [20], "a": [10]}))
    assert list(out.columns) == ["a", "b"]


def test_dtype_drift_raises_by_default():
    # dtype_drift="error" is the default: production preprocessing should
    # fail closed rather than silently applying a numeric statistic to
    # data that has drifted to a different semantic family.
    cleaner = Cleaner(CleaningPolicy(missing="mean", cap_outliers=True))
    cleaner.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}))

    drifted = pd.DataFrame({"x": ["a", "b", "c"]})
    with pytest.raises(SchemaMismatchError):
        cleaner.transform(drifted)


def test_dtype_drift_warn_mode_does_not_crash():
    cleaner = Cleaner(CleaningPolicy(missing="mean", cap_outliers=True, dtype_drift="warn"))
    cleaner.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}))

    drifted = pd.DataFrame({"x": ["a", "b", "c"]})
    with pytest.warns(UserWarning):
        cleaned, audit = cleaner.transform(drifted)

    assert audit.dtype_drift_columns == ("x",)
    # No numeric fill/cap was silently (and wrongly) applied to string data.
    assert cleaned["x"].tolist() == ["a", "b", "c"]


def test_dtype_drift_mode_is_validated_by_policy():
    with pytest.raises(ValueError):
        CleaningPolicy(dtype_drift="not_a_real_mode")


def test_dtype_family_change_between_compatible_numeric_dtypes_is_not_drift():
    # int64 -> float64 (e.g. after NaNs are introduced) is the same
    # semantic family ("numeric") and must not be flagged as drift.
    cleaner = Cleaner(CleaningPolicy(missing="mean")).fit(pd.DataFrame({"x": [1, 2, 3]}))
    cleaned, audit = cleaner.transform(pd.DataFrame({"x": [1.0, 2.0, np.nan]}))
    assert audit.dtype_drift_columns == ()


# ============================================================================
# Cleaner: persistence
# ============================================================================


def test_save_and_load_round_trip_produces_identical_output(tmp_path):
    frame = make_basic_frame()
    cleaner = Cleaner(CleaningPolicy(missing="median", cap_outliers=True))
    cleaner.fit(frame)
    expected_out, _ = cleaner.transform(frame)

    path = tmp_path / "cleaner.pkl"
    cleaner.save(path)
    reloaded = Cleaner.load(path)

    actual_out, _ = reloaded.transform(frame)
    pd.testing.assert_frame_equal(actual_out, expected_out)


def test_load_rejects_a_pickle_that_is_not_a_cleaner(tmp_path):
    path = tmp_path / "not_a_cleaner.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"just": "a dict"}, handle)
    with pytest.raises(TypeError):
        Cleaner.load(path)


def test_to_dict_contains_policy_and_learned_state():
    cleaner = Cleaner(CleaningPolicy(missing="median")).fit(make_basic_frame())
    snapshot = cleaner.to_dict()
    assert snapshot["policy"]["missing"] == "median"
    assert "age" in snapshot["learned_state"]
    assert snapshot["learned_state"]["age"]["fill_value"] is not None


# ============================================================================
# clean_frame (legacy/convenience API)
# ============================================================================


@pytest.mark.parametrize("legacy_label", list(MISSING_STRATEGIES))
def test_every_legacy_label_is_accepted(legacy_label):
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "y"]})
    out, audit = clean_frame(frame, missing=legacy_label)
    assert isinstance(out, pd.DataFrame)
    assert isinstance(audit, CleaningAudit)


def test_fill_missing_values_legacy_label_maps_to_auto_semantics():
    frame = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, np.nan], "city": ["NYC", "NYC", "LA", None]}
    )
    out, _ = clean_frame(frame, missing="Fill missing values")
    assert out["age"].iloc[3] == 20.0
    assert out["city"].iloc[3] == "NYC"


def test_unknown_missing_label_raises_value_error():
    with pytest.raises(ValueError):
        clean_frame(pd.DataFrame({"a": [1]}), missing="not a real label")


def test_clean_frame_returns_cleaning_report_alias():
    # CleaningReport is a backward-compatible alias for CleaningAudit.
    from makoding.cleaning import CleaningReport

    out, audit = clean_frame(pd.DataFrame({"a": [1, 2]}))
    assert isinstance(audit, CleaningReport)


def test_audit_backward_compatible_properties():
    out, audit = clean_frame(
        pd.DataFrame({"a": [1.0, np.nan, 3.0]}), missing="median"
    )
    assert audit.missing_removed == audit.missing_resolved
    assert audit.missing_values_before == audit.missing_before
    assert audit.missing_values_after == audit.missing_after
    assert isinstance(audit.as_markdown(), str)
    assert isinstance(audit.summary(), str)


# ============================================================================
# stats_readiness
# ============================================================================


def test_stats_readiness_flags_columns_with_too_few_observations():
    frame = pd.DataFrame({"few": [1.0, np.nan, np.nan, np.nan]})
    result = stats_readiness(frame, min_observations=3)
    assert result["few"]["valid_observations"] == 1
    assert result["few"]["enough_observations"] is False
    assert result["few"]["column_ready"] is False


def test_stats_readiness_detects_infinite_values_correctly():
    # Regression test: this must be False (Inf really is present), not
    # trivially True from checking finiteness only after stripping Inf.
    frame = pd.DataFrame({"x": [1.0, np.inf, 3.0, -np.inf, 5.0]})
    result = stats_readiness(frame)
    assert result["x"]["finite"] is False
    assert result["x"]["column_ready"] is False


def test_stats_readiness_reports_finite_true_for_clean_data():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    result = stats_readiness(frame, min_observations=3)
    assert result["x"]["finite"] is True
    assert result["x"]["column_ready"] is True


def test_stats_readiness_flags_constant_columns():
    frame = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
    result = stats_readiness(frame, min_observations=2)
    assert result["x"]["constant"] is True
    assert result["x"]["column_ready"] is False  # constant => not ready


def test_stats_readiness_rejects_non_positive_min_observations():
    with pytest.raises(ValueError):
        stats_readiness(pd.DataFrame({"a": [1]}), min_observations=0)


# ============================================================================
# profile_frame
# ============================================================================


def test_profile_frame_reports_missingness_and_duplicates():
    frame = pd.DataFrame({"a": [1, 1, np.nan], "b": ["x", "x", "y"]})
    report = profile_frame(frame)
    assert isinstance(report, QualityReport)
    assert report.rows == 3
    assert report.duplicate_rows == 1
    assert report.total_missing == 1
    assert report.columns_with_missing == 1


def test_profile_frame_flags_constant_and_identifier_columns():
    frame = pd.DataFrame({"const": [1, 1, 1], "id": [10, 20, 30]})
    report = profile_frame(frame)
    by_name = report.by_name()
    assert by_name["const"].constant is True
    assert by_name["id"].likely_identifier is True
    assert by_name["id"].constant is False


def test_profile_frame_flags_non_finite_numeric_columns():
    frame = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
    report = profile_frame(frame)
    assert report.by_name()["x"].finite is False


def test_profile_frame_missing_rate_property():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
    report = profile_frame(frame)
    assert report.missing_rate == pytest.approx(0.25)  # 1 missing / 4 cells


# ============================================================================
# Data contracts
# ============================================================================


def test_validate_contract_flags_missing_required_column():
    contract = DatasetContract(columns=(ColumnContract(name="age", required=True),))
    report = validate_contract(pd.DataFrame({"other": [1]}), contract)
    assert report.valid is False
    assert any(issue.code == "missing_column" for issue in report.issues)


def test_validate_contract_flags_extra_column_by_default():
    contract = DatasetContract(columns=(ColumnContract(name="age"),))
    report = validate_contract(pd.DataFrame({"age": [1], "extra": [2]}), contract)
    assert report.valid is False
    assert any(issue.code == "extra_column" for issue in report.issues)


def test_validate_contract_allows_extra_columns_when_configured():
    contract = DatasetContract(columns=(ColumnContract(name="age"),), allow_extra_columns=True)
    report = validate_contract(pd.DataFrame({"age": [1], "extra": [2]}), contract)
    assert report.valid is True


def test_validate_contract_flags_wrong_dtype_family():
    contract = DatasetContract(columns=(ColumnContract(name="age", dtype_family="numeric"),))
    report = validate_contract(pd.DataFrame({"age": ["not", "numeric"]}), contract)
    assert any(issue.code == "dtype" for issue in report.issues)


def test_validate_contract_flags_nulls_when_not_nullable():
    contract = DatasetContract(columns=(ColumnContract(name="age", nullable=False),))
    report = validate_contract(pd.DataFrame({"age": [1.0, np.nan]}), contract)
    assert any(issue.code == "nullability" for issue in report.issues)


def test_validate_contract_flags_out_of_range_values():
    contract = DatasetContract(columns=(ColumnContract(name="age", min_value=0, max_value=120),))
    report = validate_contract(pd.DataFrame({"age": [-5, 30, 150]}), contract)
    codes = {issue.code for issue in report.issues}
    assert "min_value" in codes
    assert "max_value" in codes


def test_validate_contract_flags_disallowed_values():
    contract = DatasetContract(columns=(ColumnContract(name="status", allowed_values=("A", "B")),))
    report = validate_contract(pd.DataFrame({"status": ["A", "B", "Z"]}), contract)
    assert any(issue.code == "allowed_values" for issue in report.issues)


def test_validate_contract_categorical_dtype_family_accepts_pandas_category():
    contract = DatasetContract(columns=(ColumnContract(name="cat", dtype_family="categorical"),))
    frame = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"])})
    report = validate_contract(frame, contract)
    assert report.valid is True


def test_validate_contract_structural_only_skips_content_checks():
    # A range violation is a "content" issue -- it must not appear when
    # structural_only=True is requested, since a Cleaner's cap_outliers
    # (or a fill strategy, for nullability) may still be able to fix it.
    contract = DatasetContract(columns=(ColumnContract(name="age", min_value=0),))
    report = validate_contract(pd.DataFrame({"age": [-5]}), contract, structural_only=True)
    assert report.valid is True


def test_validate_contract_structural_only_still_checks_dtype_and_columns():
    contract = DatasetContract(columns=(ColumnContract(name="age", dtype_family="numeric"),))
    report = validate_contract(pd.DataFrame({"age": ["not", "numeric"]}), contract, structural_only=True)
    assert report.valid is False
    assert any(issue.code == "dtype" for issue in report.issues)


def test_validate_contract_default_is_full_check():
    contract = DatasetContract(columns=(ColumnContract(name="age", min_value=0),))
    report = validate_contract(pd.DataFrame({"age": [-5]}), contract)  # structural_only defaults to False
    assert report.valid is False
    assert any(issue.code == "min_value" for issue in report.issues)


def test_validation_report_raise_if_invalid():
    report = ValidationReport(valid=False, issues=())
    with pytest.raises(ValueError):
        report.raise_if_invalid()
    ValidationReport(valid=True, issues=()).raise_if_invalid()  # must not raise


# ============================================================================
# Provenance
# ============================================================================


def test_dataset_fingerprint_is_deterministic_for_identical_data():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    assert dataset_fingerprint(frame) == dataset_fingerprint(frame.copy())


def test_dataset_fingerprint_differs_for_different_data():
    a = pd.DataFrame({"a": [1, 2, 3]})
    b = pd.DataFrame({"a": [1, 2, 4]})
    assert dataset_fingerprint(a) != dataset_fingerprint(b)


def test_dataset_fingerprint_differs_for_different_dtypes():
    a = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64")})
    b = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="float64")})
    assert dataset_fingerprint(a) != dataset_fingerprint(b)


def test_run_manifest_create_and_to_dict_round_trip():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    manifest = RunManifest.create(frame, cleaner_config={"missing": "median"}, notes="unit test")
    snapshot = manifest.to_dict()  # regression test for the missing `asdict` import
    assert snapshot["rows"] == 3
    assert snapshot["columns"] == 1
    assert snapshot["notes"] == "unit test"
    assert snapshot["cleaner_config"] == {"missing": "median"}


def test_run_manifest_cleaner_config_is_isolated_from_caller_mutation():
    frame = pd.DataFrame({"a": [1]})
    config = {"missing": "median"}
    manifest = RunManifest.create(frame, cleaner_config=config)
    config["missing"] = "mean"
    assert manifest.cleaner_config["missing"] == "median"


# ============================================================================
# AnalyticalPreprocessor
# ============================================================================


def test_analytical_preprocessor_not_fitted_raises():
    prep = AnalyticalPreprocessor(Cleaner())
    with pytest.raises(NotFittedError):
        prep.transform(pd.DataFrame({"a": [1]}))


def test_analytical_preprocessor_without_contract_behaves_like_cleaner():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    prep = AnalyticalPreprocessor(Cleaner(CleaningPolicy(missing="median")))
    result = prep.fit_transform(frame)
    assert not result.frame["a"].isna().any()
    assert result.validation is None
    assert result.manifest.rows == 3


def test_analytical_preprocessor_structural_violation_fails_before_cleaning():
    # dtype_family="numeric" against a string column is not something
    # cleaning can fix -- this must fail fast, before any cleaning runs.
    contract = DatasetContract(columns=(ColumnContract(name="x", dtype_family="numeric"),))
    prep = AnalyticalPreprocessor(Cleaner(), contract)
    with pytest.raises(ValueError):
        prep.fit(pd.DataFrame({"x": ["a", "b", "c"]}))


def test_analytical_preprocessor_lets_cap_outliers_satisfy_the_contract():
    # The core fix: a range violation that cap_outliers is configured to
    # resolve must NOT be rejected before cleaning gets a chance to run.
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    contract = DatasetContract(columns=(ColumnContract(name="x", dtype_family="numeric", max_value=10.0),))
    cleaner = Cleaner(CleaningPolicy(cap_outliers=True, outlier_multiplier=1.5))
    prep = AnalyticalPreprocessor(cleaner, contract)

    result = prep.fit_transform(frame)

    assert result.frame["x"].max() <= 10.0
    assert result.validation is not None
    assert result.validation.valid is True


def test_analytical_preprocessor_still_fails_if_cleaning_cannot_satisfy_the_contract():
    # cap_outliers is NOT enabled here, so the range violation should
    # survive cleaning and be caught by the post-clean "all" validation.
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    contract = DatasetContract(columns=(ColumnContract(name="x", dtype_family="numeric", max_value=10.0),))
    prep = AnalyticalPreprocessor(Cleaner(), contract)
    with pytest.raises(ValueError):
        prep.fit_transform(frame)


def test_analytical_preprocessor_manifest_fingerprint_matches_cleaned_output():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    prep = AnalyticalPreprocessor(Cleaner())
    result = prep.fit_transform(frame)
    assert result.manifest.dataset_fingerprint == dataset_fingerprint(result.frame)


def test_analytical_preprocessor_rejects_extra_columns_from_missing_indicators_by_default():
    # add_missing_indicators introduces new `*__missing` columns; without
    # allow_extra_columns=True, the post-clean "all" contract check
    # should correctly flag them.
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    contract = DatasetContract(columns=(ColumnContract(name="a"),), allow_extra_columns=False)
    cleaner = Cleaner(CleaningPolicy(missing="median", add_missing_indicators=True))
    prep = AnalyticalPreprocessor(cleaner, contract)
    with pytest.raises(ValueError):
        prep.fit_transform(frame)


def test_analytical_preprocessor_accepts_indicators_when_extra_columns_allowed():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    contract = DatasetContract(columns=(ColumnContract(name="a"),), allow_extra_columns=True)
    cleaner = Cleaner(CleaningPolicy(missing="median", add_missing_indicators=True))
    prep = AnalyticalPreprocessor(cleaner, contract)
    result = prep.fit_transform(frame)
    assert "a__missing" in result.frame.columns
    assert result.validation.valid is True