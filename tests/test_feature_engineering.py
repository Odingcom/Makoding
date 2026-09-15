"""Tests for the Makoding feature engineering utilities (v2 — sklearn-backed
stateful transformers plus one-shot feature-creation functions).

Each stateful transformer (``NumericScaler``, ``CategoricalEncoder``,
``NumericBinner``, ``TargetEncoder``, ``MissingIndicatorAdder``) is tested
for the property it exists to guarantee: that ``transform`` reuses
exactly what was learned at ``fit`` time, never recomputing statistics
from whatever frame it's given. Tests verifying that specific
guarantee are marked "ASSUMPTION" in their docstring, matching the
convention used throughout this test suite for behavior called out in
a docstring's Assumptions section.
"""

import numpy as np
import pandas as pd
import pytest

from makoding.feature_engineering import (
    CategoricalEncoder,
    MissingIndicatorAdder,
    NotFittedError,
    NumericBinner,
    NumericScaler,
    TargetEncoder,
    add_cyclical_features,
    add_datetime_features,
    add_frequency_features,
    add_missing_indicators,
    bin_numeric_feature,
    create_difference_features,
    create_polynomial_features,
    create_product_features,
    create_ratio_features,
    drop_correlated_features,
    drop_low_variance_features,
    encode_categorical_features,
    find_correlated_feature_pairs,
    log_transform,
    low_variance_features,
    power_transform,
    scale_numeric_features,
    sqrt_transform,
    target_encode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def train_test_numeric():
    """A train/test split with different distributions, so a leaked
    (test-fitted) scaling would produce visibly different numbers than
    a correctly train-fitted one."""
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"a": [100.0, -100.0]})
    return train, test


@pytest.fixture
def train_test_categorical():
    """A train/test split where the test split contains a category
    ('z') never seen in training."""
    train = pd.DataFrame({"cat": ["a", "b", "a", "c", None]})
    test = pd.DataFrame({"cat": ["a", "z", "b", None]})
    return train, test


@pytest.fixture
def numeric_frame():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan, 5.0],
            "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        }
    )


@pytest.fixture
def redundant_frame():
    return pd.DataFrame(
        {"const": [5, 5, 5, 5], "x": [1, 2, 3, 4], "y": [2, 4, 6, 8]}
    )


# ---------------------------------------------------------------------------
# NumericScaler
# ---------------------------------------------------------------------------

class TestNumericScaler:
    def test_rejects_invalid_method(self):
        with pytest.raises(ValueError):
            NumericScaler(method="bogus")

    def test_transform_before_fit_raises(self, numeric_frame):
        with pytest.raises(NotFittedError):
            NumericScaler().transform(numeric_frame)

    def test_fit_requires_dataframe(self):
        with pytest.raises(TypeError):
            NumericScaler().fit("nope")

    def test_fit_rejects_missing_values(self, numeric_frame):
        with pytest.raises(ValueError):
            NumericScaler().fit(numeric_frame, columns=["a"])

    def test_transform_rejects_missing_values(self):
        scaler = NumericScaler().fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
        with pytest.raises(ValueError):
            scaler.transform(pd.DataFrame({"a": [1.0, np.nan]}))

    def test_transform_rejects_missing_fitted_column(self):
        scaler = NumericScaler().fit(pd.DataFrame({"a": [1.0, 2.0]}))
        with pytest.raises(KeyError):
            scaler.transform(pd.DataFrame({"b": [1.0, 2.0]}))

    def test_standard_scaling_zero_mean_unit_variance(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = NumericScaler(method="standard").fit_transform(frame)
        assert result["a"].mean() == pytest.approx(0.0, abs=1e-9)
        # sklearn's StandardScaler uses population std (ddof=0), not
        # pandas' default sample std (ddof=1).
        assert result["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)

    def test_minmax_scaling_bounds(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = NumericScaler(method="minmax").fit_transform(frame)
        assert result["a"].min() == pytest.approx(0.0)
        assert result["a"].max() == pytest.approx(1.0)

    def test_transform_reuses_training_statistics_not_test_statistics(
        self, train_test_numeric
    ):
        """ASSUMPTION: transform applies the mean/std learned at fit time,
        not statistics recomputed from whatever frame it's given —
        this is the whole reason NumericScaler exists over a one-shot
        function."""
        train, test = train_test_numeric
        scaler = NumericScaler(method="standard").fit(train)
        # sklearn's StandardScaler uses population std (ddof=0).
        train_mean, train_std = train["a"].mean(), train["a"].std(ddof=0)

        result = scaler.transform(test)
        expected = (test["a"] - train_mean) / train_std
        pd.testing.assert_series_equal(result["a"], expected, check_names=False)

    def test_fit_transform_matches_fit_then_transform(self, train_test_numeric):
        train, _ = train_test_numeric
        a = NumericScaler().fit_transform(train.copy())
        b = NumericScaler().fit(train).transform(train)
        pd.testing.assert_frame_equal(a, b)

    def test_does_not_mutate_original(self, train_test_numeric):
        train, _ = train_test_numeric
        original = train.copy(deep=True)
        NumericScaler().fit_transform(train)
        pd.testing.assert_frame_equal(train, original)


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------

class TestCategoricalEncoder:
    def test_rejects_invalid_method(self):
        with pytest.raises(ValueError):
            CategoricalEncoder(method="bogus")

    def test_rejects_invalid_drop(self):
        with pytest.raises(ValueError):
            CategoricalEncoder(drop="last")

    def test_transform_before_fit_raises(self, train_test_categorical):
        _, test = train_test_categorical
        with pytest.raises(NotFittedError):
            CategoricalEncoder().transform(test)

    def test_fit_rejects_no_categorical_columns(self):
        with pytest.raises(ValueError):
            CategoricalEncoder().fit(pd.DataFrame({"x": [1, 2, 3]}))

    def test_onehot_unseen_category_is_all_zero(self, train_test_categorical):
        """ASSUMPTION: an unseen category at transform time produces an
        all-zero dummy row rather than raising."""
        train, test = train_test_categorical
        encoder = CategoricalEncoder(method="onehot").fit(train)
        result = encoder.transform(test)
        dummy_cols = [c for c in result.columns if c.startswith("cat_")]
        unseen_row = result.loc[test["cat"] == "z", dummy_cols]
        assert (unseen_row == 0).all(axis=None)

    def test_onehot_missing_gets_its_own_dummy(self, train_test_categorical):
        train, test = train_test_categorical
        encoder = CategoricalEncoder(method="onehot").fit(train)
        result = encoder.transform(test)
        assert "cat_nan" in result.columns
        missing_row = test["cat"].isna()
        assert result.loc[missing_row, "cat_nan"].iloc[0] == 1.0

    def test_ordinal_unseen_category_is_negative_one(self, train_test_categorical):
        """ASSUMPTION: ordinal encoding maps an unseen category to -1."""
        train, test = train_test_categorical
        encoder = CategoricalEncoder(method="ordinal").fit(train)
        result = encoder.transform(test)
        assert result.loc[test["cat"] == "z", "cat"].iloc[0] == -1.0

    def test_ordinal_missing_is_negative_two_distinct_from_unseen(
        self, train_test_categorical
    ):
        """ASSUMPTION: ordinal encoding maps missing to -2, distinguishable
        from the -1 used for unseen categories."""
        train, test = train_test_categorical
        encoder = CategoricalEncoder(method="ordinal").fit(train)
        result = encoder.transform(test)
        missing_row = test["cat"].isna()
        assert result.loc[missing_row, "cat"].iloc[0] == -2.0

    def test_frequency_reports_training_proportions(self, train_test_categorical):
        train, test = train_test_categorical
        encoder = CategoricalEncoder(method="frequency").fit(train)
        result = encoder.transform(test)
        # "a" appears 2/5 times in training.
        assert result.loc[test["cat"] == "a", "cat"].iloc[0] == pytest.approx(0.4)

    def test_frequency_unseen_category_is_zero(self, train_test_categorical):
        """ASSUMPTION: an unseen category is encoded as 0.0 frequency."""
        train, test = train_test_categorical
        encoder = CategoricalEncoder(method="frequency").fit(train)
        result = encoder.transform(test)
        assert result.loc[test["cat"] == "z", "cat"].iloc[0] == 0.0

    def test_onehot_drop_first_removes_one_column(self, train_test_categorical):
        train, _ = train_test_categorical
        with_all = CategoricalEncoder(method="onehot").fit(train)
        with_dropped = CategoricalEncoder(method="onehot", drop="first").fit(train)
        assert len(with_dropped.feature_names_) == len(with_all.feature_names_) - 1

    def test_does_not_mutate_original(self, train_test_categorical):
        train, _ = train_test_categorical
        original = train.copy(deep=True)
        CategoricalEncoder(method="onehot").fit_transform(train)
        pd.testing.assert_frame_equal(train, original)


# ---------------------------------------------------------------------------
# NumericBinner
# ---------------------------------------------------------------------------

class TestNumericBinner:
    def test_rejects_invalid_strategy(self):
        with pytest.raises(ValueError):
            NumericBinner(strategy="bogus")

    def test_rejects_n_bins_below_two(self):
        with pytest.raises(ValueError):
            NumericBinner(n_bins=1)

    def test_transform_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            NumericBinner().transform(pd.DataFrame({"v": [1, 2, 3]}))

    def test_fit_rejects_missing_values(self):
        with pytest.raises(ValueError):
            NumericBinner().fit(pd.DataFrame({"v": [1.0, np.nan, 3.0]}), "v")

    def test_uniform_binning_produces_requested_bin_count(self):
        frame = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
        binner = NumericBinner(n_bins=5, strategy="uniform").fit(frame, "v")
        result = binner.transform(frame)
        assert result["v_binned"].nunique() == 5

    def test_quantile_binning_uses_training_edges_on_new_data(self):
        """ASSUMPTION: transform reuses the exact bin edges learned at fit
        time, not edges recomputed from the new data — critical for
        quantile binning, where recomputing would give different edges."""
        train = pd.DataFrame({"v": list(range(1, 101))})
        test = pd.DataFrame({"v": [10, 60]})

        binner = NumericBinner(n_bins=4, strategy="quantile").fit(train, "v")
        result = binner.transform(test)
        # Manually verify against the stored edges rather than recomputed ones.
        expected = pd.cut(
            test["v"], bins=binner.bin_edges_, labels=False, include_lowest=True
        )
        assert result["v_binned"].tolist() == expected.tolist()

    def test_out_of_range_values_fall_into_extreme_bins(self):
        """ASSUMPTION: values beyond the training range at transform time
        are absorbed into the nearest edge bin, not treated as missing."""
        train = pd.DataFrame({"v": list(range(1, 101))})
        test = pd.DataFrame({"v": [-1000, 10000]})
        binner = NumericBinner(n_bins=4, strategy="uniform").fit(train, "v")
        result = binner.transform(test)
        assert not result["v_binned"].isna().any()
        assert result["v_binned"].iloc[0] == 0  # lowest bin
        assert result["v_binned"].iloc[1] == 3  # highest bin

    def test_does_not_mutate_original(self):
        frame = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
        original = frame.copy(deep=True)
        NumericBinner(n_bins=2).fit_transform(frame, "v")
        pd.testing.assert_frame_equal(frame, original)


# ---------------------------------------------------------------------------
# TargetEncoder
# ---------------------------------------------------------------------------

class TestTargetEncoder:
    def test_rejects_negative_smoothing(self):
        with pytest.raises(ValueError):
            TargetEncoder(smoothing=-1)

    def test_transform_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            TargetEncoder().transform(pd.DataFrame({"cat": ["a"]}))

    def test_fit_rejects_non_numeric_target(self):
        frame = pd.DataFrame({"cat": ["a", "b"], "y": ["x", "y"]})
        with pytest.raises(TypeError):
            TargetEncoder().fit(frame, "cat", "y")

    def test_fit_rejects_missing_target(self):
        frame = pd.DataFrame({"cat": ["a", "b"], "y": [1.0, np.nan]})
        with pytest.raises(ValueError):
            TargetEncoder().fit(frame, "cat", "y")

    def test_zero_smoothing_reproduces_raw_group_means(self):
        frame = pd.DataFrame({"cat": ["a", "a", "b", "b"], "y": [1.0, 3.0, 10.0, 20.0]})
        encoder = TargetEncoder(smoothing=0.0).fit(frame, "cat", "y")
        assert encoder.mapping_["a"] == pytest.approx(2.0)
        assert encoder.mapping_["b"] == pytest.approx(15.0)

    def test_transform_does_not_require_target_column(self):
        """ASSUMPTION: transform only needs the grouping column — the
        target is not required (or used) at inference time."""
        frame = pd.DataFrame({"cat": ["a", "a", "b", "b"], "y": [1.0, 3.0, 10.0, 20.0]})
        encoder = TargetEncoder().fit(frame, "cat", "y")
        inference_frame = pd.DataFrame({"cat": ["a", "b"]})  # no 'y' column at all
        result = encoder.transform(inference_frame)
        assert "cat_target_encoded" in result.columns

    def test_unseen_category_falls_back_to_global_mean(self):
        frame = pd.DataFrame({"cat": ["a", "a", "b", "b"], "y": [1.0, 3.0, 10.0, 20.0]})
        encoder = TargetEncoder(smoothing=0.0).fit(frame, "cat", "y")
        result = encoder.transform(pd.DataFrame({"cat": ["z"]}))
        assert result["cat_target_encoded"].iloc[0] == pytest.approx(encoder.global_mean_)

    def test_missing_with_no_training_precedent_falls_back_to_global_mean(self):
        """ASSUMPTION: a missing category value at transform time, when fit
        saw no missing values, falls back to the global mean like any
        other unseen category — it does not stay NaN."""
        frame = pd.DataFrame({"cat": ["a", "a", "b", "b"], "y": [1.0, 3.0, 10.0, 20.0]})
        encoder = TargetEncoder(smoothing=0.0).fit(frame, "cat", "y")
        result = encoder.transform(pd.DataFrame({"cat": [None]}))
        assert result["cat_target_encoded"].iloc[0] == pytest.approx(encoder.global_mean_)

    def test_missing_with_training_precedent_gets_its_own_group(self):
        frame = pd.DataFrame(
            {"cat": ["a", "a", None, None], "y": [1.0, 3.0, 5.0, 7.0]}
        )
        encoder = TargetEncoder(smoothing=0.0).fit(frame, "cat", "y")
        result = encoder.transform(pd.DataFrame({"cat": [None]}))
        assert result["cat_target_encoded"].iloc[0] == pytest.approx(6.0)

    def test_transform_never_uses_a_rows_own_target(self):
        """ASSUMPTION (leakage safety): encoding a validation split with a
        transformer fit only on a disjoint training split never lets a
        row's own target value influence its own encoding."""
        train = pd.DataFrame({"cat": ["a", "a", "a", "b", "b", "b"], "y": [1, 2, 3, 10, 20, 30]})
        val = pd.DataFrame({"cat": ["a", "b"], "y": [999, -999]})  # extreme values

        encoder = TargetEncoder(smoothing=0.0).fit(train, "cat", "y")
        result = encoder.transform(val)
        # Encoding must reflect only the training group means, unaffected
        # by the wildly different values in `val`.
        assert result.loc[0, "cat_target_encoded"] == pytest.approx(2.0)
        assert result.loc[1, "cat_target_encoded"] == pytest.approx(20.0)

    def test_does_not_mutate_original(self):
        frame = pd.DataFrame({"cat": ["a", "b"], "y": [1.0, 2.0]})
        original = frame.copy(deep=True)
        TargetEncoder().fit_transform(frame, "cat", "y")
        pd.testing.assert_frame_equal(frame, original)


# ---------------------------------------------------------------------------
# MissingIndicatorAdder
# ---------------------------------------------------------------------------

class TestMissingIndicatorAdder:
    def test_transform_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            MissingIndicatorAdder().transform(pd.DataFrame({"a": [1]}))

    def test_fit_requires_dataframe(self):
        with pytest.raises(TypeError):
            MissingIndicatorAdder().fit("nope")

    def test_flags_only_columns_with_missing_at_fit_time(self):
        frame = pd.DataFrame({"a": [1.0, np.nan], "b": [1, 2]})
        adder = MissingIndicatorAdder().fit(frame)
        assert adder.flagged_columns_ == ["a"]

    def test_schema_stable_across_batches_with_no_missing(self):
        """ASSUMPTION: a column flagged at fit time still gets an
        indicator at transform time, even on a batch with zero missing
        values in it — this is the whole point of the class over a
        one-shot function."""
        train = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        adder = MissingIndicatorAdder().fit(train)

        clean_batch = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # no missing here
        result = adder.transform(clean_batch)
        assert "a_is_missing" in result.columns
        assert (result["a_is_missing"] == False).all()  # noqa: E712

    def test_does_not_mutate_original(self):
        frame = pd.DataFrame({"a": [1.0, np.nan]})
        original = frame.copy(deep=True)
        MissingIndicatorAdder().fit_transform(frame)
        pd.testing.assert_frame_equal(frame, original)


# ---------------------------------------------------------------------------
# One-shot convenience wrappers (thin correctness checks; the underlying
# classes are already tested exhaustively above)
# ---------------------------------------------------------------------------

class TestOneShotWrappers:
    def test_scale_numeric_features(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = scale_numeric_features(frame, method="minmax")
        assert result["a"].min() == pytest.approx(0.0)

    def test_encode_categorical_features(self):
        frame = pd.DataFrame({"cat": ["a", "b"]})
        result = encode_categorical_features(frame, method="onehot")
        assert "cat_a" in result.columns

    def test_bin_numeric_feature(self):
        frame = pd.DataFrame({"v": [1, 2, 3, 4]})
        result = bin_numeric_feature(frame, "v", n_bins=2)
        assert "v_binned" in result.columns

    def test_target_encode(self):
        frame = pd.DataFrame({"cat": ["a", "b"], "y": [1.0, 2.0]})
        result = target_encode(frame, "cat", "y")
        assert "cat_target_encoded" in result.columns

    def test_add_missing_indicators(self):
        frame = pd.DataFrame({"a": [1.0, np.nan]})
        result = add_missing_indicators(frame)
        assert "a_is_missing" in result.columns


# ---------------------------------------------------------------------------
# log_transform / sqrt_transform / power_transform
# ---------------------------------------------------------------------------

class TestLogTransform:
    def test_rejects_missing_column(self):
        with pytest.raises(KeyError):
            log_transform(pd.DataFrame({"a": [1]}), ["nope"])

    def test_default_offset_requires_strictly_positive(self):
        with pytest.raises(ValueError):
            log_transform(pd.DataFrame({"x": [0, 1, 2]}), ["x"])

    def test_offset_one_handles_zero(self):
        """ASSUMPTION: offset=1 reproduces log1p semantics, safely
        handling zeros."""
        result = log_transform(pd.DataFrame({"x": [0, 1, 2]}), ["x"], offset=1.0)
        assert result["x_log"].tolist() == pytest.approx(np.log1p([0, 1, 2]).tolist())

    def test_missing_values_preserved(self):
        result = log_transform(pd.DataFrame({"x": [1.0, np.nan, 4.0]}), ["x"])
        assert pd.isna(result["x_log"].iloc[1])

    def test_original_column_untouched(self):
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        result = log_transform(frame, ["x"])
        pd.testing.assert_series_equal(result["x"], frame["x"])


class TestSqrtTransform:
    def test_rejects_negative_values(self):
        with pytest.raises(ValueError):
            sqrt_transform(pd.DataFrame({"x": [-1, 4, 9]}), ["x"])

    def test_computes_correct_values(self):
        result = sqrt_transform(pd.DataFrame({"x": [0, 4, 9]}), ["x"])
        assert result["x_sqrt"].tolist() == pytest.approx([0.0, 2.0, 3.0])


class TestPowerTransform:
    def test_rejects_invalid_method(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            power_transform(frame, ["a"], method="bogus")

    def test_rejects_missing_values(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError):
            power_transform(frame, ["a"])

    def test_yeo_johnson_handles_negative_values(self):
        """ASSUMPTION: yeo-johnson supports negative values, unlike box-cox."""
        frame = pd.DataFrame({"a": [-5.0, 0.0, 5.0, 10.0]})
        result = power_transform(frame, ["a"], method="yeo-johnson")
        assert not result["a_power"].isna().any()

    def test_box_cox_rejects_non_positive_via_sklearn(self):
        frame = pd.DataFrame({"a": [-1.0, 1.0, 2.0]})
        with pytest.raises(ValueError):
            power_transform(frame, ["a"], method="box-cox")

    def test_standardize_true_gives_roughly_zero_mean(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0, 16.0]})
        result = power_transform(frame, ["a"], standardize=True)
        assert result["a_power"].mean() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# create_ratio_features / create_difference_features / create_product_features
# ---------------------------------------------------------------------------

class TestPairwiseFeatureCreation:
    def test_ratio_rejects_duplicate_output_name(self):
        frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError):
            create_ratio_features(frame, {"a": ("a", "b")})

    def test_ratio_zero_division_nan(self):
        frame = pd.DataFrame({"a": [1.0], "b": [0.0]})
        result = create_ratio_features(frame, {"r": ("a", "b")}, zero_division="nan")
        assert pd.isna(result["r"].iloc[0])

    def test_ratio_zero_division_inf(self):
        frame = pd.DataFrame({"a": [1.0], "b": [0.0]})
        result = create_ratio_features(frame, {"r": ("a", "b")}, zero_division="inf")
        assert np.isinf(result["r"].iloc[0])

    def test_ratio_zero_division_raise(self):
        frame = pd.DataFrame({"a": [1.0], "b": [0.0]})
        with pytest.raises(ZeroDivisionError):
            create_ratio_features(frame, {"r": ("a", "b")}, zero_division="raise")

    def test_difference_is_order_sensitive(self):
        frame = pd.DataFrame({"a": [10.0], "b": [3.0]})
        result = create_difference_features(frame, {"d": ("a", "b")})
        assert result["d"].iloc[0] == 7.0

    def test_product_values_correct(self):
        frame = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        result = create_product_features(frame, {"p": ("a", "b")})
        assert result["p"].tolist() == [8.0, 15.0]

    def test_missing_input_propagates_to_missing_output(self):
        frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
        result = create_product_features(frame, {"p": ("a", "b")})
        assert pd.isna(result["p"].iloc[1])


class TestPolynomialFeatures:
    def test_rejects_degree_below_one(self):
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError):
            create_polynomial_features(frame, ["a"], degree=0)

    def test_produces_true_cross_terms(self):
        """ASSUMPTION: unlike single-column power features, this produces
        genuine interaction terms between different columns."""
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = create_polynomial_features(frame, ["a", "b"], degree=2)
        assert "a b" in result.columns
        assert result["a b"].tolist() == [3.0, 8.0]

    def test_does_not_duplicate_original_columns(self):
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        result = create_polynomial_features(frame, ["a"], degree=2)
        assert list(result.columns).count("a") == 1


# ---------------------------------------------------------------------------
# add_datetime_features / add_cyclical_features / add_frequency_features
# ---------------------------------------------------------------------------

class TestAddDatetimeFeatures:
    def test_default_errors_raise_on_bad_date(self):
        """ASSUMPTION: errors='raise' is the default, unlike silently
        coercing to missing."""
        frame = pd.DataFrame({"d": ["2024-01-01", "not-a-date"]})
        with pytest.raises((ValueError, TypeError)):
            add_datetime_features(frame, "d")

    def test_coerce_produces_missing_features_instead_of_raising(self):
        frame = pd.DataFrame({"d": ["2024-01-06", "not-a-date"]})
        result = add_datetime_features(frame, "d", components=["year"], errors="coerce")
        assert pd.isna(result["d_year"].iloc[1])

    def test_is_weekend_detects_saturday(self):
        frame = pd.DataFrame({"d": ["2024-01-06"]})  # a Saturday
        result = add_datetime_features(frame, "d", components=["is_weekend"])
        assert result["d_is_weekend"].iloc[0] == True  # noqa: E712

    def test_rejects_unsupported_component(self):
        frame = pd.DataFrame({"d": ["2024-01-01"]})
        with pytest.raises(ValueError):
            add_datetime_features(frame, "d", components=["century"])

    def test_prefix_used_for_output_names(self):
        frame = pd.DataFrame({"d": ["2024-01-01"]})
        result = add_datetime_features(frame, "d", components=["year"], prefix="event")
        assert "event_year" in result.columns


class TestAddCyclicalFeatures:
    def test_rejects_non_positive_period(self):
        frame = pd.DataFrame({"hour": [0, 12]})
        with pytest.raises(ValueError):
            add_cyclical_features(frame, "hour", period=0)

    def test_zero_maps_to_sin_zero_cos_one(self):
        frame = pd.DataFrame({"hour": [0]})
        result = add_cyclical_features(frame, "hour", period=24)
        assert result["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert result["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_wraparound_points_are_numerically_close(self):
        """ASSUMPTION: hour 23 and hour 0 should be close in sin/cos space,
        unlike their raw integer distance."""
        frame = pd.DataFrame({"hour": [0, 23]})
        result = add_cyclical_features(frame, "hour", period=24)
        distance = np.sqrt(
            (result["hour_sin"].iloc[0] - result["hour_sin"].iloc[1]) ** 2
            + (result["hour_cos"].iloc[0] - result["hour_cos"].iloc[1]) ** 2
        )
        assert distance < 0.5


class TestAddFrequencyFeatures:
    def test_keeps_original_column(self):
        """ASSUMPTION: unlike CategoricalEncoder(method='frequency'), the
        source column is preserved alongside the new frequency column."""
        frame = pd.DataFrame({"c": ["x", "y", "x"]})
        result = add_frequency_features(frame, ["c"])
        assert "c" in result.columns
        assert "c_frequency" in result.columns

    def test_normalized_frequencies_sum_correctly(self):
        frame = pd.DataFrame({"c": ["x", "y", "x"]})
        result = add_frequency_features(frame, ["c"], normalize=True)
        assert result.loc[0, "c_frequency"] == pytest.approx(2 / 3)

    def test_raw_counts_when_not_normalized(self):
        frame = pd.DataFrame({"c": ["x", "y", "x"]})
        result = add_frequency_features(frame, ["c"], normalize=False)
        assert result.loc[0, "c_frequency"] == 2.0


# ---------------------------------------------------------------------------
# Filter-style feature selection
# ---------------------------------------------------------------------------

class TestLowVarianceFeatures:
    def test_rejects_negative_threshold(self, redundant_frame):
        with pytest.raises(ValueError):
            low_variance_features(redundant_frame, threshold=-1)

    def test_rejects_duplicate_columns(self, redundant_frame):
        with pytest.raises(ValueError):
            low_variance_features(redundant_frame, columns=["x", "x"])

    def test_identifies_constant_column(self, redundant_frame):
        assert "const" in low_variance_features(redundant_frame)

    def test_columns_scoping_restricts_search(self, redundant_frame):
        result = low_variance_features(redundant_frame, columns=["x", "y"])
        assert result == []

    def test_ddof_zero_vs_one_can_differ_at_threshold(self):
        frame = pd.DataFrame({"a": [1, 1, 1, 2]})
        sample_var = frame["a"].var(ddof=1)
        population_var = frame["a"].var(ddof=0)
        assert sample_var != population_var  # sanity check on the fixture
        flagged_sample = low_variance_features(frame, threshold=population_var, ddof=1)
        flagged_population = low_variance_features(frame, threshold=population_var, ddof=0)
        assert "a" not in flagged_sample
        assert "a" in flagged_population

    def test_drop_low_variance_features_removes_flagged(self, redundant_frame):
        result = drop_low_variance_features(redundant_frame)
        assert "const" not in result.columns


class TestCorrelatedFeatures:
    def test_finds_perfectly_correlated_pair(self, redundant_frame):
        result = find_correlated_feature_pairs(redundant_frame, threshold=0.99)
        assert len(result) == 1
        assert set(result.iloc[0][["feature_1", "feature_2"]]) == {"x", "y"}

    def test_columns_scoping(self, redundant_frame):
        result = find_correlated_feature_pairs(redundant_frame, threshold=0.99, columns=["x", "const"])
        assert result.empty  # 'y' excluded from consideration

    def test_drop_correlated_features_keeps_first(self, redundant_frame):
        result = drop_correlated_features(redundant_frame, threshold=0.99)
        assert "x" in result.columns
        assert "y" not in result.columns

    def test_does_not_mutate_original(self, redundant_frame):
        original = redundant_frame.copy(deep=True)
        drop_correlated_features(redundant_frame, threshold=0.99)
        pd.testing.assert_frame_equal(redundant_frame, original)


# ---------------------------------------------------------------------------
# Cross-cutting: every function rejects non-DataFrame input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda f: scale_numeric_features(f),
        lambda f: encode_categorical_features(f),
        lambda f: bin_numeric_feature(f, "a"),
        lambda f: target_encode(f, "a", "b"),
        lambda f: add_missing_indicators(f),
        lambda f: log_transform(f, ["a"]),
        lambda f: sqrt_transform(f, ["a"]),
        lambda f: power_transform(f, ["a"]),
        lambda f: create_ratio_features(f, {"r": ("a", "b")}),
        lambda f: create_difference_features(f, {"d": ("a", "b")}),
        lambda f: create_product_features(f, {"p": ("a", "b")}),
        lambda f: create_polynomial_features(f, ["a"]),
        lambda f: add_datetime_features(f, "a"),
        lambda f: add_cyclical_features(f, "a", period=7),
        lambda f: add_frequency_features(f, ["a"]),
        lambda f: low_variance_features(f),
        lambda f: drop_low_variance_features(f),
        lambda f: find_correlated_feature_pairs(f),
        lambda f: drop_correlated_features(f),
    ],
)
def test_all_functions_reject_non_dataframe_input(call):
    with pytest.raises(TypeError):
        call("definitely not a dataframe")