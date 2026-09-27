import numpy as np
from hypothesis import given, strategies as st

from makoding.statistical_analysis import (
    cohens_d,
    distribution_summary,
    forecast_metrics,
    mean_confidence_interval,
    multiple_testing_correction,
    rank_biserial_from_u,
)


finite_float = st.floats(
    min_value=-1e4,
    max_value=1e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@given(st.lists(finite_float, min_size=2, max_size=100))
def test_distribution_summary_is_finite(values):
    result = distribution_summary(values)
    assert result.n_observations == len(values)
    assert np.isfinite(result.mean)
    assert np.isfinite(result.median)
    assert np.isfinite(result.variance)
    assert result.minimum <= result.maximum


@given(st.lists(finite_float, min_size=2, max_size=100))
def test_mean_confidence_interval_contains_sample_mean(values):
    lower, upper = mean_confidence_interval(values)
    mean = float(np.mean(values))
    assert lower <= mean <= upper


@given(
    st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
def test_multiple_testing_adjustments_stay_in_unit_interval(p_values):
    for method in ("bonferroni", "holm", "fdr_bh"):
        result = multiple_testing_correction(p_values, method=method)
        assert np.all(result.adjusted_p_value >= 0)
        assert np.all(result.adjusted_p_value <= 1)


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
)
def test_rank_biserial_respects_bounds(n_first, n_second):
    maximum = n_first * n_second
    u = min(maximum, maximum // 2)
    value = rank_biserial_from_u(u, n_first, n_second)
    assert -1 <= value <= 1


@given(
    st.lists(finite_float, min_size=2, max_size=50),
    st.lists(finite_float, min_size=2, max_size=50),
)
def test_cohens_d_is_antisymmetric(first, second):
    # Skip exact zero pooled variance cases.
    if np.var(first, ddof=1) == 0 and np.var(second, ddof=1) == 0:
        return
    d1 = cohens_d(first, second)
    d2 = cohens_d(second, first)
    assert np.isfinite(d1)
    assert np.isfinite(d2)
    assert np.isclose(d1, -d2)


@given(
    st.lists(finite_float, min_size=2, max_size=50),
    st.lists(finite_float, min_size=2, max_size=50),
)
def test_forecast_metrics_nonnegative_errors(actual, predicted):
    if len(actual) != len(predicted):
        return
    result = forecast_metrics(actual, predicted)
    assert result["mae"] >= 0
    assert result["mse"] >= 0
    assert result["rmse"] >= 0


def test_multiple_testing_preserves_original_order():
    p_values = [0.04, 0.001, 0.2, 0.01]
    result = multiple_testing_correction(p_values, method="holm")
    assert result["p_value"].tolist() == p_values
