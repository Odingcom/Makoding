import numpy as np
import pandas as pd
import pytest

from makoding.statistical_analysis import (
    STATSMODELS_AVAILABLE,
    __version__,
    available_distributions,
    beta_binomial_posterior,
    chi_square_test,
    cohens_d,
    cramers_v,
    describe_series,
    distribution_summary,
    epsilon_squared,
    eta_squared,
    forecast_metrics,
    hedges_g,
    independent_t_test,
    ljung_box_test,
    mean_confidence_interval,
    multiple_testing_correction,
    normal_normal_posterior,
    normality_test,
    omega_squared,
    ols_regression,
    paired_cohens_d,
    paired_t_test,
    rank_biserial_from_u,
    recommend_test,
    regression_diagnostics,
    stationarity_test,
    welch_anova,
)


def test_version_and_distribution_catalogue():
    assert __version__ == "0.2.0"
    catalogue = available_distributions()
    assert "normal" in catalogue
    assert "binomial" in catalogue
    assert len(catalogue) >= 15


def test_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        distribution_summary([1])
    with pytest.raises(ValueError):
        distribution_summary([1, np.nan])
    with pytest.raises(TypeError):
        distribution_summary([1, "bad"])
    with pytest.raises(ValueError):
        mean_confidence_interval([1, 2], confidence=1.0)


def test_descriptive_summary():
    result = distribution_summary([1, 2, 3, 4])
    assert result.n_observations == 4
    assert result.mean == pytest.approx(2.5)
    assert result.variance == pytest.approx(5 / 3)

    series = describe_series([1, 2, 3, 4, 5])
    assert series["n"] == 5
    assert series["median"] == pytest.approx(3)
    assert series["q0.5"] == pytest.approx(3)


def test_hypothesis_tests_return_structured_results():
    one = independent_t_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert one.p_value < 0.05
    assert one.reject_null

    paired = paired_t_test([10, 11, 12, 13], [11, 12, 13, 14])
    assert paired.p_value < 0.05


def test_alternative_hypotheses_are_supported():
    result = independent_t_test(
        [10, 11, 12, 13],
        [1, 2, 3, 4],
        alternative="greater",
    )
    assert result.p_value < 0.05


def test_welch_anova_is_available_without_scipy_equal_var_argument():
    result = welch_anova(
        [1, 2, 3, 4],
        [2, 3, 4, 5, 6],
        [10, 12, 14, 16, 18, 20],
    )
    assert np.isfinite(result.statistic)
    assert 0 <= result.p_value <= 1
    assert "degrees_of_freedom" in result.details


def test_chi_square_validates_counts_and_returns_expected():
    result = chi_square_test([[20, 30], [30, 20]])
    assert 0 <= result.p_value <= 1
    assert result.details["degrees_of_freedom"] == 1
    assert result.details["minimum_expected_frequency"] > 0

    with pytest.raises(ValueError):
        chi_square_test([[1.5, 2], [3, 4]])
    with pytest.raises(ValueError):
        chi_square_test([[1, 0], [0, 0]])


def test_normality_sample_size_guards():
    with pytest.raises(ValueError):
        normality_test([1, 2], method="shapiro")
    with pytest.raises(ValueError):
        normality_test([1, 2, 3, 4], method="normaltest")
    result = normality_test(np.arange(10, dtype=float), method="normaltest")
    assert 0 <= result.p_value <= 1


def test_effect_sizes():
    a = [1, 2, 3, 4, 5]
    b = [2, 3, 4, 5, 6]
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    assert d < 0
    assert abs(g) < abs(d)
    assert paired_cohens_d(a, [2, 3, 4, 5, 7]) > 0

    eta = eta_squared([1, 2, 3], [5, 6, 7], [10, 11, 12])
    omega = omega_squared([1, 2, 3], [5, 6, 7], [10, 11, 12])
    eps = epsilon_squared([1, 2, 3], [5, 6, 7], [10, 11, 12])
    assert 0 <= eta <= 1
    assert 0 <= omega <= 1
    assert 0 <= eps <= 1

    assert rank_biserial_from_u(0, 10, 10) == pytest.approx(1.0)
    assert 0 <= cramers_v([[20, 10], [10, 20]]) <= 1


def test_confidence_interval():
    low, high = mean_confidence_interval([1, 2, 3, 4, 5])
    assert low < 3 < high


def test_multiple_testing_methods():
    p = [0.001, 0.02, 0.04, 0.2]
    for method in ("bonferroni", "holm", "fdr_bh"):
        result = multiple_testing_correction(p, method=method)
        assert list(result.columns) == [
            "p_value",
            "adjusted_p_value",
            "alpha",
            "reject_null",
            "method",
        ]
        assert np.all(result.adjusted_p_value >= result.p_value - 1e-12)
        assert np.all((result.adjusted_p_value >= 0) & (result.adjusted_p_value <= 1))


def test_recommendation_engine_is_transparent():
    result = recommend_test(
        design="independent",
        groups=2,
        normality="normal",
        equal_variance="no",
    )
    assert result["test"] == "Welch independent t-test"
    assert "variance" in result["reason"]


def test_beta_binomial_strict_integer_validation():
    result = beta_binomial_posterior(30, 50)
    assert result["posterior_mean"] == pytest.approx(31 / 52)
    assert result["credible_interval_lower"] < result["posterior_mean"] < result["credible_interval_upper"]

    with pytest.raises(TypeError):
        beta_binomial_posterior(1.5, 10)
    with pytest.raises(TypeError):
        beta_binomial_posterior(1, 10.0)


def test_normal_normal_validation_and_interval():
    result = normal_normal_posterior(
        0,
        1,
        2,
        1,
        10,
        credible_level=0.95,
    )
    assert result["posterior_variance"] < 1
    assert result["credible_interval_lower"] < result["posterior_mean"] < result["credible_interval_upper"]


def test_ols_missing_data_is_explicit():
    pytest.importorskip("statsmodels")
    frame = pd.DataFrame({
        "y": [1, 2, 3, 4, 5, np.nan],
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [2, 4, 6, 8, 10, 12],
    })
    model, metadata = ols_regression(
        frame,
        target="y",
        features=["x1", "x2"],
        return_metadata=True,
    )
    assert len(model.params) == 3
    assert metadata["n_input"] == 6
    assert metadata["n_used"] == 5
    assert metadata["n_dropped"] == 1

    with pytest.raises(ValueError):
        ols_regression(frame, target="y", features=["x1", "x2"], missing="raise")


def test_regression_diagnostics():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(42)
    x1 = np.arange(50, dtype=float)
    x2 = rng.normal(size=50)
    y = 2 + 3 * x1 + 0.5 * x2 + rng.normal(scale=0.5, size=50)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    model = ols_regression(frame, target="y", features=["x1", "x2"])
    diagnostics = regression_diagnostics(model)
    assert diagnostics["n_observations"] == 50
    assert 0 <= diagnostics["r_squared"] <= 1
    assert "vif" in diagnostics
    assert set(diagnostics["vif"]) == {"x1", "x2"}


def test_stationarity_and_ljung_box():
    if not STATSMODELS_AVAILABLE:
        pytest.skip("statsmodels not installed")
    rng = np.random.default_rng(1)
    values = rng.normal(size=80)
    adf = stationarity_test(values, method="adf")
    kpss = stationarity_test(values, method="kpss")
    assert 0 <= adf.p_value <= 1
    assert 0 <= kpss.p_value <= 1

    lb = ljung_box_test(values, lags=[5, 10])
    assert list(lb.index) == [5, 10]
    assert "reject_null" in lb.columns


def test_forecast_metrics_and_mase():
    result = forecast_metrics(
        [10, 20, 30],
        [11, 19, 32],
        training=[5, 10, 15, 20, 25],
    )
    assert result["mae"] == pytest.approx(4 / 3)
    assert result["bias"] == pytest.approx(-2 / 3)
    assert result["mase"] > 0


def test_forecast_metrics_all_zero_actuals_omits_mape():
    result = forecast_metrics([0, 0, 0], [1, 2, 3])
    assert "mape" not in result
