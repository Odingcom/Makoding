"""
Makoding statistical analysis
=============================

Framework-independent statistical analysis utilities for DataLab Pro.

This module provides a first integrated statistical layer covering:

- Probability distributions
- Descriptive distribution summaries
- Classical hypothesis testing
- Paired t-tests
- ANOVA and Welch ANOVA
- Non-parametric tests
- Correlation
- Normality testing
- OLS regression
- Regression diagnostics
- Bayesian conjugate analysis
- Stationarity testing
- Autocorrelation diagnostics
- ARIMA forecasting
- Forecast evaluation

Design principles
-----------------
1. No Streamlit dependency.
2. Validate inputs before statistical operations -- including cheap,
   pure-Python checks like ``alpha``'s range, which are validated before
   any statistic is computed, not after.
3. Return structured Python objects/DataFrames.
4. Keep inference separate from presentation.
5. Do not silently transform data.
6. Make assumptions visible to the caller.
7. Keep advanced dependencies optional where practical.

A note on sample size and silent NaN results
---------------------------------------------
Some underlying SciPy routines respond to too-small samples not by
raising, but by emitting a warning and returning ``NaN`` for both the
statistic and the p-value. Left unhandled, that ``NaN`` p-value would
compare as not-less-than any ``alpha`` and silently produce a *confident-
looking* ``reject_null=False`` result -- a dangerous failure mode if
warnings happen to be suppressed (common in production logging setups).
Every test function in this module that has a hard underlying minimum
(e.g. Shapiro-Wilk's n >= 3) validates it explicitly and raises
``ValueError`` instead of letting a NaN result through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Optional statsmodels imports
# ---------------------------------------------------------------------------

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import (
        acorr_ljungbox,
        het_breuschpagan,
        het_white,
    )
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss

    STATSMODELS_AVAILABLE = True

except ImportError:  # pragma: no cover
    sm = None
    acorr_ljungbox = None
    het_breuschpagan = None
    het_white = None
    variance_inflation_factor = None
    durbin_watson = None
    jarque_bera = None
    ARIMA = None
    adfuller = None
    kpss = None
    STATSMODELS_AVAILABLE = False


__version__ = "0.2.0"


__all__ = [
    "__version__",
    "STATSMODELS_AVAILABLE",
    "DistributionSummary",
    "TestResult",
    "validate_numeric_series",
    "distribution_summary",
    "describe_series",
    "available_distributions",
    "one_sample_t_test",
    "independent_t_test",
    "paired_t_test",
    "one_way_anova",
    "welch_anova",
    "mann_whitney_test",
    "wilcoxon_test",
    "kruskal_test",
    "chi_square_test",
    "normality_test",
    "correlation_test",
    "mean_confidence_interval",
    "cohens_d",
    "hedges_g",
    "paired_cohens_d",
    "eta_squared",
    "omega_squared",
    "epsilon_squared",
    "rank_biserial_from_u",
    "cramers_v",
    "multiple_testing_correction",
    "recommend_test",
    "ols_regression",
    "regression_diagnostics",
    "beta_binomial_posterior",
    "normal_normal_posterior",
    "stationarity_test",
    "ljung_box_test",
    "fit_arima",
    "forecast_arima",
    "forecast_metrics",
]


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for an empirical sample.

    Attributes
    ----------
    name:
        Always ``"empirical"`` -- these are sample statistics, not a
        fitted parametric distribution.
    n_observations, mean, median, variance, standard_deviation,
    minimum, maximum, skewness, kurtosis:
        ``variance``/``standard_deviation`` use ``ddof=1`` (sample, not
        population, convention). ``kurtosis`` is *excess* kurtosis
        (0.0 for a normal distribution), SciPy's default.
    """

    name: str
    n_observations: int
    mean: float
    median: float
    variance: float
    standard_deviation: float
    minimum: float
    maximum: float
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class TestResult:
    """Standardized statistical test result.

    Attributes
    ----------
    test:
        Human-readable test name.
    statistic, p_value:
        The test statistic and its associated p-value.
    alpha:
        The significance level the caller supplied (or the default).
    reject_null:
        ``p_value < alpha``.
    interpretation:
        A one-line, plain-language summary of ``reject_null``. This
        describes the *statistical* conclusion only (whether to reject
        the null hypothesis) -- it does not and cannot tell you whether
        that conclusion is practically/substantively meaningful, and it
        says nothing about effect size.
    details:
        Optional extra, test-specific diagnostic information (e.g. ADF's
        critical values and lag order) that doesn't fit the common
        fields above. Empty for tests that have nothing extra to report.
        Purely additive: existing code that only reads the fields above
        is unaffected.
    """

    test: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    interpretation: str
    details: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Validation
# ============================================================================


def validate_numeric_series(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    name: str = "values",
    min_observations: int = 2,
    allow_constant: bool = True,
) -> np.ndarray:
    """Validate and return a finite numeric one-dimensional array.

    Unlike ``pandas.to_numeric(errors='coerce')``, this function never
    silently converts invalid observations to missing values. Invalid,
    missing, infinite, or non-numeric observations are rejected before
    a statistical operation is performed.

    Parameters
    ----------
    values:
        Input data.
    name:
        Name used in error messages.
    min_observations:
        Minimum required number of observations.
    allow_constant:
        Whether a zero-variance sample is allowed. Constant samples are
        accepted by default because some descriptive operations can still
        be meaningful; inferential tests that require variation perform
        their own stricter checks.
    """
    if values is None:
        raise ValueError(f"{name} cannot be None.")

    if isinstance(min_observations, bool) or min_observations < 1:
        raise ValueError("min_observations must be a positive integer.")

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < min_observations:
        raise ValueError(
            f"{name} requires at least {min_observations} observations."
        )

    try:
        array = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if not allow_constant and np.ptp(array) == 0:
        raise ValueError(f"{name} must contain more than one unique value.")

    return array



def _validate_alpha(alpha: float) -> float:
    """Validate a significance level, eagerly (called before computation)."""
    if isinstance(alpha, bool):
        raise TypeError("alpha must be numeric, not bool.")

    alpha = float(alpha)

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be strictly between 0 and 1."
        )

    return alpha


def _build_test_result(
    *,
    test: str,
    statistic: float,
    p_value: float,
    alpha: float,
    details: dict[str, Any] | None = None,
) -> TestResult:
    """Assemble a :class:`TestResult`.

    ``alpha`` is expected to already be validated by the caller (every
    public function in this module validates ``alpha`` up front, before
    computing any statistic) -- this function re-validates defensively
    rather than assuming, since it is not itself part of the public API
    boundary.
    """
    alpha = _validate_alpha(alpha)

    statistic = float(statistic)
    p_value = float(p_value)

    reject = p_value < alpha

    interpretation = (
        "Reject the null hypothesis at the selected "
        "significance level."
        if reject
        else
        "Do not reject the null hypothesis at the "
        "selected significance level."
    )

    return TestResult(
        test=test,
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        reject_null=reject,
        interpretation=interpretation,
        details=dict(details) if details else {},
    )


def _require_statsmodels() -> None:
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for this operation. "
            "Install it with: python -m pip install statsmodels"
        )


# ============================================================================
# Probability distributions
# ============================================================================


def describe_series(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    quantiles: Sequence[float] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
) -> pd.Series:
    """Return a production-oriented descriptive profile for one numeric series.

    The output includes location, spread, shape, and requested quantiles.
    Quantiles must lie in ``[0, 1]`` and are validated explicitly.
    """
    values_array = validate_numeric_series(values, name="values")
    q = np.asarray(list(quantiles), dtype=float)
    if q.ndim != 1 or q.size == 0 or np.any(~np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("quantiles must be a non-empty one-dimensional sequence in [0, 1].")

    result: dict[str, float | int] = {
        "n": int(values_array.size),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
        "std": float(np.std(values_array, ddof=1)),
        "variance": float(np.var(values_array, ddof=1)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "range": float(np.ptp(values_array)),
        "skewness": float(stats.skew(values_array, bias=False)),
        "kurtosis_excess": float(stats.kurtosis(values_array, bias=False)),
    }
    for probability in q:
        label = f"q{probability:g}"
        result[label] = float(np.quantile(values_array, probability))
    return pd.Series(result, dtype=float)


def available_distributions() -> dict[str, str]:
    """Return the main probability distributions supported conceptually
    by the DataLab statistical layer.
    """

    return {
        "normal": "Continuous symmetric measurement distribution",
        "student_t": "Small-sample inference and heavy-tailed errors",
        "chi_square": "Variance and categorical inference",
        "f": "Variance ratios and ANOVA",
        "uniform": "Bounded equal-density distribution",
        "exponential": "Waiting-time distribution",
        "gamma": "Positive continuous skewed distribution",
        "lognormal": "Positive multiplicative processes",
        "weibull": "Reliability and failure-time modelling",
        "beta": "Probabilities and proportions",
        "pareto": "Heavy-tailed phenomena",
        "bernoulli": "Single binary outcome",
        "binomial": "Number of successes in fixed trials",
        "poisson": "Event counts",
        "negative_binomial": "Overdispersed event counts",
        "geometric": "Trials until success",
    }


def distribution_summary(
    values: Sequence[float] | pd.Series | np.ndarray,
) -> DistributionSummary:
    """Calculate distribution-free descriptive statistics.

    Parameters
    ----------
    values:
        Sample data.

    Returns
    -------
    DistributionSummary

    Raises
    ------
    See :func:`validate_numeric_series`.
    """

    array = validate_numeric_series(values)

    return DistributionSummary(
        name="empirical",
        n_observations=int(array.size),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        variance=float(np.var(array, ddof=1)),
        standard_deviation=float(np.std(array, ddof=1)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        skewness=float(stats.skew(array)),
        kurtosis=float(stats.kurtosis(array)),
    )


# ============================================================================
# Classical hypothesis tests
# ============================================================================


def one_sample_t_test(
    sample: Sequence[float] | pd.Series | np.ndarray,
    population_mean: float,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Test whether a sample mean differs from a specified population mean.

    Parameters
    ----------
    sample:
        Sample data.
    population_mean:
        The hypothesized population mean under the null hypothesis.
    alpha:
        Significance level, strictly between 0 and 1.
    alternative:
        ``"two-sided"``, ``"less"``, or ``"greater"`` (forwarded to
        SciPy, which raises a clear error for any other value).

    Returns
    -------
    TestResult

    Raises
    ------
    See :func:`validate_numeric_series` and :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    values = validate_numeric_series(sample)

    result = stats.ttest_1samp(
        values,
        popmean=float(population_mean),
        alternative=alternative,
    )

    return _build_test_result(
        test="One-sample t-test",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def independent_t_test(
    first: Sequence[float] | pd.Series | np.ndarray,
    second: Sequence[float] | pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
    equal_variance: bool = False,
    alternative: str = "two-sided",
) -> TestResult:
    """Compare the means of two independent samples.

    Parameters
    ----------
    first, second:
        The two independent samples.
    alpha:
        Significance level.
    equal_variance:
        ``False`` (default) performs Welch's t-test, which does not
        assume equal population variances and is the generally safer
        default. ``True`` performs the classic Student's t-test.
    alternative:
        ``"two-sided"``, ``"less"``, or ``"greater"``.

    Returns
    -------
    TestResult

    Raises
    ------
    See :func:`validate_numeric_series` and :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    first_values = validate_numeric_series(
        first,
        name="first",
    )

    second_values = validate_numeric_series(
        second,
        name="second",
    )

    result = stats.ttest_ind(
        first_values,
        second_values,
        equal_var=equal_variance,
        alternative=alternative,
    )

    test_name = (
        "Independent t-test"
        if equal_variance
        else "Welch independent t-test"
    )

    return _build_test_result(
        test=test_name,
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def paired_t_test(
    before: Sequence[float] | pd.Series | np.ndarray,
    after: Sequence[float] | pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Compare the means of two related (paired) samples.

    Parameters
    ----------
    before, after:
        The two paired samples; must be the same length and in
        corresponding order (``before[i]`` pairs with ``after[i]``).
    alpha:
        Significance level.
    alternative:
        ``"two-sided"``, ``"less"``, or ``"greater"``.

    Returns
    -------
    TestResult

    Raises
    ------
    ValueError
        If the two samples have different lengths, in addition to the
        errors from :func:`validate_numeric_series` / :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    before_values = validate_numeric_series(
        before,
        name="before",
    )

    after_values = validate_numeric_series(
        after,
        name="after",
    )

    if before_values.shape != after_values.shape:
        raise ValueError(
            "Paired samples must have the same length."
        )

    result = stats.ttest_rel(
        before_values,
        after_values,
        alternative=alternative,
    )

    return _build_test_result(
        test="Paired t-test",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def one_way_anova(
    *groups: Sequence[float] | pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Perform standard (equal-variance-assumed) one-way ANOVA.

    Parameters
    ----------
    *groups:
        Two or more independent groups.
    alpha:
        Significance level.

    Returns
    -------
    TestResult

    Raises
    ------
    ValueError
        If fewer than two groups are supplied, in addition to the
        errors from :func:`validate_numeric_series` / :func:`_validate_alpha`.

    Assumptions
    -----------
    Assumes homogeneity of variance across groups. If that assumption
    is doubtful, prefer :func:`welch_anova`.
    """

    alpha = _validate_alpha(alpha)

    if len(groups) < 2:
        raise ValueError(
            "At least two groups are required."
        )

    validated = [
        validate_numeric_series(
            group,
            name=f"group_{index + 1}",
        )
        for index, group in enumerate(groups)
    ]

    result = stats.f_oneway(*validated)

    return _build_test_result(
        test="One-way ANOVA",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def welch_anova(
    *groups: Sequence[float] | pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Perform Welch's one-way ANOVA without assuming equal variances.

    The implementation follows the Welch ANOVA formulation using group
    sizes, means, and unbiased sample variances. This keeps the public API
    stable across SciPy versions and makes the method available even when
    ``scipy.stats.f_oneway(equal_var=False)`` is unavailable.
    """
    alpha = _validate_alpha(alpha)
    if len(groups) < 2:
        raise ValueError("At least two groups are required.")

    validated = [
        validate_numeric_series(group, name=f"group_{i + 1}", min_observations=2)
        for i, group in enumerate(groups)
    ]
    if any(np.var(group, ddof=1) == 0 for group in validated):
        raise ValueError("Welch ANOVA requires non-zero within-group variance in every group.")

    n = np.asarray([len(group) for group in validated], dtype=float)
    means = np.asarray([np.mean(group) for group in validated], dtype=float)
    variances = np.asarray([np.var(group, ddof=1) for group in validated], dtype=float)
    k = float(len(validated))

    weights = n / variances
    weight_sum = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / weight_sum)

    numerator_core = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1.0))
    correction = (
        2.0 * (k - 2.0)
        / (k**2 - 1.0)
        * float(np.sum(((1.0 - weights / weight_sum) ** 2) / (n - 1.0)))
    )
    denominator = 1.0 + correction
    statistic = numerator_core / denominator

    df1 = k - 1.0
    df2 = (k**2 - 1.0) / (
        3.0 * float(np.sum(((1.0 - weights / weight_sum) ** 2) / (n - 1.0)))
    )
    p_value = float(stats.f.sf(statistic, df1, df2))

    return _build_test_result(
        test="Welch ANOVA",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        details={"degrees_of_freedom": {"numerator": float(df1), "denominator": float(df2)}},
    )



def mann_whitney_test(
    first: Sequence[float] | pd.Series | np.ndarray,
    second: Sequence[float] | pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Mann-Whitney U test: non-parametric comparison of two independent samples.

    Parameters
    ----------
    first, second:
        The two independent samples.
    alpha:
        Significance level.

    Returns
    -------
    TestResult

    Raises
    ------
    See :func:`validate_numeric_series` and :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    first_values = validate_numeric_series(
        first,
        name="first",
    )

    second_values = validate_numeric_series(
        second,
        name="second",
    )

    result = stats.mannwhitneyu(
        first_values,
        second_values,
        alternative="two-sided",
    )

    return _build_test_result(
        test="Mann-Whitney U",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def wilcoxon_test(
    before: Sequence[float] | pd.Series | np.ndarray,
    after: Sequence[float] | pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Wilcoxon signed-rank test: non-parametric comparison of paired observations.

    Parameters
    ----------
    before, after:
        The two paired samples; must be the same length.
    alpha:
        Significance level.

    Returns
    -------
    TestResult

    Raises
    ------
    ValueError
        If the two samples have different lengths, in addition to the
        errors from :func:`validate_numeric_series` / :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    before_values = validate_numeric_series(
        before,
        name="before",
    )

    after_values = validate_numeric_series(
        after,
        name="after",
    )

    if before_values.shape != after_values.shape:
        raise ValueError(
            "Paired samples must have the same length."
        )

    result = stats.wilcoxon(
        before_values,
        after_values,
    )

    return _build_test_result(
        test="Wilcoxon signed-rank",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def kruskal_test(
    *groups: Sequence[float] | pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Kruskal-Wallis test: non-parametric comparison of independent groups.

    Parameters
    ----------
    *groups:
        Two or more independent groups.
    alpha:
        Significance level.

    Returns
    -------
    TestResult

    Raises
    ------
    ValueError
        If fewer than two groups are supplied, in addition to the
        errors from :func:`validate_numeric_series` / :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)

    if len(groups) < 2:
        raise ValueError(
            "At least two groups are required."
        )

    validated = [
        validate_numeric_series(
            group,
            name=f"group_{index + 1}",
        )
        for index, group in enumerate(groups)
    ]

    result = stats.kruskal(*validated)

    return _build_test_result(
        test="Kruskal-Wallis",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def chi_square_test(
    observed: Sequence[Sequence[float]],
    *,
    alpha: float = 0.05,
    correction: bool = True,
) -> TestResult:
    """Pearson chi-square test of independence for a contingency table.

    Parameters
    ----------
    observed:
        Two-dimensional observed frequencies. Values must be finite,
        non-negative, and integer-valued because they represent counts.
    alpha:
        Significance level.
    correction:
        Apply Yates' continuity correction for 2x2 tables, matching
        SciPy's default behavior.
    """
    alpha = _validate_alpha(alpha)
    table = np.asarray(observed, dtype=float)

    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("observed must be a two-dimensional table with at least 2 rows and 2 columns.")
    if not np.all(np.isfinite(table)):
        raise ValueError("observed contains NaN or infinite values.")
    if np.any(table < 0):
        raise ValueError("Observed frequencies cannot be negative.")
    if not np.allclose(table, np.round(table)):
        raise ValueError("Observed frequencies must be integer-valued counts.")
    if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        raise ValueError("Every row and column must have a positive marginal total.")

    result = stats.chi2_contingency(table, correction=correction)
    expected = np.asarray(result.expected_freq, dtype=float)

    return _build_test_result(
        test="Chi-square independence",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
        details={
            "degrees_of_freedom": int(result.dof),
            "expected_frequencies": expected.tolist(),
            "correction": bool(correction),
            "minimum_expected_frequency": float(expected.min()),
            "small_expected_cells_below_5": int(np.sum(expected < 5.0)),
        },
    )




# ============================================================================
# Distribution and assumption tests
# ============================================================================


def normality_test(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
    method: str = "shapiro",
) -> TestResult:
    """Test whether a sample is consistent with normality.

    Parameters
    ----------
    values:
        Sample data.
    alpha:
        Significance level.
    method:
        - ``"shapiro"`` (default): Shapiro-Wilk. Generally the most
          powerful choice for small-to-moderate samples. Requires at
          least 3 observations and at most 5,000 (this module's own
          cap, matching a documented SciPy accuracy limitation).
        - ``"normaltest"``: D'Agostino-Pearson, based on skewness and
          kurtosis. Requires at least 8 observations.
        - ``"jarque_bera"``: also skewness/kurtosis-based, asymptotic.
          This module additionally requires at least 8 observations
          for the same reason as ``"normaltest"`` -- SciPy itself does
          not enforce a minimum here, but the chi-square approximation
          the test relies on is not meaningful for very small samples.

    Returns
    -------
    TestResult

    Raises
    ------
    ValueError
        If ``method`` is unsupported, or the sample is too small/large
        for the chosen method, in addition to the errors from
        :func:`validate_numeric_series` / :func:`_validate_alpha`.

    Notes
    -----
    Every minimum-sample-size requirement above is enforced explicitly
    with a ``ValueError`` -- SciPy's own Shapiro-Wilk implementation
    responds to n < 3 with a warning and a ``NaN`` statistic/p-value
    rather than raising, which without this check would silently
    produce a confident-looking (but meaningless) ``reject_null=False``
    result. See the module docstring.
    """

    alpha = _validate_alpha(alpha)
    array = validate_numeric_series(values, allow_constant=False)

    method = method.lower().strip()

    if method == "shapiro":
        if len(array) < 3:
            raise ValueError(
                "Shapiro-Wilk requires at least 3 observations."
            )

        if len(array) > 5_000:
            raise ValueError(
                "Shapiro-Wilk is restricted to 5,000 observations "
                "in this DataLab interface."
            )

        result = stats.shapiro(array)

    elif method == "normaltest":
        if len(array) < 8:
            raise ValueError(
                "normaltest requires at least 8 observations."
            )

        result = stats.normaltest(array)

    elif method == "jarque_bera":
        if len(array) < 8:
            raise ValueError(
                "jarque_bera requires at least 8 observations for a "
                "statistically meaningful result."
            )

        result = stats.jarque_bera(array)

    else:
        raise ValueError(
            "method must be 'shapiro', 'normaltest', "
            "or 'jarque_bera'."
        )

    return _build_test_result(
        test=f"Normality test ({method})",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )


def correlation_test(
    first: Sequence[float] | pd.Series | np.ndarray,
    second: Sequence[float] | pd.Series | np.ndarray,
    *,
    method: str = "pearson",
    alpha: float = 0.05,
) -> TestResult:
    """Test whether two variables are associated.

    Parameters
    ----------
    first, second:
        The two variables; must be the same length.
    method:
        - ``"pearson"``: linear association.
        - ``"spearman"``: monotonic association (rank-based).
        - ``"kendall"``: monotonic association (concordance-based).
    alpha:
        Significance level.

    Returns
    -------
    TestResult
        ``statistic`` is the correlation coefficient itself.

    Raises
    ------
    ValueError
        If the two samples have different lengths or ``method`` is
        unsupported, in addition to the errors from
        :func:`validate_numeric_series` / :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    first_values = validate_numeric_series(
        first,
        name="first",
        allow_constant=False,
    )

    second_values = validate_numeric_series(
        second,
        name="second",
        allow_constant=False,
    )

    if first_values.shape != second_values.shape:
        raise ValueError(
            "Correlation inputs must have the same length."
        )

    method = method.lower().strip()

    if method == "pearson":
        result = stats.pearsonr(
            first_values,
            second_values,
        )

    elif method == "spearman":
        result = stats.spearmanr(
            first_values,
            second_values,
        )

    elif method == "kendall":
        result = stats.kendalltau(
            first_values,
            second_values,
        )

    else:
        raise ValueError(
            "method must be 'pearson', 'spearman', or 'kendall'."
        )

    return _build_test_result(
        test=f"{method.title()} correlation",
        statistic=result.statistic,
        p_value=result.pvalue,
        alpha=alpha,
    )



# ============================================================================
# Effect sizes, confidence intervals, and multiple-testing utilities
# ============================================================================


def mean_confidence_interval(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Student-t confidence interval for a population mean."""
    values_array = validate_numeric_series(values, name="values", min_observations=2)
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")
    mean = float(np.mean(values_array))
    sem = float(stats.sem(values_array))
    if sem == 0:
        return mean, mean
    margin = float(stats.t.ppf((1 + confidence) / 2, len(values_array) - 1) * sem)
    return mean - margin, mean + margin


def cohens_d(
    first: Sequence[float] | pd.Series | np.ndarray,
    second: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Cohen's d for two independent groups using the pooled SD."""
    a = validate_numeric_series(first, name="first", min_observations=2)
    b = validate_numeric_series(second, name="second", min_observations=2)
    pooled_variance = (
        ((len(a) - 1) * np.var(a, ddof=1))
        + ((len(b) - 1) * np.var(b, ddof=1))
    ) / (len(a) + len(b) - 2)
    if pooled_variance <= 0:
        raise ValueError("Cohen's d is undefined when pooled variance is zero.")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_variance))


def hedges_g(
    first: Sequence[float] | pd.Series | np.ndarray,
    second: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Hedges' g with small-sample bias correction."""
    a = validate_numeric_series(first, name="first", min_observations=2)
    b = validate_numeric_series(second, name="second", min_observations=2)
    d = cohens_d(a, b)
    df = len(a) + len(b) - 2
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d * correction)


def paired_cohens_d(
    before: Sequence[float] | pd.Series | np.ndarray,
    after: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Cohen's d for paired observations, using the SD of differences."""
    before_values = validate_numeric_series(before, name="before", min_observations=2)
    after_values = validate_numeric_series(after, name="after", min_observations=2)
    if before_values.shape != after_values.shape:
        raise ValueError("Paired samples must have the same length.")
    differences = after_values - before_values
    sd = float(np.std(differences, ddof=1))
    if sd == 0:
        raise ValueError("Paired Cohen's d is undefined when difference variance is zero.")
    return float(np.mean(differences) / sd)


def eta_squared(
    *groups: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Eta-squared effect size for one-way ANOVA."""
    if len(groups) < 2:
        raise ValueError("At least two groups are required.")
    validated = [validate_numeric_series(g, name=f"group_{i+1}", min_observations=2) for i, g in enumerate(groups)]
    all_values = np.concatenate(validated)
    grand_mean = float(np.mean(all_values))
    ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in validated))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    if ss_total == 0:
        raise ValueError("Eta-squared is undefined when total variance is zero.")
    return float(ss_between / ss_total)


def omega_squared(
    *groups: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Bias-adjusted omega-squared effect size for one-way ANOVA."""
    if len(groups) < 2:
        raise ValueError("At least two groups are required.")
    validated = [validate_numeric_series(g, name=f"group_{i+1}", min_observations=2) for i, g in enumerate(groups)]
    all_values = np.concatenate(validated)
    grand_mean = float(np.mean(all_values))
    ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in validated))
    ss_within = float(sum(np.sum((g - np.mean(g)) ** 2) for g in validated))
    df_between = len(validated) - 1
    mse = ss_within / (len(all_values) - len(validated))
    denominator = ss_between + ss_within + mse
    if denominator == 0:
        raise ValueError("Omega-squared is undefined when total variance is zero.")
    return float(max(0.0, (ss_between - df_between * mse) / denominator))


def epsilon_squared(
    *groups: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis."""
    if len(groups) < 2:
        raise ValueError("At least two groups are required.")
    validated = [validate_numeric_series(g, name=f"group_{i+1}", min_observations=2) for i, g in enumerate(groups)]
    result = stats.kruskal(*validated)
    n = sum(len(g) for g in validated)
    k = len(validated)
    if n <= k:
        raise ValueError("Insufficient observations for epsilon-squared.")
    return float(max(0.0, (result.statistic - k + 1) / (n - k)))


def rank_biserial_from_u(
    u_statistic: float,
    n_first: int,
    n_second: int,
) -> float:
    """Calculate rank-biserial correlation from a Mann-Whitney U statistic."""
    if n_first <= 0 or n_second <= 0:
        raise ValueError("Sample sizes must be positive.")
    u = float(u_statistic)
    if not 0 <= u <= n_first * n_second:
        raise ValueError("u_statistic is outside its valid range.")
    return float(1.0 - (2.0 * u) / (n_first * n_second))


def cramers_v(
    observed: Sequence[Sequence[float]],
) -> float:
    """Calculate Cramer's V from a contingency table."""
    table = np.asarray(observed, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("observed must be at least a 2x2 table.")
    if not np.all(np.isfinite(table)) or np.any(table < 0):
        raise ValueError("observed must contain finite non-negative counts.")
    if not np.allclose(table, np.round(table)):
        raise ValueError("observed must contain integer-valued counts.")
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = float(table.sum())
    phi2 = chi2 / n
    r, c = table.shape
    correction = ((r - 1) * (c - 1))
    if correction <= 0:
        raise ValueError("Cramer's V requires at least a 2x2 table.")
    return float(np.sqrt(phi2 / min(r - 1, c - 1)))


def multiple_testing_correction(
    p_values: Sequence[float] | np.ndarray,
    *,
    method: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Adjust a family of p-values and return reproducible decisions.

    Supported methods are ``bonferroni``, ``holm``, and ``fdr_bh``
    (Benjamini-Hochberg false-discovery-rate control).
    """
    alpha = _validate_alpha(alpha)
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be finite values in [0, 1].")

    method = method.lower().strip()
    m = len(p)
    if method == "bonferroni":
        adjusted = np.minimum(p * m, 1.0)
    elif method == "holm":
        order = np.argsort(p, kind="mergesort")
        sorted_p = p[order]
        adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    elif method in {"fdr_bh", "benjamini_hochberg", "bh"}:
        order = np.argsort(p, kind="mergesort")
        sorted_p = p[order]
        ranks = np.arange(1, m + 1, dtype=float)
        q_sorted = sorted_p * m / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        adjusted = np.empty_like(q_sorted)
        adjusted[order] = np.minimum(q_sorted, 1.0)
    else:
        raise ValueError("method must be 'bonferroni', 'holm', or 'fdr_bh'.")

    return pd.DataFrame({
        "p_value": p,
        "adjusted_p_value": adjusted,
        "alpha": alpha,
        "reject_null": adjusted < alpha,
        "method": method,
    })


def recommend_test(
    *,
    design: str,
    groups: int = 2,
    paired: bool = False,
    outcome: str = "continuous",
    normality: str = "unknown",
    equal_variance: str = "unknown",
) -> dict[str, str]:
    """Return a transparent, rule-based statistical-test suggestion.

    This is a starting point, not an automatic scientific decision-maker.
    The recommendation states the assumptions it relied on so a user can
    review them before running a test.
    """
    design = design.lower().strip()
    outcome = outcome.lower().strip()
    normality = normality.lower().strip()
    equal_variance = equal_variance.lower().strip()

    if outcome in {"binary", "proportion"}:
        return {"test": "chi-square independence or exact/proportion test", "reason": "Binary/categorical outcome requires categorical inference."}
    if design == "paired" or paired:
        if outcome != "continuous":
            return {"test": "consult a paired categorical/count method", "reason": "Paired design with a non-continuous outcome is outside the continuous-test rules."}
        if normality == "normal":
            return {"test": "paired t-test", "reason": "Two related continuous measurements with approximately normal differences."}
        return {"test": "Wilcoxon signed-rank", "reason": "Paired continuous measurements without a normality assumption."}
    if design in {"independent", "two-group"} or groups == 2:
        if normality == "normal":
            if equal_variance == "yes":
                return {"test": "independent t-test", "reason": "Two independent continuous groups with approximately normal outcomes and comparable variances."}
            return {"test": "Welch independent t-test", "reason": "Two independent continuous groups with approximately normal outcomes; equal variance is not assumed."}
        return {"test": "Mann-Whitney U", "reason": "Two independent continuous/ordinal groups without a normality assumption."}
    if groups >= 3:
        if normality == "normal" and equal_variance == "yes":
            return {"test": "one-way ANOVA", "reason": "Three or more independent continuous groups with approximately normal outcomes and homogeneous variances."}
        if normality == "normal":
            return {"test": "Welch ANOVA", "reason": "Three or more independent continuous groups with approximately normal outcomes but unequal variances."}
        return {"test": "Kruskal-Wallis", "reason": "Three or more independent groups without a normality assumption."}

    return {"test": "inspect design and assumptions", "reason": "The supplied design description is not specific enough for a defensible rule-based suggestion."}


# ============================================================================
# OLS regression
# ============================================================================


def ols_regression(
    frame: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    missing: str = "drop",
    return_metadata: bool = False,
):
    """Fit an OLS regression model with explicit missing-data handling.

    Parameters
    ----------
    frame:
        Input DataFrame.
    target:
        Outcome column.
    features:
        Explanatory columns; an intercept is added automatically.
    missing:
        ``"drop"`` performs complete-case analysis. ``"raise"`` rejects
        any missing value before fitting. No imputation is performed.
    return_metadata:
        If ``True``, return ``(model, metadata)``. The default ``False``
        preserves the original return type.

    Metadata includes input rows, used rows, dropped rows, and the exact
    columns used by the model so downstream reporting can expose the
    effective sample rather than hiding data loss.
    """
    _require_statsmodels()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} was not found.")

    features = list(features)
    if not features:
        raise ValueError("At least one explanatory variable is required.")
    if len(set(features)) != len(features):
        raise ValueError("features must not contain duplicate column names.")
    if target in features:
        raise ValueError("target cannot also appear in features.")

    missing_features = [column for column in features if column not in frame.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    missing = missing.lower().strip()
    if missing not in {"drop", "raise"}:
        raise ValueError("missing must be 'drop' or 'raise'.")

    columns = [target, *features]
    working = frame[columns].copy()
    for column in columns:
        working[column] = pd.to_numeric(working[column], errors="raise")

    n_input = int(len(working))
    missing_rows = working.isna().any(axis=1)
    n_missing = int(missing_rows.sum())
    if missing == "raise" and n_missing:
        raise ValueError(f"{n_missing} rows contain missing model inputs.")

    if missing == "drop":
        working = working.loc[~missing_rows].copy()

    n_used = int(len(working))
    n_required = len(features) + 3
    if n_used < n_required:
        raise ValueError(
            f"Insufficient complete observations for OLS: {n_used} available, {n_required} required."
        )

    y = working[target]
    X = sm.add_constant(working[features], has_constant="add")
    model = sm.OLS(y, X).fit()

    if not return_metadata:
        return model

    metadata = {
        "n_input": n_input,
        "n_used": n_used,
        "n_dropped": n_input - n_used,
        "missing_policy": missing,
        "target": target,
        "features": tuple(features),
    }
    return model, metadata




# ============================================================================
# Regression diagnostics
# ============================================================================


def regression_diagnostics(
    model,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Calculate common OLS diagnostics with explicit significance level.

    The function does not declare a model "good" or "bad". It returns
    diagnostic statistics so callers can make context-specific judgments.
    ``alpha`` is used for the p-value-based diagnostic flags.
    """
    _require_statsmodels()
    alpha = _validate_alpha(alpha)

    residuals = np.asarray(model.resid, dtype=float)
    exog = np.asarray(model.model.exog, dtype=float)
    names = list(model.model.exog_names)
    predictor_names = [name for name in names if name != "const"]

    diagnostics: dict[str, Any] = {
        "n_observations": int(len(residuals)),
        "degrees_of_freedom_resid": float(model.df_resid),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "condition_number": float(np.linalg.cond(exog)),
        "alpha": alpha,
    }

    dw = float(durbin_watson(residuals))
    diagnostics["durbin_watson"] = dw

    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    diagnostics["jarque_bera"] = {
        "statistic": float(jb_stat),
        "p_value": float(jb_pvalue),
        "skewness": float(skew),
        "kurtosis": float(kurtosis),
        "reject_normality": bool(jb_pvalue < alpha),
    }

    bp = het_breuschpagan(residuals, exog)
    diagnostics["breusch_pagan"] = {
        "lm_statistic": float(bp[0]),
        "lm_p_value": float(bp[1]),
        "f_statistic": float(bp[2]),
        "f_p_value": float(bp[3]),
        "reject_homoskedasticity": bool(bp[1] < alpha),
    }

    white = het_white(residuals, exog)
    diagnostics["white_test"] = {
        "lm_statistic": float(white[0]),
        "lm_p_value": float(white[1]),
        "f_statistic": float(white[2]),
        "f_p_value": float(white[3]),
        "reject_homoskedasticity": bool(white[1] < alpha),
    }

    if len(predictor_names) >= 2:
        vif_values: dict[str, float] = {}
        for index, name in enumerate(names):
            if name == "const":
                continue
            vif_values[name] = float(variance_inflation_factor(exog, index))
        diagnostics["vif"] = vif_values
    else:
        diagnostics["vif"] = {}

    return diagnostics




# ============================================================================
# Bayesian conjugate models
# ============================================================================


def beta_binomial_posterior(
    successes: int,
    trials: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95,
) -> dict[str, float]:
    """Calculate an analytic Beta-Binomial posterior.

    ``successes`` and ``trials`` are validated as genuine non-negative
    integer counts rather than being silently truncated with ``int()``.
    ``credible_level`` controls the equal-tailed posterior interval.
    """
    if isinstance(successes, bool) or not isinstance(successes, (int, np.integer)):
        raise TypeError("successes must be an integer count.")
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
        raise TypeError("trials must be an integer count.")
    successes = int(successes)
    trials = int(trials)

    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials.")

    prior_alpha = float(prior_alpha)
    prior_beta = float(prior_beta)
    if not np.isfinite(prior_alpha) or not np.isfinite(prior_beta):
        raise ValueError("Beta prior parameters must be finite.")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior parameters must be positive.")

    credible_level = float(credible_level)
    if not 0 < credible_level < 1:
        raise ValueError("credible_level must be strictly between 0 and 1.")

    failures = trials - successes
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    total = posterior_alpha + posterior_beta
    posterior_mean = posterior_alpha / total
    posterior_variance = posterior_alpha * posterior_beta / (total**2 * (total + 1.0))
    tail = (1.0 - credible_level) / 2.0
    lower, upper = stats.beta.ppf(
        [tail, 1.0 - tail], posterior_alpha, posterior_beta
    )

    return {
        "prior_alpha": prior_alpha,
        "prior_beta": prior_beta,
        "successes": float(successes),
        "failures": float(failures),
        "posterior_alpha": float(posterior_alpha),
        "posterior_beta": float(posterior_beta),
        "posterior_mean": float(posterior_mean),
        "posterior_variance": float(posterior_variance),
        "credible_level": credible_level,
        "credible_interval_lower": float(lower),
        "credible_interval_upper": float(upper),
    }



def normal_normal_posterior(
    prior_mean: float,
    prior_variance: float,
    sample_mean: float,
    observation_variance: float,
    sample_size: int,
    *,
    credible_level: float | None = None,
) -> dict[str, float]:
    """Calculate a Normal-Normal posterior for an unknown mean.

    ``observation_variance`` is assumed known. If ``credible_level`` is
    supplied, equal-tailed posterior credible-interval bounds are added
    to the returned dictionary; the original posterior fields remain.
    """
    values = {
        "prior_mean": float(prior_mean),
        "prior_variance": float(prior_variance),
        "sample_mean": float(sample_mean),
        "observation_variance": float(observation_variance),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Normal-Normal parameters must be finite.")
    if values["prior_variance"] <= 0:
        raise ValueError("prior_variance must be positive.")
    if values["observation_variance"] <= 0:
        raise ValueError("observation_variance must be positive.")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")

    prior_precision = 1.0 / values["prior_variance"]
    sample_precision = sample_size / values["observation_variance"]
    posterior_variance = 1.0 / (prior_precision + sample_precision)
    posterior_mean = posterior_variance * (
        prior_precision * values["prior_mean"] + sample_precision * values["sample_mean"]
    )

    result = {
        "posterior_mean": float(posterior_mean),
        "posterior_variance": float(posterior_variance),
        "posterior_standard_deviation": float(np.sqrt(posterior_variance)),
    }

    if credible_level is not None:
        credible_level = float(credible_level)
        if not 0 < credible_level < 1:
            raise ValueError("credible_level must be strictly between 0 and 1.")
        z = stats.norm.ppf((1.0 + credible_level) / 2.0)
        sd = np.sqrt(posterior_variance)
        result["credible_level"] = credible_level
        result["credible_interval_lower"] = float(posterior_mean - z * sd)
        result["credible_interval_upper"] = float(posterior_mean + z * sd)

    return result




# ============================================================================
# Time-series
# ============================================================================


def stationarity_test(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    method: str = "adf",
    alpha: float = 0.05,
) -> TestResult:
    """Test time-series stationarity.

    Parameters
    ----------
    values:
        The time series, in chronological order.
    method:
        - ``"adf"`` (default): Augmented Dickey-Fuller. H0 = unit root
          (non-stationary). Lag order is selected automatically via AIC.
        - ``"kpss"``: H0 = stationary (the **opposite** null hypothesis
          from ADF) -- ``reject_null=True`` here means non-stationary,
          the reverse of what it means for ADF. Running both and
          checking for agreement is common practice precisely because
          they can disagree.
    alpha:
        Significance level.

    Returns
    -------
    TestResult
        ``details`` includes ``"critical_values"`` (a dict of
        significance-level -> critical statistic value) for both
        methods, plus ``"used_lag"``/``"n_observations"`` for ADF.

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    ValueError
        If ``method`` is unsupported, in addition to the errors from
        :func:`validate_numeric_series` / :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    _require_statsmodels()

    values_array = validate_numeric_series(
        values,
        min_observations=20,
    )

    method = method.lower().strip()

    if method == "adf":

        result = adfuller(
            values_array,
            autolag="AIC",
        )

        return _build_test_result(
            test="Augmented Dickey-Fuller",
            statistic=result[0],
            p_value=result[1],
            alpha=alpha,
            details={
                "used_lag": int(result[2]),
                "n_observations": int(result[3]),
                "critical_values": {
                    key: float(value) for key, value in result[4].items()
                },
            },
        )

    if method == "kpss":

        statistic, p_value, used_lag, critical_values = kpss(
            values_array,
            regression="c",
            nlags="auto",
        )

        return _build_test_result(
            test="KPSS",
            statistic=statistic,
            p_value=p_value,
            alpha=alpha,
            details={
                "used_lag": int(used_lag),
                "critical_values": {
                    key: float(value) for key, value in critical_values.items()
                },
            },
        )

    raise ValueError(
        "method must be 'adf' or 'kpss'."
    )


def ljung_box_test(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    lags: int | Sequence[int] = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Ljung-Box test for residual autocorrelation, at one or more lags.

    Parameters
    ----------
    values:
        The series (typically model residuals) to test.
    lags:
        A single lag count, or a sequence of specific lag counts to
        test simultaneously (each row of the returned DataFrame is one
        lag's cumulative test through that lag).
    alpha:
        Significance level, applied to every lag.

    Returns
    -------
    pandas.DataFrame
        Indexed by lag, with columns ``lb_stat``, ``lb_pvalue`` (from
        statsmodels), plus ``alpha`` and ``reject_null`` added by this
        function so the significance decision doesn't have to be
        recomputed by the caller -- earlier versions of this function
        accepted ``alpha`` but never used it, silently ignoring
        whatever the caller passed.

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    ValueError
        If any requested lag is not positive, in addition to the
        errors from :func:`validate_numeric_series` / :func:`_validate_alpha`.
    """

    alpha = _validate_alpha(alpha)
    _require_statsmodels()

    lag_list = [lags] if isinstance(lags, int) else list(lags)

    if not lag_list:
        raise ValueError("lags must contain at least one value.")

    if any(isinstance(lag, bool) or lag <= 0 for lag in lag_list):
        raise ValueError("lags must be positive integers.")

    max_lag = max(lag_list)

    values_array = validate_numeric_series(
        values,
        min_observations=max(20, max_lag + 2),
    )

    result = acorr_ljungbox(
        values_array,
        lags=[int(lag) for lag in lag_list],
        return_df=True,
    )

    result = result.copy()
    result["alpha"] = alpha
    result["reject_null"] = result["lb_pvalue"] < alpha

    return result


def fit_arima(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (
        0,
        0,
        0,
        0,
    ),
):
    """Fit an ARIMA/SARIMA model.

    Parameters
    ----------
    values:
        The time series, in chronological order.
    order:
        ``(p, d, q)`` -- AR order, differencing order, MA order.
    seasonal_order:
        ``(P, D, Q, s)`` -- seasonal AR/differencing/MA orders and the
        seasonal period. The default ``(0, 0, 0, 0)`` disables the
        seasonal component entirely.

    Returns
    -------
    statsmodels ARIMA results object.

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    ValueError
        If ``order``/``seasonal_order`` are not the right length or
        contain negative values, in addition to the errors from
        :func:`validate_numeric_series`.
    """

    _require_statsmodels()

    values_array = validate_numeric_series(
        values,
        min_observations=30,
    )

    if len(order) != 3:
        raise ValueError(
            "order must contain (p, d, q)."
        )

    if len(seasonal_order) != 4:
        raise ValueError(
            "seasonal_order must contain (P, D, Q, s)."
        )

    if any(int(value) < 0 for value in order):
        raise ValueError(
            "ARIMA orders cannot be negative."
        )

    if any(int(value) < 0 for value in seasonal_order):
        raise ValueError(
            "Seasonal ARIMA orders cannot be negative."
        )

    return ARIMA(
        values_array,
        order=tuple(
            int(value)
            for value in order
        ),
        seasonal_order=tuple(
            int(value)
            for value in seasonal_order
        ),
    ).fit()


def forecast_arima(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    horizon: int = 12,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (
        0,
        0,
        0,
        0,
    ),
) -> pd.DataFrame:
    """Fit ARIMA/SARIMA and return a forecast DataFrame.

    Parameters
    ----------
    values:
        The time series, in chronological order.
    horizon:
        Number of future steps to forecast.
    order, seasonal_order:
        See :func:`fit_arima`.

    Returns
    -------
    pandas.DataFrame
        Columns ``forecast``, ``lower``, ``upper`` (95% confidence
        interval by statsmodels' default), one row per forecast step.

    Raises
    ------
    ValueError
        If ``horizon <= 0``, in addition to the errors from
        :func:`fit_arima`.
    """

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    result = fit_arima(
        values,
        order=order,
        seasonal_order=seasonal_order,
    )

    forecast = result.get_forecast(
        steps=int(horizon)
    )

    mean = np.asarray(
        forecast.predicted_mean,
        dtype=float,
    )

    confidence = forecast.conf_int()

    output = pd.DataFrame(
        {
            "forecast": mean,
            "lower": np.asarray(
                confidence.iloc[:, 0],
                dtype=float,
            ),
            "upper": np.asarray(
                confidence.iloc[:, 1],
                dtype=float,
            ),
        }
    )

    return output


# ============================================================================
# Forecast evaluation
# ============================================================================


def forecast_metrics(
    actual: Sequence[float] | pd.Series | np.ndarray,
    predicted: Sequence[float] | pd.Series | np.ndarray,
    *,
    training: Sequence[float] | pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate forecast-error metrics.

    Errors are defined as ``actual - predicted``; therefore positive bias
    means under-prediction on average.

    If ``training`` is supplied, MASE is also returned using the mean
    absolute one-step naive error on the training series as the scale.
    """
    actual_values = validate_numeric_series(actual, name="actual")
    predicted_values = validate_numeric_series(predicted, name="predicted")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have the same length.")

    errors = actual_values - predicted_values
    mse = float(np.mean(errors ** 2))
    result = {
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(errors)),
    }

    nonzero = actual_values != 0
    if np.any(nonzero):
        result["mape"] = float(np.mean(np.abs(errors[nonzero] / actual_values[nonzero])) * 100.0)

    if training is not None:
        training_values = validate_numeric_series(training, name="training", min_observations=2)
        naive_scale = float(np.mean(np.abs(np.diff(training_values))))
        if naive_scale == 0:
            raise ValueError("MASE is undefined because the training naive scale is zero.")
        result["mase"] = float(np.mean(np.abs(errors)) / naive_scale)

    return result
