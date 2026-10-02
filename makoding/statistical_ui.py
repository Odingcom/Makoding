"""Streamlit interface for Makoding statistical analysis.

This module is the presentation layer for ``makoding.statistical_analysis``.
The statistical engine remains framework-independent and reusable.

The UI is intentionally separated from the statistical implementation so
that statistical functions remain independently testable and suitable for
future API, notebook, reporting, or batch use.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from makoding import statistical_analysis as sa
from makoding import styling


# ============================================================================
# General helpers
# ============================================================================


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric columns from a DataFrame."""
    return frame.select_dtypes(include=np.number).columns.tolist()


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a structured statistical result into displayable data."""
    if is_dataclass(result):
        return asdict(result)

    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            converted = result.to_dict()

            if isinstance(converted, dict):
                return converted
        except Exception:
            pass

    if hasattr(result, "__dict__"):
        return {
            key: value
            for key, value in vars(result).items()
            if not key.startswith("_")
        }

    return {"result": result}


def _display_result(result: Any) -> None:
    """Render a statistical result in a readable form."""
    values = _result_to_dict(result)

    if not values:
        st.info("No statistical result was returned.")
        return

    if "test" in values:
        metric_columns = []

        if "statistic" in values:
            metric_columns.append(
                ("Statistic", values["statistic"])
            )

        if "p_value" in values:
            metric_columns.append(
                ("P-value", values["p_value"])
            )

        if "alpha" in values:
            metric_columns.append(
                ("Alpha", values["alpha"])
            )

        if metric_columns:
            columns = st.columns(
                min(len(metric_columns), 4)
            )

            for index, (label, value) in enumerate(metric_columns):
                columns[index].metric(
                    label,
                    _format_number(value),
                )

        if "reject_null" in values:
            if bool(values["reject_null"]):
                st.warning(
                    "The null hypothesis is rejected at the selected "
                    "significance level."
                )
            else:
                st.info(
                    "The null hypothesis is not rejected at the selected "
                    "significance level."
                )

        if values.get("interpretation"):
            st.caption(
                str(values["interpretation"])
            )

        details = values.get("details")

        if details:
            st.markdown("#### Details")
            st.json(details)

        return

    st.json(
        {
            key: _json_safe(value)
            for key, value in values.items()
        }
    )


def _format_number(value: Any) -> str:
    """Format numeric values consistently."""
    try:
        number = float(value)

        if not np.isfinite(number):
            return str(number)

        if abs(number) >= 1000:
            return f"{number:,.3f}"

        return f"{number:.6f}"

    except (TypeError, ValueError):
        return str(value)


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Series):
        return value.to_dict()

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


def _run_safely(
    function,
    *args,
    **kwargs,
) -> Any | None:
    """Execute a statistical operation without crashing the application."""
    try:
        return function(
            *args,
            **kwargs,
        )

    except Exception as exc:
        st.error(
            "The statistical operation could not be completed."
        )

        with st.expander("Technical details"):
            st.code(
                f"{type(exc).__name__}: {exc}"
            )

        return None


def _require_numeric_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return numeric columns or display an appropriate message."""
    columns = _numeric_columns(frame)

    if not columns:
        st.warning(
            "This analysis requires at least one numeric column."
        )

    return columns


# ============================================================================
# Descriptive statistics
# ============================================================================


def _render_descriptive(frame: pd.DataFrame) -> None:
    """Render descriptive distribution analysis."""

    styling.render_section(
        "Descriptive statistics",
        "02",
    )

    numeric = _require_numeric_columns(frame)

    if not numeric:
        return

    column = st.selectbox(
        "Variable",
        numeric,
        key="stats_descriptive_column",
    )

    result = _run_safely(
        sa.describe_series,
        frame[column],
    )

    if result is None:
        return

    display = result.to_frame(
        name="value"
    )

    st.dataframe(
        display,
        use_container_width=True,
    )

    st.markdown("#### Distribution summary")

    distribution = _run_safely(
        sa.distribution_summary,
        frame[column],
    )

    if distribution is not None:
        _display_result(distribution)


# ============================================================================
# Assumption tests
# ============================================================================


def _render_assumption_tests(frame: pd.DataFrame) -> None:
    """Render normality and correlation assumption tests."""

    styling.render_section(
        "Assumption and relationship tests",
        "02",
    )

    numeric = _require_numeric_columns(frame)

    if not numeric:
        return

    analysis = st.selectbox(
        "Analysis",
        [
            "Normality test",
            "Correlation test",
        ],
        key="stats_assumption_analysis",
    )

    alpha = st.number_input(
        "Significance level",
        min_value=0.001,
        max_value=0.50,
        value=0.05,
        step=0.01,
        key="stats_assumption_alpha",
    )

    if analysis == "Normality test":

        column = st.selectbox(
            "Variable",
            numeric,
            key="stats_normality_column",
        )

        method = st.selectbox(
            "Normality method",
            [
                "shapiro",
                "normaltest",
            ],
            key="stats_normality_method",
        )

        if st.button(
            "Run normality test",
            type="primary",
            key="run_normality",
        ):
            result = _run_safely(
                sa.normality_test,
                frame[column],
                alpha=float(alpha),
                method=method,
            )

            if result is not None:
                _display_result(result)

    else:

        if len(numeric) < 2:
            st.warning(
                "At least two numeric variables are required."
            )
            return

        first = st.selectbox(
            "First variable",
            numeric,
            key="stats_corr_first",
        )

        second_options = [
            column
            for column in numeric
            if column != first
        ]

        second = st.selectbox(
            "Second variable",
            second_options,
            key="stats_corr_second",
        )

        method = st.selectbox(
            "Correlation method",
            [
                "pearson",
                "spearman",
                "kendall",
            ],
            key="stats_corr_method",
        )

        if st.button(
            "Run correlation test",
            type="primary",
            key="run_correlation",
        ):
            result = _run_safely(
                sa.correlation_test,
                frame[first],
                frame[second],
                method=method,
                alpha=float(alpha),
            )

            if result is not None:
                _display_result(result)


# ============================================================================
# Hypothesis testing
# ============================================================================


def _render_hypothesis_tests(frame: pd.DataFrame) -> None:
    """Render classical hypothesis tests."""

    styling.render_section(
        "Hypothesis testing",
        "02",
    )

    numeric = _require_numeric_columns(frame)

    if not numeric:
        return

    test = st.selectbox(
        "Test",
        [
            "One-sample t-test",
            "Independent t-test",
            "Paired t-test",
            "One-way ANOVA",
            "Welch ANOVA",
            "Mann-Whitney U",
            "Wilcoxon signed-rank",
            "Kruskal-Wallis",
            "Chi-square",
        ],
        key="stats_hypothesis_test",
    )

    alpha = st.number_input(
        "Significance level",
        min_value=0.001,
        max_value=0.50,
        value=0.05,
        step=0.01,
        key="stats_hypothesis_alpha",
    )

    # ------------------------------------------------------------------------
    # One-sample t-test
    # ------------------------------------------------------------------------

    if test == "One-sample t-test":

        column = st.selectbox(
            "Sample variable",
            numeric,
            key="stats_one_sample_column",
        )

        population_mean = st.number_input(
            "Population mean",
            value=0.0,
            key="stats_population_mean",
        )

        alternative = st.selectbox(
            "Alternative hypothesis",
            [
                "two-sided",
                "greater",
                "less",
            ],
            key="stats_one_sample_alternative",
        )

        if st.button(
            "Run test",
            type="primary",
            key="run_one_sample",
        ):
            result = _run_safely(
                sa.one_sample_t_test,
                frame[column],
                float(population_mean),
                alpha=float(alpha),
                alternative=alternative,
            )

            if result is not None:
                _display_result(result)

    # ------------------------------------------------------------------------
    # Independent t-test
    # ------------------------------------------------------------------------

    elif test == "Independent t-test":

        first = st.selectbox(
            "First sample",
            numeric,
            key="stats_independent_first",
        )

        second = st.selectbox(
            "Second sample",
            [
                column
                for column in numeric
                if column != first
            ],
            key="stats_independent_second",
        )

        equal_variance = st.checkbox(
            "Assume equal variance",
            value=False,
            key="stats_equal_variance",
        )

        if st.button(
            "Run test",
            type="primary",
            key="run_independent",
        ):
            result = _run_safely(
                sa.independent_t_test,
                frame[first],
                frame[second],
                alpha=float(alpha),
                equal_variance=equal_variance,
            )

            if result is not None:
                _display_result(result)

    # ------------------------------------------------------------------------
    # Paired t-test
    # ------------------------------------------------------------------------

    elif test == "Paired t-test":

        before = st.selectbox(
            "Before variable",
            numeric,
            key="stats_paired_before",
        )

        after = st.selectbox(
            "After variable",
            [
                column
                for column in numeric
                if column != before
            ],
            key="stats_paired_after",
        )

        if st.button(
            "Run test",
            type="primary",
            key="run_paired",
        ):
            result = _run_safely(
                sa.paired_t_test,
                frame[before],
                frame[after],
                alpha=float(alpha),
            )

            if result is not None:
                _display_result(result)

    # ------------------------------------------------------------------------
    # ANOVA
    # ------------------------------------------------------------------------

    elif test in {
        "One-way ANOVA",
        "Welch ANOVA",
    }:

        selected = st.multiselect(
            "Group variables",
            numeric,
            min_selections=2,
            key="stats_anova_groups",
        )

        if len(selected) >= 2:

            if st.button(
                "Run test",
                type="primary",
                key="run_anova",
            ):

                groups = [
                    frame[column].dropna()
                    for column in selected
                ]

                function = (
                    sa.welch_anova
                    if test == "Welch ANOVA"
                    else sa.one_way_anova
                )

                result = _run_safely(
                    function,
                    *groups,
                    alpha=float(alpha),
                )

                if result is not None:
                    _display_result(result)

    # ------------------------------------------------------------------------
    # Mann-Whitney
    # ------------------------------------------------------------------------

    elif test == "Mann-Whitney U":

        first = st.selectbox(
            "First sample",
            numeric,
            key="stats_mann_first",
        )

        second = st.selectbox(
            "Second sample",
            [
                column
                for column in numeric
                if column != first
            ],
            key="stats_mann_second",
        )

        if st.button(
            "Run test",
            type="primary",
            key="run_mann_whitney",
        ):

            result = _run_safely(
                sa.mann_whitney_test,
                frame[first],
                frame[second],
                alpha=float(alpha),
            )

            if result is not None:
                _display_result(result)

    # ------------------------------------------------------------------------
    # Wilcoxon
    # ------------------------------------------------------------------------

    elif test == "Wilcoxon signed-rank":

        before = st.selectbox(
            "Before variable",
            numeric,
            key="stats_wilcoxon_before",
        )

        after = st.selectbox(
            "After variable",
            [
                column
                for column in numeric
                if column != before
            ],
            key="stats_wilcoxon_after",
        )

        if st.button(
            "Run test",
            type="primary",
            key="run_wilcoxon",
        ):

            result = _run_safely(
                sa.wilcoxon_test,
                frame[before],
                frame[after],
                alpha=float(alpha),
            )

            if result is not None:
                _display_result(result)

    # ------------------------------------------------------------------------
    # Kruskal-Wallis
    # ------------------------------------------------------------------------

    elif test == "Kruskal-Wallis":

        selected = st.multiselect(
            "Group variables",
            numeric,
            min_selections=2,
            key="stats_kruskal_groups",
        )

        if len(selected) >= 2:

            if st.button(
                "Run test",
                type="primary",
                key="run_kruskal",
            ):

                groups = [
                    frame[column].dropna()
                    for column in selected
                ]

                result = _run_safely(
                    sa.kruskal_test,
                    *groups,
                    alpha=float(alpha),
                )

                if result is not None:
                    _display_result(result)

    # ------------------------------------------------------------------------
    # Chi-square
    # ------------------------------------------------------------------------

    else:

        categorical = frame.select_dtypes(
            include=[
                "object",
                "category",
                "string",
                "bool",
            ]
        ).columns.tolist()

        if len(categorical) < 2:
            st.warning(
                "At least two categorical columns are required "
                "for a chi-square test."
            )
            return

        row_variable = st.selectbox(
            "Row variable",
            categorical,
            key="stats_chi_row",
        )

        column_options = [
            column
            for column in categorical
            if column != row_variable
        ]

        column_variable = st.selectbox(
            "Column variable",
            column_options,
            key="stats_chi_column",
        )

        if st.button(
            "Run test",
            type="primary",
            key="run_chi_square",
        ):

            contingency = pd.crosstab(
                frame[row_variable],
                frame[column_variable],
            )

            result = _run_safely(
                sa.chi_square_test,
                contingency,
                alpha=float(alpha),
            )

            if result is not None:
                _display_result(result)

            st.markdown("#### Contingency table")
            st.dataframe(
                contingency,
                use_container_width=True,
            )


# ============================================================================
# Effect sizes and confidence intervals
# ============================================================================


def _render_effect_sizes(frame: pd.DataFrame) -> None:
    """Render effect-size and confidence-interval calculations."""

    styling.render_section(
        "Effect sizes and confidence intervals",
        "02",
    )

    numeric = _require_numeric_columns(frame)

    if not numeric:
        return

    analysis = st.selectbox(
        "Effect-size analysis",
        [
            "Mean confidence interval",
            "Cohen's d",
            "Hedges' g",
            "Paired Cohen's d",
            "Eta squared",
            "Omega squared",
            "Epsilon squared",
            "Rank-biserial correlation",
            "Cramér's V",
        ],
        key="stats_effect_analysis",
    )

    alpha = st.number_input(
        "Significance level",
        min_value=0.001,
        max_value=0.50,
        value=0.05,
        step=0.01,
        key="stats_effect_alpha",
    )

    if analysis == "Mean confidence interval":

        column = st.selectbox(
            "Variable",
            numeric,
            key="stats_ci_column",
        )

        if st.button(
            "Calculate interval",
            type="primary",
            key="run_mean_ci",
        ):

            result = _run_safely(
                sa.mean_confidence_interval,
                frame[column],
                alpha=float(alpha),
            )

            if result is not None:
                _display_result(result)

    elif analysis in {
        "Cohen's d",
        "Hedges' g",
        "Rank-biserial correlation",
    }:

        first = st.selectbox(
            "First sample",
            numeric,
            key="stats_effect_first",
        )

        second = st.selectbox(
            "Second sample",
            [
                column
                for column in numeric
                if column != first
            ],
            key="stats_effect_second",
        )

        if st.button(
            "Calculate effect size",
            type="primary",
            key="run_effect_size",
        ):

            if analysis == "Cohen's d":
                function = sa.cohens_d
            elif analysis == "Hedges' g":
                function = sa.hedges_g
            else:
                result_u = _run_safely(
                    sa.mann_whitney_test,
                    frame[first],
                    frame[second],
                    alpha=float(alpha),
                )

                if result_u is None:
                    return

                details = _result_to_dict(result_u).get(
                    "details",
                    {},
                )

                u_statistic = details.get(
                    "u_statistic"
                )

                if u_statistic is None:
                    st.info(
                        "The Mann-Whitney result did not expose "
                        "the U statistic required for rank-biserial "
                        "calculation."
                    )
                    return

                result = _run_safely(
                    sa.rank_biserial_from_u,
                    float(u_statistic),
                    len(frame[first].dropna()),
                    len(frame[second].dropna()),
                )

                if result is not None:
                    _display_result(result)

                return

            result = _run_safely(
                function,
                frame[first],
                frame[second],
            )

            if result is not None:
                _display_result(result)

    elif analysis == "Paired Cohen's d":

        before = st.selectbox(
            "Before variable",
            numeric,
            key="stats_paired_effect_before",
        )

        after = st.selectbox(
            "After variable",
            [
                column
                for column in numeric
                if column != before
            ],
            key="stats_paired_effect_after",
        )

        if st.button(
            "Calculate effect size",
            type="primary",
            key="run_paired_effect",
        ):

            result = _run_safely(
                sa.paired_cohens_d,
                frame[before],
                frame[after],
            )

            if result is not None:
                _display_result(result)

    elif analysis in {
        "Eta squared",
        "Omega squared",
        "Epsilon squared",
    }:

        selected = st.multiselect(
            "Group variables",
            numeric,
            min_selections=2,
            key="stats_effect_groups",
        )

        if len(selected) < 2:
            return

        if st.button(
            "Calculate effect size",
            type="primary",
            key="run_anova_effect",
        ):

            groups = [
                frame[column].dropna()
                for column in selected
            ]

            if analysis == "Eta squared":
                result = _run_safely(
                    sa.eta_squared,
                    *groups,
                )
            elif analysis == "Omega squared":
                result = _run_safely(
                    sa.omega_squared,
                    *groups,
                )
            else:
                result = _run_safely(
                    sa.epsilon_squared,
                    *groups,
                )

            if result is not None:
                _display_result(result)

    else:

        categorical = frame.select_dtypes(
            include=[
                "object",
                "category",
                "string",
                "bool",
            ]
        ).columns.tolist()

        if len(categorical) < 2:
            st.warning(
                "At least two categorical variables are required."
            )
            return

        row = st.selectbox(
            "Row variable",
            categorical,
            key="stats_cramer_row",
        )

        column = st.selectbox(
            "Column variable",
            [
                value
                for value in categorical
                if value != row
            ],
            key="stats_cramer_column",
        )

        if st.button(
            "Calculate Cramér's V",
            type="primary",
            key="run_cramer",
        ):

            contingency = pd.crosstab(
                frame[row],
                frame[column],
            )

            result = _run_safely(
                sa.cramers_v,
                contingency,
            )

            if result is not None:
                _display_result(result)


# ============================================================================
# OLS regression
# ============================================================================


def _render_regression(frame: pd.DataFrame) -> None:
    """Render OLS regression and diagnostics."""

    styling.render_section(
        "OLS regression",
        "02",
    )

    numeric = _require_numeric_columns(frame)

    if len(numeric) < 2:
        return

    target = st.selectbox(
        "Target variable",
        numeric,
        key="stats_ols_target",
    )

    feature_options = [
        column
        for column in numeric
        if column != target
    ]

    features = st.multiselect(
        "Predictor variables",
        feature_options,
        default=feature_options[: min(3, len(feature_options))],
        key="stats_ols_features",
    )

    missing = st.selectbox(
        "Missing-data handling",
        [
            "drop",
        ],
        key="stats_ols_missing",
    )

    alpha = st.number_input(
        "Significance level",
        min_value=0.001,
        max_value=0.50,
        value=0.05,
        step=0.01,
        key="stats_ols_alpha",
    )

    if not features:
        st.info(
            "Select at least one predictor variable."
        )
        return

    if st.button(
        "Fit OLS regression",
        type="primary",
        key="run_ols",
    ):

        model = _run_safely(
            sa.ols_regression,
            frame,
            target=target,
            features=features,
            missing=missing,
            alpha=float(alpha),
        )

        if model is None:
            return

        st.success(
            "OLS regression completed."
        )

        summary = getattr(
            model,
            "summary",
            None,
        )

        if callable(summary):
            st.text(
                summary().as_text()
            )

        else:
            st.write(model)

        st.session_state[
            "datalab_ols_model"
        ] = model

        diagnostics = _run_safely(
            sa.regression_diagnostics,
            model,
            alpha=float(alpha),
        )

        if diagnostics is not None:

            styling.render_section(
                "Regression diagnostics",
                "03",
            )

            _render_diagnostics(
                diagnostics
            )


def _render_diagnostics(
    diagnostics: dict[str, Any],
) -> None:
    """Render regression diagnostic output."""

    scalar_values = {}

    tables = {}

    for key, value in diagnostics.items():

        if isinstance(value, pd.DataFrame):
            tables[key] = value

        elif isinstance(value, pd.Series):
            tables[key] = value

        elif np.isscalar(value):
            scalar_values[key] = value

        else:
            try:
                if np.isscalar(value):
                    scalar_values[key] = value
                else:
                    tables[key] = value
            except Exception:
                pass

    if scalar_values:

        columns = st.columns(
            min(
                4,
                len(scalar_values),
            )
        )

        for index, (
            name,
            value,
        ) in enumerate(
            scalar_values.items()
        ):

            columns[
                index % len(columns)
            ].metric(
                name.replace(
                    "_",
                    " ",
                ).title(),
                _format_number(value),
            )

    for name, value in tables.items():

        st.markdown(
            f"#### {name.replace('_', ' ').title()}"
        )

        if isinstance(
            value,
            (pd.DataFrame, pd.Series),
        ):
            st.dataframe(
                value,
                use_container_width=True,
            )
        else:
            st.write(value)


# ============================================================================
# Bayesian analysis
# ============================================================================


def _render_bayesian(frame: pd.DataFrame) -> None:
    """Render Bayesian conjugate analysis."""

    styling.render_section(
        "Bayesian analysis",
        "02",
    )

    method = st.selectbox(
        "Bayesian model",
        [
            "Beta-Binomial",
            "Normal-Normal",
        ],
        key="stats_bayesian_method",
    )

    if method == "Beta-Binomial":

        successes = st.number_input(
            "Successes",
            min_value=0,
            value=50,
            step=1,
            key="stats_successes",
        )

        trials = st.number_input(
            "Trials",
            min_value=1,
            value=100,
            step=1,
            key="stats_trials",
        )

        prior_alpha = st.number_input(
            "Prior alpha",
            min_value=0.001,
            value=1.0,
            step=0.1,
            key="stats_prior_alpha",
        )

        prior_beta = st.number_input(
            "Prior beta",
            min_value=0.001,
            value=1.0,
            step=0.1,
            key="stats_prior_beta",
        )

        if st.button(
            "Calculate posterior",
            type="primary",
            key="run_beta_binomial",
        ):

            result = _run_safely(
                sa.beta_binomial_posterior,
                int(successes),
                int(trials),
                prior_alpha=float(prior_alpha),
                prior_beta=float(prior_beta),
            )

            if result is not None:
                _display_result(result)

    else:

        numeric = _require_numeric_columns(frame)

        if not numeric:
            return

        column = st.selectbox(
            "Observed variable",
            numeric,
            key="stats_normal_normal_column",
        )

        prior_mean = st.number_input(
            "Prior mean",
            value=0.0,
            key="stats_nn_prior_mean",
        )

        prior_variance = st.number_input(
            "Prior variance",
            min_value=0.000001,
            value=1.0,
            key="stats_nn_prior_variance",
        )

        observation_variance = st.number_input(
            "Observation variance",
            min_value=0.000001,
            value=1.0,
            key="stats_nn_observation_variance",
        )

        if st.button(
            "Calculate posterior",
            type="primary",
            key="run_normal_normal",
        ):

            values = frame[column].dropna()

            result = _run_safely(
                sa.normal_normal_posterior,
                values,
                prior_mean=float(prior_mean),
                prior_variance=float(prior_variance),
                observation_variance=float(
                    observation_variance
                ),
            )

            if result is not None:
                _display_result(result)


# ============================================================================
# Time-series analysis
# ============================================================================


def _render_time_series(frame: pd.DataFrame) -> None:
    """Render stationarity, autocorrelation and ARIMA analysis."""

    styling.render_section(
        "Time-series analysis",
        "02",
    )

    numeric = _require_numeric_columns(frame)

    if not numeric:
        return

    column = st.selectbox(
        "Time-series variable",
        numeric,
        key="stats_time_series_column",
    )

    values = frame[column].dropna()

    if len(values) < 20:
        st.warning(
            "Time-series diagnostics require a longer numeric series. "
            "At least 20 observations are recommended by the statistical layer."
        )
        return

    analysis = st.selectbox(
        "Analysis",
        [
            "Stationarity",
            "Ljung-Box autocorrelation",
            "ARIMA forecast",
        ],
        key="stats_time_series_analysis",
    )

    if analysis == "Stationarity":

        method = st.selectbox(
            "Stationarity test",
            [
                "adf",
                "kpss",
            ],
            key="stats_stationarity_method",
        )

        if st.button(
            "Run stationarity test",
            type="primary",
            key="run_stationarity",
        ):

            result = _run_safely(
                sa.stationarity_test,
                values,
                method=method,
            )

            if result is not None:
                _display_result(result)

    elif analysis == "Ljung-Box autocorrelation":

        max_lag = max(
            1,
            min(
                20,
                len(values) // 4,
            ),
        )

        lag = st.slider(
            "Lag",
            min_value=1,
            max_value=max_lag,
            value=min(10, max_lag),
            key="stats_ljung_lag",
        )

        if st.button(
            "Run Ljung-Box test",
            type="primary",
            key="run_ljung_box",
        ):

            result = _run_safely(
                sa.ljung_box_test,
                values,
                lags=int(lag),
            )

            if result is not None:
                _display_result(result)

    else:

        st.markdown(
            "#### ARIMA configuration"
        )

        p = st.number_input(
            "AR order (p)",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            key="stats_arima_p",
        )

        d = st.number_input(
            "Differencing order (d)",
            min_value=0,
            max_value=2,
            value=1,
            step=1,
            key="stats_arima_d",
        )

        q = st.number_input(
            "Moving-average order (q)",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            key="stats_arima_q",
        )

        horizon = st.number_input(
            "Forecast horizon",
            min_value=1,
            max_value=100,
            value=12,
            step=1,
            key="stats_forecast_horizon",
        )

        if st.button(
            "Fit ARIMA and forecast",
            type="primary",
            key="run_arima",
        ):

            order = (
                int(p),
                int(d),
                int(q),
            )

            model = _run_safely(
                sa.fit_arima,
                values,
                order=order,
            )

            if model is None:
                return

            forecast = _run_safely(
                sa.forecast_arima,
                values,
                horizon=int(horizon),
                order=order,
            )

            if forecast is None:
                return

            st.success(
                "ARIMA forecast completed."
            )

            _display_result(
                forecast
            )


# ============================================================================
# Main statistical workspace
# ============================================================================


def render_statistical_analysis(
    frame: pd.DataFrame,
) -> None:
    """Render the complete Statistical Analysis workspace.

    Parameters
    ----------
    frame:
        Current cleaned DataFrame from DataLab Pro.

    Notes
    -----
    The function never mutates ``frame``. All statistical operations receive
    either the original Series or independent views/copies as required by the
    underlying analytical functions.
    """

    styling.render_section(
        "Statistical analysis",
        "01",
    )

    st.caption(
        "Inferential statistics, effect sizes, regression, Bayesian analysis "
        "and time-series diagnostics using the Makoding statistical engine."
    )

    if frame.empty:
        st.warning(
            "The current dataset contains no observations."
        )
        return

    workspace = st.selectbox(
        "Statistical workspace",
        [
            "Descriptive statistics",
            "Hypothesis testing",
            "Assumption and relationship tests",
            "Effect sizes and confidence intervals",
            "OLS regression",
            "Bayesian analysis",
            "Time-series analysis",
        ],
        key="statistical_workspace",
    )

    if workspace == "Descriptive statistics":
        _render_descriptive(frame)

    elif workspace == "Hypothesis testing":
        _render_hypothesis_tests(frame)

    elif workspace == "Assumption and relationship tests":
        _render_assumption_tests(frame)

    elif workspace == "Effect sizes and confidence intervals":
        _render_effect_sizes(frame)

    elif workspace == "OLS regression":
        _render_regression(frame)

    elif workspace == "Bayesian analysis":
        _render_bayesian(frame)

    elif workspace == "Time-series analysis":
        _render_time_series(frame)
