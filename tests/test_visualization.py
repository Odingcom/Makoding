"""
Tests for makoding.visualization.

The visualization API returns matplotlib Figure objects and does not
display figures directly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from makoding.visualization import (
    plot_bivariate,
    plot_box_by_group,
    plot_category_heatmap,
    plot_categorical_counts,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_numeric_distribution,
    plot_predicted_vs_actual,
    plot_precision_recall_curve,
    plot_residuals,
    plot_roc_curve,
    plot_scatter,
    plot_univariate,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def numeric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [
                18.0,
                21.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                np.nan,
                60.0,
            ],
            "income": [
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                np.nan,
            ],
            "score": [
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
            ],
        }
    )


@pytest.fixture
def categorical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": [
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
            ],
            "region": [
                "East",
                "West",
                "East",
                "North",
                "North",
                "South",
                "South",
                "East",
                "West",
                "East",
            ],
        }
    )


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [18, 21, 25, 30, 35, 40, 45, 50],
            "income": [20, 25, 30, 35, 40, 45, 50, 55],
            "gender": [
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "region": [
                "East",
                "West",
                "East",
                "North",
                "North",
                "South",
                "North",
                "South",
            ],
        }
    )


@pytest.fixture
def classification_data():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    y_score = np.array(
        [0.05, 0.15, 0.65, 0.20, 0.80, 0.90, 0.30, 0.95]
    )

    return y_true, y_pred, y_score


@pytest.fixture
def regression_data():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 18.0, 32.0, 39.0, 48.0])

    return y_true, y_pred


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ============================================================================
# Numeric distribution
# ============================================================================


class TestPlotNumericDistribution:

    def test_returns_figure(self, numeric_frame):
        fig = plot_numeric_distribution(
            numeric_frame,
            "age",
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_histogram(self, numeric_frame):
        fig = plot_numeric_distribution(
            numeric_frame,
            "age",
        )

        ax = fig.axes[0]

        assert len(ax.patches) > 0
        assert ax.get_xlabel() == "age"
        assert ax.get_ylabel() == "Count"
        assert ax.get_title() == "Distribution of age"

    def test_handles_kde(self, numeric_frame):
        fig = plot_numeric_distribution(
            numeric_frame,
            "age",
            kde=True,
        )

        assert isinstance(fig, matplotlib.figure.Figure)

        # The KDE is rendered on a second density axis.
        assert len(fig.axes) == 2

        histogram_ax = fig.axes[0]
        density_ax = fig.axes[1]

        assert histogram_ax.get_xlabel() == "age"
        assert histogram_ax.get_ylabel() == "Count"
        assert density_ax.get_ylabel() == "Density"

    def test_drops_missing_values(self, numeric_frame):
        fig = plot_numeric_distribution(
            numeric_frame,
            "age",
        )

        ax = fig.axes[0]

        total_histogram_count = sum(
            patch.get_height()
            for patch in ax.patches
        )

        # age contains 10 rows and exactly one NaN, leaving 9
        # observed numeric values.
        assert total_histogram_count == 9

    def test_rejects_missing_column(self, numeric_frame):
        with pytest.raises(KeyError):
            plot_numeric_distribution(
                numeric_frame,
                "does_not_exist",
            )

    def test_rejects_non_numeric_column(self, mixed_frame):
        with pytest.raises(TypeError, match="not numeric"):
            plot_numeric_distribution(
                mixed_frame,
                "gender",
            )

    def test_rejects_column_with_no_observed_values(self):
        frame = pd.DataFrame(
            {
                "age": [np.nan, np.nan, np.nan],
            }
        )

        with pytest.raises(ValueError, match="observed"):
            plot_numeric_distribution(
                frame,
                "age",
            )

    def test_respects_custom_bins(self, numeric_frame):
        fig = plot_numeric_distribution(
            numeric_frame,
            "age",
            bins=5,
        )

        ax = fig.axes[0]

        assert len(ax.patches) == 5


# ============================================================================
# Categorical counts
# ============================================================================


class TestPlotCategoricalCounts:

    def test_returns_figure(self, categorical_frame):
        fig = plot_categorical_counts(
            categorical_frame,
            "gender",
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_bars(self, categorical_frame):
        fig = plot_categorical_counts(
            categorical_frame,
            "gender",
        )

        ax = fig.axes[0]

        assert len(ax.patches) == 2
        assert ax.get_xlabel() == "gender"
        assert ax.get_ylabel() == "Count"
        assert ax.get_title() == "Counts of gender"

    def test_respects_top_n(self):
        frame = pd.DataFrame(
            {
                "category": [
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "C",
                    "C",
                    "D",
                    "E",
                ]
            }
        )

        fig = plot_categorical_counts(
            frame,
            "category",
            top_n=2,
            show_other=False,
        )

        ax = fig.axes[0]

        assert len(ax.patches) == 2

    def test_handles_missing_values(self):
        frame = pd.DataFrame(
            {
                "category": [
                    "A",
                    "A",
                    "B",
                    np.nan,
                    "B",
                    np.nan,
                ]
            }
        )

        fig = plot_categorical_counts(
            frame,
            "category",
        )

        ax = fig.axes[0]

        labels = [
            tick.get_text()
            for tick in ax.get_xticklabels()
        ]

        assert "A" in labels
        assert "B" in labels

    def test_rejects_missing_column(self, categorical_frame):
        with pytest.raises(KeyError):
            plot_categorical_counts(
                categorical_frame,
                "does_not_exist",
            )

    def test_rejects_empty_observed_values(self):
        frame = pd.DataFrame(
            {
                "category": [np.nan, np.nan, np.nan],
            }
        )

        with pytest.raises(ValueError, match="observed"):
            plot_categorical_counts(
                frame,
                "category",
            )


# ============================================================================
# Univariate dispatch
# ============================================================================


class TestPlotUnivariate:

    def test_numeric_column_dispatches_to_numeric_plot(
        self,
        numeric_frame,
    ):
        fig = plot_univariate(
            numeric_frame,
            "age",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_title() == "Distribution of age"

    def test_categorical_column_dispatches_to_count_plot(
        self,
        categorical_frame,
    ):
        fig = plot_univariate(
            categorical_frame,
            "gender",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_title() == "Counts of gender"


# ============================================================================
# Scatter
# ============================================================================


class TestPlotScatter:

    def test_returns_figure(self, numeric_frame):
        fig = plot_scatter(
            numeric_frame,
            "age",
            "income",
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_scatter_points(self, numeric_frame):
        fig = plot_scatter(
            numeric_frame,
            "age",
            "income",
        )

        ax = fig.axes[0]

        assert len(ax.collections) >= 1
        assert ax.get_xlabel() == "age"
        assert ax.get_ylabel() == "income"
        assert ax.get_title() == "age vs income"

    def test_supports_numeric_hue(self, numeric_frame):
        fig = plot_scatter(
            numeric_frame,
            "age",
            "income",
            hue="score",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 2

    def test_supports_categorical_hue(self, mixed_frame):
        fig = plot_scatter(
            mixed_frame,
            "age",
            "income",
            hue="gender",
        )

        assert isinstance(fig, matplotlib.figure.Figure)

        ax = fig.axes[0]

        assert len(ax.collections) >= 1
        assert ax.get_legend() is not None
        assert len(ax.get_legend().get_texts()) >= 1

    def test_rejects_non_numeric_x(self, mixed_frame):
        with pytest.raises(TypeError, match="not numeric"):
            plot_scatter(
                mixed_frame,
                "gender",
                "income",
            )

    def test_rejects_non_numeric_y(self, mixed_frame):
        with pytest.raises(TypeError, match="not numeric"):
            plot_scatter(
                mixed_frame,
                "age",
                "gender",
            )

    def test_handles_missing_x_and_y(self):
        frame = pd.DataFrame(
            {
                "x": [1.0, 2.0, np.nan, 4.0],
                "y": [10.0, np.nan, 30.0, 40.0],
            }
        )

        fig = plot_scatter(
            frame,
            "x",
            "y",
        )

        ax = fig.axes[0]

        assert len(ax.collections) >= 1


# ============================================================================
# Box plot by group
# ============================================================================


class TestPlotBoxByGroup:

    def test_returns_figure(self, mixed_frame):
        fig = plot_box_by_group(
            mixed_frame,
            "income",
            "region",
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_boxplot(self, mixed_frame):
        fig = plot_box_by_group(
            mixed_frame,
            "income",
            "region",
        )

        ax = fig.axes[0]

        assert ax.get_xlabel() == "region"
        assert ax.get_ylabel() == "income"
        assert ax.get_title() == "income by region"

    def test_rejects_non_numeric_value_column(self, mixed_frame):
        with pytest.raises(TypeError, match="not numeric"):
            plot_box_by_group(
                mixed_frame,
                "gender",
                "region",
            )

    def test_handles_missing_values(self):
        frame = pd.DataFrame(
            {
                "value": [10, 20, np.nan, 40, 50],
                "group": ["A", "A", "A", "B", "B"],
            }
        )

        fig = plot_box_by_group(
            frame,
            "value",
            "group",
        )

        assert isinstance(fig, matplotlib.figure.Figure)


# ============================================================================
# Category heatmap
# ============================================================================


class TestPlotCategoryHeatmap:

    def test_returns_figure(self, categorical_frame):
        fig = plot_category_heatmap(
            categorical_frame,
            "gender",
            "region",
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_heatmap(self, categorical_frame):
        fig = plot_category_heatmap(
            categorical_frame,
            "gender",
            "region",
        )

        ax = fig.axes[0]

        assert len(ax.images) == 1
        assert ax.get_xlabel() == "region"
        assert ax.get_ylabel() == "gender"

    def test_handles_missing_values(self):
        frame = pd.DataFrame(
            {
                "a": ["A", "A", "B", np.nan, "B"],
                "b": ["X", "Y", "X", "Y", np.nan],
            }
        )

        fig = plot_category_heatmap(
            frame,
            "a",
            "b",
        )

        assert isinstance(fig, matplotlib.figure.Figure)


# ============================================================================
# Bivariate dispatch
# ============================================================================


class TestPlotBivariate:

    def test_numeric_numeric_dispatches_to_scatter(
        self,
        numeric_frame,
    ):
        fig = plot_bivariate(
            numeric_frame,
            "age",
            "income",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_title() == "age vs income"

    def test_numeric_categorical_dispatches_to_boxplot(
        self,
        mixed_frame,
    ):
        fig = plot_bivariate(
            mixed_frame,
            "income",
            "region",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_title() == "income by region"

    def test_categorical_numeric_dispatches_to_boxplot(
        self,
        mixed_frame,
    ):
        fig = plot_bivariate(
            mixed_frame,
            "region",
            "income",
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_categorical_categorical_dispatches_to_heatmap(
        self,
        categorical_frame,
    ):
        fig = plot_bivariate(
            categorical_frame,
            "gender",
            "region",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes[0].images) == 1


# ============================================================================
# Confusion matrix
# ============================================================================


class TestPlotConfusionMatrix:

    def test_returns_figure(self, classification_data):
        y_true, y_pred, _ = classification_data

        fig = plot_confusion_matrix(
            y_true,
            y_pred,
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_confusion_matrix(self, classification_data):
        y_true, y_pred, _ = classification_data

        fig = plot_confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2

        matrix_ax = fig.axes[0]
        colorbar_ax = fig.axes[1]

        assert matrix_ax.get_xlabel() == "Predicted label"
        assert matrix_ax.get_ylabel() == "True label"
        assert matrix_ax.get_title() == "Confusion matrix"
        assert colorbar_ax.get_label() == "<colorbar>"

    def test_supports_normalization(self, classification_data):
        y_true, y_pred, _ = classification_data

        fig = plot_confusion_matrix(
            y_true,
            y_pred,
            normalize="true",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            plot_confusion_matrix(
                [],
                [],
            )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            plot_confusion_matrix(
                [0, 1, 1],
                [0, 1],
            )


# ============================================================================
# ROC curve
# ============================================================================


class TestPlotRocCurve:

    def test_returns_figure(self, classification_data):
        y_true, _, y_score = classification_data

        fig = plot_roc_curve(
            y_true,
            y_score,
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_roc_curve(self, classification_data):
        y_true, _, y_score = classification_data

        fig = plot_roc_curve(
            y_true,
            y_score,
        )

        ax = fig.axes[0]

        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"

    def test_includes_chance_line(self, classification_data):
        y_true, _, y_score = classification_data

        fig = plot_roc_curve(
            y_true,
            y_score,
        )

        ax = fig.axes[0]

        lines = ax.get_lines()

        assert len(lines) >= 2

    def test_rejects_non_binary_target(self):
        y_true = [0, 1, 2, 0]
        y_score = [0.1, 0.8, 0.7, 0.2]

        with pytest.raises(ValueError):
            plot_roc_curve(
                y_true,
                y_score,
            )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            plot_roc_curve(
                [0, 1, 1],
                [0.1, 0.8],
            )


# ============================================================================
# Precision-recall curve
# ============================================================================


class TestPlotPrecisionRecallCurve:

    def test_returns_figure(self, classification_data):
        y_true, _, y_score = classification_data

        fig = plot_precision_recall_curve(
            y_true,
            y_score,
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_precision_recall_curve(
        self,
        classification_data,
    ):
        y_true, _, y_score = classification_data

        fig = plot_precision_recall_curve(
            y_true,
            y_score,
        )

        ax = fig.axes[0]

        # Recent scikit-learn versions include the positive class
        # in the axis label, e.g. "Recall (Positive label: 1)".
        assert ax.get_xlabel().startswith("Recall")
        assert ax.get_ylabel().startswith("Precision")

    def test_rejects_non_binary_target(self):
        y_true = [0, 1, 2, 0]
        y_score = [0.1, 0.8, 0.7, 0.2]

        with pytest.raises(ValueError):
            plot_precision_recall_curve(
                y_true,
                y_score,
            )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            plot_precision_recall_curve(
                [0, 1, 1],
                [0.1, 0.8],
            )


# ============================================================================
# Residuals
# ============================================================================


class TestPlotResiduals:

    def test_returns_figure(self, regression_data):
        y_true, y_pred = regression_data

        fig = plot_residuals(
            y_true,
            y_pred,
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_residual_plot(self, regression_data):
        y_true, y_pred = regression_data

        fig = plot_residuals(
            y_true,
            y_pred,
        )

        ax = fig.axes[0]

        assert ax.get_xlabel() == "Predicted"
        assert ax.get_ylabel() == "Residual"

    def test_includes_zero_line(self, regression_data):
        y_true, y_pred = regression_data

        fig = plot_residuals(
            y_true,
            y_pred,
        )

        ax = fig.axes[0]

        lines = ax.get_lines()

        assert len(lines) >= 1

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            plot_residuals(
                [1, 2, 3],
                [1, 2],
            )


# ============================================================================
# Predicted vs actual
# ============================================================================


class TestPlotPredictedVsActual:

    def test_returns_figure(self, regression_data):
        y_true, y_pred = regression_data

        fig = plot_predicted_vs_actual(
            y_true,
            y_pred,
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_plot(self, regression_data):
        y_true, y_pred = regression_data

        fig = plot_predicted_vs_actual(
            y_true,
            y_pred,
        )

        ax = fig.axes[0]

        assert ax.get_xlabel() == "Actual"
        assert ax.get_ylabel() == "Predicted"

    def test_includes_reference_line(self, regression_data):
        y_true, y_pred = regression_data

        fig = plot_predicted_vs_actual(
            y_true,
            y_pred,
        )

        ax = fig.axes[0]

        lines = ax.get_lines()

        assert len(lines) >= 1

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            plot_predicted_vs_actual(
                [1, 2, 3],
                [1, 2],
            )


# ============================================================================
# Feature importance
# ============================================================================


class TestPlotFeatureImportance:

    @pytest.fixture
    def importance_frame(self):
        return pd.DataFrame(
            {
                "feature": [
                    "age",
                    "income",
                    "score",
                    "tenure",
                    "region",
                ],
                "importance": [
                    0.35,
                    0.25,
                    0.20,
                    0.12,
                    0.08,
                ],
            }
        )

    def test_returns_figure(self, importance_frame):
        fig = plot_feature_importance(
            importance_frame,
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_horizontal_bars(self, importance_frame):
        fig = plot_feature_importance(
            importance_frame,
        )

        ax = fig.axes[0]

        assert len(ax.patches) == 5
        assert ax.get_xlabel() == "Importance"
        assert ax.get_ylabel() == "Feature"

    def test_respects_top_n(self, importance_frame):
        fig = plot_feature_importance(
            importance_frame,
            top_n=3,
        )

        ax = fig.axes[0]

        assert len(ax.patches) == 3

    def test_sorts_by_importance(self, importance_frame):
        fig = plot_feature_importance(
            importance_frame,
        )

        ax = fig.axes[0]

        labels = [
            tick.get_text()
            for tick in ax.get_yticklabels()
        ]

        assert labels[0] == "region"
        assert labels[-1] == "age"

    def test_supports_index_as_feature_column(self):
        frame = pd.DataFrame(
            {
                "importance": [0.1, 0.3, 0.2],
            },
            index=["a", "b", "c"],
        )

        fig = plot_feature_importance(
            frame,
            feature_column="index",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes[0].patches) == 3

    def test_rejects_missing_importance_column(self):
        frame = pd.DataFrame(
            {
                "feature": ["a", "b"],
            }
        )

        with pytest.raises(KeyError):
            plot_feature_importance(
                frame,
            )

    def test_rejects_empty_dataframe(self):
        frame = pd.DataFrame(
            columns=["feature", "importance"],
        )

        with pytest.raises(ValueError):
            plot_feature_importance(
                frame,
            )


# ============================================================================
# Axis reuse
# ============================================================================


class TestAxisReuse:

    def test_numeric_distribution_uses_supplied_axis(
        self,
        numeric_frame,
    ):
        fig, ax = plt.subplots()

        returned_fig = plot_numeric_distribution(
            numeric_frame,
            "age",
            ax=ax,
        )

        assert returned_fig is fig
        assert returned_fig.axes[0] is ax

    def test_categorical_counts_uses_supplied_axis(
        self,
        categorical_frame,
    ):
        fig, ax = plt.subplots()

        returned_fig = plot_categorical_counts(
            categorical_frame,
            "gender",
            ax=ax,
        )

        assert returned_fig is fig
        assert returned_fig.axes[0] is ax

    def test_scatter_uses_supplied_axis(
        self,
        numeric_frame,
    ):
        fig, ax = plt.subplots()

        returned_fig = plot_scatter(
            numeric_frame,
            "age",
            "income",
            ax=ax,
        )

        assert returned_fig is fig
        assert returned_fig.axes[0] is ax

    def test_feature_importance_uses_supplied_axis(self):
        frame = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "importance": [0.3, 0.2, 0.1],
            }
        )

        fig, ax = plt.subplots()

        returned_fig = plot_feature_importance(
            frame,
            ax=ax,
        )

        assert returned_fig is fig
        assert returned_fig.axes[0] is ax