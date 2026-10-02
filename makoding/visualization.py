"""Visualization utilities for Makoding.

Matplotlib-based chart generation for univariate analysis, bivariate
analysis, data-quality and cleaning diagnostics, statistical
diagnostics, unsupervised-learning diagnostics, and supervised-model
diagnostics. Every function returns a ``matplotlib.figure.Figure`` —
none of them call ``plt.show()`` — so callers decide how to display
it: ``st.pyplot(fig)`` in Streamlit, ``fig.savefig(...)`` to a file,
or direct inspection of ``fig.axes`` in a test.

Families of plots
-----------------
- **Univariate**: :func:`plot_histogram` / :func:`plot_numeric_distribution`,
  :func:`plot_categorical_counts` (with an optional ``hue`` for
  grouped/stacked counts), :func:`plot_violin`, and the dispatcher
  :func:`plot_univariate`.
- **Bivariate**: :func:`plot_scatter`, :func:`plot_box_by_group`,
  :func:`plot_category_heatmap`, :func:`plot_pairplot`, and the
  dispatcher :func:`plot_bivariate`.
- **Data quality and cleaning**: :func:`plot_missingness`,
  :func:`plot_missing_matrix`, :func:`plot_correlation_heatmap`,
  :func:`plot_correlation_matrix`, :func:`plot_cleaning_impact`,
  :func:`plot_outlier_fences`, :func:`plot_before_after`. These pair with
  :mod:`makoding.cleaning` (``CleaningAudit``, the fences learned by a
  fitted ``Cleaner``) and :mod:`makoding.eda`.
- **Statistical diagnostics**: :func:`plot_qq`,
  :func:`plot_regression_diagnostics`, :func:`plot_time_series`,
  :func:`plot_forecast`. These pair with the statistical-analysis layer
  (normality tests, ``ols_regression`` results, ``forecast_arima`` output).
- **Unsupervised diagnostics**: :func:`plot_pca_scatter`,
  :func:`plot_cluster_scatter`, :func:`plot_elbow`,
  :func:`plot_silhouette_by_k`. These take the DataFrames/arrays
  produced by :mod:`makoding.unsupervised` (``PCAResult.transformed``,
  ``ClusteringResult.labels``, ``kmeans_elbow``'s output) rather than
  a raw frame and column name.
- **Supervised model diagnostics**: :func:`plot_confusion_matrix`,
  :func:`plot_roc_curve`, :func:`plot_precision_recall_curve`,
  :func:`plot_calibration_curve`, :func:`plot_residuals`,
  :func:`plot_prediction_error`, :func:`plot_predicted_vs_actual`,
  :func:`plot_feature_importance`, :func:`plot_cross_validation_scores`,
  :func:`plot_model_comparison`. These take ``y_true``/``y_pred``/``y_score``
  arrays or a ``Model.cross_validate()``-shaped DataFrame, not a frame and
  column name.

Design principles
-----------------
- Most functions accept an optional ``ax`` (a matplotlib ``Axes``) so
  plots can be composed into a grid; when omitted, a new figure is
  created. Either way, the function returns the owning ``Figure``.
  :func:`plot_pairplot`, :func:`plot_before_after` and
  :func:`plot_regression_diagnostics` are the exceptions — they always own a
  multi-panel grid, so they do not accept ``ax``.
- Figures are built with ``matplotlib.figure.Figure`` directly rather than
  through ``pyplot``, so they are never registered in pyplot's global
  figure registry. A figure that a caller forgets to close is simply
  garbage-collected; it cannot accumulate across Streamlit reruns and
  trigger matplotlib's "more than 20 figures" warning. (Calling
  ``plt.close(fig)`` on one of these figures is harmless.)
- Colours come from :mod:`makoding.palette`, the same tokens the Streamlit
  theme uses, so charts match the page they are embedded in.
- Functions validate input and raise typed errors (``TypeError``,
  ``ValueError``, ``KeyError``), matching the rest of this package.
- Nothing here mutates a caller's DataFrame or array.
- Every function's docstring has an ``Assumptions`` section — read it
  before trusting a plot's shape on data you haven't looked at yet
  (in particular, several functions cap how many categories/points
  they draw, for readability, and document exactly how).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: safe on servers, CI, and tests

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from .feature_engineering import (
    _resolve_numeric_columns,
    _validate_columns_exist,
    _validate_dataframe,
    _validate_numeric_columns,
)
from .palette import (
    ACCENT,
    BORDER,
    CATEGORICAL,
    INK,
    MUTED,
    NAVY,
    TEAL,
    TEAL_DARK,
    TEAL_LIGHT,
    WHITE,
)

__all__ = [
    # Univariate
    "plot_histogram",
    "plot_numeric_distribution",
    "plot_categorical_counts",
    "plot_violin",
    "plot_univariate",
    # Bivariate
    "plot_scatter",
    "plot_box_by_group",
    "plot_category_heatmap",
    "plot_pairplot",
    "plot_bivariate",
    # Data quality and cleaning
    "plot_missingness",
    "plot_missing_matrix",
    "plot_correlation_heatmap",
    "plot_correlation_matrix",
    "plot_cleaning_impact",
    "plot_outlier_fences",
    "plot_before_after",
    # Statistical diagnostics
    "plot_qq",
    "plot_regression_diagnostics",
    "plot_time_series",
    "plot_forecast",
    # Unsupervised diagnostics
    "plot_pca_scatter",
    "plot_cluster_scatter",
    "plot_elbow",
    "plot_silhouette_by_k",
    # Supervised model diagnostics
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    "plot_residuals",
    "plot_prediction_error",
    "plot_predicted_vs_actual",
    "plot_feature_importance",
    "plot_cross_validation_scores",
    "plot_model_comparison",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_DEFAULT_FIGSIZE = (8, 5)
_PRIMARY = TEAL
_ANNOTATION_LIMIT = 25  # correlation cells are annotated up to this many columns

_SEQUENTIAL = LinearSegmentedColormap.from_list("datalab_sequential", [WHITE, TEAL])
_DIVERGING = LinearSegmentedColormap.from_list("datalab_diverging", [ACCENT, WHITE, TEAL])
_POINTS = LinearSegmentedColormap.from_list("datalab_points", ["#9EC8D4", NAVY])
_BINARY = ListedColormap([TEAL_LIGHT, ACCENT])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cat(index: int) -> str:
    """Return the ``index``-th categorical colour, cycling if needed."""
    return CATEGORICAL[index % len(CATEGORICAL)]


def _get_figure_and_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Return ``(figure, axes)``, creating a new figure only if ``ax`` is ``None``.

    New figures are created with ``Figure`` directly (not ``pyplot``) so they
    are never held in pyplot's global registry.
    """
    if ax is None:
        fig = Figure(figsize=figsize, facecolor=WHITE)
        ax = fig.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _style_axes(ax: Axes, label_size: float = 9) -> None:
    """Apply the design-system look to one axes (no-op for colorbar axes)."""
    if ax.get_label() == "<colorbar>":
        ax.tick_params(colors=MUTED, labelsize=label_size)
        return

    ax.set_facecolor(WHITE)
    for name in ("top", "right"):
        spine = ax.spines.get(name)
        if spine is not None and not ax.images:
            spine.set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=label_size)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)

    title = ax.get_title()
    if title:
        ax.set_title(
            title,
            loc="center",
            fontsize=12,
            fontweight="bold",
            color=NAVY,
            pad=10,
        )

    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_color(BORDER)


def _finalize(fig: Figure, label_size: float = 9) -> Figure:
    """Style every axes, lay the figure out, and return it."""
    for axes in fig.axes:
        _style_axes(axes, label_size=label_size)
    fig.tight_layout()
    return fig


def _boxplot(ax: Axes, data: Sequence[Any], labels: Sequence[str]) -> Any:
    """Draw a themed box plot, tolerating the ``labels`` -> ``tick_labels`` rename.

    Matplotlib 3.9 renamed ``boxplot``'s ``labels`` argument to ``tick_labels``
    (``labels`` was later deprecated); trying the new name first and falling
    back keeps this working on both sides of that change.
    """
    style = dict(
        patch_artist=True,
        boxprops=dict(facecolor=TEAL_LIGHT, edgecolor=TEAL),
        medianprops=dict(color=ACCENT, linewidth=2),
        whiskerprops=dict(color=TEAL),
        capprops=dict(color=TEAL),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor=MUTED),
    )
    try:
        return ax.boxplot(data, tick_labels=labels, **style)
    except TypeError:
        return ax.boxplot(data, labels=labels, **style)


def _style_violin(parts: dict[str, Any]) -> None:
    for body in parts["bodies"]:
        body.set_facecolor(_PRIMARY)
        body.set_edgecolor(TEAL_DARK)
        body.set_alpha(0.55)
    for key in ("cmeans", "cmedians"):
        if key in parts:
            parts[key].set_color(ACCENT)
    for key in ("cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(TEAL_DARK)


def _note_axes(ax: Axes, message: str) -> None:
    """Replace an empty plot area with a centred explanatory message."""
    ax.text(
        0.5, 0.5, message, ha="center", va="center",
        color=MUTED, fontsize=11, transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _int_attr(obj: Any, name: str) -> int:
    """Read an integer attribute defensively (missing/None/garbage -> 0)."""
    try:
        return int(getattr(obj, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _validate_array_pair(y_true: Sequence[Any], y_pred: Sequence[Any], names: tuple[str, str]) -> None:
    """Validate that two prediction-style arrays are non-empty and equal length."""
    name_a, name_b = names
    if len(y_true) == 0:
        raise ValueError(f"{name_a} must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{name_a} and {name_b} must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )


def _validate_numeric_array(values: Sequence[Any], *, name: str) -> np.ndarray:
    """Return a finite 1-D float array, or raise; never silently drops values."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains missing or non-finite values.")
    return numeric


def _plot_scatter_by_group(
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[Any],
    max_categories: int = 10,
    legend_title: str | None = None,
) -> None:
    """Draw a scatter plot on ``ax``, colored by discrete ``groups``, with a legend.

    ``x``, ``y``, and ``groups`` must already be equal-length,
    aligned, and free of missing values the caller cares about —
    this helper does no filtering of its own. Groups beyond
    ``max_categories`` (by frequency) are merged into ``"Other"``.
    Shared by :func:`plot_scatter`, :func:`plot_pca_scatter`, and
    :func:`plot_cluster_scatter` so their hue-coloring behaves
    identically.
    """
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    group_series = pd.Series(np.asarray(groups, dtype=object)).astype(str)

    top_groups = group_series.value_counts().index[:max_categories]
    grouped = group_series.where(group_series.isin(top_groups), "Other")

    for index, group in enumerate(sorted(grouped.unique())):
        mask = (grouped == group).to_numpy()
        ax.scatter(
            x_array[mask], y_array[mask],
            alpha=0.6, edgecolor="none", color=_cat(index), label=group,
        )
    ax.legend(title=legend_title, fontsize=8, loc="best")


# ---------------------------------------------------------------------------
# Univariate
# ---------------------------------------------------------------------------

def plot_numeric_distribution(
    frame: pd.DataFrame,
    column: str,
    bins: int = 30,
    kde: bool = False,
    ax: Axes | None = None,
) -> Figure:
    """Plot a histogram of a numeric column.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    column:
        Numeric column to plot.
    bins:
        Number of histogram bins.
    kde:
        If ``True``, overlay a kernel-density-estimate curve (via
        ``scipy.stats.gaussian_kde``) on a secondary y-axis. Requires
        at least 2 non-missing values with nonzero variance; silently
        skipped (not an error) if the data can't support a KDE — see
        Assumptions.
    ax:
        Existing matplotlib ``Axes`` to draw on. A new figure is
        created if omitted.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or ``column`` is not numeric.
    KeyError
        If ``column`` does not exist.
    ValueError
        If ``column`` has no non-missing values.

    Assumptions
    -----------
    - Missing values are dropped before plotting, silently — the plot
      reflects only observed values, with no on-chart indication of
      how many were missing (pair this with
      ``makoding.eda.missing_values_summary`` if that matters for your
      audience).
    - ``kde=True`` is skipped without raising when the data has fewer
      than 2 distinct values (a KDE is undefined for a single unique
      value) — the histogram is still drawn either way.
    """
    _validate_columns_exist(frame, [column])
    _validate_numeric_columns(frame, [column])

    values = frame[column].dropna()
    if values.empty:
        raise ValueError(f"Column {column!r} has no observed (non-missing) values to plot")

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.hist(values, bins=bins, color=_PRIMARY, edgecolor="white", alpha=0.85)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {column}")
    ax.grid(axis="y", alpha=0.3)

    if kde and values.nunique() >= 2 and values.std() > 0:
        from scipy.stats import gaussian_kde

        kde_estimator = gaussian_kde(values)
        x_grid = np.linspace(values.min(), values.max(), 200)
        density = kde_estimator(x_grid)

        kde_ax = ax.twinx()
        kde_ax.plot(x_grid, density, color=ACCENT, linewidth=2)
        kde_ax.set_ylabel("Density")
        kde_ax.set_yticks([])

    return _finalize(fig)


def plot_histogram(
    frame: pd.DataFrame,
    column: str,
    bins: int = 30,
    ax: Axes | None = None,
) -> Figure:
    """Plot a histogram of a numeric column.

    A plainly-named alias for :func:`plot_numeric_distribution` with
    ``kde=False``, for callers who want a histogram specifically and
    don't need the KDE overlay option. See
    :func:`plot_numeric_distribution` for parameters, return value,
    errors, and assumptions — they're identical here.
    """
    return plot_numeric_distribution(frame, column, bins=bins, kde=False, ax=ax)


def plot_categorical_counts(
    frame: pd.DataFrame,
    column: str,
    hue: str | None = None,
    top_n: int = 20,
    show_other: bool = True,
    stacked: bool = False,
    ax: Axes | None = None,
) -> Figure:
    """Plot a bar chart of value counts for a categorical (or any discrete) column.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    column:
        Column to count values for. Any dtype is accepted.
    hue:
        Optional second column to split each count by, producing
        grouped (or ``stacked=True``) bars — e.g. counts of ``city``
        broken down by ``plan``. Any dtype is accepted.
    top_n:
        Maximum number of individual ``column`` categories to show,
        ranked by total frequency (summed across ``hue`` groups when
        ``hue`` is given) descending.
    show_other:
        If ``True`` (default) and more than ``top_n`` distinct values
        exist, the remaining values are summed into one additional
        ``"Other"`` bar (or bar group). If ``False``, they are simply
        omitted.
    stacked:
        Only relevant when ``hue`` is given. If ``True``, hue groups
        are stacked into a single bar per ``column`` category; if
        ``False`` (default), they're drawn side by side.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If ``column`` or ``hue`` does not exist.
    ValueError
        If there are no non-missing values to plot.

    Assumptions
    -----------
    - Missing values in ``column`` are excluded entirely, and (when
      ``hue`` is given) a row missing either ``column`` or ``hue`` is
      dropped — neither ever appears as its own bar/segment.
    - The ``"Other"`` bar (when shown) is a **sum** of every category
      beyond the top ``top_n`` — it is not itself broken down further,
      and its presence can be misleading if it happens to be the
      tallest bar; check ``frame[column].nunique()`` if you suspect
      this.
    - With ``hue``, every distinct ``hue`` value gets its own color,
      with no cap on how many — a high-cardinality ``hue`` column
      will produce an unreadable legend; cap it yourself (e.g. by
      pre-grouping rare categories) if that's a risk.
    """
    if hue is None:
        _validate_columns_exist(frame, [column])
        counts = frame[column].dropna().value_counts()
        if counts.empty:
            raise ValueError(f"Column {column!r} has no observed (non-missing) values to plot")

        if len(counts) > top_n:
            shown = counts.iloc[:top_n]
            if show_other:
                other_total = counts.iloc[top_n:].sum()
                shown = pd.concat([shown, pd.Series({"Other": other_total})])
            counts = shown

        fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
        labels = [str(v) for v in counts.index]
        ax.bar(labels, counts.values, color=_PRIMARY, edgecolor="white", alpha=0.85)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"Counts of {column}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        return _finalize(fig)

    _validate_columns_exist(frame, [column, hue])
    working = frame[[column, hue]].dropna()
    if working.empty:
        raise ValueError(f"No rows have both {column!r} and {hue!r} non-missing")

    totals = working[column].value_counts()
    kept = totals.index[:top_n].tolist()
    working = working.copy()
    if len(totals) > top_n:
        if show_other:
            working[column] = working[column].where(working[column].isin(kept), "Other")
            row_order = kept + ["Other"]
        else:
            working = working[working[column].isin(kept)]
            row_order = kept
    else:
        row_order = kept

    crosstab = pd.crosstab(working[column], working[hue]).reindex(row_order)

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    x_positions = np.arange(len(crosstab.index))
    hue_categories = crosstab.columns.tolist()
    row_labels = [str(i) for i in crosstab.index]

    if stacked:
        bottom = np.zeros(len(crosstab.index))
        for index, hue_value in enumerate(hue_categories):
            values = crosstab[hue_value].to_numpy()
            ax.bar(x_positions, values, bottom=bottom, color=_cat(index), label=str(hue_value))
            bottom = bottom + values
    else:
        width = 0.8 / max(len(hue_categories), 1)
        for index, hue_value in enumerate(hue_categories):
            offset = (index - (len(hue_categories) - 1) / 2) * width
            ax.bar(
                x_positions + offset, crosstab[hue_value].to_numpy(), width=width,
                color=_cat(index), label=str(hue_value),
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(row_labels, rotation=45, ha="right")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Counts of {column} by {hue}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=hue, fontsize=8, loc="best")
    return _finalize(fig)


def plot_violin(
    frame: pd.DataFrame,
    numeric_column: str,
    group_column: str | None = None,
    max_groups: int = 15,
    ax: Axes | None = None,
) -> Figure:
    """Plot a violin plot of a numeric column, optionally split by group.

    A violin plot shows the same distributional shape a KDE would,
    mirrored into a symmetric "violin" — useful for comparing
    distribution shape (not just median/quartiles, unlike a box plot)
    across groups.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    numeric_column:
        Numeric column to summarize.
    group_column:
        Optional column defining groups, each drawn as its own
        violin. If omitted, a single violin for the whole column is
        drawn.
    max_groups:
        When ``group_column`` is given, the maximum number of groups
        shown, kept by descending group size. Groups beyond this are
        omitted entirely (no merged "Other" violin, for the same
        reason as :func:`plot_box_by_group`).
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or ``numeric_column`` is not
        numeric.
    KeyError
        If ``numeric_column`` or ``group_column`` does not exist.
    ValueError
        If there are no non-missing values to plot.

    Assumptions
    -----------
    - Rows missing either column involved are dropped.
    - A violin's shape comes from a kernel density estimate, which
      needs at least 2 distinct values in that group; a group with
      only one distinct value renders as a degenerate (near-flat)
      shape rather than raising.
    """
    _validate_columns_exist(frame, [numeric_column] + ([group_column] if group_column else []))
    _validate_numeric_columns(frame, [numeric_column])

    if group_column is None:
        values = frame[numeric_column].dropna()
        if values.empty:
            raise ValueError(f"Column {numeric_column!r} has no observed (non-missing) values to plot")

        fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
        parts = ax.violinplot([values.to_numpy()], showmeans=True, showmedians=True)
        _style_violin(parts)
        ax.set_xticks([1])
        ax.set_xticklabels([numeric_column])
        ax.set_ylabel(numeric_column)
        ax.set_title(f"Distribution of {numeric_column}")
    else:
        working = frame[[numeric_column, group_column]].dropna()
        if working.empty:
            raise ValueError(
                f"No rows have both {numeric_column!r} and {group_column!r} non-missing"
            )

        group_sizes = working[group_column].value_counts()
        kept_groups = group_sizes.index[:max_groups]
        data = [
            working.loc[working[group_column] == group, numeric_column].to_numpy()
            for group in kept_groups
        ]
        labels = [str(g) for g in kept_groups]

        fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        _style_violin(parts)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(numeric_column)
        ax.set_xlabel(group_column)
        ax.set_title(f"{numeric_column} by {group_column}")

    ax.grid(axis="y", alpha=0.3)
    return _finalize(fig)


def plot_univariate(frame: pd.DataFrame, column: str, ax: Axes | None = None, **kwargs: Any) -> Figure:
    """Plot a single column, auto-dispatching by dtype.

    Numeric columns get :func:`plot_numeric_distribution`; everything
    else gets :func:`plot_categorical_counts`.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    column:
        Column to plot.
    ax:
        Existing matplotlib ``Axes`` to draw on.
    **kwargs:
        Forwarded to whichever underlying function is dispatched to
        (e.g. ``bins``/``kde`` for numeric, ``top_n``/``hue`` for
        categorical). Passing a keyword the dispatched function
        doesn't accept raises ``TypeError`` from that function, not
        from this one.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    See :func:`plot_numeric_distribution` / :func:`plot_categorical_counts`.

    Assumptions
    -----------
    Dispatch is based purely on ``pandas.api.types.is_numeric_dtype``
    — a numeric column that is actually a low-cardinality coded
    category (e.g. a ``0``/``1``/``2`` status code) is plotted as a
    histogram, not value counts, unless you call
    :func:`plot_categorical_counts` directly.
    """
    _validate_columns_exist(frame, [column])
    if pd.api.types.is_numeric_dtype(frame[column]):
        return plot_numeric_distribution(frame, column, ax=ax, **kwargs)
    return plot_categorical_counts(frame, column, ax=ax, **kwargs)


# ---------------------------------------------------------------------------
# Bivariate
# ---------------------------------------------------------------------------

def plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    max_hue_categories: int = 10,
    ax: Axes | None = None,
) -> Figure:
    """Plot a scatter chart of two numeric columns, optionally colored by a third.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    x, y:
        Numeric columns for the horizontal and vertical axes.
    hue:
        Optional column to color points by. If numeric, points are
        colored on a continuous colormap with a colorbar. If
        non-numeric, points are colored by discrete category with a
        legend (see ``max_hue_categories``).
    max_hue_categories:
        For a non-numeric ``hue`` with more distinct values than this,
        the least frequent categories beyond the top
        ``max_hue_categories`` are grouped into a single ``"Other"``
        legend entry, so the legend stays readable.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or ``x``/``y`` is not numeric.
    KeyError
        If ``x``, ``y``, or ``hue`` does not exist.
    ValueError
        If no rows have both ``x`` and ``y`` non-missing.

    Assumptions
    -----------
    - A row is dropped from the plot entirely if either ``x`` or ``y``
      is missing; a row with a missing ``hue`` value is still plotted
      when ``hue`` is non-numeric (grouped in with whichever category
      it falls into after ``fillna``) but dropped when ``hue`` is
      numeric (a continuous colormap has no way to represent a
      missing value).
    - Overlapping points are drawn with partial transparency
      (``alpha=0.6``) to make density visible, but this plot does not
      compute or display exact overlap counts — for a dense dataset,
      consider binning first.
    """
    _validate_columns_exist(frame, [x, y] + ([hue] if hue else []))
    _validate_numeric_columns(frame, [x, y])

    working = frame[[x, y] + ([hue] if hue else [])].copy()
    working = working.dropna(subset=[x, y])
    if working.empty:
        raise ValueError(f"No rows have both {x!r} and {y!r} non-missing")

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)

    if hue is None:
        ax.scatter(working[x], working[y], color=_PRIMARY, alpha=0.6, edgecolor="none")
    elif pd.api.types.is_numeric_dtype(working[hue]):
        working = working.dropna(subset=[hue])
        scatter = ax.scatter(
            working[x], working[y], c=working[hue], cmap=_POINTS, alpha=0.8, edgecolor="none"
        )
        fig.colorbar(scatter, ax=ax, label=hue)
    else:
        hue_values = working[hue].fillna("Missing").astype(str)
        _plot_scatter_by_group(
            ax, working[x], working[y], hue_values, max_categories=max_hue_categories, legend_title=hue
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_box_by_group(
    frame: pd.DataFrame,
    numeric_column: str,
    group_column: str,
    max_groups: int = 15,
    ax: Axes | None = None,
) -> Figure:
    """Plot a box plot of a numeric column, split by groups of a second column.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    numeric_column:
        Numeric column to summarize per group.
    group_column:
        Column defining the groups. Any dtype is accepted.
    max_groups:
        Maximum number of groups to show, kept by descending group
        size (largest groups first). Groups beyond this are omitted
        entirely (unlike :func:`plot_categorical_counts`, there is no
        ``"Other"`` box, since merging box-plot statistics across
        unrelated groups would misrepresent both).
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or ``numeric_column`` is not
        numeric.
    KeyError
        If ``numeric_column`` or ``group_column`` does not exist.
    ValueError
        If no rows have both columns non-missing.

    Assumptions
    -----------
    - A row is dropped if either ``numeric_column`` or ``group_column``
      is missing.
    - Groups beyond ``max_groups`` are silently omitted (not merged),
      so this plot alone does not represent 100% of the data on
      high-cardinality group columns — check
      ``frame[group_column].nunique()`` if that matters.
    - Boxes are ordered by descending group size, not alphabetically
      or by median value.
    """
    _validate_columns_exist(frame, [numeric_column, group_column])
    _validate_numeric_columns(frame, [numeric_column])

    working = frame[[numeric_column, group_column]].dropna()
    if working.empty:
        raise ValueError(
            f"No rows have both {numeric_column!r} and {group_column!r} non-missing"
        )

    group_sizes = working[group_column].value_counts()
    kept_groups = group_sizes.index[:max_groups]

    data = [
        working.loc[working[group_column] == group, numeric_column].to_numpy()
        for group in kept_groups
    ]
    labels = [str(g) for g in kept_groups]

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    _boxplot(ax, data, labels)
    ax.set_xlabel(group_column)
    ax.set_ylabel(numeric_column)
    ax.set_title(f"{numeric_column} by {group_column}")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    return _finalize(fig)


def plot_category_heatmap(
    frame: pd.DataFrame,
    column_a: str,
    column_b: str,
    max_categories: int = 20,
    ax: Axes | None = None,
) -> Figure:
    """Plot a heatmap of co-occurrence counts between two discrete columns.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    column_a, column_b:
        Columns to cross-tabulate. Any dtype is accepted.
    max_categories:
        Maximum number of categories kept per axis (by descending
        frequency); this bounds the heatmap to at most
        ``max_categories x max_categories`` cells for readability.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If ``column_a`` or ``column_b`` does not exist.
    ValueError
        If no rows have both columns non-missing.

    Assumptions
    -----------
    - A row is dropped if either column is missing.
    - Categories beyond ``max_categories`` (per axis, independently)
      are dropped, not aggregated into an "Other" row/column — the
      displayed cell counts do not necessarily sum to the total row
      count for a high-cardinality column.
    """
    _validate_columns_exist(frame, [column_a, column_b])

    working = frame[[column_a, column_b]].dropna()
    if working.empty:
        raise ValueError(f"No rows have both {column_a!r} and {column_b!r} non-missing")

    top_a = working[column_a].value_counts().index[:max_categories]
    top_b = working[column_b].value_counts().index[:max_categories]
    working = working[working[column_a].isin(top_a) & working[column_b].isin(top_b)]

    crosstab = pd.crosstab(working[column_a], working[column_b])

    figsize = (max(6, 0.5 * len(crosstab.columns) + 2), max(5, 0.5 * len(crosstab.index) + 2))
    fig, ax = _get_figure_and_axes(ax, figsize)
    image = ax.imshow(crosstab.values, cmap=_SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels([str(c) for c in crosstab.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels([str(i) for i in crosstab.index])
    ax.set_xlabel(column_b)
    ax.set_ylabel(column_a)
    ax.set_title(f"{column_a} vs {column_b}")
    fig.colorbar(image, ax=ax, label="Count")
    return _finalize(fig)


def plot_pairplot(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    hue: str | None = None,
    max_columns: int = 6,
    max_hue_categories: int = 10,
) -> Figure:
    """Plot a grid of pairwise relationships among several numeric columns.

    Diagonal cells show each column's histogram; off-diagonal cells
    show a scatter plot of the row column against the column column.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    columns:
        Numeric columns to include. Defaults to every numeric column
        in ``frame`` (subject to ``max_columns``).
    hue:
        Optional column to color scatter points by (discrete
        categories with a legend; see ``max_hue_categories``). The
        diagonal histograms are not split by ``hue``.
    max_columns:
        Maximum number of columns included in the grid — the panel
        count grows with the square of this, so keep it modest
        (default 6 -> up to 36 panels). Columns beyond this are
        dropped, with a warning logged listing which ones.
    max_hue_categories:
        For a non-numeric ``hue`` with more distinct values than this,
        the rest are grouped into ``"Other"`` in the legend.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or an explicitly requested
        column is not numeric.
    KeyError
        If a requested column (or ``hue``) does not exist.
    ValueError
        If fewer than 2 numeric columns are available to plot.

    Assumptions
    -----------
    - This function does **not** accept ``ax`` — it always creates
      and owns an ``n x n`` grid of axes as a new figure, unlike
      most other plotting functions in this module.
    - A row missing any of the plotted ``columns`` is dropped from
      every panel (not just the ones involving the missing value),
      so all panels represent exactly the same set of rows.
    - Rendering time and panel legibility both degrade quickly beyond
      roughly 6-8 columns; ``max_columns`` exists specifically to
      keep this from silently producing an impractically large figure.
    """
    _validate_dataframe(frame)

    if columns is None:
        columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = list(columns)
        _validate_columns_exist(frame, columns + ([hue] if hue else []))
        _validate_numeric_columns(frame, columns)

    if len(columns) > max_columns:
        dropped = columns[max_columns:]
        columns = columns[:max_columns]
        logger.warning(
            "plot_pairplot: showing only the first %d columns; dropped: %s", max_columns, dropped
        )

    if len(columns) < 2:
        raise ValueError(
            f"plot_pairplot requires at least 2 numeric columns, got {len(columns)}"
        )

    subset_columns = columns + ([hue] if hue else [])
    working = frame[subset_columns].dropna(subset=columns)
    if working.empty:
        raise ValueError("No rows have every plotted column non-missing")

    hue_groups: pd.Series | None = None
    unique_groups: list[str] = []
    if hue:
        hue_series = working[hue].fillna("Missing").astype(str)
        top_groups = hue_series.value_counts().index[:max_hue_categories]
        hue_groups = hue_series.where(hue_series.isin(top_groups), "Other")
        unique_groups = sorted(hue_groups.unique())

    n = len(columns)
    fig = Figure(figsize=(2.3 * n, 2.3 * n), facecolor=WHITE)
    axes = fig.subplots(n, n)

    for row_index, row_column in enumerate(columns):
        for col_index, col_column in enumerate(columns):
            panel = axes[row_index, col_index]

            if row_index == col_index:
                panel.hist(working[row_column], bins=15, color=_PRIMARY, alpha=0.85)
            elif hue:
                for group_index, group in enumerate(unique_groups):
                    mask = (hue_groups == group).to_numpy()
                    panel.scatter(
                        working[col_column].to_numpy()[mask],
                        working[row_column].to_numpy()[mask],
                        s=10, alpha=0.6, color=_cat(group_index), edgecolor="none",
                    )
            else:
                panel.scatter(
                    working[col_column], working[row_column], s=10, alpha=0.5,
                    color=_PRIMARY, edgecolor="none",
                )

            if row_index == n - 1:
                panel.set_xlabel(col_column, fontsize=8)
            else:
                panel.set_xticklabels([])
            if col_index == 0:
                panel.set_ylabel(row_column, fontsize=8)
            else:
                panel.set_yticklabels([])

    if hue:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_cat(i), label=group, markersize=6)
            for i, group in enumerate(unique_groups)
        ]
        fig.legend(handles=legend_handles, title=hue, loc="upper right", fontsize=8)

    fig.suptitle("Pairwise relationships", color=NAVY, fontweight="bold")
    return _finalize(fig, label_size=7)


def plot_bivariate(frame: pd.DataFrame, x: str, y: str, ax: Axes | None = None, **kwargs: Any) -> Figure:
    """Plot the relationship between two columns, auto-dispatching by dtype combination.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    x, y:
        Columns to relate.
    ax:
        Existing matplotlib ``Axes`` to draw on.
    **kwargs:
        Forwarded to whichever underlying function is dispatched to.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    See the dispatched function.

    Assumptions
    -----------
    - numeric + numeric -> :func:`plot_scatter` (``x``/``y`` as given).
    - numeric + non-numeric -> :func:`plot_box_by_group`, with the
      numeric column as ``numeric_column`` and the other as
      ``group_column``, regardless of which of ``x``/``y`` was numeric.
    - non-numeric + non-numeric -> :func:`plot_category_heatmap`, with
      ``column_a=x`` and ``column_b=y``.
    - This dispatch logic is purely dtype-based, the same caveat as
      :func:`plot_univariate` about coded numeric categories applies
      here too.
    """
    _validate_columns_exist(frame, [x, y])
    x_numeric = pd.api.types.is_numeric_dtype(frame[x])
    y_numeric = pd.api.types.is_numeric_dtype(frame[y])

    if x_numeric and y_numeric:
        return plot_scatter(frame, x, y, ax=ax, **kwargs)
    if x_numeric and not y_numeric:
        return plot_box_by_group(frame, x, y, ax=ax, **kwargs)
    if y_numeric and not x_numeric:
        return plot_box_by_group(frame, y, x, ax=ax, **kwargs)
    return plot_category_heatmap(frame, x, y, ax=ax, **kwargs)


# ---------------------------------------------------------------------------
# Data quality and cleaning
# ---------------------------------------------------------------------------

def plot_missingness(
    frame: pd.DataFrame,
    *,
    top_n: int | None = 20,
    percentage: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot missing-value counts by column, sorted from highest to lowest.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    top_n:
        Maximum number of columns shown (those with the most missing
        values), or ``None`` for every column that has any.
    percentage:
        If ``True`` (default) the bars show percent of rows missing;
        if ``False``, raw counts.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    ValueError
        If ``frame`` has no rows, or ``top_n`` is less than 1.

    Assumptions
    -----------
    - Only columns with at least one missing value are drawn; a frame
      with no missing values produces a figure carrying a "No missing
      values" note rather than an axis of zero-length bars or an error,
      since "nothing is missing" is a valid, useful answer for a
      data-quality chart.
    - Only ``NaN``/``None``/``NaT`` count as missing; empty strings and
      placeholder values (``"N/A"``, ``-999``) do not.
    """
    _validate_dataframe(frame)
    if frame.empty:
        raise ValueError("frame is empty.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be >= 1 or None.")

    counts = frame.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    total_with_missing = len(counts)
    if top_n is not None:
        counts = counts.head(top_n)

    height = max(3.0, 0.38 * max(len(counts), 1) + 1.6)
    fig, ax = _get_figure_and_axes(ax, (9, height))
    ax.set_title("Missingness by column")

    if counts.empty:
        _note_axes(ax, "No missing values")
        return _finalize(fig)

    values = counts / len(frame) * 100 if percentage else counts.astype(float)
    shown = pd.DataFrame(
        {"column": counts.index.astype(str), "value": values.to_numpy(), "count": counts.to_numpy()}
    ).iloc[::-1]

    bars = ax.barh(shown["column"], shown["value"], color=_PRIMARY, edgecolor="none")
    if percentage:
        labels = [f"{v:.1f}%  ({n:,})" for v, n in zip(shown["value"], shown["count"])]
        ax.set_xlabel("Missing observations (%)")
    else:
        labels = [f"{int(n):,}" for n in shown["count"]]
        ax.set_xlabel("Missing observations")
    ax.bar_label(bars, labels=labels, padding=4, fontsize=8, color=MUTED)
    ax.set_xlim(0, max(float(shown["value"].max()) * 1.3, 1.0))
    ax.grid(axis="x", alpha=0.3)
    if top_n is not None and total_with_missing > top_n:
        ax.set_ylabel(f"Top {top_n} of {total_with_missing} columns with missing values")
    return _finalize(fig)


def plot_missing_matrix(
    frame: pd.DataFrame,
    max_rows: int = 200,
    max_columns: int = 40,
    ax: Axes | None = None,
) -> Figure:
    """Plot where values are missing, as a rows-by-columns pattern.

    Unlike :func:`plot_missingness` (how much is missing per column), this
    shows *which rows* — so you can see whether missing values cluster
    together, e.g. one upstream source failing for a stretch of rows.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    max_rows:
        Maximum rows drawn. Larger frames are sampled at evenly spaced
        row positions (deterministic, keeps original order).
    max_columns:
        Maximum columns drawn, keeping those with the most missing values.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    ValueError
        If ``frame`` has no rows, or ``max_rows``/``max_columns`` is less than 1.

    Assumptions
    -----------
    - Only columns with at least one missing value are drawn; if there are
      none, the figure carries a "No missing values" note.
    - Row sampling is by position, not random, so the picture is
      reproducible; short missing runs between sampled positions can be
      missed on very tall frames. The y-axis label says when sampling
      occurred.
    """
    _validate_dataframe(frame)
    if frame.empty:
        raise ValueError("frame is empty.")
    if max_rows < 1 or max_columns < 1:
        raise ValueError("max_rows and max_columns must be >= 1.")

    counts = frame.isna().sum()
    with_missing = counts[counts > 0].sort_values(ascending=False)

    fig, ax = _get_figure_and_axes(ax, (9, 5))
    ax.set_title("Missing-value pattern")

    if with_missing.empty:
        _note_axes(ax, "No missing values")
        return _finalize(fig)

    columns = with_missing.index[:max_columns]
    data = frame[columns]
    sampled = len(data) > max_rows
    if sampled:
        positions = np.linspace(0, len(data) - 1, max_rows).astype(int)
        data = data.iloc[positions]

    matrix = data.isna().to_numpy(dtype=int)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=_BINARY, vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([str(c) for c in columns], rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel("Rows (evenly sampled)" if sampled else "Rows")
    ax.legend(
        handles=[Patch(facecolor=TEAL_LIGHT, label="Present"), Patch(facecolor=ACCENT, label="Missing")],
        loc="upper right", fontsize=8, framealpha=1,
    )
    return _finalize(fig)


def plot_correlation_matrix(
    corr: pd.DataFrame,
    annotate: bool | None = None,
    title: str = "Correlation matrix",
    ax: Axes | None = None,
) -> Figure:
    """Plot an already-computed correlation matrix as a heatmap.

    Designed for ``makoding.eda.correlation_matrix(...)`` output. To compute
    the correlations from raw data in the same call, use
    :func:`plot_correlation_heatmap`.

    Parameters
    ----------
    corr:
        A square DataFrame of correlation coefficients in ``[-1, 1]``.
    annotate:
        Write each coefficient in its cell. ``None`` (default) annotates
        automatically when there are 25 or fewer columns.
    title:
        Plot title.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``corr`` is not a DataFrame.
    ValueError
        If ``corr`` is empty or not square.

    Assumptions
    -----------
    - The colour scale is fixed to ``[-1, 1]`` and diverging around zero
      (warm = negative, teal = positive), so plots from different datasets
      are directly comparable; values are not rescaled to the data's range.
    - ``NaN`` coefficients (e.g. a constant column) are left blank, not
      drawn as zero.
    - Nothing checks that ``corr`` is symmetric or in range — it is drawn
      as given.
    """
    _validate_dataframe(corr)
    if corr.empty:
        raise ValueError("corr is empty.")
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square correlation matrix.")

    n = corr.shape[0]
    values = corr.to_numpy(dtype=float)
    if annotate is None:
        annotate = n <= _ANNOTATION_LIMIT

    size = max(5.5, 0.5 * n + 2.5)
    fig, ax = _get_figure_and_axes(ax, (size * 1.1, size))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=_DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(c) for c in corr.columns], rotation=60, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i) for i in corr.index])

    if annotate and n <= _ANNOTATION_LIMIT:
        for row in range(n):
            for col in range(n):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(
                        col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color=WHITE if abs(value) > 0.65 else INK,
                    )

    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    return _finalize(fig)


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: str = "pearson",
    annotate: bool | None = True,
    ax: Axes | None = None,
) -> Figure:
    """Compute and plot a numeric correlation heatmap.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    columns:
        Numeric columns to include. Defaults to every numeric column.
    method:
        ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    annotate:
        Write each coefficient in its cell (up to 25 columns). ``None``
        decides automatically.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or a requested column is not numeric.
    KeyError
        If a requested column does not exist.
    ValueError
        If ``method`` is unsupported, or fewer than two numeric columns are
        available.

    Assumptions
    -----------
    - Correlations use pairwise-complete observations (pandas' default), so
      different cells can rest on different numbers of rows.
    - See :func:`plot_correlation_matrix` for how colours, ``NaN`` cells and
      annotation are handled.
    """
    _validate_dataframe(frame)
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    numeric = _resolve_numeric_columns(frame, columns)
    if len(numeric) < 2:
        raise ValueError("At least two numeric columns are required for a correlation heatmap.")

    correlation = frame[numeric].corr(method=method)
    return plot_correlation_matrix(
        correlation, annotate=annotate, title=f"{method.title()} correlation", ax=ax
    )


def plot_cleaning_impact(audit: Any, ax: Axes | None = None) -> Figure:
    """Plot how much each cleaning step changed, from a ``CleaningAudit``.

    Parameters
    ----------
    audit:
        A ``makoding.cleaning.CleaningAudit`` (or any object with the same
        attributes). Read defensively: missing attributes count as zero.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Assumptions
    -----------
    - Shows counts of *changes*, not data quality: a large "values filled"
      bar means cleaning did a lot, not that the result is good.
    - "Rows dropped (missing values)" is derived as
      ``rows_before - rows_after - duplicates_removed`` and is omitted when
      not positive.
    - If cleaning changed nothing, the figure carries a "Cleaning made no
      changes" note instead of an empty chart.
    """
    dropped_rows = max(
        _int_attr(audit, "rows_before") - _int_attr(audit, "rows_after")
        - _int_attr(audit, "duplicates_removed"),
        0,
    )
    items = [
        ("Whitespace trimmed", _int_attr(audit, "whitespace_trimmed")),
        ("Infinite values replaced", _int_attr(audit, "infinite_replaced")),
        ("Duplicate rows removed", _int_attr(audit, "duplicates_removed")),
        ("Rows dropped (missing values)", dropped_rows),
        ("Missing values filled", _int_attr(audit, "values_filled")),
        ("Outliers capped", _int_attr(audit, "values_capped")),
        ("Columns dropped", len(getattr(audit, "columns_dropped", ()) or ())),
    ]

    fig, ax = _get_figure_and_axes(ax, (8, 4.2))
    ax.set_title("What cleaning changed")

    if not any(count for _, count in items):
        _note_axes(ax, "Cleaning made no changes")
        return _finalize(fig)

    labels = [name for name, _ in items][::-1]
    counts = [count for _, count in items][::-1]
    bars = ax.barh(labels, counts, color=_PRIMARY, edgecolor="none")
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=4, fontsize=8, color=MUTED)
    ax.set_xlim(0, max(max(counts) * 1.2, 1))
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.3)
    return _finalize(fig)


def plot_outlier_fences(
    frame: pd.DataFrame,
    column: str,
    lower_fence: float | None = None,
    upper_fence: float | None = None,
    multiplier: float = 1.5,
    bins: int = 30,
    ax: Axes | None = None,
) -> Figure:
    """Plot a histogram with outlier fences and the regions beyond them shaded.

    Pair with the fences a fitted ``Cleaner`` learned from training data
    (``cleaner.learned_state[column].lower_fence`` / ``.upper_fence``) to see
    exactly which values ``cap_outliers`` will clip.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    column:
        Numeric column to plot.
    lower_fence, upper_fence:
        Fence positions. If **both** are ``None``, Tukey fences are computed
        from ``column`` itself as ``Q1 - multiplier*IQR`` / ``Q3 + multiplier*IQR``.
        If only one is given, only that one is drawn.
    multiplier:
        IQR multiplier for the computed fences. Ignored when fences are given.
    bins:
        Number of histogram bins.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame or ``column`` is not numeric.
    KeyError
        If ``column`` does not exist.
    ValueError
        If ``column`` has no finite values, ``multiplier`` is not positive and
        finite, or a supplied fence is not finite.

    Assumptions
    -----------
    - Missing and infinite values are dropped before plotting (as with
      :func:`plot_numeric_distribution`).
    - Computed fences use this frame's own quartiles. That is right for
      exploring one dataset, but for judging a *validation/test* frame
      against a training-time rule, pass the training fences explicitly —
      computing them from the frame being judged would recreate exactly the
      leakage a fitted ``Cleaner`` avoids.
    - A column with IQR of zero yields fences equal to the quartiles, so
      nearly every distinct value falls outside them.
    """
    _validate_columns_exist(frame, [column])
    _validate_numeric_columns(frame, [column])

    values = pd.to_numeric(frame[column], errors="raise").astype("float64")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError(f"Column {column!r} has no finite values to plot")

    for name, fence in (("lower_fence", lower_fence), ("upper_fence", upper_fence)):
        if fence is not None and not np.isfinite(float(fence)):
            raise ValueError(f"{name} must be finite.")

    if lower_fence is None and upper_fence is None:
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("multiplier must be positive and finite.")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = float(q1 - multiplier * iqr)
        upper_fence = float(q3 + multiplier * iqr)

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.hist(values, bins=bins, color=_PRIMARY, edgecolor="white", alpha=0.85)

    if lower_fence is not None:
        below = int((values < float(lower_fence)).sum())
        ax.axvline(float(lower_fence), color=ACCENT, linestyle="--", linewidth=1.5,
                   label=f"Lower fence ({below:,} below)")
        ax.axvspan(min(float(values.min()), float(lower_fence)), float(lower_fence),
                   color=ACCENT, alpha=0.10)
    if upper_fence is not None:
        above = int((values > float(upper_fence)).sum())
        ax.axvline(float(upper_fence), color=ACCENT, linestyle="--", linewidth=1.5,
                   label=f"Upper fence ({above:,} above)")
        ax.axvspan(float(upper_fence), max(float(values.max()), float(upper_fence)),
                   color=ACCENT, alpha=0.10)

    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Outlier fences for {column}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return _finalize(fig)


def plot_before_after(
    before: pd.DataFrame,
    after: pd.DataFrame,
    column: str,
    bins: int = 30,
    labels: tuple[str, str] = ("Before", "After"),
) -> Figure:
    """Plot one column's distribution before and after a transformation, side by side.

    Useful for showing the effect of cleaning (outlier capping, filling) or a
    scaler/power transform on the same column.

    Parameters
    ----------
    before, after:
        DataFrames that both contain ``column`` (typically the raw and the
        cleaned/transformed frame).
    column:
        Numeric column to compare.
    bins:
        Histogram bins per panel.
    labels:
        Panel titles.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If either frame is not a DataFrame or ``column`` is not numeric in both.
    KeyError
        If ``column`` is missing from either frame.
    ValueError
        If either frame has no finite values in ``column``.

    Assumptions
    -----------
    - The two panels have **independent** x-axes: a scaler changes the value
      range, so a shared axis would squash one panel. Compare shape, not
      position.
    - Missing/infinite values are dropped from each panel, and each panel's
      title states its own row count so drops are visible.
    - This function owns a two-panel figure and does not accept ``ax``.
    """
    for frame in (before, after):
        _validate_columns_exist(frame, [column])
        _validate_numeric_columns(frame, [column])

    def _clean(frame: pd.DataFrame) -> pd.Series:
        series = pd.to_numeric(frame[column], errors="raise").astype("float64")
        return series.replace([np.inf, -np.inf], np.nan).dropna()

    before_values, after_values = _clean(before), _clean(after)
    if before_values.empty or after_values.empty:
        raise ValueError(f"Column {column!r} has no finite values in one of the frames")

    fig = Figure(figsize=(10, 4), facecolor=WHITE)
    left, right = fig.subplots(1, 2)

    left.hist(before_values, bins=bins, color=_cat(5), edgecolor="white", alpha=0.9)
    left.set_title(f"{labels[0]} (n={len(before_values):,})")
    left.set_xlabel(column)
    left.set_ylabel("Count")
    left.grid(axis="y", alpha=0.3)

    right.hist(after_values, bins=bins, color=_PRIMARY, edgecolor="white", alpha=0.9)
    right.set_title(f"{labels[1]} (n={len(after_values):,})")
    right.set_xlabel(column)
    right.grid(axis="y", alpha=0.3)

    return _finalize(fig)


# ---------------------------------------------------------------------------
# Statistical diagnostics
# ---------------------------------------------------------------------------

def plot_qq(values: Sequence[float], *, ax: Axes | None = None) -> Figure:
    """Plot a normal Q-Q diagnostic.

    Points close to the dashed reference line mean the sample's quantiles
    match a normal distribution with the sample's own mean and standard
    deviation. A visual companion to the normality tests.

    Parameters
    ----------
    values:
        One-dimensional numeric sample.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``values`` cannot be converted to numbers.
    ValueError
        If ``values`` is empty, contains missing/infinite values, has fewer
        than 3 observations, or is constant.

    Assumptions
    -----------
    - Missing or infinite values raise rather than being dropped, matching
      the statistical layer's rule of never silently transforming data.
    - Theoretical quantiles use plotting positions ``(i - 0.5) / n``.
    - A Q-Q plot is a judgement aid, not a test; it does not replace
      ``normality_test``.
    """
    numeric = _validate_numeric_array(values, name="values")
    if numeric.size < 3:
        raise ValueError("plot_qq requires at least 3 observations.")
    if float(np.ptp(numeric)) == 0:
        raise ValueError("plot_qq requires values that are not all identical.")

    observed = np.sort(numeric)
    probabilities = (np.arange(1, observed.size + 1) - 0.5) / observed.size
    theoretical = float(np.mean(observed)) + float(np.std(observed, ddof=1)) * stats.norm.ppf(probabilities)

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.scatter(theoretical, observed, alpha=0.7, color=_PRIMARY, edgecolor="none")
    lower = min(theoretical.min(), observed.min())
    upper = max(theoretical.max(), observed.max())
    ax.plot([lower, upper], [lower, upper], color=INK, linestyle="--", linewidth=1.1)
    ax.set_title("Normal Q-Q plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed quantiles")
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_regression_diagnostics(model: Any) -> Figure:
    """Plot the four standard OLS residual diagnostics in one figure.

    Panels: residuals vs fitted (curvature / non-linearity), a normal Q-Q of
    standardized residuals (normality), scale-location (constant variance),
    and a histogram of residuals.

    Parameters
    ----------
    model:
        A fitted regression results object exposing ``resid`` and
        ``fittedvalues`` — e.g. the ``statsmodels`` results returned by
        ``ols_regression``. Duck-typed; ``statsmodels`` is not imported here.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``model`` does not expose ``resid`` and ``fittedvalues``.
    ValueError
        If residuals/fitted values are empty, non-finite, of different lengths,
        fewer than 3, or the residuals have zero variance.

    Assumptions
    -----------
    - Residuals are standardized as ``(r - mean) / sd`` (``ddof=1``), a
      simple stand-in for studentized residuals that needs no leverage
      information.
    - These panels support judgement; for formal tests use the
      ``regression_diagnostics`` function in the statistical layer.
    - This function owns a four-panel figure and does not accept ``ax``.
    """
    if not (hasattr(model, "resid") and hasattr(model, "fittedvalues")):
        raise TypeError("model must expose 'resid' and 'fittedvalues' (e.g. a statsmodels OLS result).")

    residuals = _validate_numeric_array(model.resid, name="model.resid")
    fitted = _validate_numeric_array(model.fittedvalues, name="model.fittedvalues")
    if residuals.size != fitted.size:
        raise ValueError("model.resid and model.fittedvalues must have the same length.")
    if residuals.size < 3:
        raise ValueError("At least 3 observations are required.")

    spread = float(np.std(residuals, ddof=1))
    if spread == 0:
        raise ValueError("Residuals have zero variance; diagnostics are undefined.")
    standardized = (residuals - float(np.mean(residuals))) / spread

    fig = Figure(figsize=(10, 8), facecolor=WHITE)
    (a, b), (c, d) = fig.subplots(2, 2)

    a.scatter(fitted, residuals, alpha=0.6, color=_PRIMARY, edgecolor="none")
    a.axhline(0, color=ACCENT, linestyle="--", linewidth=1.3)
    a.set_title("Residuals vs fitted")
    a.set_xlabel("Fitted values")
    a.set_ylabel("Residuals")
    a.grid(alpha=0.3)

    ordered = np.sort(standardized)
    positions = (np.arange(1, ordered.size + 1) - 0.5) / ordered.size
    theoretical = stats.norm.ppf(positions)
    b.scatter(theoretical, ordered, alpha=0.7, color=_PRIMARY, edgecolor="none")
    limit = max(abs(theoretical).max(), abs(ordered).max())
    b.plot([-limit, limit], [-limit, limit], color=INK, linestyle="--", linewidth=1.1)
    b.set_title("Normal Q-Q (standardized residuals)")
    b.set_xlabel("Theoretical quantiles")
    b.set_ylabel("Standardized residuals")
    b.grid(alpha=0.3)

    c.scatter(fitted, np.sqrt(np.abs(standardized)), alpha=0.6, color=_PRIMARY, edgecolor="none")
    c.set_title("Scale-location")
    c.set_xlabel("Fitted values")
    c.set_ylabel("sqrt(|standardized residuals|)")
    c.grid(alpha=0.3)

    d.hist(residuals, bins=min(30, max(5, residuals.size // 5)), color=_PRIMARY,
           edgecolor="white", alpha=0.85)
    d.set_title("Residual distribution")
    d.set_xlabel("Residuals")
    d.set_ylabel("Count")
    d.grid(axis="y", alpha=0.3)

    return _finalize(fig)


def plot_time_series(
    frame: pd.DataFrame,
    time_column: str,
    value_column: str,
    *,
    rolling_window: int | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot an ordered numeric series over time, with an optional rolling mean.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    time_column:
        Column parsed as datetimes.
    value_column:
        Numeric column to plot.
    rolling_window:
        If given (>= 2), overlay a rolling mean over this many observations.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame or ``value_column`` is not numeric.
    KeyError
        If a column does not exist.
    ValueError
        If ``rolling_window`` is less than 2, or no valid observations remain.

    Assumptions
    -----------
    - ``time_column`` is parsed with ``errors="coerce"``: rows whose time
      cannot be parsed, or whose value is missing, are dropped silently.
      Check the parse yourself if a badly-formatted date column is possible.
    - Rows are sorted by time before plotting, so out-of-order input is
      drawn chronologically. Irregular spacing is not resampled: gaps in
      time are drawn as straight lines between observed points.
    - The rolling mean counts observations, not calendar time.
    """
    _validate_dataframe(frame)
    _validate_columns_exist(frame, [time_column, value_column])
    if not pd.api.types.is_numeric_dtype(frame[value_column]):
        raise TypeError(f"Column {value_column!r} must be numeric.")
    if rolling_window is not None and rolling_window < 2:
        raise ValueError("rolling_window must be >= 2 or None.")

    working = frame[[time_column, value_column]].copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.dropna().sort_values(time_column)
    if working.empty:
        raise ValueError("No valid observations remain after parsing the time column.")

    fig, ax = _get_figure_and_axes(ax, (9, 5))
    ax.plot(working[time_column], working[value_column], color=_PRIMARY, linewidth=1.8, label=value_column)
    if rolling_window is not None:
        rolling = working[value_column].rolling(rolling_window, min_periods=rolling_window).mean()
        ax.plot(working[time_column], rolling, color=ACCENT, linestyle="--", linewidth=1.4,
                label=f"{rolling_window}-period rolling mean")
        ax.legend(frameon=False)
    ax.set_title(f"{value_column} over time")
    ax.set_xlabel(time_column)
    ax.set_ylabel(value_column)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _finalize(fig)


def plot_forecast(
    history: Sequence[float],
    forecast: pd.DataFrame,
    history_tail: int | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a series' history followed by its forecast and prediction interval.

    Designed for the DataFrame returned by ``forecast_arima`` (columns
    ``forecast``, ``lower``, ``upper``).

    Parameters
    ----------
    history:
        The observed series the model was fit on, in chronological order.
    forecast:
        A DataFrame with a ``forecast`` column and, optionally, ``lower`` and
        ``upper`` columns, one row per future step.
    history_tail:
        If given, draw only the last ``history_tail`` observations, so a long
        history does not flatten the forecast.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``forecast`` is not a DataFrame.
    KeyError
        If ``forecast`` has no ``forecast`` column.
    ValueError
        If ``history`` is empty or non-finite, ``forecast`` has no rows or
        non-finite values, or ``history_tail`` is less than 1.

    Assumptions
    -----------
    - The x-axis is the step index (history ``0..n-1``, forecast ``n..n+h-1``),
      not calendar time; ``forecast_arima`` does not carry dates.
    - The shaded band is labelled "Prediction interval" without a percentage,
      because its level is set by the model that produced it (95% by
      statsmodels' default) and is not recorded in the frame.
    - Missing or infinite values raise rather than being dropped, since
      dropping one would silently shift every later point along the x-axis.
    """
    values = _validate_numeric_array(history, name="history")
    if not isinstance(forecast, pd.DataFrame):
        raise TypeError("forecast must be a pandas DataFrame.")
    _validate_columns_exist(forecast, ["forecast"])
    if forecast.empty:
        raise ValueError("forecast has no rows.")
    if history_tail is not None and history_tail < 1:
        raise ValueError("history_tail must be >= 1 or None.")

    predicted = _validate_numeric_array(forecast["forecast"], name="forecast['forecast']")
    has_band = {"lower", "upper"}.issubset(forecast.columns)
    if has_band:
        lower = _validate_numeric_array(forecast["lower"], name="forecast['lower']")
        upper = _validate_numeric_array(forecast["upper"], name="forecast['upper']")

    n, horizon = values.size, predicted.size
    x_history = np.arange(n)
    x_forecast = np.arange(n, n + horizon)

    if history_tail is not None and history_tail < n:
        x_history = x_history[-history_tail:]
        values = values[-history_tail:]

    fig, ax = _get_figure_and_axes(ax, (9, 5))
    ax.plot(x_history, values, color=_PRIMARY, linewidth=1.8, label="Observed")
    ax.plot(
        np.r_[x_history[-1], x_forecast], np.r_[values[-1], predicted],
        color=ACCENT, linewidth=1.8, linestyle="--", label="Forecast",
    )
    if has_band:
        ax.fill_between(x_forecast, lower, upper, color=ACCENT, alpha=0.18, label="Prediction interval")
    ax.axvline(n - 0.5, color=BORDER, linestyle=":", linewidth=1.2)
    ax.set_title("Forecast")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    return _finalize(fig)


# ---------------------------------------------------------------------------
# Unsupervised diagnostics
# ---------------------------------------------------------------------------

def plot_pca_scatter(
    transformed: pd.DataFrame,
    x_component: str = "PC1",
    y_component: str = "PC2",
    hue: Sequence[Any] | None = None,
    explained_variance_ratio: Sequence[float] | None = None,
    max_hue_categories: int = 10,
    ax: Axes | None = None,
) -> Figure:
    """Plot a scatter of two PCA components, optionally colored by group.

    Designed to plot ``makoding.unsupervised.apply_pca(...).transformed``
    directly.

    Parameters
    ----------
    transformed:
        A DataFrame of PCA component columns (e.g. ``PCAResult.transformed``).
    x_component, y_component:
        Which component columns to plot on each axis (e.g. ``"PC1"``,
        ``"PC2"``).
    hue:
        Optional array-like, the same length as ``transformed``, of
        group labels to color points by (e.g. cluster assignments, or
        a target column from the original data). Always treated as
        discrete categories, regardless of dtype — unlike
        :func:`plot_scatter`, there is no continuous-colormap mode
        here.
    explained_variance_ratio:
        Optional array of explained-variance-ratio values (e.g.
        ``PCAResult.explained_variance_ratio``), used only to label
        the axes with the percentage of variance each shown component
        explains. Looked up by ``transformed.columns`` position, so
        it must be in the same component order ``apply_pca`` produced
        it in.
    max_hue_categories:
        For a ``hue`` with more distinct values than this, the rest
        are grouped into ``"Other"`` in the legend.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``transformed`` is not a DataFrame.
    KeyError
        If ``x_component``/``y_component`` does not exist in
        ``transformed``.
    ValueError
        If no rows have both components non-missing, or ``hue`` is
        given with a different length than ``transformed``.

    Assumptions
    -----------
    A row missing either plotted component is dropped; if ``hue`` is
    given, its values are matched to ``transformed`` by **position**
    (not by index label) after that filtering, so ``hue`` must be
    supplied in the same row order as ``transformed`` was originally
    built in.
    """
    _validate_columns_exist(transformed, [x_component, y_component])

    working = transformed[[x_component, y_component]].reset_index(drop=True).dropna()
    if working.empty:
        raise ValueError(
            f"No rows have both {x_component!r} and {y_component!r} non-missing"
        )

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)

    if hue is None:
        ax.scatter(working[x_component], working[y_component], alpha=0.6, color=_PRIMARY, edgecolor="none")
    else:
        if len(hue) != len(transformed):
            raise ValueError(
                f"hue must have one entry per row of transformed ({len(transformed)}), got {len(hue)}"
            )
        hue_aligned = pd.Series(np.asarray(hue)).reset_index(drop=True).loc[working.index]
        _plot_scatter_by_group(
            ax, working[x_component], working[y_component], hue_aligned,
            max_categories=max_hue_categories, legend_title="Group",
        )

    x_label, y_label = x_component, y_component
    if explained_variance_ratio is not None:
        component_order = transformed.columns.tolist()
        try:
            x_index = component_order.index(x_component)
            y_index = component_order.index(y_component)
            x_label = f"{x_component} ({explained_variance_ratio[x_index] * 100:.1f}% variance)"
            y_label = f"{y_component} ({explained_variance_ratio[y_index] * 100:.1f}% variance)"
        except (ValueError, IndexError):
            pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("PCA components")
    ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.5)
    ax.axvline(0, color=MUTED, linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_cluster_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    labels: Sequence[int],
    max_clusters_shown: int = 15,
    ax: Axes | None = None,
) -> Figure:
    """Plot two features colored by cluster assignment.

    Designed to plot the output of, e.g.,
    ``makoding.unsupervised.fit_kmeans(...).labels`` against two of
    the original (or standardized) feature columns.

    Parameters
    ----------
    frame:
        Input pandas DataFrame (typically the same features clustering
        was fit on).
    x, y:
        Numeric columns for the horizontal and vertical axes.
    labels:
        Cluster assignment per row of ``frame``, same length and row
        order. A label of ``-1`` (scikit-learn's convention for
        DBSCAN noise points) is shown as its own ``"Noise"`` category
        rather than a numbered cluster.
    max_clusters_shown:
        Maximum number of distinct clusters given their own color;
        the rest are grouped into ``"Other"`` in the legend (see
        Assumptions — this is rarely relevant in practice).
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or ``x``/``y`` is not numeric.
    KeyError
        If ``x``/``y`` does not exist.
    ValueError
        If ``labels`` has a different length than ``frame``, or no
        rows have both ``x`` and ``y`` non-missing.

    Assumptions
    -----------
    - ``max_clusters_shown`` rarely triggers in practice, since
      clustering runs typically produce far fewer than 15 clusters —
      it exists mainly as a safety net against an unexpectedly large
      ``n_clusters``, not a feature you're expected to tune routinely.
    - A row missing ``x`` or ``y`` is dropped, along with its cluster
      label.
    """
    _validate_columns_exist(frame, [x, y])
    _validate_numeric_columns(frame, [x, y])
    if len(labels) != len(frame):
        raise ValueError(
            f"labels must have one entry per row of frame ({len(frame)}), got {len(labels)}"
        )

    working = frame[[x, y]].reset_index(drop=True).copy()
    working["__cluster__"] = np.asarray(labels)
    working = working.dropna(subset=[x, y])
    if working.empty:
        raise ValueError(f"No rows have both {x!r} and {y!r} non-missing")

    display_labels = working["__cluster__"].map(
        lambda value: "Noise" if value == -1 else f"Cluster {value}"
    )

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    _plot_scatter_by_group(
        ax, working[x], working[y], display_labels,
        max_categories=max_clusters_shown, legend_title="Cluster",
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Cluster assignments")
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_elbow(
    elbow_frame: pd.DataFrame,
    k_column: str = "n_clusters",
    score_column: str = "inertia",
    ax: Axes | None = None,
) -> Figure:
    """Plot a clustering score (e.g. inertia) across candidate values of k.

    Designed to plot the output of
    ``makoding.unsupervised.kmeans_elbow`` directly.

    Parameters
    ----------
    elbow_frame:
        A DataFrame with one row per candidate ``k`` and a score
        column (default column names match ``kmeans_elbow``'s output).
    k_column:
        Column holding the number-of-clusters value.
    score_column:
        Column holding the score to plot (e.g. ``"inertia"``).
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``elbow_frame`` is not a DataFrame.
    KeyError
        If ``k_column`` or ``score_column`` does not exist.
    ValueError
        If ``elbow_frame`` is empty.

    Assumptions
    -----------
    This function draws the line and points only — it does not
    algorithmically detect or mark the "elbow" itself; identifying
    the bend is left to visual inspection, the traditional (if
    informal) way this plot is used.
    """
    _validate_columns_exist(elbow_frame, [k_column, score_column])
    if elbow_frame.empty:
        raise ValueError("elbow_frame is empty")

    ordered = elbow_frame.sort_values(k_column)

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.plot(ordered[k_column], ordered[score_column], marker="o", color=_PRIMARY)
    ax.set_xlabel(k_column.replace("_", " ").title())
    ax.set_ylabel(score_column.replace("_", " ").title())
    ax.set_title("Elbow plot")
    ax.set_xticks(ordered[k_column])
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_silhouette_by_k(
    k_values: Sequence[int],
    scores: Sequence[float],
    ax: Axes | None = None,
) -> Figure:
    """Plot silhouette score across candidate values of k.

    Unlike :func:`plot_elbow`, there is no single ``makoding.unsupervised``
    function that produces this directly — compute each score yourself
    by looping ``fit_kmeans`` (or another clustering function) and
    ``silhouette_score`` over a range of ``k``, then pass the two
    parallel sequences here::

        k_values, scores = [], []
        for k in range(2, 11):
            result = unsupervised.fit_kmeans(data, n_clusters=k)
            k_values.append(k)
            scores.append(unsupervised.silhouette_score(data, result.labels))
        plot_silhouette_by_k(k_values, scores)

    Parameters
    ----------
    k_values:
        Candidate numbers of clusters tried.
    scores:
        Silhouette score for each entry in ``k_values``, same order.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``k_values`` is empty, or ``k_values``/``scores`` have
        different lengths.

    Assumptions
    -----------
    Points are plotted in the order given in ``k_values``, not sorted
    — pass them pre-sorted if you want the line drawn left-to-right by
    increasing k (the typical presentation).
    """
    if len(k_values) == 0:
        raise ValueError("k_values must not be empty")
    if len(k_values) != len(scores):
        raise ValueError(
            f"k_values and scores must have the same length, got {len(k_values)} and {len(scores)}"
        )

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.plot(list(k_values), list(scores), marker="o", color=_PRIMARY)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score by k")
    ax.set_xticks(list(k_values))
    ax.grid(alpha=0.3)
    return _finalize(fig)


# ---------------------------------------------------------------------------
# Supervised model diagnostics
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    normalize: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a confusion matrix for classification predictions.

    Backed by scikit-learn's ``ConfusionMatrixDisplay``.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted class labels (e.g. from
        ``makoding.modeling.Model.predict``). Any number of classes.
    labels:
        Explicit class ordering for the axes. Defaults to the sorted
        union of classes seen in ``y_true``/``y_pred``.
    normalize:
        ``None`` (default, raw counts), ``"true"`` (row-normalize),
        ``"pred"`` (column-normalize), or ``"all"`` — scikit-learn's
        own normalization options.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``y_true``/``y_pred`` are empty, of mismatched length, or
        ``normalize`` is invalid (surfaced from scikit-learn).

    Assumptions
    -----------
    Works for any number of classes, not just binary — for a
    high-cardinality target the matrix can become hard to read; that
    is a property of confusion matrices in general, not something
    this function mitigates.
    """
    _validate_array_pair(y_true, y_pred, ("y_true", "y_pred"))

    fig, ax = _get_figure_and_axes(ax, (6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize=normalize, ax=ax,
        colorbar=True, cmap=_SEQUENTIAL,
    )
    ax.set_title("Confusion matrix")
    return _finalize(fig)


def plot_roc_curve(
    y_true: Sequence[Any],
    y_score: Sequence[float],
    pos_label: Any = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot an ROC curve for binary classification.

    Backed by scikit-learn's ``RocCurveDisplay``.

    Parameters
    ----------
    y_true:
        True binary class labels.
    y_score:
        Predicted probability (or decision score) of the positive
        class — e.g. ``model.predict_proba(X)[classes_[1]]``.
    pos_label:
        Which class in ``y_true`` is the positive class. Defaults to
        scikit-learn's own inference (the greater of the two labels
        when they're ``{0, 1}``-like; required explicitly otherwise).
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``y_true`` has other than 2 distinct classes, is empty, or
        of mismatched length with ``y_score``.

    Assumptions
    -----------
    This is a binary-only plot, matching the same restriction as
    ``makoding.modeling.Model.evaluate``'s ``"roc_auc"`` metric — a
    multi-class target raises rather than silently picking a
    one-vs-rest averaging strategy.
    """
    _validate_array_pair(y_true, y_score, ("y_true", "y_score"))

    unique_classes = pd.unique(pd.Series(y_true).dropna())
    if len(unique_classes) != 2:
        raise ValueError(
            f"plot_roc_curve requires a binary target (2 classes); got {len(unique_classes)}"
        )

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    RocCurveDisplay.from_predictions(y_true, y_score, pos_label=pos_label, ax=ax)
    # Recolour after the fact rather than passing colour kwargs: scikit-learn's
    # display API for line styling has changed between releases.
    if ax.lines:
        ax.lines[0].set_color(_PRIMARY)
        ax.lines[0].set_linewidth(2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Chance")
    # Override scikit-learn's auto-generated labels (which append
    # "(Positive label: X)" whenever pos_label was inferred rather than
    # given explicitly) with the plain, stable label text.
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_precision_recall_curve(
    y_true: Sequence[Any],
    y_score: Sequence[float],
    pos_label: Any = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a precision-recall curve for binary classification.

    Backed by scikit-learn's ``PrecisionRecallDisplay``.

    Parameters
    ----------
    y_true, y_score, pos_label:
        See :func:`plot_roc_curve` — identical meaning here.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``y_true`` has other than 2 distinct classes, is empty, or
        of mismatched length with ``y_score``.

    Assumptions
    -----------
    - Binary-only, same as :func:`plot_roc_curve`.
    - Precision-recall curves are generally more informative than ROC
      curves for imbalanced classes (where the negative class heavily
      outnumbers the positive) — this function does not choose one
      over the other for you; use both where class balance is a
      concern.
    """
    _validate_array_pair(y_true, y_score, ("y_true", "y_score"))

    unique_classes = pd.unique(pd.Series(y_true).dropna())
    if len(unique_classes) != 2:
        raise ValueError(
            f"plot_precision_recall_curve requires a binary target (2 classes); "
            f"got {len(unique_classes)}"
        )

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    PrecisionRecallDisplay.from_predictions(y_true, y_score, pos_label=pos_label, ax=ax)
    if ax.lines:
        ax.lines[0].set_color(_PRIMARY)
        ax.lines[0].set_linewidth(2)
    ax.set_title("Precision-recall curve")
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_calibration_curve(
    y_true: Sequence[Any],
    y_probability: Sequence[float],
    *,
    n_bins: int = 10,
    pos_label: Any = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot predicted probability against observed event frequency.

    A well-calibrated classifier's points lie on the diagonal: among cases
    given a predicted probability of about 0.7, roughly 70% are positive.

    Parameters
    ----------
    y_true:
        True binary class labels.
    y_probability:
        Predicted probability of the positive class, each in ``[0, 1]``.
    n_bins:
        Number of equal-width probability bins (at least 2).
    pos_label:
        Which class is "positive". Defaults to the greater of the two labels
        (scikit-learn's convention for ``{0, 1}``-like labels); required
        explicitly if the labels cannot be ordered.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the inputs are empty or of mismatched length, ``n_bins`` is less
        than 2, probabilities are non-finite or outside ``[0, 1]``,
        ``y_true`` does not have exactly two classes, or ``pos_label`` is not
        one of them.

    Assumptions
    -----------
    - Rows whose ``y_true`` is missing are dropped from both inputs.
    - The positive class is chosen by **label value**, not by order of
      appearance. An earlier version used whichever label happened to occur
      last in the data, so the curve could flip between mirror images
      depending on row order.
    - Bins with no predictions are skipped, so the line can have fewer than
      ``n_bins`` points; sparsely populated bins give noisy points.
    """
    _validate_array_pair(y_true, y_probability, ("y_true", "y_probability"))
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    probability = _validate_numeric_array(y_probability, name="y_probability")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("y_probability values must be between 0 and 1.")

    target = pd.Series(np.asarray(y_true, dtype=object))
    valid = target.notna().to_numpy()
    target, probability = target[valid], probability[valid]
    if target.empty:
        raise ValueError("y_true has no non-missing values.")

    classes = list(target.unique())
    if len(classes) != 2:
        raise ValueError(f"Calibration curves require exactly two target classes; got {len(classes)}.")
    if pos_label is None:
        try:
            positive = sorted(classes)[-1]
        except TypeError as exc:
            raise ValueError("Class labels cannot be ordered; pass pos_label explicitly.") from exc
    elif pos_label in classes:
        positive = pos_label
    else:
        raise ValueError(f"pos_label {pos_label!r} is not one of the classes {classes!r}.")

    observed = (target == positive).to_numpy(dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers: list[float] = []
    fractions: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (probability >= lower) & ((probability <= upper) if upper == 1 else (probability < upper))
        if np.any(in_bin):
            centers.append(float(probability[in_bin].mean()))
            fractions.append(float(observed[in_bin].mean()))

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    ax.plot(centers, fractions, marker="o", color=_PRIMARY, linewidth=1.8, label="Model")
    ax.set_title("Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return _finalize(fig)


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    ax: Axes | None = None,
) -> Figure:
    """Plot regression residuals (``y_true - y_pred``) against predicted values.

    A well-fitting model shows residuals scattered randomly around
    zero with no systematic pattern; a curve, funnel shape, or trend
    in this plot signals model misspecification or non-constant
    error variance.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted numeric target values.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the inputs are empty or of mismatched length.

    Assumptions
    -----------
    Both arrays are cast to ``float`` via ``numpy.asarray`` — a
    non-numeric value in either raises a numpy/pandas casting error,
    not a message from this function specifically.
    """
    _validate_array_pair(y_true, y_pred, ("y_true", "y_pred"))

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residuals = y_true_array - y_pred_array

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.scatter(y_pred_array, residuals, alpha=0.6, color=_PRIMARY, edgecolor="none")
    ax.axhline(0, color=ACCENT, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs predicted")
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_prediction_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot residual error against predicted values, with strict input checks.

    The same picture as :func:`plot_residuals`, but inputs that contain
    missing or non-finite values raise instead of propagating ``NaN`` into
    the plot — the stricter variant, for pipelines that should fail loudly.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted numeric target values.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If either input contains non-numeric values.
    ValueError
        If the inputs are empty, of mismatched length, or non-finite.

    Assumptions
    -----------
    Residuals are ``y_true - y_pred``, so points above zero are
    under-predictions.
    """
    _validate_array_pair(y_true, y_pred, ("y_true", "y_pred"))
    actual = _validate_numeric_array(y_true, name="y_true")
    predicted = _validate_numeric_array(y_pred, name="y_pred")
    residuals = actual - predicted

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.scatter(predicted, residuals, alpha=0.65, color=_PRIMARY, edgecolor="none")
    ax.axhline(0, color=INK, linestyle="--", linewidth=1.1)
    ax.set_title("Prediction error")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_predicted_vs_actual(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    ax: Axes | None = None,
) -> Figure:
    """Plot predicted values against actual values, with a ``y = x`` reference line.

    Points on the reference line are perfect predictions; the tighter
    the scatter hugs the line, the better the fit.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted numeric target values.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the inputs are empty or of mismatched length.

    Assumptions
    -----------
    The reference line spans the combined observed range of
    ``y_true`` and ``y_pred`` — it is a visual aid, not a fitted
    regression line through the points.
    """
    _validate_array_pair(y_true, y_pred, ("y_true", "y_pred"))

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    ax.scatter(y_true_array, y_pred_array, alpha=0.6, color=_PRIMARY, edgecolor="none")

    combined_min = min(y_true_array.min(), y_pred_array.min())
    combined_max = max(y_true_array.max(), y_pred_array.max())
    ax.plot(
        [combined_min, combined_max], [combined_min, combined_max],
        color=ACCENT, linestyle="--", linewidth=1.5, label="Perfect prediction",
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return _finalize(fig)


def plot_feature_importance(
    importance_frame: pd.DataFrame,
    feature_column: str = "feature",
    importance_column: str = "importance",
    top_n: int = 15,
    ax: Axes | None = None,
) -> Figure:
    """Plot a horizontal bar chart of feature importances.

    Designed to plot the output of
    ``makoding.modeling.Model.feature_importance()`` or
    ``makoding.modeling.get_feature_importance()`` directly.

    Parameters
    ----------
    importance_frame:
        A DataFrame with a feature-name column and a numeric
        importance column — either as regular columns (matching
        ``get_feature_importance``'s output) or with the feature name
        as the index (matching ``Model.feature_importance()``'s
        output; pass ``feature_column="index"`` in that case — see
        Assumptions).
    feature_column:
        Name of the column holding feature names, or the literal
        string ``"index"`` to use ``importance_frame.index`` instead.
    importance_column:
        Name of the column holding numeric importance values.
    top_n:
        Maximum number of features to show, kept by descending
        importance.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``importance_frame`` is not a DataFrame.
    KeyError
        If ``feature_column`` (when not ``"index"``) or
        ``importance_column`` does not exist.
    ValueError
        If ``importance_frame`` is empty.

    Assumptions
    -----------
    - Bars are drawn with the **largest importance at the top** —
      standard convention for a ranked horizontal bar chart, achieved
      by reversing plotting order (matplotlib draws horizontal bars
      bottom-to-top by default).
    - If more than one row shares the same feature name, all are
      plotted as separate bars with the same label — this function
      does not deduplicate or aggregate.
    """
    _validate_dataframe(importance_frame)
    if importance_frame.empty:
        raise ValueError("importance_frame is empty")

    if feature_column == "index":
        features = importance_frame.index.astype(str)
    else:
        _validate_columns_exist(importance_frame, [feature_column])
        features = importance_frame[feature_column].astype(str)

    _validate_columns_exist(importance_frame, [importance_column])

    working = pd.DataFrame(
        {"feature": features.to_numpy(), "importance": importance_frame[importance_column].to_numpy()}
    )
    working = working.sort_values("importance", ascending=False).head(top_n)
    working = working.iloc[::-1]  # reverse so largest ends up at the top when plotted

    fig, ax = _get_figure_and_axes(ax, (8, max(4, 0.4 * len(working))))
    ax.barh(working["feature"], working["importance"], color=_PRIMARY, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature importance")
    ax.grid(axis="x", alpha=0.3)
    return _finalize(fig)


def plot_cross_validation_scores(
    cv_results: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a box plot of cross-validation fold scores, one box per metric.

    Designed to plot the output of
    ``makoding.modeling.Model.cross_validate()`` directly.

    Parameters
    ----------
    cv_results:
        A DataFrame with one row per fold and one column per scoring
        metric (scikit-learn's convention: metric columns are named
        ``f"test_{scorer_name}"``, e.g. ``"test_accuracy"``).
    metrics:
        Explicit list of column names to plot. Defaults to every
        column whose name starts with ``"test_"`` — matching
        ``cross_validate``'s own naming, and excluding its
        ``fit_time``/``score_time`` columns automatically.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``cv_results`` is not a DataFrame.
    KeyError
        If an explicitly requested metric column does not exist.
    ValueError
        If ``cv_results`` is empty, or no ``"test_*"`` columns are
        found and ``metrics`` was not given explicitly.

    Assumptions
    -----------
    - Individual fold scores are overlaid as jittered points on top of
      each box, so you can see the actual per-fold values, not just
      the box's summary statistics — useful with a small number of
      folds (e.g. ``cv=5``), where a box plot alone can be misleading.
    - Metric column labels are cleaned for display (the ``"test_"``
      prefix stripped, underscores turned to spaces) but the
      underlying data is plotted exactly as given, with no unit
      conversion — a mix of accuracy-like metrics (0 to 1) and
      unbounded error metrics (e.g. negative MSE) on the same plot
      will have wildly different scales; consider plotting them
      separately in that case.
    """
    _validate_dataframe(cv_results)
    if cv_results.empty:
        raise ValueError("cv_results is empty")

    if metrics is None:
        score_columns = [c for c in cv_results.columns if c.startswith("test_")]
        if not score_columns:
            raise ValueError(
                "No 'test_*' columns found in cv_results; pass metrics explicitly"
            )
    else:
        _validate_columns_exist(cv_results, list(metrics))
        score_columns = list(metrics)

    data = [cv_results[column].to_numpy(dtype=float) for column in score_columns]
    labels = [column.replace("test_", "").replace("_", " ") for column in score_columns]

    fig, ax = _get_figure_and_axes(ax, _DEFAULT_FIGSIZE)
    _boxplot(ax, data, labels)

    jitter_rng = np.random.RandomState(0)
    for position, values in enumerate(data, start=1):
        jitter = jitter_rng.normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter, values,
            alpha=0.7, color=TEAL_DARK, s=15, zorder=3, edgecolor="none",
        )

    ax.set_ylabel("Score")
    ax.set_title("Cross-validation scores by fold")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    return _finalize(fig)


def plot_model_comparison(
    comparison: pd.DataFrame,
    *,
    model_column: str = "model",
    metric_column: str = "score",
    ascending: bool = False,
    top_n: int | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot ranked model-comparison scores as a horizontal bar chart.

    Parameters
    ----------
    comparison:
        A DataFrame with one row per model.
    model_column:
        Column holding model names.
    metric_column:
        Numeric column holding the score to rank by.
    ascending:
        ``False`` (default) ranks higher scores first, for metrics where
        bigger is better (accuracy, R²). Use ``True`` for error metrics
        (RMSE, log-loss) where smaller is better.
    top_n:
        Show only the best ``top_n`` models after ranking.
    ax:
        Existing matplotlib ``Axes`` to draw on.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        If ``comparison`` is not a DataFrame or ``metric_column`` is not numeric.
    KeyError
        If a named column does not exist.
    ValueError
        If ``comparison`` is empty or ``top_n`` is less than 1.

    Assumptions
    -----------
    - The best model is drawn at the **top**. Which direction is "best" is
      the caller's choice via ``ascending``; nothing here knows what the
      metric means.
    - Scores are single numbers, so the chart carries no uncertainty. Models
      separated by less than the fold-to-fold variation (see
      :func:`plot_cross_validation_scores`) are not meaningfully ranked.
    """
    _validate_dataframe(comparison)
    _validate_columns_exist(comparison, [model_column, metric_column])
    if comparison.empty:
        raise ValueError("comparison is empty.")
    if not pd.api.types.is_numeric_dtype(comparison[metric_column]):
        raise TypeError(f"Column {metric_column!r} must be numeric.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be >= 1 or None.")

    working = comparison[[model_column, metric_column]].sort_values(metric_column, ascending=ascending)
    if top_n is not None:
        working = working.head(top_n)
    working = working.iloc[::-1]

    fig, ax = _get_figure_and_axes(ax, (8, max(4.5, 0.42 * len(working))))
    bars = ax.barh(working[model_column].astype(str), working[metric_column], color=_PRIMARY, edgecolor="none")
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in working[metric_column]], padding=4, fontsize=8, color=MUTED)
    ax.set_title("Model comparison")
    ax.set_xlabel(metric_column.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.3)
    return _finalize(fig)