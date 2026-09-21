"""Visualization utilities for Makoding.

Matplotlib-based chart generation for univariate analysis, bivariate
analysis, unsupervised-learning diagnostics, and supervised-model
diagnostics. Every function returns a ``matplotlib.figure.Figure`` —
none of them call ``plt.show()`` — so callers decide how to display
it: ``st.pyplot(fig)`` in Streamlit, ``fig.savefig(...)`` to a file,
or direct inspection of ``fig.axes`` in a test.

Four families of plots
------------------------
- **Univariate**: :func:`plot_histogram` / :func:`plot_numeric_distribution`,
  :func:`plot_categorical_counts` (with an optional ``hue`` for
  grouped/stacked counts), :func:`plot_violin`, and the dispatcher
  :func:`plot_univariate`.
- **Bivariate**: :func:`plot_scatter`, :func:`plot_box_by_group`,
  :func:`plot_category_heatmap`, :func:`plot_pairplot`, and the
  dispatcher :func:`plot_bivariate`.
- **Unsupervised diagnostics**: :func:`plot_pca_scatter`,
  :func:`plot_cluster_scatter`, :func:`plot_elbow`,
  :func:`plot_silhouette_by_k`. These take the DataFrames/arrays
  produced by :mod:`makoding.unsupervised` (``PCAResult.transformed``,
  ``ClusteringResult.labels``, ``kmeans_elbow``'s output) rather than
  a raw frame and column name.
- **Supervised model diagnostics**: :func:`plot_confusion_matrix`,
  :func:`plot_roc_curve`, :func:`plot_precision_recall_curve`,
  :func:`plot_residuals`, :func:`plot_predicted_vs_actual`,
  :func:`plot_feature_importance`, :func:`plot_cross_validation_scores`.
  These take ``y_true``/``y_pred``/``y_score`` arrays or a
  ``Model.cross_validate()``-shaped DataFrame, not a frame and column
  name.

Design principles
------------------
- Most functions accept an optional ``ax`` (a matplotlib ``Axes``) so
  plots can be composed into a grid; when omitted, a new figure is
  created. Either way, the function returns the owning ``Figure``.
  :func:`plot_pairplot` is the one exception — it always owns a
  multi-panel grid, so it does not accept ``ax``.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from .feature_engineering import (
    _validate_columns_exist,
    _validate_dataframe,
    _validate_numeric_columns,
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
    # Unsupervised diagnostics
    "plot_pca_scatter",
    "plot_cluster_scatter",
    "plot_elbow",
    "plot_silhouette_by_k",
    # Supervised model diagnostics
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_residuals",
    "plot_predicted_vs_actual",
    "plot_feature_importance",
    "plot_cross_validation_scores",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_DEFAULT_FIGSIZE = (8, 5)
_TAB10 = plt.get_cmap("tab10")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_figure_and_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Return ``(figure, axes)``, creating a new figure only if ``ax`` is ``None``."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


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
            alpha=0.6, edgecolor="none", color=_TAB10(index % 10), label=group,
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
    ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
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
        kde_ax.plot(x_grid, density, color="#C44E52", linewidth=2)
        kde_ax.set_ylabel("Density")
        kde_ax.set_yticks([])

    fig.tight_layout()
    return fig


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
        ax.bar(labels, counts.values, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"Counts of {column}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        return fig

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
            ax.bar(x_positions, values, bottom=bottom, color=_TAB10(index % 10), label=str(hue_value))
            bottom = bottom + values
    else:
        width = 0.8 / max(len(hue_categories), 1)
        for index, hue_value in enumerate(hue_categories):
            offset = (index - (len(hue_categories) - 1) / 2) * width
            ax.bar(
                x_positions + offset, crosstab[hue_value].to_numpy(), width=width,
                color=_TAB10(index % 10), label=str(hue_value),
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(row_labels, rotation=45, ha="right")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Counts of {column} by {hue}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=hue, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


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
        ax.violinplot([values.to_numpy()], showmeans=True, showmedians=True)
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
        ax.violinplot(data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(numeric_column)
        ax.set_xlabel(group_column)
        ax.set_title(f"{numeric_column} by {group_column}")

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


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
        ax.scatter(working[x], working[y], color="#4C72B0", alpha=0.6, edgecolor="none")
    elif pd.api.types.is_numeric_dtype(working[hue]):
        working = working.dropna(subset=[hue])
        scatter = ax.scatter(
            working[x], working[y], c=working[hue], cmap="viridis", alpha=0.7, edgecolor="none"
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
    fig.tight_layout()
    return fig


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
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel(group_column)
    ax.set_ylabel(numeric_column)
    ax.set_title(f"{numeric_column} by {group_column}")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    return fig


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
    image = ax.imshow(crosstab.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels([str(c) for c in crosstab.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels([str(i) for i in crosstab.index])
    ax.set_xlabel(column_b)
    ax.set_ylabel(column_a)
    ax.set_title(f"{column_a} vs {column_b}")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    return fig


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
      every other plotting function in this module.
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
    fig, axes = plt.subplots(n, n, figsize=(2.3 * n, 2.3 * n))

    for row_index, row_column in enumerate(columns):
        for col_index, col_column in enumerate(columns):
            panel = axes[row_index, col_index]

            if row_index == col_index:
                panel.hist(working[row_column], bins=15, color="#4C72B0", alpha=0.85)
            elif hue:
                for group_index, group in enumerate(unique_groups):
                    mask = (hue_groups == group).to_numpy()
                    panel.scatter(
                        working[col_column].to_numpy()[mask],
                        working[row_column].to_numpy()[mask],
                        s=10, alpha=0.6, color=_TAB10(group_index % 10), edgecolor="none",
                    )
            else:
                panel.scatter(
                    working[col_column], working[row_column], s=10, alpha=0.5,
                    color="#4C72B0", edgecolor="none",
                )

            if row_index == n - 1:
                panel.set_xlabel(col_column, fontsize=8)
            else:
                panel.set_xticklabels([])
            if col_index == 0:
                panel.set_ylabel(row_column, fontsize=8)
            else:
                panel.set_yticklabels([])
            panel.tick_params(labelsize=7)

    if hue:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_TAB10(i % 10), label=group, markersize=6)
            for i, group in enumerate(unique_groups)
        ]
        fig.legend(handles=legend_handles, title=hue, loc="upper right", fontsize=8)

    fig.suptitle("Pairwise relationships")
    fig.tight_layout()
    return fig


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
        ax.scatter(working[x_component], working[y_component], alpha=0.6, color="#4C72B0", edgecolor="none")
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
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
    fig.tight_layout()
    return fig


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
    ax.plot(ordered[k_column], ordered[score_column], marker="o", color="#4C72B0")
    ax.set_xlabel(k_column.replace("_", " ").title())
    ax.set_ylabel(score_column.replace("_", " ").title())
    ax.set_title("Elbow plot")
    ax.set_xticks(ordered[k_column])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
    ax.plot(list(k_values), list(scores), marker="o", color="#4C72B0")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score by k")
    ax.set_xticks(list(k_values))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
        y_true, y_pred, labels=labels, normalize=normalize, ax=ax, colorbar=True
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


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
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    # Override scikit-learn's auto-generated labels (which append
    # "(Positive label: X)" whenever pos_label was inferred rather than
    # given explicitly) with the plain, stable label text.
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
    ax.set_title("Precision-recall curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
    ax.scatter(y_pred_array, residuals, alpha=0.6, color="#4C72B0", edgecolor="none")
    ax.axhline(0, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs predicted")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
    ax.scatter(y_true_array, y_pred_array, alpha=0.6, color="#4C72B0", edgecolor="none")

    combined_min = min(y_true_array.min(), y_pred_array.min())
    combined_max = max(y_true_array.max(), y_pred_array.max())
    ax.plot(
        [combined_min, combined_max], [combined_min, combined_max],
        color="#C44E52", linestyle="--", linewidth=1.5, label="Perfect prediction",
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
    ax.barh(working["feature"], working["importance"], color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


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
    ax.boxplot(data, tick_labels=labels)

    jitter_rng = np.random.RandomState(0)
    for position, values in enumerate(data, start=1):
        jitter = jitter_rng.normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter, values,
            alpha=0.6, color="#4C72B0", s=15, zorder=3, edgecolor="none",
        )

    ax.set_ylabel("Score")
    ax.set_title("Cross-validation scores by fold")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig