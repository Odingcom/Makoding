"""Feature engineering utilities for Makoding.

Framework-independent transformations for preparing tabular data for
analysis and machine learning.

Two APIs, by design
--------------------
- **Stateful transformers** (:class:`NumericScaler`, :class:`CategoricalEncoder`,
  :class:`NumericBinner`, :class:`TargetEncoder`, :class:`MissingIndicatorAdder`):
  explicit ``fit``/``transform``/``fit_transform`` methods. **Use these in
  any pipeline that has a train/test or train/inference split.** Fit
  once on training data; call ``transform`` (never ``fit`` again) on
  validation, test, or live data, so every split sees the exact same
  learned statistics, categories, and bin edges. Fitted state is
  stored on plain, inspectable instance attributes (``mean_``,
  ``categories_``, ``bin_edges_``, ...) for easy debugging or
  serialization with your framework of choice (``pickle``, ``joblib``).
- **One-shot functions** (everything else — ``power_transform``,
  ``create_ratio_features``, ``add_datetime_features``, ``add_cyclical_features``,
  ``add_frequency_features``, the low-variance/correlation filters, and
  convenience wrappers ``scale_numeric_features`` / `encode_categorical_features`` /
  ``bin_numeric_feature`` / ``target_encode`` / ``add_missing_indicators``
  around the classes above): fit-and-apply in a single call, for
  exploratory analysis on one dataset. The convenience wrappers exist
  purely for quick one-off use — reach for the class directly the
  moment you have more than one dataset split to process consistently.

Design principles
------------------
- Public functions and classes validate their inputs and raise
  informative, typed errors (``TypeError``, ``ValueError``, ``KeyError``).
- Caller DataFrames are never mutated.
- Missing values are never silently imputed — functions raise rather
  than guess, except where a transform's own semantics make "missing
  stays missing" the obviously correct behavior (scaling, binning,
  datetime extraction).
- Output column names are deterministic and suitable for downstream
  selection (see :mod:`makoding.feature_selection`).
- Every function/class docstring has an ``Assumptions`` section
  describing statistical or structural assumptions and edge-case
  behavior. Read it before applying a transform to a new dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

__all__ = [
    # Stateful transformers
    "NumericScaler",
    "CategoricalEncoder",
    "NumericBinner",
    "TargetEncoder",
    "MissingIndicatorAdder",
    "NotFittedError",
    # One-shot convenience wrappers around the transformers above
    "scale_numeric_features",
    "encode_categorical_features",
    "bin_numeric_feature",
    "target_encode",
    "add_missing_indicators",
    # One-shot feature-creation functions
    "log_transform",
    "sqrt_transform",
    "power_transform",
    "create_ratio_features",
    "create_difference_features",
    "create_product_features",
    "create_polynomial_features",
    "add_datetime_features",
    "add_cyclical_features",
    "add_frequency_features",
    # Filter-style feature selection
    "low_variance_features",
    "drop_low_variance_features",
    "find_correlated_feature_pairs",
    "drop_correlated_features",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ScaleMethod = Literal["standard", "minmax", "robust"]
EncodeMethod = Literal["onehot", "ordinal", "frequency"]
BinStrategy = Literal["uniform", "quantile", "kmeans"]

_DATETIME_COMPONENTS = {
    "year",
    "month",
    "quarter",
    "day",
    "dayofweek",
    "weekofyear",
    "hour",
    "is_weekend",
    "is_month_start",
    "is_month_end",
}


class NotFittedError(RuntimeError):
    """Raised when ``transform`` is called before ``fit`` on a stateful transformer."""


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

def _validate_dataframe(frame: Any) -> None:
    """Validate that ``frame`` is a pandas DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__!r}")


def _validate_columns_exist(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that requested columns exist in ``frame`` and contain no duplicates."""
    _validate_dataframe(frame)
    columns = list(columns)

    if len(columns) != len(set(columns)):
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        raise ValueError(f"columns contains duplicates: {duplicates}")

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Column(s) not found in DataFrame: {missing}")


def _validate_numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise TypeError listing any non-numeric columns."""
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise TypeError(f"Column(s) are not numeric: {non_numeric}")


def _resolve_numeric_columns(frame: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """Resolve the numeric-column list, defaulting to every numeric column."""
    _validate_dataframe(frame)

    if columns is None:
        result = frame.select_dtypes(include=[np.number]).columns.tolist()
    else:
        result = list(columns)
        _validate_columns_exist(frame, result)
        _validate_numeric_columns(frame, result)

    if not result:
        raise ValueError("No numeric columns were selected.")
    return result


def _validate_no_missing(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ValueError listing any columns that contain missing values."""
    missing = frame[list(columns)].isna().sum()
    affected = missing[missing > 0].index.tolist()
    if affected:
        raise ValueError(
            f"Column(s) contain missing values: {affected}. "
            "Impute or remove missing values explicitly before transformation."
        )


# ---------------------------------------------------------------------------
# NumericScaler
# ---------------------------------------------------------------------------

def _make_scaler(method: ScaleMethod) -> Any:
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler()
    raise ValueError(f"method must be one of ['standard', 'minmax', 'robust'], got {method!r}")


class NumericScaler:
    """Fit-once, apply-many-times numeric scaling (z-score / min-max / robust).

    Backed by scikit-learn's ``StandardScaler`` / ``MinMaxScaler`` /
    ``RobustScaler``. Fit this transformer on training data only, then
    reuse it — via ``transform``, never ``fit`` again — on validation,
    test, and inference data, so every split is scaled with the exact
    same learned statistics.

    Parameters
    ----------
    method:
        - ``"standard"``: z-score, ``(x - mean) / std``.
        - ``"minmax"``: ``(x - min) / (max - min)``, onto ``[0, 1]``.
        - ``"robust"``: ``(x - median) / IQR``, less sensitive to outliers.

    Attributes
    ----------
    columns_:
        Feature columns fit on. ``None`` until ``fit`` is called.
    n_features_in_:
        Number of columns fit on.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or a column is not numeric.
    KeyError
        If a requested column does not exist.
    ValueError
        If ``method`` is invalid, a column contains a duplicate name,
        or a fit/transform column contains missing values.
    NotFittedError
        If ``transform`` is called before ``fit``.

    Assumptions
    -----------
    - Statistics are learned entirely from the data passed to ``fit``;
      calling ``transform`` never updates them, even implicitly.
    - Both ``fit`` and ``transform`` reject columns containing missing
      values outright (unlike this module's underlying scikit-learn
      scalers, which would silently propagate ``NaN``) — impute first.
      This is a deliberate difference from a plain one-shot scaling
      function: at fit time you decide your imputation strategy once;
      silently accepting ``NaN`` here would let a data-quality
      regression at inference time pass through unnoticed.
    - A zero-variance (constant) training column is handled the way
      the underlying scikit-learn scaler handles it (``StandardScaler``
      maps it to all zeros rather than dividing by zero); this
      transformer does not add its own zero-variance guard on top.
    - ``method="standard"`` divides by the **population** standard
      deviation (``ddof=0``, scikit-learn's convention), not pandas'
      default sample standard deviation (``ddof=1``) — if you
      manually recompute a column's std with plain
      ``frame[column].std()`` to sanity-check this transformer's
      output, use ``frame[column].std(ddof=0)`` instead, or the
      numbers won't quite match for small samples. This module's own
      :func:`low_variance_features` defaults to ``ddof=1`` — the two
      are independent conventions from different underlying libraries,
      not a bug in either one.
    """

    def __init__(self, method: ScaleMethod = "standard") -> None:
        if method not in {"standard", "minmax", "robust"}:
            raise ValueError(
                f"method must be one of ['standard', 'minmax', 'robust'], got {method!r}"
            )
        self.method = method
        self._scaler = _make_scaler(method)
        self.columns_: list[str] | None = None
        self.n_features_in_: int | None = None
        self._is_fitted = False

    def fit(self, frame: pd.DataFrame, columns: Sequence[str] | None = None) -> "NumericScaler":
        """Learn scaling statistics from ``frame``."""
        self.columns_ = _resolve_numeric_columns(frame, columns)
        _validate_no_missing(frame, self.columns_)

        self._scaler = _make_scaler(self.method)
        self._scaler.fit(frame[self.columns_])
        self.n_features_in_ = len(self.columns_)
        self._is_fitted = True

        logger.debug(
            "NumericScaler.fit: method=%s, %d columns", self.method, self.n_features_in_
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the previously-learned scaling to ``frame``."""
        if not self._is_fitted:
            raise NotFittedError("NumericScaler is not fitted. Call 'fit' before 'transform'.")

        _validate_columns_exist(frame, self.columns_)
        _validate_no_missing(frame, self.columns_)

        result = frame.copy(deep=True)
        result.loc[:, self.columns_] = self._scaler.transform(frame[self.columns_])
        return result

    def fit_transform(
        self, frame: pd.DataFrame, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Equivalent to calling :meth:`fit` then :meth:`transform` on ``frame``."""
        return self.fit(frame, columns).transform(frame)


def scale_numeric_features(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: ScaleMethod = "standard",
) -> pd.DataFrame:
    """Scale numeric columns in a copy of ``frame`` (one-shot convenience wrapper).

    Fits and transforms on the same data in a single call. For a
    train/test or train/inference workflow, use :class:`NumericScaler`
    directly so the same fitted statistics apply to every split.

    Parameters
    ----------
    frame:
        Input DataFrame.
    columns:
        Numeric columns to scale. Defaults to every numeric column.
    method:
        See :class:`NumericScaler`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified columns scaled.

    Raises
    ------
    See :class:`NumericScaler`.
    """
    return NumericScaler(method).fit_transform(frame, columns)


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------

class CategoricalEncoder:
    """Fit-once, apply-many-times categorical encoding (one-hot / ordinal / frequency).

    One-hot and ordinal encoding are backed by scikit-learn's
    ``OneHotEncoder`` / ``OrdinalEncoder`` with ``handle_unknown``
    configured so a category never seen at fit time does not crash
    ``transform`` — see Assumptions for exactly what happens instead.

    Parameters
    ----------
    method:
        - ``"onehot"``: one binary column per category seen at fit
          time, via ``OneHotEncoder(handle_unknown="ignore")``.
        - ``"ordinal"``: each category mapped to an integer (order
          determined by scikit-learn's ``OrdinalEncoder`` — sorted by
          category value, not by any semantic meaning), via
          ``OrdinalEncoder(handle_unknown="use_encoded_value")``.
        - ``"frequency"``: each category replaced by its relative
          frequency (proportion of rows) at fit time.
    drop:
        For ``method="onehot"`` only: ``None`` (default, keep every
        dummy column) or ``"first"`` (drop the first category's dummy
        column per feature, to avoid perfect multicollinearity for
        linear models). Ignored for other methods.

    Attributes
    ----------
    columns_:
        Feature columns fit on.
    feature_names_:
        Output column names produced by ``transform`` (only meaningful
        for ``method="onehot"``, where it differs from ``columns_``).

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If a requested column does not exist.
    ValueError
        If ``method``/``drop`` is invalid, no categorical columns are
        found, or a fit/transform column contains a duplicate name.
    NotFittedError
        If ``transform`` is called before ``fit``.

    Assumptions
    -----------
    - **Unseen categories at transform time** are handled differently
      per method, and this is the main reason to prefer this class
      over hand-rolling one-hot encoding yourself:

      - ``"onehot"``: an unseen category produces an all-zero row
        across that feature's dummy columns — indistinguishable from
        a genuinely missing value unless you also run
        :class:`MissingIndicatorAdder` beforehand.
      - ``"ordinal"``: an unseen category is encoded as ``-1``; a
        missing value is encoded as ``-2`` (distinct from ``-1``, so
        the two cases remain distinguishable even though neither
        matches a genuine learned category code, which are always
        ``>= 0``).
      - ``"frequency"``: an unseen category is encoded as ``0.0``
        (it was observed zero times in the training data), which can
        collide with a training category that also happened to be
        very rare — a known limitation of frequency encoding.

    - Missing values in a source column are themselves treated as
      **their own category** at fit time for ``"onehot"`` and
      ``"frequency"`` (so they get a learned encoding, not a
      propagated ``NaN``), as long as at least one missing value was
      present at fit time. ``"ordinal"`` is the exception: scikit-learn's
      ``OrdinalEncoder`` always encodes missing values as the fixed
      sentinel ``-2`` regardless of whether missing values were
      present at fit time — see above.
    - ``"ordinal"`` integers carry no ordinal/semantic meaning unless
      your categories happen to sort correctly — this is the same
      caveat as any label encoder.
    """

    def __init__(
        self,
        method: EncodeMethod = "onehot",
        drop: str | None = None,
    ) -> None:
        if method not in {"onehot", "ordinal", "frequency"}:
            raise ValueError(
                f"method must be one of ['onehot', 'ordinal', 'frequency'], got {method!r}"
            )
        if drop not in {None, "first"}:
            raise ValueError(f"drop must be None or 'first', got {drop!r}")

        self.method = method
        self.drop = drop
        self._encoder: Any = None
        self._frequency_maps: dict[str, dict[Any, float]] = {}
        self.columns_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self._is_fitted = False

    def fit(
        self, frame: pd.DataFrame, columns: Sequence[str] | None = None
    ) -> "CategoricalEncoder":
        """Learn categories (and, for onehot/ordinal, the sklearn encoder) from ``frame``."""
        _validate_dataframe(frame)

        if columns is None:
            columns = frame.select_dtypes(
                include=["object", "category", "string", "bool"]
            ).columns.tolist()
        else:
            columns = list(columns)
            _validate_columns_exist(frame, columns)

        if not columns:
            raise ValueError("No categorical columns were selected.")

        self.columns_ = columns

        if self.method == "onehot":
            self._encoder = OneHotEncoder(
                handle_unknown="ignore", drop=self.drop, sparse_output=False
            )
            self._encoder.fit(frame[self.columns_])
            names = self._encoder.get_feature_names_out(self.columns_)
            self.feature_names_ = [str(name) for name in names]
        elif self.method == "ordinal":
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            self._encoder.fit(frame[self.columns_])
            self.feature_names_ = list(self.columns_)
        else:  # frequency
            self._frequency_maps = {
                column: frame[column].value_counts(normalize=True, dropna=False).to_dict()
                for column in self.columns_
            }
            self.feature_names_ = list(self.columns_)

        self._is_fitted = True
        logger.debug(
            "CategoricalEncoder.fit: method=%s, %d columns", self.method, len(self.columns_)
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the previously-learned encoding to ``frame``."""
        if not self._is_fitted:
            raise NotFittedError(
                "CategoricalEncoder is not fitted. Call 'fit' before 'transform'."
            )

        _validate_columns_exist(frame, self.columns_)
        result = frame.drop(columns=self.columns_).copy(deep=True)

        if self.method == "onehot":
            encoded = self._encoder.transform(frame[self.columns_])
            encoded_frame = pd.DataFrame(
                encoded, index=frame.index, columns=self.feature_names_
            )
            return pd.concat([result, encoded_frame], axis=1)
        elif self.method == "ordinal":
            encoded = self._encoder.transform(frame[self.columns_])
            encoded_frame = pd.DataFrame(
                encoded, index=frame.index, columns=self.feature_names_
            )
            return pd.concat([result, encoded_frame], axis=1)
        else:  # frequency
            unseen_counts = {}
            encoded_columns = {}
            for column in self.columns_:
                mapping = self._frequency_maps[column]
                encoded_columns[column] = frame[column].map(mapping).fillna(0.0)
                unseen = ~frame[column].isin(mapping.keys())
                if unseen.any():
                    unseen_counts[column] = int(unseen.sum())
            if unseen_counts:
                logger.warning(
                    "CategoricalEncoder(frequency).transform: unseen categories "
                    "encountered (encoded as 0.0): %s",
                    unseen_counts,
                )
            encoded_frame = pd.DataFrame(encoded_columns, index=frame.index)
            return pd.concat([result, encoded_frame], axis=1)

    def fit_transform(
        self, frame: pd.DataFrame, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Equivalent to calling :meth:`fit` then :meth:`transform` on ``frame``."""
        return self.fit(frame, columns).transform(frame)


def encode_categorical_features(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: EncodeMethod = "onehot",
    drop: str | None = None,
) -> pd.DataFrame:
    """Encode categorical columns in a copy of ``frame`` (one-shot convenience wrapper).

    Fits and transforms on the same data in a single call. For a
    train/test or train/inference workflow, use :class:`CategoricalEncoder`
    directly so unseen categories at inference are handled by the
    encoding learned from training, not silently re-derived.

    Parameters
    ----------
    frame, columns, method, drop:
        See :class:`CategoricalEncoder`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified columns encoded.

    Raises
    ------
    See :class:`CategoricalEncoder`.
    """
    return CategoricalEncoder(method=method, drop=drop).fit_transform(frame, columns)


# ---------------------------------------------------------------------------
# NumericBinner
# ---------------------------------------------------------------------------

class NumericBinner:
    """Fit-once, apply-many-times discretization of a numeric column.

    Backed by scikit-learn's ``KBinsDiscretizer``. Bin edges are
    learned from the data passed to ``fit`` and reused, unchanged, by
    ``transform`` — critical for ``strategy="quantile"``, where
    recomputing quantiles on a new dataset (as a one-shot binning
    function necessarily does) would silently produce different bin
    boundaries each time.

    Parameters
    ----------
    n_bins:
        Number of bins requested.
    strategy:
        - ``"uniform"``: equal-width bins across the observed range.
        - ``"quantile"``: equal-frequency bins.
        - ``"kmeans"``: bin edges placed via 1D k-means cluster
          centers — clusters together values that are numerically
          close, which can produce uneven-width, uneven-frequency
          bins that better reflect natural groupings in the data.

    Attributes
    ----------
    column_:
        The column fit on.
    bin_edges_:
        The learned bin edge array (length ``n_bins + 1``); the
        outermost edges are extended to ``-inf``/``+inf`` at transform
        time so out-of-range values at inference fall into the
        nearest bin rather than becoming missing.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or the column is not numeric.
    KeyError
        If the column does not exist.
    ValueError
        If ``strategy`` is invalid, ``n_bins`` is less than 2, or a
        fit/transform column contains missing values.
    NotFittedError
        If ``transform`` is called before ``fit``.

    Assumptions
    -----------
    - ``fit`` and ``transform`` both reject missing values outright
      (unlike a one-shot ``pandas.cut``/``qcut``-based binner, which
      would leave them as missing silently) — impute first, or bin
      the missingness itself with :class:`MissingIndicatorAdder`.
    - ``strategy="quantile"`` on a column with many repeated values
      can produce **fewer distinct bins than requested**, since
      duplicate quantile edges collapse; check
      ``len(set(binner.bin_edges_))`` if you need an exact count.
    - Values outside the range observed at fit time are assigned to
      the nearest edge bin at transform time (via the ``-inf``/``+inf``
      edge extension), not treated as missing or out-of-domain —
      appropriate for most modeling uses, but means a wildly
      out-of-range value at inference is silently absorbed into the
      extreme bin rather than flagged.
    """

    def __init__(self, n_bins: int = 5, strategy: BinStrategy = "uniform") -> None:
        if strategy not in {"uniform", "quantile", "kmeans"}:
            raise ValueError(
                f"strategy must be one of ['uniform', 'quantile', 'kmeans'], got {strategy!r}"
            )
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        self.n_bins = n_bins
        self.strategy = strategy
        self.column_: str | None = None
        self.bin_edges_: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, frame: pd.DataFrame, column: str) -> "NumericBinner":
        """Learn bin edges for ``column`` from ``frame``."""
        _validate_columns_exist(frame, [column])
        _validate_numeric_columns(frame, [column])
        _validate_no_missing(frame, [column])

        discretizer_kwargs = dict(
            n_bins=self.n_bins, encode="ordinal", strategy=self.strategy, subsample=None
        )
        if self.strategy == "quantile":
            import inspect

            if "quantile_method" in inspect.signature(KBinsDiscretizer.__init__).parameters:
                discretizer_kwargs["quantile_method"] = "averaged_inverted_cdf"
        discretizer = KBinsDiscretizer(**discretizer_kwargs)
        discretizer.fit(frame[[column]])

        self.column_ = column
        edges = discretizer.bin_edges_[0].copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.bin_edges_ = edges
        self._is_fitted = True

        logger.debug(
            "NumericBinner.fit: column=%s, strategy=%s, %d bin edges",
            column,
            self.strategy,
            len(self.bin_edges_),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add a ``f"{column}_binned"`` column using the previously-learned edges."""
        if not self._is_fitted:
            raise NotFittedError("NumericBinner is not fitted. Call 'fit' before 'transform'.")

        _validate_columns_exist(frame, [self.column_])
        _validate_no_missing(frame, [self.column_])

        result = frame.copy(deep=True)
        result[f"{self.column_}_binned"] = pd.cut(
            frame[self.column_], bins=self.bin_edges_, labels=False, include_lowest=True
        )
        return result

    def fit_transform(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        """Equivalent to calling :meth:`fit` then :meth:`transform` on ``frame``."""
        return self.fit(frame, column).transform(frame)


def bin_numeric_feature(
    frame: pd.DataFrame,
    column: str,
    n_bins: int = 5,
    strategy: BinStrategy = "uniform",
) -> pd.DataFrame:
    """Discretize a numeric column in a copy of ``frame`` (one-shot convenience wrapper).

    Fits and transforms on the same data in a single call. For a
    train/test or train/inference workflow, use :class:`NumericBinner`
    directly, especially for ``strategy="quantile"``, whose bin edges
    must come from training data alone.

    Parameters
    ----------
    frame, column, n_bins, strategy:
        See :class:`NumericBinner`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with an added ``f"{column}_binned"`` column.

    Raises
    ------
    See :class:`NumericBinner`.
    """
    return NumericBinner(n_bins=n_bins, strategy=strategy).fit_transform(frame, column)


# ---------------------------------------------------------------------------
# TargetEncoder
# ---------------------------------------------------------------------------

class TargetEncoder:
    """Fit-once, apply-many-times smoothed mean target encoding.

    This class exists specifically to close the leakage gap of a
    one-shot target-encoding function: fit it on **training folds
    only**, then call ``transform`` (never ``fit``) on validation/test
    data. A row's encoded value never depends on its own target value
    once you follow this fit/transform split — the single biggest
    practical difference from computing target encoding in one call
    on your whole dataset.

    Parameters
    ----------
    smoothing:
        Non-negative weight, in "equivalent number of rows", given to
        the global target mean when blending it with each category's
        own mean:
        ``encoded = (category_mean * category_count + global_mean * smoothing)
        / (category_count + smoothing)``.
        ``smoothing=0`` reduces to the raw per-category mean.

    Attributes
    ----------
    column_, target_:
        The grouping and target columns fit on.
    mapping_:
        Dict of category -> smoothed encoded value, learned at fit time.
    global_mean_:
        The training target mean, used as the fallback encoding for
        any category not seen at fit time.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or ``target`` is not numeric
        (fit only).
    KeyError
        If ``column``/``target`` does not exist.
    ValueError
        If ``smoothing`` is negative.
    NotFittedError
        If ``transform`` is called before ``fit``.

    Assumptions
    -----------
    - **This class is what makes target encoding leakage-safe in this
      module** — the one-shot :func:`target_encode` function below
      fits and applies in a single call and is documented as leaking
      target information into its own output; prefer this class
      whenever you have a train/validation/test split, which is
      whenever leakage would actually matter.
    - Missing values in ``column`` are treated as their own category
      at fit time, exactly as in :func:`target_encode`.
    - A category seen at ``transform`` time but not at ``fit`` time —
      including a **missing value** when ``fit`` saw no missing
      values in ``column`` — is encoded as ``global_mean_`` (the
      overall training target mean). This is logged at ``WARNING``
      level with a row count, since a large number of unseen
      categories at transform time often signals category drift
      between training and inference data worth investigating. A
      missing value *is* resolved to its own learned smoothed mean
      (not the fallback) whenever ``fit`` did see at least one missing
      value in ``column``.
    - ``transform`` does not require (or use) a ``target`` column at
      all — only ``fit`` does — which is exactly the point: at
      inference time you don't have the target yet.
    """

    def __init__(self, smoothing: float = 1.0) -> None:
        if smoothing < 0:
            raise ValueError(f"smoothing must be non-negative, got {smoothing}")
        self.smoothing = smoothing
        self.column_: str | None = None
        self.target_: str | None = None
        self.mapping_: dict[Any, float] = {}
        self.global_mean_: float | None = None
        self._is_fitted = False

    def fit(self, frame: pd.DataFrame, column: str, target: str) -> "TargetEncoder":
        """Learn smoothed per-category target means from ``frame``."""
        _validate_columns_exist(frame, [column, target])
        if not pd.api.types.is_numeric_dtype(frame[target]):
            raise TypeError(f"target column {target!r} must be numeric")
        if frame[target].isna().any():
            raise ValueError(f"target column {target!r} contains missing values")

        self.column_ = column
        self.target_ = target
        self.global_mean_ = float(frame[target].mean())

        stats = frame.groupby(column, dropna=False)[target].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
            stats["count"] + self.smoothing
        )

        index_is_na = pd.isna(stats.index.to_series())
        self.mapping_ = smoothed[~index_is_na.values].to_dict()
        if index_is_na.any():
            self.mapping_[None] = float(smoothed[index_is_na.values].iloc[0])

        self._is_fitted = True
        logger.debug(
            "TargetEncoder.fit: column=%s, %d categories, global_mean=%.4f",
            column,
            len(self.mapping_),
            self.global_mean_,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add a ``f"{column}_target_encoded"`` column using the previously-learned mapping."""
        if not self._is_fitted:
            raise NotFittedError("TargetEncoder is not fitted. Call 'fit' before 'transform'.")

        _validate_columns_exist(frame, [self.column_])

        result = frame.copy(deep=True)
        is_na = frame[self.column_].isna()

        encoded = frame[self.column_].map(self.mapping_)
        if is_na.any() and None in self.mapping_:
            encoded = encoded.where(~is_na, self.mapping_[None])

        # Anything still unresolved — an unseen non-null category, or a
        # missing value with no learned NaN-group (because fit saw no
        # missing values) — falls back to the global training mean.
        unresolved = encoded.isna()
        if unresolved.any():
            logger.warning(
                "TargetEncoder.transform: %d row(s) had a category not seen at fit "
                "time (including missing values, if fit saw none); encoded with the "
                "global training mean (%.4f)",
                int(unresolved.sum()),
                self.global_mean_,
            )
            encoded = encoded.where(~unresolved, self.global_mean_)

        result[f"{self.column_}_target_encoded"] = encoded
        return result

    def fit_transform(self, frame: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
        """Equivalent to calling :meth:`fit` then :meth:`transform` on ``frame``.

        Still leaks target information into its own output — see the
        class-level Assumptions. Prefer ``fit`` on a training split and
        ``transform`` on other splits whenever leakage would matter.
        """
        return self.fit(frame, column, target).transform(frame)


def target_encode(
    frame: pd.DataFrame,
    column: str,
    target: str,
    smoothing: float = 1.0,
) -> pd.DataFrame:
    """Target-encode a column in a copy of ``frame`` (one-shot convenience wrapper).

    Fits and applies on the same data in a single call, which means
    each row's encoded value is influenced by its own target value —
    using this directly as a training feature without any
    cross-validation scheme risks target leakage. Use
    :class:`TargetEncoder` (fit on training folds, transform other
    splits) whenever that risk would matter.

    Parameters
    ----------
    frame, column, target, smoothing:
        See :class:`TargetEncoder`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with an added ``f"{column}_target_encoded"`` column.

    Raises
    ------
    See :class:`TargetEncoder`.
    """
    return TargetEncoder(smoothing=smoothing).fit_transform(frame, column, target)


# ---------------------------------------------------------------------------
# MissingIndicatorAdder
# ---------------------------------------------------------------------------

class MissingIndicatorAdder:
    """Fit-once, apply-many-times missing-value indicator flags.

    Unlike a one-shot "flag columns that currently have missing
    values" function, this class remembers **which columns had
    missing values at fit time** and always adds an indicator for
    exactly those columns at transform time — even if a particular
    transform batch happens to have zero missing values in one of
    them. This keeps the output schema (column names) identical across
    every batch you transform, which matters because most downstream
    models require a fixed feature schema.

    Parameters
    ----------
    columns:
        Columns to consider. Defaults to every column in the frame
        passed to ``fit``.
    suffix:
        Suffix appended to each source column's name to form the
        indicator column's name.

    Attributes
    ----------
    flagged_columns_:
        The columns that had at least one missing value at fit time
        (a subset of ``columns``); only these get an indicator added
        at transform time.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If a requested column does not exist.
    NotFittedError
        If ``transform`` is called before ``fit``.

    Assumptions
    -----------
    - Columns with **zero** missing values at fit time never get an
      indicator, even if ``columns`` was passed explicitly — to avoid
      manufacturing a constant, useless column. This mirrors the
      one-shot function's default-columns behavior, but here it also
      applies to explicitly-passed columns, since the schema is fixed
      at fit time either way.
    - This class only adds indicators; it never imputes or drops the
      underlying missing values.
    """

    def __init__(self, columns: Sequence[str] | None = None, suffix: str = "_is_missing") -> None:
        self.columns = list(columns) if columns is not None else None
        self.suffix = suffix
        self.flagged_columns_: list[str] | None = None
        self._is_fitted = False

    def fit(self, frame: pd.DataFrame) -> "MissingIndicatorAdder":
        """Determine which columns had missing values in ``frame``."""
        _validate_dataframe(frame)

        columns = self.columns if self.columns is not None else list(frame.columns)
        _validate_columns_exist(frame, columns)

        self.flagged_columns_ = [c for c in columns if frame[c].isna().any()]
        self._is_fitted = True

        logger.debug(
            "MissingIndicatorAdder.fit: flagged %d/%d columns",
            len(self.flagged_columns_),
            len(columns),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns for every column flagged at fit time."""
        if not self._is_fitted:
            raise NotFittedError(
                "MissingIndicatorAdder is not fitted. Call 'fit' before 'transform'."
            )

        _validate_columns_exist(frame, self.flagged_columns_)

        result = frame.copy(deep=True)
        for column in self.flagged_columns_:
            result[f"{column}{self.suffix}"] = frame[column].isna()
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Equivalent to calling :meth:`fit` then :meth:`transform` on ``frame``."""
        return self.fit(frame).transform(frame)


def add_missing_indicators(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    suffix: str = "_is_missing",
) -> pd.DataFrame:
    """Add missing-value indicators to a copy of ``frame`` (one-shot convenience wrapper).

    Fits and transforms on the same data in a single call. For a
    train/test or train/inference workflow, use
    :class:`MissingIndicatorAdder` directly so the same set of
    indicator columns is added to every split, even a split that
    happens to have no missing values in a column that was missing in
    training.

    Parameters
    ----------
    frame, columns, suffix:
        See :class:`MissingIndicatorAdder`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added boolean column per flagged
        source column.

    Raises
    ------
    See :class:`MissingIndicatorAdder`.
    """
    return MissingIndicatorAdder(columns=columns, suffix=suffix).fit_transform(frame)


# ---------------------------------------------------------------------------
# Log / power transforms
# ---------------------------------------------------------------------------

def log_transform(
    frame: pd.DataFrame,
    columns: Sequence[str],
    offset: float = 0.0,
    suffix: str = "_log",
) -> pd.DataFrame:
    """Add natural-log features: ``log(x + offset)``.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    columns:
        Numeric columns to transform.
    offset:
        Added to each value before logging. ``offset=0`` is a plain
        natural log (values must be strictly positive); ``offset=1``
        reproduces the common "log1p" transform (values must be
        greater than ``-1``, so zeros are safe).
    suffix:
        Suffix appended to each source column's name for the new
        column. The source column is left unchanged.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added ``f"{column}{suffix}"`` column
        per requested column.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, a column is not numeric, or
        ``offset`` is not numeric.
    KeyError
        If a column does not exist.
    ValueError
        If any non-missing value in a column is ``<= 0`` after adding
        ``offset``.

    Assumptions
    -----------
    - Domain violations are checked only on non-missing values and
      raise immediately, rather than silently producing ``-inf`` — if
      your data contains zeros, pass ``offset=1`` (equivalent to
      ``log1p``) rather than adjusting after the fact.
    - Missing values are preserved as missing in the output.
    - Unlike :class:`NumericScaler` and friends, this is not a
      stateful transformer — there is no "statistic learned from
      training data" here; the same fixed ``offset`` you choose
      applies identically to every dataset you call this on.
    """
    _validate_columns_exist(frame, columns)
    if not isinstance(offset, (int, float, np.number)):
        raise TypeError("offset must be numeric.")

    result = frame.copy(deep=True)
    for column in columns:
        values = pd.to_numeric(result[column], errors="raise") + float(offset)
        observed = values.dropna()
        if (observed <= 0).any():
            raise ValueError(
                f"Column {column!r} contains values <= 0 after applying offset={offset}; "
                "natural log is undefined there. Consider a larger offset."
            )
        result[f"{column}{suffix}"] = np.log(values)
    return result


def sqrt_transform(
    frame: pd.DataFrame,
    columns: Sequence[str],
    suffix: str = "_sqrt",
) -> pd.DataFrame:
    """Add square-root features.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    columns:
        Numeric columns to transform.
    suffix:
        Suffix appended to each source column's name for the new
        column. The source column is left unchanged.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added ``f"{column}{suffix}"`` column
        per requested column.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or a column is not numeric.
    KeyError
        If a column does not exist.
    ValueError
        If any non-missing value in a column is negative.

    Assumptions
    -----------
    - Domain violations (negative values) raise rather than silently
      producing ``NaN``.
    - Missing values are preserved as missing in the output.
    """
    _validate_columns_exist(frame, columns)
    _validate_numeric_columns(frame, columns)

    result = frame.copy(deep=True)
    for column in columns:
        observed = result[column].dropna()
        if (observed < 0).any():
            raise ValueError(
                f"Column {column!r} contains negative values; square root is undefined there."
            )
        result[f"{column}{suffix}"] = np.sqrt(result[column])
    return result


def power_transform(
    frame: pd.DataFrame,
    columns: Sequence[str],
    method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
    standardize: bool = True,
) -> pd.DataFrame:
    """Add power-transformed features that reduce skew toward a normal shape.

    Backed by scikit-learn's ``PowerTransformer``.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    columns:
        Numeric columns to transform.
    method:
        - ``"yeo-johnson"`` (default): supports zero and negative
          values.
        - ``"box-cox"``: requires strictly positive values; often
          gives a slightly better fit when your data qualifies.
    standardize:
        If ``True`` (default), the power-transformed output is also
        standardized to zero mean / unit variance (scikit-learn's own
        default behavior).

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added ``f"{column}_power"`` column
        per requested column.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or a column is not numeric.
    KeyError
        If a column does not exist.
    ValueError
        If ``method`` is invalid, a column contains missing values, or
        (``method="box-cox"``) a column contains non-positive values.

    Assumptions
    -----------
    - This function fits a fresh ``PowerTransformer`` on ``frame``
      every call — it is a one-shot function, not a stateful
      transformer, even though scikit-learn's ``PowerTransformer``
      itself has fit/transform methods. If you need the *same*
      learned transform (in particular the same optimal lambda
      parameter) applied consistently to train and test data, use
      ``sklearn.preprocessing.PowerTransformer`` directly with its own
      ``fit``/``transform`` rather than calling this function twice.
    - Requires fully non-missing input (see ``_validate_no_missing``);
      unlike the log/sqrt transforms above, scikit-learn's
      ``PowerTransformer`` does not have well-defined missing-value
      passthrough behavior, so this function does not attempt it.
    """
    _validate_columns_exist(frame, columns)
    _validate_numeric_columns(frame, columns)
    _validate_no_missing(frame, columns)

    if method not in {"yeo-johnson", "box-cox"}:
        raise ValueError(f"method must be 'yeo-johnson' or 'box-cox', got {method!r}")

    transformer = PowerTransformer(method=method, standardize=standardize)
    values = transformer.fit_transform(frame[list(columns)])

    result = frame.copy(deep=True)
    for i, column in enumerate(columns):
        result[f"{column}_power"] = values[:, i]
    return result


# ---------------------------------------------------------------------------
# Pairwise / polynomial feature creation
# ---------------------------------------------------------------------------

def create_ratio_features(
    frame: pd.DataFrame,
    ratios: dict[str, tuple[str, str]],
    zero_division: Literal["nan", "inf", "raise"] = "nan",
) -> pd.DataFrame:
    """Create ``numerator / denominator`` ratio features.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    ratios:
        Mapping of new column name -> ``(numerator_column, denominator_column)``.
    zero_division:
        - ``"nan"`` (default): a zero denominator produces ``NaN``.
        - ``"inf"``: a zero denominator produces ``inf``/``-inf``
          (or ``NaN`` for ``0 / 0``), ordinary floating-point division
          behavior.
        - ``"raise"``: a zero denominator raises ``ZeroDivisionError``
          immediately.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added column per entry in ``ratios``.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If a referenced column does not exist.
    ValueError
        If ``zero_division`` is invalid, or an output column name
        already exists in ``frame``.
    ZeroDivisionError
        If ``zero_division="raise"`` and a denominator contains zero.

    Assumptions
    -----------
    - Column order in each tuple matters — ratios are not symmetric.
    - If either input is missing, the resulting ratio is missing,
      regardless of ``zero_division``.
    - This function does not validate that referenced columns are
      numeric before dividing; a non-numeric column will raise from
      pandas' own arithmetic rather than this function's validation,
      with a less specific error message.
    """
    _validate_dataframe(frame)
    if zero_division not in {"nan", "inf", "raise"}:
        raise ValueError(f"zero_division must be 'nan', 'inf', or 'raise', got {zero_division!r}")

    all_columns = [c for pair in ratios.values() for c in pair]
    _validate_columns_exist(frame, all_columns)

    result = frame.copy(deep=True)
    for new_name, (numerator, denominator) in ratios.items():
        if new_name in result.columns:
            raise ValueError(f"Output column already exists: {new_name!r}")

        denominator_series = result[denominator]
        if zero_division == "raise" and (denominator_series == 0).any():
            raise ZeroDivisionError(f"Column {denominator!r} contains zero values.")

        with np.errstate(divide="ignore", invalid="ignore"):
            values = result[numerator] / denominator_series

        if zero_division == "nan":
            values = values.replace([np.inf, -np.inf], np.nan)

        result[new_name] = values
    return result


def create_difference_features(
    frame: pd.DataFrame, differences: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Create ``left - right`` difference features.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    differences:
        Mapping of new column name -> ``(left_column, right_column)``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added column per entry in ``differences``.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If a referenced column does not exist.
    ValueError
        If an output column name already exists in ``frame``.

    Assumptions
    -----------
    Column order matters (subtraction is not symmetric). If either
    input is missing, the result is missing.
    """
    _validate_dataframe(frame)
    _validate_columns_exist(frame, [c for pair in differences.values() for c in pair])
    result = frame.copy(deep=True)

    for new_name, (left, right) in differences.items():
        if new_name in result.columns:
            raise ValueError(f"Output column already exists: {new_name!r}")
        result[new_name] = result[left] - result[right]
    return result


def create_product_features(
    frame: pd.DataFrame, products: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Create pairwise product (interaction) features: ``left * right``.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    products:
        Mapping of new column name -> ``(left_column, right_column)``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added column per entry in ``products``.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If a referenced column does not exist.
    ValueError
        If an output column name already exists in ``frame``.

    Assumptions
    -----------
    Multiplication is symmetric, so column order only affects the
    output column *name*, not its values. If either input is missing,
    the result is missing.
    """
    _validate_dataframe(frame)
    _validate_columns_exist(frame, [c for pair in products.values() for c in pair])
    result = frame.copy(deep=True)

    for new_name, (left, right) in products.items():
        if new_name in result.columns:
            raise ValueError(f"Output column already exists: {new_name!r}")
        result[new_name] = result[left] * result[right]
    return result


def create_polynomial_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    degree: int = 2,
    include_bias: bool = False,
) -> pd.DataFrame:
    """Add polynomial powers *and* cross-term interactions for the given columns.

    Backed by scikit-learn's ``PolynomialFeatures``, so — unlike
    generating each column's powers independently — this also
    produces genuine cross terms between different columns (e.g.
    ``a * b``, ``a^2 * b``) up to ``degree``.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    columns:
        Numeric columns to expand.
    degree:
        Highest total polynomial degree to generate. Must be ``>= 1``
        (``degree=1`` reproduces the original columns, added under
        scikit-learn's generated names — see Assumptions).
    include_bias:
        If ``True``, also adds a constant ``1`` column (scikit-learn's
        own bias term). Default ``False``, since most modeling
        contexts add their own intercept separately.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with scikit-learn-named polynomial/interaction
        columns added (e.g. ``"a^2"``, ``"a b"``). The original
        ``columns`` are **not duplicated** — scikit-learn always
        regenerates them at degree 1, and this function skips adding
        a column whose name already exists in ``frame``.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or a column is not numeric.
    KeyError
        If a column does not exist.
    ValueError
        If ``degree`` is less than 1.

    Assumptions
    -----------
    - Column count grows combinatorially with both the number of
      input columns and ``degree`` — ``PolynomialFeatures`` on 10
      columns at ``degree=3`` produces hundreds of output columns.
      Start with a small, pre-filtered column set.
    - This function does not require or preserve any fitted state
      across calls — every call independently fits a fresh
      ``PolynomialFeatures`` on whatever ``frame`` it's given. For
      polynomial degree/structure this is harmless (the generated
      formula doesn't depend on the specific data values, only on
      which columns and what degree), unlike scaling or binning.
    """
    _validate_columns_exist(frame, columns)
    _validate_numeric_columns(frame, columns)
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")

    transformer = PolynomialFeatures(degree=degree, include_bias=include_bias)
    values = transformer.fit_transform(frame[list(columns)])
    names = [str(name) for name in transformer.get_feature_names_out(columns)]

    result = frame.copy(deep=True)
    for i, name in enumerate(names):
        if name in result.columns:
            continue
        result[name] = values[:, i]
    return result


# ---------------------------------------------------------------------------
# Datetime / cyclical / frequency feature creation
# ---------------------------------------------------------------------------

def add_datetime_features(
    frame: pd.DataFrame,
    column: str,
    components: list[str] | None = None,
    prefix: str | None = None,
    errors: Literal["raise", "coerce"] = "raise",
) -> pd.DataFrame:
    """Decompose a datetime column into individual calendar feature columns.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    column:
        Column to parse as datetimes and decompose. Does not need to
        already have a datetime dtype — it is passed through
        ``pandas.to_datetime``.
    components:
        Which components to add. Defaults to every supported
        component: ``"year"``, ``"month"``, ``"quarter"``, ``"day"``,
        ``"dayofweek"``, ``"weekofyear"``, ``"hour"``, ``"is_weekend"``,
        ``"is_month_start"``, ``"is_month_end"``.
    prefix:
        Prefix for the new column names. Defaults to ``column``.
    errors:
        - ``"raise"`` (default): a value that fails to parse as a
          date raises immediately — the safe default, since a
          silently-mostly-missing datetime feature set is easy to miss.
        - ``"coerce"``: unparseable values become ``NaT``, and every
          derived feature for that row is missing, without raising.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added column per requested component,
        named ``f"{prefix}_{component}"``. The original column is left
        unchanged.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If ``column`` does not exist.
    ValueError
        If ``components`` contains an unsupported value, or
        ``errors`` is invalid.

    Assumptions
    -----------
    - **``errors="raise"`` is the default**, unlike scanning a column
      and silently coercing failures to missing — a badly-formatted
      date column will surface immediately as an error here rather
      than as a mysteriously mostly-missing feature set discovered
      much later. Pass ``errors="coerce"`` explicitly once you've
      decided that's the right handling for a specific column.
    - ``"is_weekend"`` is ``True`` for Saturday/Sunday, ``False`` for
      weekdays, and missing (not ``False``) for a row that failed to
      parse under ``errors="coerce"``.
    - Time-zone information, if present in the source column, is
      preserved by ``pandas.to_datetime`` and affects ``"hour"``;
      naive and timezone-aware timestamps are not reconciled for you.
    """
    _validate_columns_exist(frame, [column])
    if errors not in {"raise", "coerce"}:
        raise ValueError(f"errors must be 'raise' or 'coerce', got {errors!r}")

    if components is None:
        components = sorted(_DATETIME_COMPONENTS)
    else:
        unknown = set(components) - _DATETIME_COMPONENTS
        if unknown:
            raise ValueError(
                f"Unsupported datetime component(s): {sorted(unknown)}. "
                f"Supported: {sorted(_DATETIME_COMPONENTS)}"
            )

    result = frame.copy(deep=True)
    dates = pd.to_datetime(result[column], errors=errors)
    still_missing = dates.isna()
    name = prefix or column

    for component in components:
        target_name = f"{name}_{component}"
        if component == "year":
            result[target_name] = dates.dt.year
        elif component == "month":
            result[target_name] = dates.dt.month
        elif component == "quarter":
            result[target_name] = dates.dt.quarter
        elif component == "day":
            result[target_name] = dates.dt.day
        elif component == "dayofweek":
            result[target_name] = dates.dt.dayofweek
        elif component == "weekofyear":
            result[target_name] = dates.dt.isocalendar().week.astype("Int64")
            result.loc[still_missing, target_name] = pd.NA
        elif component == "hour":
            result[target_name] = dates.dt.hour
        elif component == "is_weekend":
            is_weekend = dates.dt.dayofweek.isin([5, 6])
            result[target_name] = is_weekend.mask(still_missing)
        elif component == "is_month_start":
            result[target_name] = dates.dt.is_month_start.where(~still_missing, other=pd.NA)
        elif component == "is_month_end":
            result[target_name] = dates.dt.is_month_end.where(~still_missing, other=pd.NA)

    return result


def add_cyclical_features(
    frame: pd.DataFrame,
    column: str,
    period: float,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Represent a periodic numeric variable with sine/cosine features.

    Encodes a value like "hour of day" (period 24) or "day of week"
    (period 7) so that the numeric distance between adjacent points
    near the wraparound (e.g. hour 23 and hour 0) is small, unlike the
    raw integer, where they look maximally far apart.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    column:
        Numeric column to encode.
    period:
        The cycle length in the same units as ``column`` (e.g. ``24``
        for hour-of-day, ``7`` for day-of-week, ``12`` for month).
    suffix:
        Prefix for the two new column names (``f"{suffix}_sin"``,
        ``f"{suffix}_cos"``). Defaults to ``column``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with two added columns.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or the column is not numeric.
    KeyError
        If the column does not exist.
    ValueError
        If ``period`` is not a positive number.

    Assumptions
    -----------
    - Values are assumed to already be on the stated cycle (e.g.
      hour-of-day values in ``[0, 24)``) — this function does not
      validate or wrap out-of-range values, it simply applies
      ``sin``/``cos`` of ``2*pi*value/period`` regardless.
    - Missing values propagate to missing ``sin``/``cos`` output.
    - The two output columns are meant to be used **together** as a
      pair; a model given only ``_sin`` or only ``_cos`` loses the
      ability to distinguish some pairs of points on the cycle.
    """
    _validate_columns_exist(frame, [column])
    _validate_numeric_columns(frame, [column])
    if not isinstance(period, (int, float, np.number)) or float(period) <= 0:
        raise ValueError(f"period must be a positive number, got {period}")

    result = frame.copy(deep=True)
    label = suffix or column
    angle = 2 * np.pi * result[column] / float(period)
    result[f"{label}_sin"] = np.sin(angle)
    result[f"{label}_cos"] = np.cos(angle)
    return result


def add_frequency_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    normalize: bool = True,
    suffix: str = "_frequency",
) -> pd.DataFrame:
    """Add category-frequency features without replacing the source columns.

    Unlike ``CategoricalEncoder(method="frequency")``, which replaces
    each source column with its frequency, this function adds a new
    column alongside the original — useful when you want the
    frequency as a signal while keeping the raw category around for
    inspection or for a different encoding elsewhere in your pipeline.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    columns:
        Columns to compute frequency features for.
    normalize:
        If ``True`` (default), frequency is a proportion of rows in
        ``frame`` (between 0 and 1). If ``False``, it is the raw
        count.
    suffix:
        Suffix appended to each source column's name for the new
        column.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one added ``f"{column}{suffix}"`` column
        per requested column.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If a column does not exist.

    Assumptions
    -----------
    - This is a one-shot function, not a stateful transformer — it
      computes frequencies fresh from whatever ``frame`` you pass each
      call. For a train/inference split where you need the *training*
      frequencies applied to new data, use
      ``CategoricalEncoder(method="frequency")`` instead (which has
      fit/transform and handles unseen categories explicitly), even
      though it replaces rather than augments the source column.
    - Missing values are treated as their own category and given
      their own frequency (via ``dropna=False``), not dropped.
    """
    _validate_columns_exist(frame, columns)
    result = frame.copy(deep=True)

    for column in columns:
        counts = result[column].value_counts(dropna=False)
        values = result[column].map(counts)
        if normalize:
            values = values / len(result)
        result[f"{column}{suffix}"] = values.astype(float)
    return result


# ---------------------------------------------------------------------------
# Filter-style feature selection
# ---------------------------------------------------------------------------

def low_variance_features(
    frame: pd.DataFrame,
    threshold: float = 0.0,
    columns: Sequence[str] | None = None,
    ddof: int = 1,
) -> list[str]:
    """Return numeric column names whose variance is at or below a threshold.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    threshold:
        Variance threshold. Columns with variance ``<= threshold`` are
        returned. The default, ``0.0``, flags only truly constant
        columns.
    columns:
        Numeric columns to consider. Defaults to every numeric column.
    ddof:
        Delta degrees of freedom for the variance calculation.
        ``1`` (default) is pandas' sample variance; ``0`` is
        population variance. Only affects the computed value, not
        which columns are considered — pick whichever convention your
        downstream analysis expects.

    Returns
    -------
    list[str]
        Names of numeric columns at or below the variance threshold,
        in the DataFrame's original column order.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or an explicitly requested
        column is not numeric.
    KeyError
        If an explicitly requested column does not exist.
    ValueError
        If ``threshold`` is negative, or ``columns`` contains
        duplicate names.

    Assumptions
    -----------
    - A column that is entirely missing, or has too few non-null
      values to compute variance under the given ``ddof``, is treated
      as having variance ``0.0`` and will be flagged at the default
      threshold — such a column carries no usable variation for
      modeling, the same practical conclusion as a constant column.
    - Only numeric columns are considered; categorical/text columns
      are never included in the result, regardless of how many
      distinct values they have.
    """
    numeric_columns = _resolve_numeric_columns(frame, columns)
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    variances = frame[numeric_columns].var(axis=0, ddof=ddof, skipna=True).fillna(0.0)
    return variances[variances <= threshold].index.tolist()


def drop_low_variance_features(
    frame: pd.DataFrame,
    threshold: float = 0.0,
    columns: Sequence[str] | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with low-variance numeric columns removed.

    Parameters
    ----------
    frame, threshold, columns, ddof:
        See :func:`low_variance_features`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame without the columns identified by
        :func:`low_variance_features`.

    Raises
    ------
    See :func:`low_variance_features`.

    Assumptions
    -----------
    Identical to :func:`low_variance_features`.
    """
    columns_to_drop = low_variance_features(frame, threshold=threshold, columns=columns, ddof=ddof)
    return frame.drop(columns=columns_to_drop).copy(deep=True)


def find_correlated_feature_pairs(
    frame: pd.DataFrame,
    threshold: float = 0.95,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Identify pairs of numeric features that are highly correlated.

    Parameters
    ----------
    frame:
        Input pandas DataFrame.
    threshold:
        Absolute correlation value at or above which a pair is
        reported. Must be between 0 and 1 inclusive.
    method:
        Correlation method: ``"pearson"``, ``"spearman"``, or
        ``"kendall"``.
    columns:
        Numeric columns to consider. Defaults to every numeric column.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature_1``, ``feature_2``, ``correlation`` (the
        absolute correlation value), one row per pair found. Empty
        (with these columns) if no pair meets the threshold or fewer
        than two numeric columns exist.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame, or an explicitly requested
        column is not numeric.
    KeyError
        If an explicitly requested column does not exist.
    ValueError
        If ``threshold`` is outside ``[0, 1]``, ``method`` is
        unsupported, or ``columns`` contains duplicate names.

    Assumptions
    -----------
    - Correlation is compared on the **absolute value** — strongly
      negatively correlated pairs (e.g. correlation of -0.99) are
      reported just as a strongly positive pair would be, since both
      indicate redundant information.
    - Each unordered pair is reported once (``feature_1`` appears
      before ``feature_2`` in the DataFrame's original column order),
      not twice in both directions.
    - Pairs involving a column with undefined correlation (e.g. a
      constant column, which produces ``NaN`` correlation with
      everything) are never reported as "correlated".
    """
    _validate_dataframe(frame)

    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    valid_methods = {"pearson", "kendall", "spearman"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}, got {method!r}")

    numeric_columns = _resolve_numeric_columns(frame, columns) if columns is not None else (
        frame.select_dtypes(include="number").columns.tolist()
    )
    empty_result = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    if len(numeric_columns) < 2:
        return empty_result

    correlation = frame[numeric_columns].corr(method=method).abs()

    records = []
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            value = correlation.iloc[i, j]
            if pd.notna(value) and value >= threshold:
                records.append(
                    {
                        "feature_1": numeric_columns[i],
                        "feature_2": numeric_columns[j],
                        "correlation": float(value),
                    }
                )

    if not records:
        return empty_result
    return pd.DataFrame(records, columns=["feature_1", "feature_2", "correlation"])


def drop_correlated_features(
    frame: pd.DataFrame,
    threshold: float = 0.95,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with redundant correlated columns removed.

    Parameters
    ----------
    frame, threshold, method, columns:
        See :func:`find_correlated_feature_pairs`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with one column from each highly-correlated
        pair removed.

    Raises
    ------
    See :func:`find_correlated_feature_pairs`.

    Assumptions
    -----------
    - For every reported pair, this function **always keeps the
      first column and drops the second**, in the DataFrame's original
      column order. This is a deterministic but arbitrary rule — it
      does not consider which column is more predictive of any target,
      more interpretable, or cheaper to compute at inference. Review
      the dropped columns before finalizing a feature set.
    - If column A is correlated with both B and C above the
      threshold, only A is kept and both B and C are dropped, even if
      B and C are not correlated with each other.
    """
    pairs = find_correlated_feature_pairs(frame, threshold=threshold, method=method, columns=columns)

    columns_to_drop: set[str] = set()
    for _, row in pairs.iterrows():
        if row["feature_1"] not in columns_to_drop:
            columns_to_drop.add(row["feature_2"])

    return frame.drop(columns=list(columns_to_drop)).copy(deep=True)