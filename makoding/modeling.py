"""Production-oriented supervised modelling utilities for Makoding DataLab Pro.

This module provides a single, consistent modelling layer built on top of
scikit-learn (with optional XGBoost / LightGBM / CatBoost support). It is
organized into three layers:

1. Low-level functional helpers (``fit_model``, ``evaluate_classification``,
   ``cross_validate_model``, ...) that operate directly on
   ``pandas.DataFrame`` / ``pandas.Series`` and plain arrays.

2. ``Model`` — a stateful, DataFrame-first wrapper around a single
   scikit-learn-compatible estimator that remembers its feature schema,
   target name, and fitted class labels, and exposes ``save``/``load``
   for round-tripping a fitted model to disk.

3. Comparison and tuning helpers (``compare_models``, ``grid_search``,
   ``random_search``) for choosing between candidate models/hyperparameters
   under a consistent cross-validation protocol.

Design boundary
----------------
This module deliberately does not perform implicit preprocessing. Feature
engineering / preprocessing should be fitted on training data only and then
reused to transform validation, test, and inference data, e.g. via a
separate ``FeaturePipeline``:

    raw data -> train/test split -> FeaturePipeline.fit_transform(train)
                                     FeaturePipeline.transform(test)
                                  -> estimator -> evaluation

This keeps learned preprocessing state from leaking test-set information
into training.
"""
from __future__ import annotations

import hashlib
import importlib
import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "Model",
    "ModelResult",
    "ModelEvaluation",
    "ModelArtifact",
    "ModelComparison",
    "NotFittedError",
    "split_train_test",
    "train_test_split_data",
    "fit_model",
    "predict",
    "evaluate_classification",
    "evaluate_regression",
    "cross_validate_model",
    "generate_oof_predictions",
    "get_feature_importance",
    "permutation_feature_importance",
    "compare_models",
    "grid_search",
    "random_search",
    "calibrate_model",
    "find_optimal_threshold",
    "classification_diagnostics",
    "regression_diagnostics",
    "dataset_fingerprint",
    "compute_artifact_checksum",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
VALID_TASKS = {TASK_CLASSIFICATION, TASK_REGRESSION}
DEFAULT_SCORING = {TASK_CLASSIFICATION: "accuracy", TASK_REGRESSION: "r2"}

_ARTIFACT_FORMAT = "makoding-model"
_ARTIFACT_FORMAT_VERSIONS = {1}
_UNSET = object()  # sentinel distinguishing "not passed" from "passed as None"


class NotFittedError(SklearnNotFittedError):
    """Raised when a model operation requires a fitted estimator."""


# ---------------------------------------------------------------------------
# Result / artifact containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelResult:
    """Container returned by :func:`fit_model`.

    Parameters
    ----------
    model:
        Fitted estimator.
    metrics:
        Evaluation metrics computed against ``X_eval``/``y_eval`` if those
        were supplied to :func:`fit_model`; empty otherwise.
    """

    model: Any
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvaluation:
    """Structured evaluation result bundling metrics with raw outputs.

    Parameters
    ----------
    task:
        ``"classification"`` or ``"regression"``.
    metrics:
        Calculated evaluation metrics.
    predictions:
        Model predictions.
    probabilities:
        Class probabilities (classification only), when available.
    y_true:
        Observed target values.
    """

    task: str
    metrics: Mapping[str, float]
    predictions: Any
    probabilities: Any | None = None
    y_true: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        return {
            "task": self.task,
            "metrics": dict(self.metrics),
            "predictions": np.asarray(self.predictions).tolist(),
            "probabilities": (
                np.asarray(self.probabilities).tolist()
                if self.probabilities is not None
                else None
            ),
            "y_true": (
                np.asarray(self.y_true).tolist()
                if self.y_true is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ModelArtifact:
    """Metadata describing a trained model, for reporting and lineage.

    The artifact retains a reference to the fitted estimator but is
    primarily meant as a serialization-friendly *description* of a
    training event (for reporting/lineage) — not a persisted file. For an
    actual reloadable model file on disk, use :meth:`Model.save` /
    :meth:`Model.load`, which write the estimator itself (via joblib)
    alongside the wrapper's schema.
    """

    model: Any
    task: str
    target: str
    feature_columns: tuple[str, ...]
    model_name: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    random_state: int | None = None
    training_rows: int | None = None
    feature_count: int = 0
    created_at: str = ""
    package_version: str = "0.1.0"
    python_version: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a serialization-friendly artifact summary."""
        return {
            "task": self.task,
            "target": self.target,
            "feature_columns": list(self.feature_columns),
            "feature_count": self.feature_count,
            "model_name": self.model_name,
            "metrics": dict(self.metrics),
            "random_state": self.random_state,
            "training_rows": self.training_rows,
            "created_at": self.created_at,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelComparison:
    """Structured result from comparing multiple models.

    ``results`` is sorted best-first according to ``mean_score`` (assuming
    the higher-is-better scikit-learn scoring convention, which holds for
    all built-in scorers including ``neg_*`` variants).
    """

    task: str
    target: str
    results: pd.DataFrame
    models: Mapping[str, Any]

    def best_model(
        self,
        metric: str | None = None,
        higher_is_better: bool = True,
    ) -> tuple[str, float]:
        """Return ``(model_name, metric_value)`` for the best candidate."""
        if self.results.empty:
            raise ValueError("No model comparison results are available.")

        metric = metric or "mean_score"

        if metric not in self.results.columns:
            raise ValueError(
                f"Metric {metric!r} is not present in comparison results."
            )

        values = pd.to_numeric(self.results[metric], errors="coerce")

        if values.notna().sum() == 0:
            raise ValueError(
                f"No finite values are available for metric {metric!r}."
            )

        index = values.idxmax() if higher_is_better else values.idxmin()
        row = self.results.loc[index]

        return str(row["model"]), float(row[metric])


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_task(task: str) -> str:
    """Validate a modelling task string."""
    if task not in VALID_TASKS:
        raise ValueError(
            f"task must be one of {sorted(VALID_TASKS)}; got {task!r}."
        )
    return task


def _validate_cv_folds(cv: int) -> int:
    """Validate a cross-validation fold count."""
    if isinstance(cv, bool) or not isinstance(cv, (int, np.integer)) or cv < 2:
        raise ValueError("cv must be an integer of at least 2.")
    return int(cv)


def _validate_test_size(value: float) -> float:
    """Validate a train/test split proportion."""
    if isinstance(value, bool):
        raise TypeError("test_size must be numeric.")
    value = float(value)
    if not 0 < value < 1:
        raise ValueError("test_size must be between 0 and 1.")
    return value


def _validate_random_state(value: int | None) -> int | None:
    """Validate a random_state argument (rejecting bool)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return int(value)


def _validate_n_jobs(value: int | None) -> int | None:
    """Validate an n_jobs argument (rejecting bool and zero)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("n_jobs must be an integer or None.")
    if value == 0:
        raise ValueError("n_jobs must not be 0.")
    return int(value)


def _validate_dataframe(X: Any) -> pd.DataFrame:
    """Validate a DataFrame's basic shape (non-empty, unique columns)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X must not be empty.")
    if X.columns.duplicated().any():
        raise ValueError("X contains duplicate column names.")
    return X


def _validate_numeric_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame is fully numeric, complete, and finite."""
    X = _validate_dataframe(X)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(
            f"X must contain numeric columns. Non-numeric columns: "
            f"{non_numeric}."
        )

    if X.isna().any().any():
        raise ValueError("X must not contain missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X must contain only finite numeric values.")

    return X


def _validate_target(y: Sequence[Any], task: str) -> np.ndarray:
    """Validate a target vector for the given task."""
    arr = np.asarray(y)

    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("y must be a non-empty one-dimensional sequence.")

    if pd.isna(arr).any():
        raise ValueError("y must not contain missing values.")

    if task == TASK_REGRESSION:
        try:
            values = arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError("Regression targets must be numeric.") from exc

        if not np.isfinite(values).all():
            raise ValueError("Regression targets must be finite.")

    elif len(np.unique(arr)) < 2:
        raise ValueError(
            "Classification target must contain at least two classes."
        )

    return arr


def _validate_X_y(
    X: pd.DataFrame,
    y: Sequence[Any],
    task: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate a feature/target pair together."""
    _validate_task(task)
    X = _validate_numeric_dataframe(X)
    y = _validate_target(y, task)

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    return X, y


def _validate_sample_weight(
    weights: Sequence[float] | None,
    n: int,
) -> np.ndarray | None:
    """Validate an optional sample-weight vector."""
    if weights is None:
        return None

    arr = np.asarray(weights, dtype=float)

    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(
            "sample_weight must be one-dimensional and match y length."
        )

    if not np.isfinite(arr).all() or (arr < 0).any() or arr.sum() <= 0:
        raise ValueError(
            "sample_weight must be finite, non-negative, and have "
            "positive total weight."
        )

    return arr


def _prepare_prediction_frame(
    X: pd.DataFrame,
    expected: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate and (optionally) reorder a frame for prediction."""
    X = _validate_numeric_dataframe(X)

    if expected is not None:
        expected = list(expected)
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]

        if missing:
            raise ValueError(f"Missing feature columns: {missing}.")
        if extra:
            raise ValueError(f"Unexpected feature columns: {extra}.")

        X = X.loc[:, expected]

    return X


def _is_fitted(estimator: Any) -> bool:
    """Return whether an estimator passes sklearn fitted-state checks."""
    if estimator is None:
        return False
    try:
        check_is_fitted(estimator)
        return True
    except (SklearnNotFittedError, TypeError, AttributeError):
        return False


def _require_fitted(estimator: Any) -> None:
    """Raise :class:`NotFittedError` unless ``estimator`` is fitted."""
    if not _is_fitted(estimator):
        raise NotFittedError("Model is not fitted.")


# ---------------------------------------------------------------------------
# Estimator resolution
# ---------------------------------------------------------------------------

def _build_estimator(
    task: str,
    model_name: str,
    random_state: int | None = 42,
    **params: Any,
) -> Any:
    """Build a supported estimator from a simple model name.

    Core scikit-learn models are always available. XGBoost, LightGBM, and
    CatBoost are optional dependencies imported lazily on first use.
    Extra ``params`` are forwarded to the estimator constructor, on top of
    sensible defaults (e.g. ``n_estimators=200``), so callers can request
    e.g. ``model_name="random_forest", n_estimators=500`` directly.
    """
    _validate_task(task)
    random_state = _validate_random_state(random_state)

    name = model_name.strip().lower().replace("-", "_").replace(" ", "_")

    if task == TASK_CLASSIFICATION:
        registry = {
            "logistic_regression": LogisticRegression,
            "logistic": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "random_forest_classifier": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "gradient_boosting_classifier": GradientBoostingClassifier,
        }
    else:
        registry = {
            "linear_regression": LinearRegression,
            "linear": LinearRegression,
            "random_forest": RandomForestRegressor,
            "random_forest_regressor": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "gradient_boosting_regressor": GradientBoostingRegressor,
        }

    if name in registry:
        estimator_cls = registry[name]

        if estimator_cls is LogisticRegression:
            params.setdefault("max_iter", 1000)

        accepts_random_state = "random_state" in estimator_cls().get_params()
        if accepts_random_state and random_state is not None:
            params.setdefault("random_state", random_state)

        return estimator_cls(**params)

    if name in {"xgboost", "xgb"}:
        try:
            module = importlib.import_module("xgboost")
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required to use model_name='xgboost'. "
                "Install it with: pip install xgboost"
            ) from exc

        estimator_cls = (
            module.XGBClassifier
            if task == TASK_CLASSIFICATION
            else module.XGBRegressor
        )
        params.setdefault("n_estimators", 200)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("n_jobs", 1)
        if random_state is not None:
            params.setdefault("random_state", random_state)
        if task == TASK_CLASSIFICATION:
            params.setdefault("eval_metric", "logloss")

        return estimator_cls(**params)

    if name in {"lightgbm", "light_gbm", "lgbm"}:
        try:
            module = importlib.import_module("lightgbm")
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required to use model_name='lightgbm'. "
                "Install it with: pip install lightgbm"
            ) from exc

        estimator_cls = (
            module.LGBMClassifier
            if task == TASK_CLASSIFICATION
            else module.LGBMRegressor
        )
        params.setdefault("n_estimators", 200)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("verbosity", -1)
        if random_state is not None:
            params.setdefault("random_state", random_state)

        return estimator_cls(**params)

    if name in {"catboost", "cat_boost"}:
        try:
            module = importlib.import_module("catboost")
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required to use model_name='catboost'. "
                "Install it with: pip install catboost"
            ) from exc

        estimator_cls = (
            module.CatBoostClassifier
            if task == TASK_CLASSIFICATION
            else module.CatBoostRegressor
        )
        params.setdefault("iterations", 300)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("verbose", False)
        if random_state is not None:
            params.setdefault("random_seed", random_state)

        return estimator_cls(**params)

    raise ValueError(f"Unknown model_name {model_name!r} for task {task!r}.")


def _resolve_estimator(
    task: str,
    model: Any,
    random_state: int | None = 42,
    **params: Any,
) -> Any:
    """Resolve ``model`` to a fresh, unfitted estimator.

    ``model`` may be a registry name (see :func:`_build_estimator`) or an
    already-instantiated scikit-learn-compatible estimator, in which case
    it is validated against ``task`` and cloned so callers always receive
    an independent, unfitted instance.
    """
    if isinstance(model, str):
        return _build_estimator(task, model, random_state, **params)

    if hasattr(model, "fit"):
        _validate_estimator_task(model, task)
        fresh = clone(model)
        if params:
            fresh.set_params(**params)
        return fresh

    raise TypeError(
        "model must be either a registered model name (str) or a "
        "scikit-learn-compatible estimator instance."
    )


def _validate_estimator_task(estimator: Any, task: str) -> None:
    """Validate that an estimator's type matches the requested task."""
    _validate_task(task)

    if task == TASK_CLASSIFICATION and is_regressor(estimator):
        raise TypeError(
            f"Estimator {type(estimator).__name__!r} is a regressor but "
            "task='classification' was requested."
        )

    if task == TASK_REGRESSION and is_classifier(estimator):
        raise TypeError(
            f"Estimator {type(estimator).__name__!r} is a classifier but "
            "task='regression' was requested."
        )


def _routed_cross_validate(estimator, X, y, *, fit_params, **kwargs):
    """``cross_validate`` with a clear error if weight routing fails.

    ``sample_weight`` passthrough relies on scikit-learn's metadata-routing
    ``params=`` argument, whose exact behavior/requirements have changed
    across sklearn versions (tested here against sklearn %s). If routing
    fails, surface a message that names the actual cause instead of
    sklearn's internal routing traceback.
    """
    try:
        return cross_validate(estimator, X, y, params=fit_params, **kwargs)
    except TypeError as exc:
        if fit_params:
            raise TypeError(
                "sample_weight could not be routed to cross_validate on "
                f"sklearn {sklearn.__version__} (developed/tested against "
                "sklearn 1.8). Either upgrade/pin sklearn, or retry without "
                f"sample_weight. Original error: {exc}"
            ) from exc
        raise


def _routed_cross_val_predict(estimator, X, y, *, fit_params, **kwargs):
    """``cross_val_predict`` with a clear error if weight routing fails."""
    try:
        return cross_val_predict(estimator, X, y, params=fit_params, **kwargs)
    except TypeError as exc:
        if fit_params:
            raise TypeError(
                "sample_weight could not be routed to cross_val_predict on "
                f"sklearn {sklearn.__version__} (developed/tested against "
                "sklearn 1.8). Either upgrade/pin sklearn, or retry without "
                f"sample_weight. Original error: {exc}"
            ) from exc
        raise


def _resolve_cv(
    task: str,
    cv: Any,
    random_state: int | None,
):
    """Resolve ``cv`` to a scikit-learn cross-validator.

    ``cv`` may be an integer fold count (stratified for classification,
    plain K-fold for regression) or an already-built splitter object (e.g.
    ``TimeSeriesSplit``, ``GroupKFold``) with a ``split`` method, which is
    passed through unchanged.
    """
    if hasattr(cv, "split"):
        return cv

    folds = _validate_cv_folds(cv)
    random_state = _validate_random_state(random_state)

    if task == TASK_CLASSIFICATION:
        return StratifiedKFold(folds, shuffle=True, random_state=random_state)

    return KFold(folds, shuffle=True, random_state=random_state)


# ---------------------------------------------------------------------------
# Dataset lineage
# ---------------------------------------------------------------------------

def dataset_fingerprint(X: pd.DataFrame, y: Sequence[Any] | None = None) -> str:
    """Return a stable hash summarizing a dataset's content and dtypes.

    Useful for recording exactly which data a model artifact was trained
    on, without storing the data itself.

    Scope: this targets *validated modelling datasets* — numeric, complete
    DataFrames of the kind this module already requires before fitting or
    evaluating (see ``_validate_numeric_dataframe``) — not arbitrary
    DataFrames. Everywhere this module calls ``dataset_fingerprint``
    internally (``Model.fit``), ``X``/``y`` have already passed that
    validation. Called directly on a frame with unhashable object-dtype
    values (e.g. lists or dicts in a cell), it raises a clear ``TypeError``
    rather than propagating pandas' underlying hashing error.
    """
    X = _validate_dataframe(X)

    digest = hashlib.sha256()
    digest.update(
        "|".join(f"{c}:{d}" for c, d in X.dtypes.items()).encode()
    )

    try:
        digest.update(
            pd.util.hash_pandas_object(X, index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        )
    except TypeError as exc:
        raise TypeError(
            "dataset_fingerprint could not hash X: it contains values "
            "pandas cannot hash (e.g. lists/dicts in object columns). "
            "This function targets validated, numeric modelling datasets, "
            "not arbitrary DataFrames."
        ) from exc

    if y is not None:
        try:
            digest.update(
                pd.util.hash_pandas_object(pd.Series(y), index=True)
                .to_numpy(dtype=np.uint64)
                .tobytes()
            )
        except TypeError as exc:
            raise TypeError(
                "dataset_fingerprint could not hash y: it contains "
                "unhashable values."
            ) from exc

    return digest.hexdigest()


def compute_artifact_checksum(path: str | Path) -> str:
    """Return the sha256 checksum of a saved model artifact's raw bytes.

    Compute this right after :meth:`Model.save` and record/transmit the
    checksum through a channel independent of the file itself (e.g.
    commit it alongside code, or a signed manifest). Passing it back to
    :meth:`Model.load` via ``expected_sha256`` verifies the file's raw
    bytes *before* any deserialization is attempted — a file that has
    been swapped or tampered with in transit is rejected up front, before
    joblib/pickle ever executes anything from it.

    This does not make loading an artifact from an *originally* malicious
    or untrusted source safe — joblib/pickle deserialization can execute
    arbitrary code regardless of checksum, so only load artifacts whose
    origin you trust. It protects against post-save tampering, not
    against a bad artifact you were given in the first place.
    """
    path = Path(path)
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Train/test splitting
# ---------------------------------------------------------------------------

def split_train_test(
    X: pd.DataFrame,
    y: Sequence[Any],
    test_size: float = 0.20,
    random_state: int | None = 42,
    stratify: bool = False,
    task: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split predictors and target into train/test partitions.

    ``task`` only affects target validation; if omitted it is inferred
    as ``"classification"`` when ``stratify=True``, else ``"regression"``.
    """
    task = task or (TASK_CLASSIFICATION if stratify else TASK_REGRESSION)
    X, y = _validate_X_y(X, y, task)

    y_series = pd.Series(y, index=X.index, name=getattr(y, "name", "target"))
    stratify_values = y_series if stratify else None

    return train_test_split(
        X,
        y_series,
        test_size=_validate_test_size(test_size),
        random_state=_validate_random_state(random_state),
        stratify=stratify_values,
    )


def train_test_split_data(
    X: pd.DataFrame,
    y: Sequence[Any],
    test_size: float = 0.20,
    random_state: int | None = 42,
    stratify: bool = False,
    task: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Alias for :func:`split_train_test`."""
    return split_train_test(X, y, test_size, random_state, stratify, task)


# ---------------------------------------------------------------------------
# Functional fit / predict / evaluate
# ---------------------------------------------------------------------------

def fit_model(
    X: pd.DataFrame,
    y: Sequence[Any],
    task: str,
    model_name: str | Any,
    random_state: int | None = 42,
    sample_weight: Sequence[float] | None = None,
    metrics: Sequence[str] | None = None,
    X_eval: pd.DataFrame | None = None,
    y_eval: Sequence[Any] | None = None,
    **model_params: Any,
) -> ModelResult:
    """Fit a model and return a :class:`ModelResult`.

    ``model_name`` may be a registry name (e.g. ``"random_forest"``) or an
    already-instantiated estimator. If ``X_eval``/``y_eval`` are supplied,
    the fitted model is immediately evaluated against them and the
    resulting metrics are attached to the returned :class:`ModelResult`.
    """
    X, y = _validate_X_y(X, y, task)
    weights = _validate_sample_weight(sample_weight, len(X))

    estimator = _resolve_estimator(task, model_name, random_state, **model_params)

    logger.info(
        "fit_model: fitting %s (task=%s) on %d rows / %d features",
        type(estimator).__name__,
        task,
        len(X),
        X.shape[1],
    )

    if weights is None:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, sample_weight=weights)

    evaluated: dict[str, float] = {}

    if X_eval is not None or y_eval is not None:
        if X_eval is None or y_eval is None:
            raise ValueError("X_eval and y_eval must be supplied together.")

        X_eval = _prepare_prediction_frame(X_eval, X.columns)
        # Training, validation, and inference data all go through the same
        # validation boundary — y_eval is not exempt just because
        # downstream metrics would eventually surface most problems with
        # it anyway.
        y_eval = _validate_target(y_eval, task)
        if len(X_eval) != len(y_eval):
            raise ValueError(
                "X_eval and y_eval must contain the same number of rows."
            )

        pred = estimator.predict(X_eval)

        if task == TASK_CLASSIFICATION:
            proba = (
                estimator.predict_proba(X_eval)
                if hasattr(estimator, "predict_proba")
                else None
            )
            if proba is not None and np.asarray(proba).shape[1] == 2:
                proba = np.asarray(proba)[:, 1]

            evaluated = evaluate_classification(y_eval, pred, proba, metrics=metrics)
        else:
            evaluated = evaluate_regression(y_eval, pred, metrics=metrics)

    return ModelResult(estimator, evaluated)


def predict(model: Any, X: pd.DataFrame) -> pd.Series:
    """Generate predictions from a fitted estimator."""
    _require_fitted(model)
    X = _prepare_prediction_frame(X, getattr(model, "feature_columns_", None))
    return pd.Series(model.predict(X), index=X.index, name="prediction")


def evaluate_classification(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_probability: Sequence[float] | None = None,
    metrics: Sequence[str] | None = None,
    average: str = "weighted",
    zero_division: int = 0,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, float]:
    """Calculate classification metrics.

    ``accuracy``/``balanced_accuracy``/``precision``/``recall``/``f1``
    support binary and multiclass targets. ``roc_auc`` and
    ``average_precision`` require ``y_probability`` — a 1D vector of
    positive-class probabilities for binary targets, or a 2D array of
    per-class probabilities (evaluated one-vs-rest) for multiclass.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred):
        raise ValueError(
            "y_true and y_pred must be one-dimensional and have equal "
            "length."
        )

    weights = _validate_sample_weight(sample_weight, len(true))
    requested = list(metrics) if metrics is not None else [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    out: dict[str, float] = {}

    for metric in requested:
        if metric == "accuracy":
            out[metric] = float(
                accuracy_score(true, pred, sample_weight=weights)
            )
        elif metric == "balanced_accuracy":
            out[metric] = float(
                balanced_accuracy_score(true, pred, sample_weight=weights)
            )
        elif metric == "precision":
            out[metric] = float(
                precision_score(
                    true, pred, average=average, zero_division=zero_division,
                    sample_weight=weights,
                )
            )
        elif metric == "recall":
            out[metric] = float(
                recall_score(
                    true, pred, average=average, zero_division=zero_division,
                    sample_weight=weights,
                )
            )
        elif metric == "f1":
            out[metric] = float(
                f1_score(
                    true, pred, average=average, zero_division=zero_division,
                    sample_weight=weights,
                )
            )
        elif metric in {"roc_auc", "average_precision"}:
            if y_probability is None:
                raise ValueError(f"y_probability is required for {metric}.")

            proba = np.asarray(y_probability, dtype=float)

            if metric == "roc_auc":
                if proba.ndim == 1:
                    out[metric] = float(
                        roc_auc_score(true, proba, sample_weight=weights)
                    )
                else:
                    out[metric] = float(
                        roc_auc_score(
                            true, proba, multi_class="ovr", average=average,
                            sample_weight=weights,
                        )
                    )
            else:
                if proba.ndim != 1:
                    raise ValueError(
                        "average_precision requires a binary probability "
                        "vector."
                    )
                out[metric] = float(
                    average_precision_score(true, proba, sample_weight=weights)
                )
        else:
            raise ValueError(f"Unsupported classification metric: {metric!r}.")

    return out


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metrics: Sequence[str] | None = None,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, float]:
    """Calculate standard regression metrics."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred):
        raise ValueError(
            "y_true and y_pred must be one-dimensional and have equal "
            "length."
        )

    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Regression values must be finite.")

    weights = _validate_sample_weight(sample_weight, len(true))
    requested = list(metrics) if metrics is not None else [
        "mae",
        "mse",
        "rmse",
        "r2",
    ]

    out: dict[str, float] = {}

    for metric in requested:
        if metric == "mae":
            out[metric] = float(
                mean_absolute_error(true, pred, sample_weight=weights)
            )
        elif metric == "mse":
            out[metric] = float(
                mean_squared_error(true, pred, sample_weight=weights)
            )
        elif metric == "rmse":
            out[metric] = float(
                np.sqrt(mean_squared_error(true, pred, sample_weight=weights))
            )
        elif metric == "r2":
            out[metric] = float(r2_score(true, pred, sample_weight=weights))
        else:
            raise ValueError(f"Unsupported regression metric: {metric!r}.")

    return out


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    X: pd.DataFrame,
    y: Sequence[Any],
    task: str,
    model_name: str | Any,
    cv: int | Any = 5,
    scoring: str | Sequence[str] | None = None,
    random_state: int | None = 42,
    n_jobs: int | None = None,
    sample_weight: Sequence[float] | None = None,
    groups: Sequence[Any] | None = None,
    return_train_score: bool = True,
    **model_params: Any,
) -> dict[str, Any]:
    """Cross-validate a model and return aggregate, JSON-friendly results.

    ``cv`` may be an integer fold count or a pre-built splitter (e.g.
    ``TimeSeriesSplit``). ``groups`` is forwarded to the splitter for
    group-aware strategies (e.g. ``GroupKFold``) and ignored otherwise.

    The returned dict always exposes ``metrics`` as
    ``{metric_name: {"scores": [...], "mean": ..., "std": ...}}``
    regardless of whether one or several scoring metrics were requested,
    plus top-level ``scores``/``mean_score``/``std_score`` mirroring the
    first requested metric for convenience.
    """
    X, y = _validate_X_y(X, y, task)
    weights = _validate_sample_weight(sample_weight, len(X))
    scoring = scoring or DEFAULT_SCORING[task]

    estimator = _resolve_estimator(task, model_name, random_state, **model_params)
    cv_strategy = _resolve_cv(task, cv, random_state)

    fit_params = {"sample_weight": weights} if weights is not None else None

    logger.info(
        "cross_validate_model: %s (task=%s), cv=%s, scoring=%s",
        type(estimator).__name__,
        task,
        cv,
        scoring,
    )

    raw = _routed_cross_validate(
        estimator,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=_validate_n_jobs(n_jobs),
        return_train_score=return_train_score,
        groups=groups,
        fit_params=fit_params,
    )

    metric_names = (
        [scoring] if isinstance(scoring, str) else list(scoring)
    )

    metrics: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        key = "test_score" if isinstance(scoring, str) else f"test_{name}"
        scores = np.asarray(raw[key], dtype=float)
        metrics[name] = {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }

    primary = metrics[metric_names[0]]

    return {
        "metrics": metrics,
        "scores": primary["scores"],
        "mean_score": primary["mean"],
        "std_score": primary["std"],
        "fit_time": np.asarray(raw.get("fit_time", [])).tolist(),
        "score_time": np.asarray(raw.get("score_time", [])).tolist(),
        "cv": cv if isinstance(cv, int) else type(cv).__name__,
        "task": task,
        "model": (
            model_name if isinstance(model_name, str)
            else type(estimator).__name__
        ),
    }


def generate_oof_predictions(
    X: pd.DataFrame,
    y: Sequence[Any],
    task: str,
    model_name: str | Any,
    cv: int | Any = 5,
    random_state: int | None = 42,
    method: str = "predict",
    n_jobs: int | None = None,
    sample_weight: Sequence[float] | None = None,
    groups: Sequence[Any] | None = None,
    **model_params: Any,
) -> dict[str, Any]:
    """Generate out-of-fold predictions and evaluate them.

    Useful for stacking/blending and for an unbiased estimate of model
    performance on every row without holding out a separate test set.
    """
    X, y = _validate_X_y(X, y, task)

    if method not in {"predict", "predict_proba"}:
        raise ValueError("method must be 'predict' or 'predict_proba'.")

    weights = _validate_sample_weight(sample_weight, len(X))
    fit_params = {"sample_weight": weights} if weights is not None else None

    estimator = _resolve_estimator(task, model_name, random_state, **model_params)
    cv_strategy = _resolve_cv(task, cv, random_state)

    pred = _routed_cross_val_predict(
        estimator,
        X,
        y,
        cv=cv_strategy,
        method=method,
        n_jobs=_validate_n_jobs(n_jobs),
        groups=groups,
        fit_params=fit_params,
    )

    if task == TASK_CLASSIFICATION:
        if method == "predict_proba":
            proba = np.asarray(pred)
            # cross_val_predict guarantees probability columns follow the
            # sorted-unique-label order (its internal label binarizer),
            # matching np.unique(y) — NOT column positions 0..k-1 as raw
            # class values, so we must map argmax indices back through the
            # actual class labels rather than using them directly.
            class_labels = np.unique(y)
            labels = class_labels[np.argmax(proba, axis=1)]
            positive_proba = proba[:, 1] if proba.shape[1] == 2 else None
        else:
            proba = None
            labels = pred
            positive_proba = None

        metrics = evaluate_classification(y, labels, positive_proba)
    else:
        proba = None
        metrics = evaluate_regression(y, pred)

    return {
        "predictions": pred,
        "metrics": metrics,
        "cv": cv if isinstance(cv, int) else type(cv_strategy).__name__,
        "task": task,
        "model": (
            model_name if isinstance(model_name, str)
            else type(estimator).__name__
        ),
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _native_importance(model: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Extract native (impurity- or coefficient-based) feature importance."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        raise ValueError(
            "Estimator does not expose native feature importance; use "
            "permutation_feature_importance() instead."
        )

    values = values.ravel()

    if len(values) != len(feature_names):
        raise ValueError(
            "Feature importance length does not match feature names."
        )

    return pd.DataFrame(
        {"feature": list(feature_names), "importance": values}
    ).sort_values(
        "importance", ascending=False, kind="stable"
    ).reset_index(drop=True)


def get_feature_importance(
    model: Any,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return native feature importance, sorted descending.

    ``model`` may be a raw fitted scikit-learn estimator or a fitted
    :class:`Model` wrapper (its ``.estimator`` is used automatically,
    matching how :func:`calibrate_model` already handles both).

    ``feature_names`` is used whenever it is not ``None`` — including an
    explicitly empty sequence, which is a real (if unusual) argument, not
    "not provided". The previous ``feature_names or ...`` fallback chain
    conflated the two, and its first fallback,
    ``getattr(model, "feature_columns_", None)``, was dead code in
    practice: ``model`` here is typically a raw estimator, which never
    carries that attribute (only a :class:`Model` wrapper does). Falling
    back now checks the wrapper's schema when ``model`` actually is one,
    then scikit-learn's own ``feature_names_in_`` (set automatically when
    an estimator is fit on a DataFrame with named columns).
    """
    estimator = model.estimator if isinstance(model, Model) else model
    _require_fitted(estimator)

    if feature_names is None:
        if isinstance(model, Model):
            feature_names = model.feature_columns_
        if feature_names is None:
            feature_names = getattr(estimator, "feature_names_in_", None)

    if feature_names is None:
        raise ValueError("feature_names must be supplied for this estimator.")

    return _native_importance(estimator, feature_names)


def permutation_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: Sequence[Any],
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int | None = 42,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Return permutation-based feature importance (model-agnostic).

    Slower than :func:`get_feature_importance` but works for any
    estimator, including linear models with correlated features and
    boosting models where impurity-based importance can be misleading.
    """
    _require_fitted(model)
    X = _prepare_prediction_frame(X, getattr(model, "feature_columns_", None))

    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=int(n_repeats),
        random_state=random_state,
        n_jobs=_validate_n_jobs(n_jobs),
    )

    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(
        "importance_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    X: pd.DataFrame,
    y: Sequence[Any],
    task: str,
    model_names: Sequence[str | Any] | None = None,
    cv: int | Any = 5,
    scoring: str | None = None,
    random_state: int | None = 42,
    n_jobs: int | None = None,
    sample_weight: Sequence[float] | None = None,
    groups: Sequence[Any] | None = None,
    refit: bool = True,
    target: str | None = None,
) -> ModelComparison:
    """Compare multiple candidate models under the same CV protocol.

    Models requiring an uninstalled optional dependency (XGBoost, LightGBM,
    CatBoost) are skipped rather than failing the whole comparison; skipped
    models and the reason are recorded in
    ``comparison.results.attrs["skipped_models"]``.

    Results are sorted best-first by ``mean_score`` (see
    :class:`ModelComparison`).

    Note on ``comparison.models`` when ``refit=True``: each stored
    estimator is fit once on the *entire* ``X``/``y``, not on any single
    CV fold — the ``mean_score``/``std_score`` in ``results`` describe
    cross-validation performance, while ``comparison.models[name]`` is a
    separate, subsequently-refit estimator meant for downstream use (e.g.
    ``comparison.models[comparison.best_model()[0]]``). They are related
    but not literally the same fitted object that produced the CV scores.
    """
    X, y = _validate_X_y(X, y, task)

    if model_names is None:
        model_names = (
            [
                "logistic_regression", "random_forest", "gradient_boosting",
                "xgboost", "lightgbm", "catboost",
            ]
            if task == TASK_CLASSIFICATION
            else [
                "linear_regression", "random_forest", "gradient_boosting",
                "xgboost", "lightgbm", "catboost",
            ]
        )

    model_names = list(model_names)
    name_labels = [
        m if isinstance(m, str) else f"{type(m).__name__}_{i}"
        for i, m in enumerate(model_names)
    ]

    if len(name_labels) != len(set(name_labels)):
        raise ValueError("model_names must not contain duplicate labels.")

    scoring = scoring or DEFAULT_SCORING[task]

    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for label, candidate in zip(name_labels, model_names):
        try:
            result = cross_validate_model(
                X, y, task, candidate, cv=cv, scoring=scoring,
                random_state=random_state, n_jobs=n_jobs,
                sample_weight=sample_weight, groups=groups,
                return_train_score=False,
            )
        except ImportError as exc:
            logger.warning("compare_models: skipping %r (%s)", label, exc)
            skipped[label] = str(exc)
            continue

        rows.append({
            "model": label,
            "mean_score": result["mean_score"],
            "std_score": result["std_score"],
            "cv": result["cv"],
            "scoring": scoring,
        })

        if refit:
            estimator = _resolve_estimator(task, candidate, random_state)
            if sample_weight is not None:
                weights = _validate_sample_weight(sample_weight, len(X))
                estimator.fit(X, y, sample_weight=weights)
            else:
                estimator.fit(X, y)
            models[label] = estimator

    if not rows:
        raise ValueError(f"No candidate models could be evaluated: {skipped}")

    table = pd.DataFrame(rows).sort_values(
        "mean_score", ascending=False, kind="stable"
    ).reset_index(drop=True)
    table.attrs["skipped_models"] = skipped

    logger.info(
        "compare_models: evaluated %d model(s), skipped %d",
        len(rows), len(skipped),
    )

    return ModelComparison(
        task, str(target or getattr(y, "name", "") or ""), table, models
    )


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def grid_search(
    X: pd.DataFrame,
    y: Sequence[Any],
    task: str,
    estimator: Any,
    param_grid: Mapping[str, Any],
    cv: int | Any = 5,
    scoring: str | None = None,
    random_state: int | None = 42,
    n_jobs: int | None = None,
    refit: bool = True,
    **kwargs: Any,
) -> GridSearchCV:
    """Run an exhaustive grid search over ``param_grid``.

    Note: ``sample_weight`` is not threaded through here — scikit-learn's
    metadata-routing setup required for that adds meaningful fragility
    across sklearn versions/scorer configurations for comparatively small
    benefit in a grid search context. Use class/sample weighting baked
    into the estimator (e.g. ``class_weight="balanced"``) instead, or use
    :func:`cross_validate_model`, which does support ``sample_weight``,
    to evaluate a small number of hand-picked configurations.
    """
    X, y = _validate_X_y(X, y, task)

    if not isinstance(param_grid, Mapping) or not param_grid:
        raise TypeError("param_grid must be a non-empty mapping.")

    _validate_estimator_task(estimator, task)

    search = GridSearchCV(
        clone(estimator),
        dict(param_grid),
        scoring=scoring or DEFAULT_SCORING[task],
        cv=_resolve_cv(task, cv, random_state),
        n_jobs=_validate_n_jobs(n_jobs),
        refit=refit,
        **kwargs,
    )

    logger.info(
        "grid_search: %s over %d parameter combination(s)",
        type(estimator).__name__,
        int(np.prod([len(v) for v in param_grid.values()])),
    )

    search.fit(X, y)

    logger.info(
        "grid_search: best_score_=%.6f, best_params_=%s",
        search.best_score_, search.best_params_,
    )

    return search


def random_search(
    X: pd.DataFrame,
    y: Sequence[Any],
    task: str,
    estimator: Any,
    param_distributions: Mapping[str, Any],
    cv: int | Any = 5,
    n_iter: int = 20,
    scoring: str | None = None,
    random_state: int | None = 42,
    n_jobs: int | None = None,
    refit: bool = True,
    **kwargs: Any,
) -> RandomizedSearchCV:
    """Run a randomized hyperparameter search over ``param_distributions``.

    See :func:`grid_search` for why ``sample_weight`` is intentionally not
    supported here.
    """
    X, y = _validate_X_y(X, y, task)

    if (
        not isinstance(n_iter, (int, np.integer))
        or isinstance(n_iter, bool)
        or n_iter < 1
    ):
        raise ValueError("n_iter must be a positive integer.")

    if not isinstance(param_distributions, Mapping) or not param_distributions:
        raise TypeError("param_distributions must be a non-empty mapping.")

    _validate_estimator_task(estimator, task)

    search = RandomizedSearchCV(
        clone(estimator),
        dict(param_distributions),
        n_iter=int(n_iter),
        scoring=scoring or DEFAULT_SCORING[task],
        cv=_resolve_cv(task, cv, random_state),
        random_state=random_state,
        n_jobs=_validate_n_jobs(n_jobs),
        refit=refit,
        **kwargs,
    )

    logger.info(
        "random_search: %s, n_iter=%d", type(estimator).__name__, n_iter,
    )

    search.fit(X, y)

    logger.info(
        "random_search: best_score_=%.6f, best_params_=%s",
        search.best_score_, search.best_params_,
    )

    return search


# ---------------------------------------------------------------------------
# Calibration and thresholding
# ---------------------------------------------------------------------------

def calibrate_model(
    model: Any,
    X: pd.DataFrame,
    y: Sequence[Any],
    method: str = "sigmoid",
    cv: int | Any = 5,
) -> CalibratedClassifierCV:
    """Fit a probability-calibrated wrapper around a fitted classifier.

    ``model`` may be a raw fitted scikit-learn classifier or a fitted
    :class:`Model` wrapper (its ``.estimator`` is used automatically). The
    classifier's hyperparameters are cloned and refit internally by
    ``CalibratedClassifierCV`` using its own cross-validation, so the
    passed-in ``model`` need only be fitted in order to validate its type
    and hyperparameters — its fitted weights are not reused directly.
    """
    raw_estimator = model.estimator if isinstance(model, Model) else model
    _require_fitted(raw_estimator)

    if not is_classifier(raw_estimator):
        raise TypeError("calibrate_model requires a classifier.")

    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'.")

    feature_columns = getattr(model, "feature_columns_", None)
    X = _prepare_prediction_frame(X, feature_columns)

    calibrated = CalibratedClassifierCV(
        estimator=clone(raw_estimator),
        method=method,
        cv=_resolve_cv(TASK_CLASSIFICATION, cv, None),
    )
    calibrated.fit(X, y)

    return calibrated


def find_optimal_threshold(
    y_true: Sequence[Any],
    probabilities: Sequence[float],
    objective: str = "f1",
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    """Find the binary decision threshold maximizing ``objective``.

    When ``thresholds`` is not supplied, candidates are the distinct
    observed probability values (plus 0.0), which is both exact (the
    optimum over any threshold grid is always attained at one of these
    points for a monotone decision rule) and typically far cheaper than a
    fixed dense grid.

    Statistical note: only optimize a threshold against probabilities the
    model did *not* see during training — e.g. out-of-fold predictions
    from :func:`generate_oof_predictions`, or a held-out validation set.
    Optimizing against in-sample (training-set) probabilities will
    overfit the threshold the same way tuning a hyperparameter on the
    training set overfits the model itself.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(probabilities, dtype=float)

    if (
        proba.ndim != 1
        or len(proba) != len(y_true)
        or not np.isfinite(proba).all()
        or ((proba < 0) | (proba > 1)).any()
    ):
        raise ValueError(
            "probabilities must be a finite vector between 0 and 1."
        )

    classes = np.unique(y_true)

    if len(classes) != 2:
        raise ValueError("Threshold optimization requires binary classification.")

    # ``classes[1]`` is treated as the positive class throughout this
    # module (matching predict_proba's column order and roc_auc_score's
    # binary convention), so thresholded predictions must be mapped back
    # through the actual labels rather than assumed to be {0, 1} —
    # y_true may be any two labels (e.g. "cat"/"dog"), not just 0/1.
    negative_label, positive_label = classes[0], classes[1]

    if thresholds is None:
        thresholds = np.unique(np.concatenate([[0.0], proba]))
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    scorers = {
        "f1": lambda a, b: f1_score(a, b, pos_label=positive_label, zero_division=0),
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "recall": lambda a, b: recall_score(a, b, pos_label=positive_label, zero_division=0),
        "precision": lambda a, b: precision_score(a, b, pos_label=positive_label, zero_division=0),
    }

    if objective not in scorers:
        raise ValueError(f"Unsupported threshold objective: {objective!r}.")

    best_score, best_threshold = -np.inf, 0.5

    for threshold in thresholds:
        predicted_labels = np.where(proba >= threshold, positive_label, negative_label)
        score = float(scorers[objective](y_true, predicted_labels))
        if score > best_score:
            best_score, best_threshold = score, float(threshold)

    return {
        "threshold": best_threshold,
        "score": best_score,
        "objective": objective,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def classification_diagnostics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_probability: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return a compact diagnostic bundle for a classification result."""
    requested = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    if y_probability is not None:
        requested.append("roc_auc")

    metrics = evaluate_classification(y_true, y_pred, y_probability, requested)

    classes = np.unique(y_true)

    return {
        "metrics": metrics,
        "n_observations": len(y_true),
        "classes": classes.tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=classes
        ).tolist(),
    }


def regression_diagnostics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, Any]:
    """Return a compact diagnostic bundle for a regression result."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residuals = true - pred

    return {
        "metrics": evaluate_regression(true, pred),
        "n_observations": len(true),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
    }


# ---------------------------------------------------------------------------
# Object-oriented Model wrapper
# ---------------------------------------------------------------------------

class Model:
    """Stateful, DataFrame-first wrapper around a single estimator.

    Remembers the feature schema and target used during ``fit`` and
    validates that data passed to ``predict``/``evaluate`` matches it.
    Supports persistence via :meth:`save`/:meth:`load`.
    """

    def __init__(
        self,
        task: str,
        estimator: Any | None = None,
        model_name: str | None = None,
        random_state: int | None = 42,
        target: str | None = None,
        **model_params: Any,
    ) -> None:
        self.task = _validate_task(task)
        self.random_state = _validate_random_state(random_state)
        self.target = target
        self.model_name = model_name or (
            type(estimator).__name__ if estimator is not None else "custom"
        )
        # Explicit identity check rather than `estimator or _build_estimator(...)`:
        # a custom estimator implementing __bool__/__len__ could evaluate
        # falsy while still being a perfectly valid, deliberately-supplied
        # estimator, which truthiness would silently discard.
        if estimator is not None:
            _validate_estimator_task(estimator, self.task)
            self.estimator = estimator
        else:
            self.estimator = _build_estimator(
                self.task, self.model_name, self.random_state, **model_params
            )

        self.feature_columns_: tuple[str, ...] | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.training_rows_: int | None = None
        self.training_fingerprint_: str | None = None
        self.metrics_: dict[str, float] = {}
        self._fitted = False

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        detail = (
            f", target={self.target!r}, n_features={self.n_features_in_}"
            if self._fitted
            else ""
        )
        return (
            f"Model(estimator={type(self.estimator).__name__}, "
            f"task={self.task!r}, status={status}{detail})"
        )

    def _check_fitted(self) -> None:
        if not self._fitted or not _is_fitted(self.estimator):
            raise NotFittedError("Model is not fitted.")

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        sample_weight: Sequence[float] | None = None,
    ) -> "Model":
        """Fit the wrapped estimator and record the feature/target schema."""
        X, y = _validate_X_y(X, y, self.task)

        self.feature_columns_ = tuple(X.columns)
        self.n_features_in_ = X.shape[1]
        self.training_rows_ = len(X)
        self.training_fingerprint_ = dataset_fingerprint(X, y)

        weights = _validate_sample_weight(sample_weight, len(X))

        logger.info(
            "Model.fit: fitting %s (task=%s) on %d rows / %d features, "
            "target=%r",
            type(self.estimator).__name__, self.task, len(X),
            self.n_features_in_, self.target,
        )

        if weights is None:
            self.estimator.fit(X, y)
        else:
            self.estimator.fit(X, y, sample_weight=weights)

        if self.task == TASK_CLASSIFICATION and hasattr(self.estimator, "classes_"):
            self.classes_ = np.asarray(self.estimator.classes_)

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        self._check_fitted()
        X = _prepare_prediction_frame(X, self.feature_columns_)
        return pd.Series(
            self.estimator.predict(X), index=X.index, name="prediction"
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate classification probabilities."""
        self._check_fitted()

        if self.task != TASK_CLASSIFICATION:
            raise TypeError("predict_proba is available only for classification.")

        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Estimator does not provide predict_proba().")

        X = _prepare_prediction_frame(X, self.feature_columns_)
        proba = np.asarray(self.estimator.predict_proba(X))
        columns = (
            list(self.classes_) if self.classes_ is not None
            else list(range(proba.shape[1]))
        )

        return pd.DataFrame(proba, index=X.index, columns=columns)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        metrics: Sequence[str] | None = None,
        average: str = "weighted",
        sample_weight: Sequence[float] | None = None,
    ) -> dict[str, float]:
        """Evaluate the fitted model against held-out data."""
        self._check_fitted()
        pred = self.predict(X)

        if self.task == TASK_CLASSIFICATION:
            proba = None
            if metrics and any(
                m in {"roc_auc", "average_precision"} for m in metrics
            ):
                proba_frame = self.predict_proba(X)
                proba = (
                    proba_frame.iloc[:, 1].to_numpy()
                    if proba_frame.shape[1] == 2
                    else proba_frame.to_numpy()
                )
            return evaluate_classification(
                y, pred, proba, metrics, average, sample_weight=sample_weight
            )

        return evaluate_regression(y, pred, metrics, sample_weight=sample_weight)

    def evaluate_result(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        metrics: Sequence[str] | None = None,
        average: str = "weighted",
    ) -> ModelEvaluation:
        """Return a structured evaluation result with raw predictions."""
        pred = self.predict(X)
        proba = None

        if self.task == TASK_CLASSIFICATION:
            needs_proba = metrics and any(
                m in {"roc_auc", "average_precision"} for m in metrics
            )
            if needs_proba:
                proba_frame = self.predict_proba(X)
                proba = (
                    proba_frame.iloc[:, 1].to_numpy()
                    if proba_frame.shape[1] == 2
                    else proba_frame.to_numpy()
                )
                scores = evaluate_classification(y, pred, proba, metrics, average)
            else:
                scores = evaluate_classification(y, pred, metrics=metrics, average=average)
        else:
            scores = evaluate_regression(y, pred, metrics=metrics)

        return ModelEvaluation(self.task, scores, pred, proba, y)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        cv: int | Any = 5,
        scoring: str | Sequence[str] | None = None,
        n_jobs: int | None = None,
        sample_weight: Sequence[float] | None = None,
        groups: Sequence[Any] | None = None,
    ) -> pd.DataFrame:
        """Cross-validate a fresh clone of the wrapped estimator."""
        self._check_fitted()
        X, y = _validate_X_y(X, y, self.task)
        weights = _validate_sample_weight(sample_weight, len(X))
        fit_params = {"sample_weight": weights} if weights is not None else None

        results = _routed_cross_validate(
            clone(self.estimator),
            X,
            y,
            cv=_resolve_cv(self.task, cv, self.random_state),
            scoring=scoring or DEFAULT_SCORING[self.task],
            n_jobs=_validate_n_jobs(n_jobs),
            return_train_score=True,
            groups=groups,
            fit_params=fit_params,
        )

        return pd.DataFrame(results)

    def feature_importance(self) -> pd.DataFrame:
        """Return native (impurity- or coefficient-based) feature importance."""
        self._check_fitted()
        return get_feature_importance(self.estimator, self.feature_columns_)

    def permutation_importance(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        scoring: str | None = None,
        n_repeats: int = 10,
        n_jobs: int | None = None,
    ) -> pd.DataFrame:
        """Return permutation-based (model-agnostic) feature importance."""
        self._check_fitted()
        return permutation_feature_importance(
            self.estimator, X, y, scoring, n_repeats, self.random_state, n_jobs
        )

    def artifact(
        self,
        metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelArtifact:
        """Create a structured, serialization-friendly artifact summary."""
        self._check_fitted()

        try:
            version = importlib.import_module("makoding").__version__
        except Exception:
            version = "0.1.0"

        return ModelArtifact(
            self.estimator,
            self.task,
            self.target or "",
            tuple(self.feature_columns_ or ()),
            self.model_name,
            dict(metrics or self.metrics_),
            self.random_state,
            self.training_rows_,
            self.n_features_in_ or 0,
            datetime.now(timezone.utc).isoformat(),
            version,
            platform.python_version(),
            {
                "training_fingerprint": self.training_fingerprint_,
                **dict(metadata or {}),
            },
        )

    def save(self, path: str | Path) -> Path:
        """Persist the fitted estimator and wrapper schema to ``path``.

        Requires ``joblib``. Use :meth:`load` to restore an equivalent,
        fully-usable :class:`Model` instance.
        """
        try:
            joblib = importlib.import_module("joblib")
        except ImportError as exc:
            raise ImportError("joblib is required to save models.") from exc

        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "format": _ARTIFACT_FORMAT,
                "format_version": max(_ARTIFACT_FORMAT_VERSIONS),
                "task": self.task,
                "model_name": self.model_name,
                "random_state": self.random_state,
                "target": self.target,
                "estimator": self.estimator,
                "feature_columns": self.feature_columns_,
                "classes": self.classes_,
                "n_features_in": self.n_features_in_,
                "training_rows": self.training_rows_,
                "training_fingerprint": self.training_fingerprint_,
                "metrics": self.metrics_,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            path,
        )

        logger.info("Model.save: wrote fitted model to %s", path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_sha256: str | None = None,
    ) -> "Model":
        """Load a model previously written by :meth:`save`.

        Security note: this uses ``joblib.load``, which is pickle-based —
        deserializing a pickle can execute arbitrary code. Only load
        artifacts from sources you trust; this is a property of
        pickle/joblib-style serialization in general, not something a
        format check here can fully guard against. If you have a checksum
        recorded through a channel independent of the file itself (see
        :func:`compute_artifact_checksum`, called right after
        :meth:`save`), pass it as ``expected_sha256`` — the file's raw
        bytes are verified against it *before* deserialization is
        attempted, so a file that was swapped or tampered with in transit
        is rejected up front. This protects against post-save tampering;
        it does not make loading an artifact that was malicious from the
        start safe.
        """
        try:
            joblib = importlib.import_module("joblib")
        except ImportError as exc:
            raise ImportError("joblib is required to load models.") from exc

        path = Path(path)

        if expected_sha256 is not None:
            actual_sha256 = compute_artifact_checksum(path)
            if actual_sha256.lower() != expected_sha256.strip().lower():
                raise ValueError(
                    f"Checksum mismatch for {path}: expected "
                    f"{expected_sha256!r}, got {actual_sha256!r}. Refusing "
                    "to deserialize a file that does not match the "
                    "trusted checksum."
                )

        payload = joblib.load(path)

        if not isinstance(payload, dict) or payload.get("format") != _ARTIFACT_FORMAT:
            raise ValueError(f"{path!r} is not a valid Makoding model artifact.")

        version = payload.get("format_version")
        if version not in _ARTIFACT_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported Makoding model artifact format_version "
                f"{version!r}; supported: {sorted(_ARTIFACT_FORMAT_VERSIONS)}."
            )

        instance = cls(
            payload["task"],
            estimator=payload["estimator"],
            model_name=payload["model_name"],
            random_state=payload.get("random_state"),
            target=payload.get("target"),
        )

        instance.feature_columns_ = tuple(payload["feature_columns"])
        instance.classes_ = payload.get("classes")
        instance.n_features_in_ = payload.get("n_features_in")
        instance.training_rows_ = payload.get("training_rows")
        instance.training_fingerprint_ = payload.get("training_fingerprint")
        instance.metrics_ = dict(payload.get("metrics", {}))
        instance._fitted = _is_fitted(instance.estimator)

        if not instance._fitted:
            raise NotFittedError(
                "The loaded artifact does not contain a fitted estimator."
            )

        logger.info("Model.load: restored fitted model from %s", path)
        return instance

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return this wrapper's parameters, flattened with the estimator's.

        The returned dict is symmetric with :meth:`set_params`: any key
        here (other than ``task``/``model_name``/``random_state``/
        ``target``, which are consumed by the constructor) can be passed
        back to ``set_params`` to update the underlying estimator.
        """
        params: dict[str, Any] = {
            "task": self.task,
            "model_name": self.model_name,
            "random_state": self.random_state,
            "target": self.target,
        }
        if hasattr(self.estimator, "get_params"):
            params.update(self.estimator.get_params(deep=deep))
        return params

    def set_params(self, **params: Any) -> "Model":
        """Update wrapper and/or estimator parameters, keeping them in sync.

        A naive implementation that just does
        ``setattr(self, key, value)`` for wrapper-level keys lets the
        wrapper's declared state and the actual wrapped estimator drift
        apart — e.g. ``set_params(random_state=123)`` updating
        ``self.random_state`` without touching ``self.estimator``'s own
        ``random_state``, or ``set_params(model_name=...)`` relabeling the
        wrapper without swapping in a different estimator. This method
        instead keeps them consistent:

        - ``estimator=...`` replaces the wrapped estimator outright
          (validated against the current/new ``task``).
        - ``task=...`` and/or ``model_name=...`` (with no ``estimator``
          given) rebuild a fresh estimator from the registry using the
          current/new ``random_state``, so the declared task/model_name
          and the actual estimator can never disagree.
        - ``random_state=...`` alone (no task/model_name/estimator change)
          is synced into the *existing* estimator too, if it accepts a
          ``random_state`` parameter — so ``self.random_state`` cannot
          drift from what the estimator will actually use.
        - ``target=...`` updates the wrapper's target-column label.
        - Any remaining keys are forwarded as flat estimator
          hyperparameters via the estimator's own ``set_params``.

        A change that fails partway must not leave the wrapper half
        updated — e.g. a rejected ``task`` change must not leave
        ``self.task`` mutated while ``self.estimator`` stays on the old
        task. Every new value is validated/built *before* anything is
        committed to ``self``.
        """
        if "target" in params:
            self.target = params.pop("target")

        new_estimator = params.pop("estimator", _UNSET)
        new_task = params.pop("task", _UNSET)
        new_model_name = params.pop("model_name", _UNSET)
        new_random_state = params.pop("random_state", _UNSET)

        resolved_task = (
            _validate_task(new_task) if new_task is not _UNSET else self.task
        )
        resolved_model_name = (
            new_model_name if new_model_name is not _UNSET else self.model_name
        )
        resolved_random_state = (
            _validate_random_state(new_random_state)
            if new_random_state is not _UNSET
            else self.random_state
        )

        task_changed = resolved_task != self.task
        model_name_changed = resolved_model_name != self.model_name

        if new_estimator is not _UNSET:
            _validate_estimator_task(new_estimator, resolved_task)
            resolved_estimator = new_estimator
        elif task_changed or model_name_changed:
            # Build the replacement estimator before touching `self` at
            # all -- if this raises (e.g. task/model_name is not a valid
            # registry combination), nothing about the wrapper has
            # changed yet.
            resolved_estimator = _build_estimator(
                resolved_task, resolved_model_name, resolved_random_state
            )
        elif new_random_state is not _UNSET and (
            "random_state" in self.estimator.get_params()
        ):
            resolved_estimator = clone(self.estimator)
            resolved_estimator.set_params(random_state=resolved_random_state)
        else:
            resolved_estimator = self.estimator

        # Everything above succeeded -- commit atomically.
        self.task = resolved_task
        self.model_name = resolved_model_name
        self.random_state = resolved_random_state
        self.estimator = resolved_estimator

        if params:
            self.estimator.set_params(**params)

        self._fitted = False
        return self