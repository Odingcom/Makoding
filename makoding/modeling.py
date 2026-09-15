"""Production-ready modeling utilities for Makoding.

This module provides two complementary modeling interfaces:

1. ``Model`` — a DataFrame-first object-oriented wrapper around
   scikit-learn-compatible estimators.

2. Functional helpers such as ``fit_model`` and ``evaluate_classification``
   for simple application workflows such as DataLab Pro.

The module supports classification and regression and is designed to work
with scikit-learn-compatible estimators, including optional third-party
estimators such as XGBoost, LightGBM, and CatBoost when installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
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
    RandomizedSearchCV,
    cross_validate as _sklearn_cross_validate,
    train_test_split as _sklearn_train_test_split,
)

from .feature_engineering import (
    _validate_columns_exist,
    _validate_dataframe,
    _validate_no_missing,
    _validate_numeric_columns,
)

__all__ = [
    "Model",
    "ModelResult",
    "NotFittedError",
    "split_train_test",
    "train_test_split_data",
    "fit_model",
    "predict",
    "evaluate_classification",
    "evaluate_regression",
    "cross_validate_model",
    "get_feature_importance",
    "grid_search",
    "random_search",
]


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TASKS = {"classification", "regression"}

_CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
}

_REGRESSION_METRICS = {
    "mae",
    "mse",
    "rmse",
    "r2",
}

_DEFAULT_CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
]

_DEFAULT_REGRESSION_METRICS = [
    "mae",
    "mse",
    "rmse",
    "r2",
]

_DEFAULT_CLASSIFICATION_SCORING = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]

_DEFAULT_REGRESSION_SCORING = [
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "r2",
]


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Container returned by ``fit_model``.

    Parameters
    ----------
    model:
        Fitted estimator.
    metrics:
        Optional evaluation metrics associated with the model.
    """

    model: Any
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NotFittedError(RuntimeError):
    """Raised when a fitted-model operation is used before fitting."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_task(task: str) -> None:
    """Validate a modeling task."""
    if task not in _TASKS:
        raise ValueError(
            f"task must be one of {sorted(_TASKS)}, got {task!r}"
        )


def _validate_estimator(estimator: Any) -> None:
    """Validate a supplied estimator."""
    if not hasattr(estimator, "fit"):
        raise TypeError(
            "estimator must be a scikit-learn-compatible object "
            "with a 'fit' method, "
            f"got {type(estimator).__name__!r}"
        )


def _resolve_feature_columns(
    frame: pd.DataFrame,
    target: str,
    columns: Sequence[str] | None,
) -> list[str]:
    """Resolve and validate model feature columns."""
    if columns is None:
        resolved = [
            column for column in frame.columns
            if column != target
        ]
    else:
        resolved = list(columns)
        _validate_columns_exist(frame, resolved)

        if target in resolved:
            raise ValueError(
                f"target column {target!r} must not appear "
                "in feature_columns"
            )

    if not resolved:
        raise ValueError(
            "No feature columns available. "
            "The frame must contain at least one predictor."
        )

    return resolved


def _validate_training_data(
    frame: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
) -> None:
    """Validate features and target before fitting."""
    _validate_dataframe(frame)

    _validate_columns_exist(
        frame,
        [target, *feature_columns],
    )

    _validate_numeric_columns(
        frame,
        feature_columns,
    )

    _validate_no_missing(
        frame,
        feature_columns,
    )

    if frame[target].isna().any():
        raise ValueError(
            f"target column {target!r} contains missing values"
        )


def _validate_prediction_data(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    """Validate a frame used for prediction."""
    _validate_dataframe(frame)

    _validate_columns_exist(
        frame,
        feature_columns,
    )

    _validate_numeric_columns(
        frame,
        feature_columns,
    )

    _validate_no_missing(
        frame,
        feature_columns,
    )


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

def _build_estimator(
    task: str,
    model_name: str,
    random_state: int | None = 42,
) -> Any:
    """Build a supported estimator from a simple model name.

    Core scikit-learn models are always available. XGBoost, LightGBM,
    and CatBoost are optional dependencies and are imported only when
    the corresponding model is requested.
    """

    _validate_task(task)

    name = model_name.strip().lower()

    # ------------------------------------------------------------------
    # Core scikit-learn models
    # ------------------------------------------------------------------

    classification_models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }

    regression_models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=random_state,
        ),
    }

    models = (
        classification_models
        if task == "classification"
        else regression_models
    )

    if name in models:
        return models[name]

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required to use model_name='xgboost'. "
                "Install it with: pip install xgboost"
            ) from exc

        if task == "classification":
            return XGBClassifier(
                n_estimators=200,
                random_state=random_state,
                eval_metric="logloss",
            )

        return XGBRegressor(
            n_estimators=200,
            random_state=random_state,
            objective="reg:squarederror",
        )

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required to use model_name='lightgbm'. "
                "Install it with: pip install lightgbm"
            ) from exc

        if task == "classification":
            return LGBMClassifier(
                n_estimators=200,
                random_state=random_state,
                verbosity=-1,
            )

        return LGBMRegressor(
            n_estimators=200,
            random_state=random_state,
            verbosity=-1,
        )

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------

    if name == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required to use model_name='catboost'. "
                "Install it with: pip install catboost"
            ) from exc

        if task == "classification":
            return CatBoostClassifier(
                iterations=200,
                random_seed=random_state,
                verbose=False,
            )

        return CatBoostRegressor(
            iterations=200,
            random_seed=random_state,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Unknown model
    # ------------------------------------------------------------------

    supported_models = sorted(
        set(classification_models)
        if task == "classification"
        else set(regression_models)
    )

    supported_models.extend(
        ["xgboost", "lightgbm", "catboost"]
    )

    raise ValueError(
        f"Unknown model {model_name!r} for task={task!r}. "
        f"Supported models: {sorted(supported_models)}"
    )
# ---------------------------------------------------------------------------
# Train/test splitting
# ---------------------------------------------------------------------------

def split_train_test(
    frame: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test DataFrames."""
    _validate_dataframe(frame)
    _validate_columns_exist(frame, [target])

    if not isinstance(test_size, (int, float)):
        raise TypeError("test_size must be numeric")

    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size must be between 0 and 1 (exclusive), "
            f"got {test_size}"
        )

    stratification_target = frame[target] if stratify else None

    train_frame, test_frame = _sklearn_train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratification_target,
    )

    return (
        train_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split predictors and target into train/test partitions.

    This is the functional counterpart to ``split_train_test``.
    """
    _validate_dataframe(X)

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or DataFrame")

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of observations"
        )

    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size must be between 0 and 1 (exclusive), "
            f"got {test_size}"
        )

    stratification_target = y if stratify else None

    X_train, X_test, y_train, y_test = _sklearn_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratification_target,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Functional model fitting
# ---------------------------------------------------------------------------

def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    model_name: str,
    random_state: int | None = 42,
) -> ModelResult:
    """Fit a supported model and return a ``ModelResult``."""
    _validate_task(task)
    _validate_dataframe(X)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of observations"
        )

    if X.isna().any().any():
        raise ValueError("X contains missing values")

    estimator = _build_estimator(
        task=task,
        model_name=model_name,
        random_state=random_state,
    )

    estimator.fit(X, y)

    return ModelResult(
        model=estimator,
        metrics={},
    )


# ---------------------------------------------------------------------------
# Functional prediction
# ---------------------------------------------------------------------------

def _is_fitted(estimator: Any) -> bool:
    """Return whether an estimator appears to be fitted."""
    if not hasattr(estimator, "fit"):
        return False

    fitted_attributes = (
        "n_features_in_",
        "classes_",
        "coef_",
        "feature_importances_",
        "intercept_",
    )

    return any(
        hasattr(estimator, attribute)
        for attribute in fitted_attributes
    )


def predict(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate predictions from a fitted estimator."""
    if not _is_fitted(model):
        raise NotFittedError(
            "Model is not fitted. Call fit before predict."
        )

    _validate_dataframe(X)

    if X.isna().any().any():
        raise ValueError("X contains missing values")

    return np.asarray(model.predict(X))


# ---------------------------------------------------------------------------
# Functional evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_probability: Sequence[float] | None = None,
) -> dict[str, float]:
    """Calculate standard classification metrics."""
    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    if y_probability is not None:
        unique_classes = np.unique(y_true)

        if len(unique_classes) != 2:
            raise ValueError(
                "ROC-AUC probability evaluation requires "
                "binary classification."
            )

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        )

    return metrics


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Calculate standard regression metrics."""
    mse = float(
        mean_squared_error(y_true, y_pred)
    )

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(
            r2_score(y_true, y_pred)
        ),
    }


# ---------------------------------------------------------------------------
# Functional cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    model_name: str,
    cv: int = 5,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """Perform cross-validation and return aggregate results."""
    _validate_task(task)
    _validate_dataframe(X)

    if cv < 2:
        raise ValueError("cv must be at least 2")

    estimator = _build_estimator(
        task=task,
        model_name=model_name,
        random_state=random_state,
    )

    if task == "classification":
        scoring = "accuracy"
    else:
        scoring = "r2"

    results = _sklearn_cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )

    scores = np.asarray(
        results["test_score"],
        dtype=float,
    )

    return {
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Functional feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: Any,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Return feature importance values sorted descending."""
    if not _is_fitted(model):
        raise NotFittedError(
            "Model is not fitted. Fit the model before "
            "requesting feature importance."
        )

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

    elif hasattr(model, "coef_"):
        coefficients = np.asarray(
            model.coef_,
            dtype=float,
        )

        if coefficients.ndim == 1:
            importance = np.abs(coefficients)
        else:
            importance = np.max(
                np.abs(coefficients),
                axis=0,
            )

    else:
        raise ValueError(
            f"{type(model).__name__} does not expose "
            "feature_importances_ or coef_."
        )

    feature_names = list(feature_names)

    if len(importance) != len(feature_names):
        raise ValueError(
            "Number of feature importance values does not match "
            "number of feature names."
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Object-oriented Model wrapper
# ---------------------------------------------------------------------------

class Model:
    """DataFrame-first wrapper around a single estimator."""

    def __init__(
        self,
        estimator: Any,
        task: str = "classification",
    ) -> None:
        _validate_task(task)
        _validate_estimator(estimator)

        self.estimator = estimator
        self.task = task

        self.feature_columns_: list[str] | None = None
        self.target_: str | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

        self._is_fitted = False

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(
                f"{type(self).__name__} is not fitted. "
                "Call 'fit' before using this method."
            )

    def _invalidate_fit(self) -> None:
        self._is_fitted = False
        self.feature_columns_ = None
        self.target_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        feature_columns: Sequence[str] | None = None,
    ) -> "Model":
        """Fit the wrapped estimator."""
        _validate_dataframe(frame)

        resolved_columns = _resolve_feature_columns(
            frame,
            target,
            feature_columns,
        )

        _validate_training_data(
            frame,
            target,
            resolved_columns,
        )

        if self.task == "classification":
            if frame[target].nunique() < 2:
                raise ValueError(
                    f"target column {target!r} has fewer than "
                    "2 distinct classes."
                )

        if self.task == "regression":
            if not pd.api.types.is_numeric_dtype(frame[target]):
                raise TypeError(
                    f"target column {target!r} must be numeric "
                    "for regression."
                )

        self.estimator.fit(
            frame[resolved_columns],
            frame[target],
        )

        self.feature_columns_ = resolved_columns
        self.target_ = target
        self.n_features_in_ = len(resolved_columns)

        if self.task == "classification":
            if not hasattr(self.estimator, "classes_"):
                raise ValueError(
                    f"{type(self.estimator).__name__} does not expose "
                    "'classes_' after fitting."
                )

            self.classes_ = np.asarray(
                self.estimator.classes_
            )
        else:
            self.classes_ = None

        self._is_fitted = True

        return self

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Generate predictions."""
        self._check_fitted()

        _validate_prediction_data(
            frame,
            self.feature_columns_,
        )

        predictions = self.estimator.predict(
            frame[self.feature_columns_]
        )

        return pd.Series(
            predictions,
            index=frame.index,
            name=f"{self.target_}_predicted",
        )

    def predict_proba(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate classification probabilities."""
        self._check_fitted()

        if self.task != "classification":
            raise ValueError(
                "predict_proba is only available for "
                "classification models"
            )

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                f"{type(self.estimator).__name__} does not support "
                "predict_proba"
            )

        _validate_prediction_data(
            frame,
            self.feature_columns_,
        )

        probabilities = self.estimator.predict_proba(
            frame[self.feature_columns_]
        )

        return pd.DataFrame(
            probabilities,
            index=frame.index,
            columns=self.classes_,
        )

    def evaluate(
        self,
        frame: pd.DataFrame,
        target: str | None = None,
        metrics: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate the fitted model."""
        self._check_fitted()
        _validate_dataframe(frame)

        target = target or self.target_

        _validate_columns_exist(
            frame,
            [target],
        )

        if frame[target].isna().any():
            raise ValueError(
                f"target column {target!r} contains missing values"
            )

        if metrics is None:
            metrics = (
                _DEFAULT_CLASSIFICATION_METRICS
                if self.task == "classification"
                else _DEFAULT_REGRESSION_METRICS
            )

        supported = (
            _CLASSIFICATION_METRICS
            if self.task == "classification"
            else _REGRESSION_METRICS
        )

        unsupported = [
            metric
            for metric in metrics
            if metric not in supported
        ]

        if unsupported:
            raise ValueError(
                f"Unsupported metric(s): {unsupported}"
            )

        y_true = frame[target]
        y_pred = self.predict(frame)

        results: dict[str, float] = {}

        for metric in metrics:
            if metric == "accuracy":
                results[metric] = float(
                    accuracy_score(y_true, y_pred)
                )

            elif metric == "precision":
                results[metric] = float(
                    precision_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                )

            elif metric == "recall":
                results[metric] = float(
                    recall_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                )

            elif metric == "f1":
                results[metric] = float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                )

            elif metric == "roc_auc":
                if self.classes_ is None or len(self.classes_) != 2:
                    raise ValueError(
                        "roc_auc currently requires binary classification."
                    )

                probabilities = self.predict_proba(frame)

                results[metric] = float(
                    roc_auc_score(
                        y_true,
                        probabilities[self.classes_[1]],
                    )
                )

            elif metric == "mae":
                results[metric] = float(
                    mean_absolute_error(y_true, y_pred)
                )

            elif metric == "mse":
                results[metric] = float(
                    mean_squared_error(y_true, y_pred)
                )

            elif metric == "rmse":
                results[metric] = float(
                    np.sqrt(
                        mean_squared_error(
                            y_true,
                            y_pred,
                        )
                    )
                )

            elif metric == "r2":
                results[metric] = float(
                    r2_score(y_true, y_pred)
                )

        return results

    def cross_validate(
        self,
        frame: pd.DataFrame,
        target: str,
        feature_columns: Sequence[str] | None = None,
        cv: int = 5,
        scoring: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Cross-validate cloned estimators."""
        _validate_dataframe(frame)

        if not isinstance(cv, int) or cv < 2:
            raise ValueError(
                f"cv must be an integer >= 2, got {cv!r}"
            )

        resolved_columns = _resolve_feature_columns(
            frame,
            target,
            feature_columns,
        )

        _validate_training_data(
            frame,
            target,
            resolved_columns,
        )

        if self.task == "regression":
            if not pd.api.types.is_numeric_dtype(
                frame[target]
            ):
                raise TypeError(
                    "Regression target must be numeric."
                )

        if scoring is None:
            scoring = (
                _DEFAULT_CLASSIFICATION_SCORING
                if self.task == "classification"
                else _DEFAULT_REGRESSION_SCORING
            )

        fresh_estimator = clone(self.estimator)

        results = _sklearn_cross_validate(
            fresh_estimator,
            frame[resolved_columns],
            frame[target],
            cv=cv,
            scoring=list(scoring),
            error_score="raise",
        )

        result_frame = pd.DataFrame(results)
        result_frame.index.name = "fold"

        return result_frame

    def feature_importance(self) -> pd.DataFrame:
        """Return model-based feature importance."""
        self._check_fitted()

        if hasattr(self.estimator, "coef_"):
            coefficients = np.asarray(
                self.estimator.coef_
            )

            if coefficients.ndim == 1:
                importances = np.abs(coefficients)
            else:
                importances = np.max(
                    np.abs(coefficients),
                    axis=0,
                )

        elif hasattr(self.estimator, "feature_importances_"):
            importances = np.asarray(
                self.estimator.feature_importances_
            )

        else:
            raise ValueError(
                f"{type(self.estimator).__name__} exposes neither "
                "'coef_' nor 'feature_importances_'."
            )

        if len(importances) != len(self.feature_columns_):
            raise ValueError(
                "Estimator importance vector does not match "
                "the number of feature columns."
            )

        result = pd.DataFrame(
            {
                "importance": importances,
            },
            index=pd.Index(
                self.feature_columns_,
                name="feature",
            ),
        )

        return result.sort_values(
            "importance",
            ascending=False,
        )

    def get_params(self) -> dict[str, Any]:
        """Return estimator hyperparameters."""
        return self.estimator.get_params()

    def set_params(
        self,
        **params: Any,
    ) -> "Model":
        """Update estimator parameters and invalidate fitting."""
        self.estimator.set_params(**params)
        self._invalidate_fit()
        return self


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def grid_search(
    estimator: Any,
    frame: pd.DataFrame,
    target: str,
    param_grid: dict[str, Any],
    task: str = "classification",
    feature_columns: Sequence[str] | None = None,
    cv: int = 5,
    scoring: str | None = None,
    n_jobs: int | None = None,
) -> GridSearchCV:
    """Run GridSearchCV."""
    _validate_task(task)
    _validate_estimator(estimator)

    resolved_columns = _resolve_feature_columns(
        frame,
        target,
        feature_columns,
    )

    _validate_training_data(
        frame,
        target,
        resolved_columns,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        error_score="raise",
    )

    search.fit(
        frame[resolved_columns],
        frame[target],
    )

    return search


def random_search(
    estimator: Any,
    frame: pd.DataFrame,
    target: str,
    param_distributions: dict[str, Any],
    task: str = "classification",
    feature_columns: Sequence[str] | None = None,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str | None = None,
    random_state: int | None = 42,
    n_jobs: int | None = None,
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV."""
    _validate_task(task)
    _validate_estimator(estimator)

    if n_iter < 1:
        raise ValueError(
            f"n_iter must be >= 1, got {n_iter}"
        )

    resolved_columns = _resolve_feature_columns(
        frame,
        target,
        feature_columns,
    )

    _validate_training_data(
        frame,
        target,
        resolved_columns,
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        error_score="raise",
    )

    search.fit(
        frame[resolved_columns],
        frame[target],
    )

    return search