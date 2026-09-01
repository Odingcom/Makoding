"""Central configuration for DataLab Pro.

Keeping configuration in one place makes the product easier to
white-label, tune, test, deploy, and audit.

Environment variables can override selected operational settings,
which is useful for local development, CI/CD, and production
deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer from an environment variable.

    Falls back to ``default`` if the variable is missing or invalid.
    """

    raw_value = os.environ.get(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        return default

    return max(value, minimum)


def _get_float_env(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Read a bounded float from an environment variable."""

    raw_value = os.environ.get(name)

    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError:
        return default

    return min(max(value, minimum), maximum)


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppMeta:
    """Public application metadata."""

    name: str = "DataLab Pro"

    tagline: str = (
        "Collect, clean, explore, engineer features, and model "
        "— in one workspace."
    )

    version: str = "1.0.0"

    icon: str = "🧪"

    accent_color: str = "#4F46E5"

    support_email: str = "support@example.com"


# ---------------------------------------------------------------------------
# Operational limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Limits:
    """Application resource and validation limits."""

    max_upload_mb: int = _get_int_env(
        "DATALAB_MAX_UPLOAD_MB",
        default=200,
        minimum=1,
    )

    max_rows: int = _get_int_env(
        "DATALAB_MAX_ROWS",
        default=1_000_000,
        minimum=1,
    )

    max_preview_rows: int = _get_int_env(
        "DATALAB_MAX_PREVIEW_ROWS",
        default=100,
        minimum=1,
    )

    max_categorical_levels: int = _get_int_env(
        "DATALAB_MAX_CATEGORICAL_LEVELS",
        default=30,
        minimum=2,
    )

    url_timeout_seconds: int = _get_int_env(
        "DATALAB_URL_TIMEOUT_SECONDS",
        default=30,
        minimum=1,
    )

    allowed_url_schemes: tuple[str, ...] = (
        "http",
        "https",
    )

    allowed_upload_extensions: tuple[str, ...] = (
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
    )


# ---------------------------------------------------------------------------
# Machine-learning defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelDefaults:
    """Default machine-learning configuration."""

    cv_folds: int = _get_int_env(
        "DATALAB_CV_FOLDS",
        default=3,
        minimum=2,
    )

    random_state: int = _get_int_env(
        "DATALAB_RANDOM_STATE",
        default=42,
        minimum=0,
    )

    max_features_default: int = _get_int_env(
        "DATALAB_MAX_FEATURES",
        default=5,
        minimum=1,
    )

    classification_uniqueness_threshold: int = _get_int_env(
        "DATALAB_CLASSIFICATION_THRESHOLD",
        default=10,
        minimum=2,
    )

    test_size: float = _get_float_env(
        "DATALAB_TEST_SIZE",
        default=0.20,
        minimum=0.05,
        maximum=0.50,
    )


# ---------------------------------------------------------------------------
# Application-wide configuration instances
# ---------------------------------------------------------------------------

APP: Final = AppMeta()

LIMITS: Final = Limits()

MODEL_DEFAULTS: Final = ModelDefaults()


# ---------------------------------------------------------------------------
# Supervised learning model registry
# ---------------------------------------------------------------------------

SUPERVISED_MODELS: Final[dict[str, dict[str, object]]] = {

    "Random Forest": {
        "classifier": "sklearn.ensemble.RandomForestClassifier",
        "regressor": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "n_estimators": 200,
            "random_state": MODEL_DEFAULTS.random_state,
            "n_jobs": -1,
        },
    },

    "Logistic / Linear Regression": {
        "classifier": "sklearn.linear_model.LogisticRegression",
        "regressor": "sklearn.linear_model.Ridge",
        "params": {
            "max_iter": 1000,
        },
    },

    "Gradient Boosting": {
        "classifier": "sklearn.ensemble.GradientBoostingClassifier",
        "regressor": "sklearn.ensemble.GradientBoostingRegressor",
        "params": {
            "random_state": MODEL_DEFAULTS.random_state,
        },
    },

    "XGBoost": {
        "classifier": "xgboost.XGBClassifier",
        "regressor": "xgboost.XGBRegressor",
        "params": {
            "random_state": MODEL_DEFAULTS.random_state,
            "n_jobs": -1,
        },
    },

    "LightGBM": {
        "classifier": "lightgbm.LGBMClassifier",
        "regressor": "lightgbm.LGBMRegressor",
        "params": {
            "random_state": MODEL_DEFAULTS.random_state,
            "n_jobs": -1,
            "verbosity": -1,
        },
    },

    "CatBoost": {
        "classifier": "catboost.CatBoostClassifier",
        "regressor": "catboost.CatBoostRegressor",
        "params": {
            "random_seed": MODEL_DEFAULTS.random_state,
            "verbose": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: Final[str] = os.environ.get(
    "DATALAB_LOG_LEVEL",
    "INFO",
).upper()