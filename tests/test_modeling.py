import numpy as np
import pandas as pd
import pytest

from makoding.modeling import (
    ModelResult,
    NotFittedError,
    train_test_split_data,
    fit_model,
    predict,
    evaluate_classification,
    evaluate_regression,
    cross_validate_model,
    get_feature_importance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    X = pd.DataFrame(
        {
            "age": [20, 22, 25, 30, 35, 40, 45, 50, 55, 60],
            "income": [20, 25, 30, 40, 45, 50, 60, 70, 80, 90],
        }
    )

    y = pd.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        name="target",
    )

    return X, y


@pytest.fixture
def regression_data():
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )

    y = pd.Series(
        [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        name="target",
    )

    return X, y


# ---------------------------------------------------------------------------
# ModelResult
# ---------------------------------------------------------------------------

def test_model_result_stores_model_and_metrics():
    result = ModelResult(
        model="test_model",
        metrics={"accuracy": 0.95},
    )

    assert result.model == "test_model"
    assert result.metrics["accuracy"] == 0.95


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def test_train_test_split_returns_four_objects(classification_data):
    X, y = classification_data

    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3


def test_train_test_split_preserves_features(classification_data):
    X, y = classification_data

    X_train, X_test, _, _ = train_test_split_data(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    assert list(X_train.columns) == ["age", "income"]
    assert list(X_test.columns) == ["age", "income"]


def test_train_test_split_is_reproducible(classification_data):
    X, y = classification_data

    split1 = train_test_split_data(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    split2 = train_test_split_data(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    pd.testing.assert_frame_equal(split1[0], split2[0])
    pd.testing.assert_frame_equal(split1[1], split2[1])


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def test_fit_model_classification(classification_data):
    X, y = classification_data

    result = fit_model(
        X,
        y,
        task="classification",
        model_name="logistic_regression",
        random_state=42,
    )

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert hasattr(result.model, "predict")


def test_fit_model_regression(regression_data):
    X, y = regression_data

    result = fit_model(
        X,
        y,
        task="regression",
        model_name="linear_regression",
        random_state=42,
    )

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert hasattr(result.model, "predict")


def test_fit_model_rejects_invalid_task(classification_data):
    X, y = classification_data

    with pytest.raises(ValueError):
        fit_model(
            X,
            y,
            task="invalid",
            model_name="logistic_regression",
        )


def test_fit_model_rejects_unknown_model(classification_data):
    X, y = classification_data

    with pytest.raises(ValueError):
        fit_model(
            X,
            y,
            task="classification",
            model_name="does_not_exist",
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_returns_predictions(classification_data):
    X, y = classification_data

    result = fit_model(
        X,
        y,
        task="classification",
        model_name="logistic_regression",
        random_state=42,
    )

    predictions = predict(result.model, X)

    assert len(predictions) == len(X)
    assert isinstance(predictions, np.ndarray)


def test_predict_unfitted_model_raises():
    class DummyModel:
        pass

    with pytest.raises(NotFittedError):
        predict(DummyModel(), pd.DataFrame({"x": [1, 2, 3]}))


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = evaluate_classification(
        y_true,
        y_pred,
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    assert metrics["accuracy"] == 0.75


def test_classification_metrics_with_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_probability = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_classification(
        y_true,
        y_pred,
        y_probability,
    )

    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == 1.0


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def test_regression_metrics():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 4, 4])

    metrics = evaluate_regression(
        y_true,
        y_pred,
    )

    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

    assert metrics["mae"] == 0.25


def test_perfect_regression_metrics():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])

    metrics = evaluate_regression(
        y_true,
        y_pred,
    )

    assert metrics["mae"] == 0
    assert metrics["mse"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def test_cross_validate_classification(classification_data):
    X, y = classification_data

    result = cross_validate_model(
        X,
        y,
        task="classification",
        model_name="logistic_regression",
        cv=3,
        random_state=42,
    )

    assert isinstance(result, dict)
    assert "mean_score" in result
    assert "std_score" in result
    assert "scores" in result
    assert len(result["scores"]) == 3


def test_cross_validate_regression(regression_data):
    X, y = regression_data

    result = cross_validate_model(
        X,
        y,
        task="regression",
        model_name="linear_regression",
        cv=3,
        random_state=42,
    )

    assert isinstance(result, dict)
    assert "mean_score" in result
    assert "std_score" in result
    assert "scores" in result
    assert len(result["scores"]) == 3


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def test_feature_importance_returns_dataframe(classification_data):
    X, y = classification_data

    result = fit_model(
        X,
        y,
        task="classification",
        model_name="random_forest",
        random_state=42,
    )

    importance = get_feature_importance(
        result.model,
        feature_names=X.columns,
    )

    assert isinstance(importance, pd.DataFrame)
    assert "feature" in importance.columns
    assert "importance" in importance.columns


def test_feature_importance_contains_all_features(classification_data):
    X, y = classification_data

    result = fit_model(
        X,
        y,
        task="classification",
        model_name="random_forest",
        random_state=42,
    )

    importance = get_feature_importance(
        result.model,
        feature_names=X.columns,
    )

    assert set(importance["feature"]) == {"age", "income"}


def test_feature_importance_is_sorted(classification_data):
    X, y = classification_data

    result = fit_model(
        X,
        y,
        task="classification",
        model_name="random_forest",
        random_state=42,
    )

    importance = get_feature_importance(
        result.model,
        feature_names=X.columns,
    )

    values = importance["importance"].tolist()

    assert values == sorted(values, reverse=True)