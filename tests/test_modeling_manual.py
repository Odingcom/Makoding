"""Full test suite for ``makoding.modeling``.

Covers: input validation, functional fit/predict/evaluate, cross-validation
(including custom splitters and sample_weight), out-of-fold predictions
across all label-encoding combinations, feature importance (native and
permutation), model comparison, hyperparameter search, calibration,
threshold optimization, diagnostics, dataset fingerprinting, the ``Model``
wrapper's full lifecycle (fit/predict/evaluate/save/load/get_params/
set_params), and every bug fix identified in review (OOF label mapping,
threshold label mapping, compare_models sort order, get_params/set_params
round-trip, set_params task/model_name/random_state consistency, the
``estimator or ...`` truthiness bug, dataset_fingerprint's error contract,
fit_model's y_eval validation, checksum-verified load, and
get_feature_importance's fallback logic).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from makoding.modeling import (
    Model,
    ModelArtifact,
    ModelComparison,
    ModelEvaluation,
    ModelResult,
    NotFittedError,
    calibrate_model,
    classification_diagnostics,
    compare_models,
    compute_artifact_checksum,
    cross_validate_model,
    dataset_fingerprint,
    evaluate_classification,
    evaluate_regression,
    find_optimal_threshold,
    fit_model,
    generate_oof_predictions,
    get_feature_importance,
    grid_search,
    permutation_feature_importance,
    predict,
    random_search,
    regression_diagnostics,
    split_train_test,
    train_test_split_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_int_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=5, n_informative=4, n_redundant=0,
        random_state=0, weights=[0.5, 0.5],
    )
    return (
        pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]),
        pd.Series(y, name="target"),
    )


@pytest.fixture
def binary_str_data(binary_int_data):
    X, y = binary_int_data
    return X, y.map({0: "cat", 1: "dog"}).rename("target")


@pytest.fixture
def multiclass_int_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=6, n_classes=3, n_informative=4,
        n_redundant=0, random_state=1,
    )
    return (
        pd.DataFrame(X, columns=[f"g{i}" for i in range(6)]),
        pd.Series(y, name="target"),
    )


@pytest.fixture
def multiclass_str_data(multiclass_int_data):
    X, y = multiclass_int_data
    label_map = {0: "red", 1: "green", 2: "blue"}
    return X, y.map(label_map).rename("target")


@pytest.fixture
def regression_data():
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=4, noise=5.0, random_state=2)
    return (
        pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]),
        pd.Series(y, name="target"),
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_invalid_task(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="task must be one of"):
            fit_model(X, y, task="not_a_task", model_name="logistic_regression")

    def test_rejects_non_dataframe_X(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            fit_model(np.zeros((10, 2)), pd.Series(np.arange(10) % 2), task="classification", model_name="logistic_regression")

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError, match="must not be empty"):
            fit_model(pd.DataFrame(), pd.Series([0, 1]), task="classification", model_name="logistic_regression")

    def test_rejects_duplicate_columns(self):
        X = pd.DataFrame(np.zeros((5, 2)), columns=["a", "a"])
        with pytest.raises(ValueError, match="duplicate column"):
            fit_model(X, pd.Series([0, 1, 0, 1, 0]), task="classification", model_name="logistic_regression")

    def test_rejects_non_numeric_features(self):
        X = pd.DataFrame({"a": ["x", "y", "z", "x", "y"], "b": [1, 2, 3, 4, 5]})
        with pytest.raises(TypeError, match="numeric columns"):
            fit_model(X, pd.Series([0, 1, 0, 1, 0]), task="classification", model_name="logistic_regression")

    def test_rejects_missing_values_in_X(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="missing values"):
            fit_model(X, pd.Series([0, 1, 0, 1, 0]), task="classification", model_name="logistic_regression")

    def test_rejects_non_finite_values_in_X(self):
        X = pd.DataFrame({"a": [1.0, np.inf, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="finite"):
            fit_model(X, pd.Series([0, 1, 0, 1, 0]), task="classification", model_name="logistic_regression")

    def test_rejects_missing_values_in_target(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="missing values"):
            fit_model(X, pd.Series([0, np.nan, 0, 1, 0]), task="classification", model_name="logistic_regression")

    def test_rejects_single_class_target(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(ValueError, match="at least two classes"):
            fit_model(X, pd.Series([1, 1, 1, 1, 1]), task="classification", model_name="logistic_regression")

    def test_rejects_non_numeric_regression_target(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(TypeError, match="numeric"):
            fit_model(X, pd.Series(["a", "b", "c", "d", "e"]), task="regression", model_name="linear_regression")

    def test_rejects_mismatched_row_counts(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="same number of rows"):
            fit_model(X, y, task="classification", model_name="logistic_regression")

    def test_rejects_bool_random_state(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(TypeError, match="random_state"):
            fit_model(X, y, task="classification", model_name="logistic_regression", random_state=True)

    def test_rejects_bad_sample_weight_length(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="sample_weight"):
            fit_model(X, y, task="classification", model_name="logistic_regression", sample_weight=[1.0, 2.0])

    def test_rejects_negative_sample_weight(self, binary_int_data):
        X, y = binary_int_data
        weights = np.ones(len(X))
        weights[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            fit_model(X, y, task="classification", model_name="logistic_regression", sample_weight=weights)


# ---------------------------------------------------------------------------
# split_train_test / train_test_split_data
# ---------------------------------------------------------------------------

class TestSplitting:
    def test_split_train_test_shapes(self, binary_int_data):
        X, y = binary_int_data
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.25, random_state=0, stratify=True)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - 0.25) < 0.02

    def test_split_preserves_class_balance_when_stratified(self, binary_int_data):
        X, y = binary_int_data
        _, _, y_train, y_test = split_train_test(X, y, test_size=0.3, random_state=0, stratify=True)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.1

    def test_train_test_split_data_is_alias(self, binary_int_data):
        X, y = binary_int_data
        a = split_train_test(X, y, test_size=0.2, random_state=1, stratify=True)
        b = train_test_split_data(X, y, test_size=0.2, random_state=1, stratify=True)
        for left, right in zip(a, b):
            pd.testing.assert_frame_equal(left, right) if isinstance(left, pd.DataFrame) else pd.testing.assert_series_equal(left, right)

    def test_rejects_bad_test_size(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="test_size"):
            split_train_test(X, y, test_size=1.5, stratify=True)


# ---------------------------------------------------------------------------
# fit_model / predict
# ---------------------------------------------------------------------------

class TestFitModel:
    def test_fit_model_classification_returns_fitted_estimator(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(X, y, task="classification", model_name="logistic_regression")
        assert isinstance(result, ModelResult)
        assert is_classifier(result.model)
        assert result.metrics == {}

    def test_fit_model_regression_returns_fitted_estimator(self, regression_data):
        X, y = regression_data
        result = fit_model(X, y, task="regression", model_name="linear_regression")
        assert is_regressor(result.model)

    def test_fit_model_accepts_estimator_instance_not_just_name(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(X, y, task="classification", model_name=RandomForestClassifier(n_estimators=15, random_state=0))
        assert isinstance(result.model, RandomForestClassifier)

    def test_fit_model_forwards_estimator_params(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(X, y, task="classification", model_name="random_forest", n_estimators=17)
        assert result.model.n_estimators == 17

    def test_fit_model_with_eval_data_attaches_metrics(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(
            X, y, task="classification", model_name="logistic_regression",
            X_eval=X, y_eval=y, metrics=["accuracy", "f1"],
        )
        assert set(result.metrics) == {"accuracy", "f1"}
        assert result.metrics["accuracy"] > 0.5

    def test_fit_model_requires_both_eval_args_together(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="together"):
            fit_model(X, y, task="classification", model_name="logistic_regression", X_eval=X)

    def test_fit_model_validates_y_eval_missing_values(self, binary_int_data):
        X, y = binary_int_data
        bad_y_eval = pd.Series([np.nan] * 10)
        with pytest.raises(ValueError, match="missing values"):
            fit_model(
                X, y, task="classification", model_name="logistic_regression",
                X_eval=X.iloc[:10], y_eval=bad_y_eval,
            )

    def test_fit_model_validates_eval_row_count_match(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="same number of rows"):
            fit_model(
                X, y, task="classification", model_name="logistic_regression",
                X_eval=X.iloc[:10], y_eval=y.iloc[:5],
            )

    def test_fit_model_with_sample_weight(self, binary_int_data):
        X, y = binary_int_data
        weights = np.ones(len(X))
        result = fit_model(X, y, task="classification", model_name="logistic_regression", sample_weight=weights)
        assert is_classifier(result.model)

    def test_predict_requires_fitted_model(self, binary_int_data):
        X, _ = binary_int_data
        with pytest.raises(NotFittedError):
            predict(LogisticRegression(), X)

    def test_predict_on_fitted_model(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(X, y, task="classification", model_name="logistic_regression")
        preds = predict(result.model, X)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# evaluate_classification / evaluate_regression
# ---------------------------------------------------------------------------

class TestEvaluateClassification:
    def test_default_metrics(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1]
        metrics = evaluate_classification(y_true, y_pred)
        assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
        assert 0 <= metrics["accuracy"] <= 1

    def test_roc_auc_requires_probability(self):
        with pytest.raises(ValueError, match="y_probability is required"):
            evaluate_classification([0, 1, 0, 1], [0, 1, 0, 0], metrics=["roc_auc"])

    def test_roc_auc_binary(self):
        y_true = [0, 1, 0, 1]
        proba = [0.1, 0.9, 0.2, 0.8]
        metrics = evaluate_classification(y_true, [0, 1, 0, 1], y_probability=proba, metrics=["roc_auc"])
        assert metrics["roc_auc"] == 1.0

    def test_roc_auc_multiclass(self):
        y_true = [0, 1, 2, 0, 1, 2]
        proba = np.array([
            [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
        ])
        metrics = evaluate_classification(y_true, [0, 1, 2, 0, 1, 2], y_probability=proba, metrics=["roc_auc"])
        assert metrics["roc_auc"] > 0.9

    def test_average_precision_requires_binary_vector(self):
        proba_2d = np.array([[0.5, 0.5]] * 4)
        with pytest.raises(ValueError, match="binary probability vector"):
            evaluate_classification([0, 1, 0, 1], [0, 1, 0, 1], y_probability=proba_2d, metrics=["average_precision"])

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="Unsupported classification metric"):
            evaluate_classification([0, 1], [0, 1], metrics=["not_a_metric"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            evaluate_classification([0, 1, 0], [0, 1])

    def test_sample_weight_changes_result(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        unweighted = evaluate_classification(y_true, y_pred)
        weighted = evaluate_classification(y_true, y_pred, sample_weight=[10, 0.001, 1, 1])
        assert unweighted["accuracy"] != weighted["accuracy"]


class TestEvaluateRegression:
    def test_default_metrics(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        metrics = evaluate_regression(y_true, y_pred)
        assert set(metrics) == {"mae", "mse", "rmse", "r2"}
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))

    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0]
        metrics = evaluate_regression(y, y)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_rejects_non_finite_predictions(self):
        with pytest.raises(ValueError, match="finite"):
            evaluate_regression([1.0, 2.0], [1.0, np.inf])

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="Unsupported regression metric"):
            evaluate_regression([1.0, 2.0], [1.0, 2.0], metrics=["not_a_metric"])


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

class TestCrossValidateModel:
    def test_single_metric_shape(self, binary_int_data):
        X, y = binary_int_data
        result = cross_validate_model(X, y, task="classification", model_name="logistic_regression", cv=3)
        assert "accuracy" in result["metrics"]
        assert set(result["metrics"]["accuracy"]) == {"scores", "mean", "std"}
        assert len(result["metrics"]["accuracy"]["scores"]) == 3
        assert result["mean_score"] == result["metrics"]["accuracy"]["mean"]

    def test_multi_metric_shape(self, binary_int_data):
        X, y = binary_int_data
        result = cross_validate_model(
            X, y, task="classification", model_name="logistic_regression",
            cv=3, scoring=["accuracy", "f1_weighted"],
        )
        assert set(result["metrics"]) == {"accuracy", "f1_weighted"}

    def test_result_is_json_serializable(self, binary_int_data):
        X, y = binary_int_data
        result = cross_validate_model(X, y, task="classification", model_name="logistic_regression", cv=3, sample_weight=np.ones(len(X)))
        json.dumps(result)  # must not raise

    def test_sample_weight_accepted(self, regression_data):
        X, y = regression_data
        weights = np.ones(len(X))
        result = cross_validate_model(X, y, task="regression", model_name="linear_regression", cv=3, sample_weight=weights)
        assert "r2" in result["metrics"]

    def test_accepts_custom_cv_splitter(self, regression_data):
        X, y = regression_data
        result = cross_validate_model(X, y, task="regression", model_name="linear_regression", cv=TimeSeriesSplit(n_splits=4))
        assert result["cv"] == "TimeSeriesSplit"
        assert len(result["scores"]) == 4

    def test_accepts_estimator_instance(self, binary_int_data):
        X, y = binary_int_data
        result = cross_validate_model(X, y, task="classification", model_name=RandomForestClassifier(n_estimators=10, random_state=0), cv=3)
        assert result["model"] == "RandomForestClassifier"

    def test_rejects_cv_below_two(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="cv must be"):
            cross_validate_model(X, y, task="classification", model_name="logistic_regression", cv=1)


# ---------------------------------------------------------------------------
# generate_oof_predictions -- including the label-mapping bug fix matrix
# ---------------------------------------------------------------------------

class TestOutOfFoldPredictions:
    def test_oof_regression(self, regression_data):
        X, y = regression_data
        result = generate_oof_predictions(X, y, task="regression", model_name="linear_regression", cv=4)
        assert len(result["predictions"]) == len(X)
        assert result["metrics"]["r2"] > 0.5

    def test_oof_binary_predict_method(self, binary_int_data):
        X, y = binary_int_data
        result = generate_oof_predictions(X, y, task="classification", model_name="logistic_regression", cv=4, method="predict")
        assert result["metrics"]["accuracy"] > 0.7

    @pytest.mark.parametrize(
        "fixture_name",
        ["binary_int_data", "binary_str_data", "multiclass_int_data", "multiclass_str_data"],
    )
    def test_oof_predict_proba_label_mapping_across_encodings(self, fixture_name, request):
        # Regression test for the bug where OOF predict_proba results were
        # mapped back via raw np.argmax indices instead of actual class
        # labels -- silently wrong for any non-{0,1}-int-coded target.
        X, y = request.getfixturevalue(fixture_name)
        result = generate_oof_predictions(X, y, task="classification", model_name="logistic_regression", cv=4, method="predict_proba")
        predicted_labels = set(np.unique(result["predictions"].argmax(axis=1)))
        # predictions here are still raw probability arrays; check the
        # metrics (which internally perform the same label mapping) are
        # sane rather than near-chance, which is what the bug produced.
        assert result["metrics"]["accuracy"] > 0.5, (
            f"OOF accuracy for {fixture_name} looks like the label-mapping "
            "bug regressed (near-chance accuracy)."
        )

    def test_oof_rejects_bad_method(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="method must be"):
            generate_oof_predictions(X, y, task="classification", model_name="logistic_regression", method="not_a_method")

    def test_oof_accepts_custom_cv_splitter(self, regression_data):
        X, y = regression_data
        from sklearn.model_selection import KFold
        result = generate_oof_predictions(X, y, task="regression", model_name="linear_regression", cv=KFold(n_splits=4, shuffle=True, random_state=0))
        assert len(result["predictions"]) == len(X)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_native_importance_tree_model(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(X, y, task="classification", model_name="random_forest")
        fi = get_feature_importance(result.model, feature_names=list(X.columns))
        assert list(fi.columns) == ["feature", "importance"]
        assert list(fi["importance"]) == sorted(fi["importance"], reverse=True)
        assert set(fi["feature"]) == set(X.columns)

    def test_native_importance_linear_model_uses_coef(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(X, y, task="classification", model_name="logistic_regression")
        fi = get_feature_importance(result.model, feature_names=list(X.columns))
        assert len(fi) == X.shape[1]

    def test_requires_fitted_model(self, binary_int_data):
        X, _ = binary_int_data
        with pytest.raises(NotFittedError):
            get_feature_importance(RandomForestClassifier(), feature_names=list(X.columns))

    def test_accepts_model_wrapper_directly_resolving_feature_columns(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="random_forest").fit(X, y)
        fi = get_feature_importance(m)  # no feature_names -- must resolve via m.feature_columns_
        assert set(fi["feature"]) == set(X.columns)

    def test_falls_back_to_sklearn_feature_names_in(self, binary_int_data):
        X, y = binary_int_data
        raw = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        fi = get_feature_importance(raw)  # no feature_names -- must resolve via feature_names_in_
        assert set(fi["feature"]) == set(X.columns)

    def test_raises_when_no_feature_name_source_available(self, binary_int_data):
        X, y = binary_int_data
        raw = RandomForestClassifier(n_estimators=10, random_state=0).fit(X.to_numpy(), y)
        with pytest.raises(ValueError, match="feature_names must be supplied"):
            get_feature_importance(raw)

    def test_explicit_empty_feature_names_is_respected_not_overridden(self, binary_int_data):
        X, y = binary_int_data
        raw = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="does not match"):
            get_feature_importance(raw, feature_names=[])

    def test_explicit_feature_names_take_priority_over_fallbacks(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="random_forest").fit(X, y)
        custom_names = [f"custom_{i}" for i in range(X.shape[1])]
        fi = get_feature_importance(m, feature_names=custom_names)
        assert set(fi["feature"]) == set(custom_names)

    def test_permutation_importance(self, binary_int_data):
        X, y = binary_int_data
        result = fit_model(X, y, task="classification", model_name="logistic_regression")
        fi = permutation_feature_importance(result.model, X, y, n_repeats=5, random_state=0)
        assert set(fi.columns) == {"feature", "importance_mean", "importance_std"}
        assert len(fi) == X.shape[1]


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_comparison_sorted_best_first(self, binary_int_data):
        X, y = binary_int_data
        comparison = compare_models(
            X, y, task="classification",
            model_names=["logistic_regression", "random_forest"], cv=3,
        )
        assert isinstance(comparison, ModelComparison)
        scores = comparison.results["mean_score"].tolist()
        assert scores == sorted(scores, reverse=True), "results must be sorted best-first"

    def test_best_model_matches_top_row(self, binary_int_data):
        X, y = binary_int_data
        comparison = compare_models(X, y, task="classification", model_names=["logistic_regression", "random_forest"], cv=3)
        name, score = comparison.best_model()
        assert name == comparison.results.iloc[0]["model"]
        assert score == comparison.results.iloc[0]["mean_score"]

    def test_skips_models_needing_uninstalled_libraries(self, binary_int_data):
        X, y = binary_int_data
        # "xgboost" is a real registry entry but the library itself may
        # not be installed in this environment -- that should be skipped
        # gracefully, not crash the whole comparison.
        comparison = compare_models(
            X, y, task="classification",
            model_names=["logistic_regression", "xgboost"],
            cv=3,
        )
        assert "logistic_regression" in comparison.results["model"].values
        # if xgboost genuinely isn't installed here, it must show up as skipped
        import importlib
        try:
            importlib.import_module("xgboost")
            xgboost_installed = True
        except ImportError:
            xgboost_installed = False
        if not xgboost_installed:
            assert "xgboost" in comparison.results.attrs.get("skipped_models", {})

    def test_unknown_model_name_raises_loudly_rather_than_silently_skipped(self, binary_int_data):
        # A typo'd/unregistered model_name is a user error, not a missing
        # optional dependency -- it must not be silently swallowed the
        # same way an ImportError for xgboost/lightgbm/catboost is.
        X, y = binary_int_data
        with pytest.raises(ValueError, match="Unknown model_name"):
            compare_models(X, y, task="classification", model_names=["logistic_regression", "totally_made_up_name"], cv=3)

    def test_refit_true_produces_models_fit_on_full_data(self, binary_int_data):
        X, y = binary_int_data
        comparison = compare_models(X, y, task="classification", model_names=["logistic_regression"], cv=3, refit=True)
        assert "logistic_regression" in comparison.models
        fitted = comparison.models["logistic_regression"]
        assert is_classifier(fitted)
        preds = fitted.predict(X)
        assert len(preds) == len(X)

    def test_raises_if_no_models_could_be_evaluated(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError):
            compare_models(X, y, task="classification", model_names=["totally_unknown_model_xyz"], cv=3)


# ---------------------------------------------------------------------------
# grid_search / random_search
# ---------------------------------------------------------------------------

class TestHyperparameterSearch:
    def test_grid_search_returns_best_params(self, binary_int_data):
        X, y = binary_int_data
        search = grid_search(
            X, y, task="classification", estimator=LogisticRegression(max_iter=500),
            param_grid={"C": [0.1, 1.0, 10.0]}, cv=3,
        )
        assert search.best_params_["C"] in {0.1, 1.0, 10.0}

    def test_grid_search_rejects_empty_param_grid(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(TypeError, match="non-empty mapping"):
            grid_search(X, y, task="classification", estimator=LogisticRegression(), param_grid={}, cv=3)

    def test_random_search_returns_best_params(self, binary_int_data):
        X, y = binary_int_data
        search = random_search(
            X, y, task="classification", estimator=LogisticRegression(max_iter=500),
            param_distributions={"C": [0.1, 1.0, 10.0]}, n_iter=2, cv=3, random_state=0,
        )
        assert search.best_params_["C"] in {0.1, 1.0, 10.0}

    def test_random_search_rejects_bad_n_iter(self, binary_int_data):
        X, y = binary_int_data
        with pytest.raises(ValueError, match="n_iter"):
            random_search(X, y, task="classification", estimator=LogisticRegression(), param_distributions={"C": [1.0]}, n_iter=0, cv=3)


# ---------------------------------------------------------------------------
# Calibration and thresholding
# ---------------------------------------------------------------------------

class TestCalibrationAndThreshold:
    def test_calibrate_raw_estimator(self, binary_int_data):
        X, y = binary_int_data
        fitted = LogisticRegression(max_iter=500).fit(X, y)
        calibrated = calibrate_model(fitted, X, y, method="sigmoid", cv=3)
        proba = calibrated.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_calibrate_model_wrapper(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        calibrated = calibrate_model(m, X, y, method="sigmoid", cv=3)
        assert calibrated.predict_proba(X).shape[0] == len(X)

    def test_calibrate_rejects_regressor(self, regression_data):
        X, y = regression_data
        fitted = LinearRegression().fit(X, y)
        with pytest.raises(TypeError, match="classifier"):
            calibrate_model(fitted, X, y)

    def test_calibrate_rejects_bad_method(self, binary_int_data):
        X, y = binary_int_data
        fitted = LogisticRegression(max_iter=500).fit(X, y)
        with pytest.raises(ValueError, match="method must be"):
            calibrate_model(fitted, X, y, method="not_a_method")

    def test_find_optimal_threshold_int_labels(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        proba = m.predict_proba(X).iloc[:, 1].to_numpy()
        result = find_optimal_threshold(y, proba, objective="f1")
        assert 0.0 <= result["threshold"] <= 1.0
        assert result["score"] > 0.5

    def test_find_optimal_threshold_string_labels(self, binary_str_data):
        # Regression test for the bug where thresholded 0/1 predictions
        # were compared directly against non-{0,1} labels.
        X, y = binary_str_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        proba = m.predict_proba(X).iloc[:, 1].to_numpy()
        result = find_optimal_threshold(y, proba, objective="f1")
        assert result["score"] > 0.5

    def test_find_optimal_threshold_rejects_non_binary_target(self, multiclass_int_data):
        X, y = multiclass_int_data
        proba = np.random.default_rng(0).uniform(size=len(y))
        with pytest.raises(ValueError, match="binary classification"):
            find_optimal_threshold(y, proba)

    def test_find_optimal_threshold_rejects_bad_probabilities(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            find_optimal_threshold([0, 1, 0, 1], [0.1, 1.5, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_classification_diagnostics_includes_confusion_matrix(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        pred = m.predict(X)
        proba = m.predict_proba(X).iloc[:, 1].to_numpy()
        diag = classification_diagnostics(y, pred, proba)
        assert "confusion_matrix" in diag
        assert np.array(diag["confusion_matrix"]).sum() == len(X)
        assert "roc_auc" in diag["metrics"]

    def test_regression_diagnostics_residual_stats(self, regression_data):
        X, y = regression_data
        m = Model(task="regression", model_name="linear_regression").fit(X, y)
        pred = m.predict(X)
        diag = regression_diagnostics(y, pred)
        assert diag["residual_min"] <= diag["residual_mean"] <= diag["residual_max"]
        assert diag["n_observations"] == len(X)


# ---------------------------------------------------------------------------
# dataset_fingerprint / compute_artifact_checksum
# ---------------------------------------------------------------------------

class TestFingerprintingAndChecksums:
    def test_fingerprint_is_stable(self, binary_int_data):
        X, y = binary_int_data
        assert dataset_fingerprint(X, y) == dataset_fingerprint(X, y)

    def test_fingerprint_changes_with_data(self, binary_int_data):
        X, y = binary_int_data
        X2 = X.copy()
        X2.iloc[0, 0] += 1.0
        assert dataset_fingerprint(X, y) != dataset_fingerprint(X2, y)

    def test_fingerprint_raises_clear_error_on_unhashable_data(self):
        bad = pd.DataFrame({"a": [[1, 2], [3, 4]]})
        with pytest.raises(TypeError, match="could not hash"):
            dataset_fingerprint(bad)

    def test_checksum_matches_for_unchanged_file(self, tmp_path, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        path = m.save(tmp_path / "model.joblib")
        assert compute_artifact_checksum(path) == compute_artifact_checksum(path)

    def test_checksum_differs_for_different_files(self, tmp_path, binary_int_data, regression_data):
        Xc, yc = binary_int_data
        Xr, yr = regression_data
        m1 = Model(task="classification", model_name="logistic_regression").fit(Xc, yc)
        m2 = Model(task="regression", model_name="linear_regression").fit(Xr, yr)
        p1 = m1.save(tmp_path / "m1.joblib")
        p2 = m2.save(tmp_path / "m2.joblib")
        assert compute_artifact_checksum(p1) != compute_artifact_checksum(p2)


# ---------------------------------------------------------------------------
# Model wrapper: fit / predict / evaluate
# ---------------------------------------------------------------------------

class TestModelLifecycle:
    def test_fit_predict_classification(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds.unique()).issubset(set(y.unique()))

    def test_fit_predict_regression(self, regression_data):
        X, y = regression_data
        m = Model(task="regression", model_name="linear_regression").fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)

    def test_predict_before_fit_raises(self, binary_int_data):
        X, _ = binary_int_data
        m = Model(task="classification", model_name="logistic_regression")
        with pytest.raises(NotFittedError):
            m.predict(X)

    def test_predict_proba_classification(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_rejects_regression_task(self, regression_data):
        X, y = regression_data
        m = Model(task="regression", model_name="linear_regression").fit(X, y)
        with pytest.raises(TypeError, match="classification"):
            m.predict_proba(X)

    def test_schema_protection_rejects_missing_columns(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        with pytest.raises(ValueError, match="Missing feature columns"):
            m.predict(X.drop(columns=[X.columns[0]]))

    def test_schema_protection_rejects_extra_columns(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        X_extra = X.copy()
        X_extra["extra_col"] = 0.0
        with pytest.raises(ValueError, match="Unexpected feature columns"):
            m.predict(X_extra)

    def test_evaluate_classification(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        metrics = m.evaluate(X, y, metrics=["accuracy", "roc_auc"])
        assert metrics["accuracy"] > 0.5
        assert 0 <= metrics["roc_auc"] <= 1

    def test_evaluate_regression(self, regression_data):
        X, y = regression_data
        m = Model(task="regression", model_name="linear_regression").fit(X, y)
        metrics = m.evaluate(X, y)
        assert "r2" in metrics

    def test_evaluate_result_carries_predictions_and_probabilities(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        result = m.evaluate_result(X, y, metrics=["accuracy", "roc_auc"])
        assert isinstance(result, ModelEvaluation)
        assert result.probabilities is not None
        as_dict = result.to_dict()
        json.dumps(as_dict)  # must be JSON serializable

    def test_cross_validate_on_model(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        df = m.cross_validate(X, y, cv=3)
        assert "test_accuracy" in df.columns or "test_score" in df.columns

    def test_cross_validate_with_sample_weight(self, regression_data):
        X, y = regression_data
        m = Model(task="regression", model_name="linear_regression").fit(X, y)
        df = m.cross_validate(X, y, cv=3, sample_weight=np.ones(len(X)))
        assert len(df) == 3

    def test_feature_importance_method(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="random_forest").fit(X, y)
        fi = m.feature_importance()
        assert set(fi["feature"]) == set(X.columns)

    def test_permutation_importance_method(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        fi = m.permutation_importance(X, y, n_repeats=5)
        assert len(fi) == X.shape[1]

    def test_artifact_summary(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression", target="target").fit(X, y)
        artifact = m.artifact(metrics={"accuracy": 0.9})
        assert isinstance(artifact, ModelArtifact)
        summary = artifact.summary()
        assert summary["target"] == "target"
        assert summary["feature_count"] == X.shape[1]
        json.dumps(summary)

    def test_repr_reflects_fitted_state(self, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression")
        assert "not fitted" in repr(m)
        m.fit(X, y)
        assert "status=fitted" in repr(m)


# ---------------------------------------------------------------------------
# Model wrapper: save / load
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        path = m.save(tmp_path / "model.joblib")
        loaded = Model.load(path)
        pd.testing.assert_series_equal(m.predict(X), loaded.predict(X))
        assert loaded.feature_columns_ == m.feature_columns_
        assert loaded.task == m.task

    def test_load_with_correct_checksum_succeeds(self, tmp_path, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        path = m.save(tmp_path / "model.joblib")
        checksum = compute_artifact_checksum(path)
        loaded = Model.load(path, expected_sha256=checksum)
        assert loaded._fitted

    def test_load_with_wrong_checksum_raises(self, tmp_path, binary_int_data):
        X, y = binary_int_data
        m = Model(task="classification", model_name="logistic_regression").fit(X, y)
        path = m.save(tmp_path / "model.joblib")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            Model.load(path, expected_sha256="0" * 64)

    def test_save_requires_fitted_model(self, tmp_path):
        m = Model(task="classification", model_name="logistic_regression")
        with pytest.raises(NotFittedError):
            m.save(tmp_path / "model.joblib")

    def test_load_rejects_non_makoding_file(self, tmp_path):
        import joblib

        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"unrelated": "payload"}, path)
        with pytest.raises(ValueError, match="not a valid Makoding model artifact"):
            Model.load(path)


# ---------------------------------------------------------------------------
# Model wrapper: get_params / set_params
# ---------------------------------------------------------------------------

class TestModelParams:
    def test_get_params_includes_wrapper_and_estimator_keys(self):
        m = Model(task="classification", model_name="logistic_regression", random_state=42)
        params = m.get_params()
        assert params["task"] == "classification"
        assert params["model_name"] == "logistic_regression"
        assert params["random_state"] == 42
        assert "C" in params  # flattened estimator params

    def test_get_params_set_params_round_trip(self):
        m1 = Model(task="classification", model_name="logistic_regression", random_state=7)
        params = m1.get_params()
        m2 = Model(task="classification", model_name="logistic_regression")
        estimator_only = {
            k: v for k, v in params.items()
            if k not in {"task", "model_name", "random_state", "target"}
        }
        m2.set_params(**estimator_only)
        assert m2.estimator.get_params()["C"] == params["C"]

    def test_set_params_flat_estimator_kwarg(self):
        m = Model(task="classification", model_name="logistic_regression").fit(
            pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}), pd.Series([0, 1, 0, 1])
        )
        m.set_params(C=0.5)
        assert m.estimator.get_params()["C"] == 0.5
        assert not m._fitted  # any param change invalidates the fit

    def test_set_params_random_state_syncs_into_estimator(self):
        m = Model(task="classification", model_name="logistic_regression", random_state=42)
        assert m.estimator.get_params()["random_state"] == 42
        m.set_params(random_state=123)
        assert m.random_state == 123
        assert m.estimator.get_params()["random_state"] == 123

    def test_set_params_model_name_change_rebuilds_matching_estimator(self):
        m = Model(task="classification", model_name="logistic_regression")
        m.set_params(model_name="random_forest")
        assert isinstance(m.estimator, RandomForestClassifier)

    def test_set_params_task_change_rebuilds_task_compatible_estimator(self):
        m = Model(task="classification", model_name="random_forest", random_state=42)
        m.set_params(task="regression")
        assert m.task == "regression"
        assert is_regressor(m.estimator)
        assert not m._fitted

    def test_set_params_task_change_raises_clearly_when_model_name_incompatible(self):
        m = Model(task="classification", model_name="logistic_regression", random_state=42)
        with pytest.raises(ValueError, match="Unknown model_name"):
            m.set_params(task="regression")
        # a failed attempt must leave the model untouched
        assert m.task == "classification"
        assert is_classifier(m.estimator)

    def test_set_params_explicit_estimator_swap(self):
        m = Model(task="classification", model_name="logistic_regression")
        m.set_params(estimator=RandomForestClassifier(n_estimators=10))
        assert isinstance(m.estimator, RandomForestClassifier)

    def test_set_params_explicit_estimator_validated_against_task(self):
        m = Model(task="classification", model_name="logistic_regression")
        with pytest.raises(TypeError, match="regressor"):
            m.set_params(estimator=LinearRegression())

    def test_set_params_target_update(self):
        m = Model(task="classification", model_name="logistic_regression", target="old_target")
        m.set_params(target="new_target")
        assert m.target == "new_target"


# ---------------------------------------------------------------------------
# Model wrapper: constructor edge cases
# ---------------------------------------------------------------------------

class TestModelConstruction:
    def test_estimator_task_mismatch_detected_at_construction(self):
        with pytest.raises(TypeError, match="regressor"):
            Model(task="classification", estimator=LinearRegression())

    def test_falsy_but_valid_estimator_is_not_discarded(self):
        class FalsyEstimator(LogisticRegression):
            def __len__(self):
                return 0  # falsy, but a perfectly valid estimator

        fe = FalsyEstimator(max_iter=200)
        m = Model(task="classification", estimator=fe)
        assert m.estimator is fe

    def test_default_model_name_from_estimator_class(self):
        m = Model(task="classification", estimator=RandomForestClassifier())
        assert m.model_name == "RandomForestClassifier"

    def test_forwards_model_params_to_registry_build(self):
        m = Model(task="classification", model_name="random_forest", n_estimators=33)
        assert m.estimator.n_estimators == 33