"""Tests for classification module."""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.classification.classifiers import (
    get_classifiers, train_and_evaluate, run_all_classifiers, compare_feature_sets
)
from src.utils.evaluation import compute_classification_metrics


def _make_dummy_data():
    """Create small dummy sparse data for classifier testing."""
    np.random.seed(42)
    X_train = csr_matrix(np.random.rand(60, 20))
    X_val = csr_matrix(np.random.rand(15, 20))
    X_test = csr_matrix(np.random.rand(15, 20))
    y_train = np.array([0, 1, 2] * 20)
    y_val = np.array([0, 1, 2] * 5)
    y_test = np.array([0, 1, 2] * 5)
    return X_train, y_train, X_val, y_val, X_test, y_test


class TestGetClassifiers:
    def test_returns_three(self):
        clfs = get_classifiers(mode="cpu")
        assert len(clfs) == 3

    def test_expected_names(self):
        clfs = get_classifiers(mode="cpu")
        expected = {"RandomForest", "HistGBT_sklearn", "LinearSVC"}
        assert set(clfs.keys()) == expected

    def test_all_have_fit_predict(self):
        for name, clf in get_classifiers(mode="cpu").items():
            assert hasattr(clf, "fit"), f"{name} missing fit()"
            assert hasattr(clf, "predict"), f"{name} missing predict()"


class TestTrainAndEvaluate:
    def test_returns_metrics_and_predictions(self):
        X_train, y_train, _, _, X_test, y_test = _make_dummy_data()
        clf = get_classifiers(mode="cpu")["RandomForest"]
        metrics, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, clf, "RF")
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "classifier" in metrics
        assert len(y_pred) == len(y_test)

    def test_predictions_valid(self):
        X_train, y_train, _, _, X_test, y_test = _make_dummy_data()
        clf = get_classifiers(mode="cpu")["RandomForest"]
        _, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, clf, "RF")
        assert set(y_pred).issubset(set(y_train))


class TestRunAllClassifiers:
    def test_returns_dataframe(self):
        X_train, y_train, _, _, X_test, y_test = _make_dummy_data()
        results_df, predictions = run_all_classifiers(
            X_train, y_train, X_test, y_test, mode="test",
            classifiers=get_classifiers(mode="cpu"),
        )
        assert len(results_df) == 3
        assert "classifier" in results_df.columns
        assert "mode" in results_df.columns
        assert all(results_df["mode"] == "test")

    def test_all_classifiers_ran(self):
        X_train, y_train, _, _, X_test, y_test = _make_dummy_data()
        results_df, predictions = run_all_classifiers(
            X_train, y_train, X_test, y_test, classifiers=get_classifiers(mode="cpu"),
        )
        assert len(predictions) == 3

    def test_uses_validation_when_available(self):
        X_train, y_train, X_val, y_val, X_test, y_test = _make_dummy_data()
        results_df, _ = run_all_classifiers(
            X_train, y_train, X_test, y_test, mode="test", X_val=X_val, y_val=y_val,
            classifiers=get_classifiers(mode="cpu"),
        )
        assert results_df["used_validation"].all()
        assert "selection_macro_f1" in results_df.columns
        assert "selected_params" in results_df.columns


class TestCompareFeatureSets:
    def test_supports_multiple_feature_modes(self):
        X_train, y_train, X_val, y_val, X_test, y_test = _make_dummy_data()
        feature_sets = {
            "flat": {"X_train": X_train, "X_val": X_val, "X_test": X_test},
            "section": {"X_train": X_train, "X_val": X_val, "X_test": X_test},
            "hybrid": {"X_train": X_train, "X_val": X_val, "X_test": X_test},
        }
        labels = {"y_train": y_train, "y_val": y_val, "y_test": y_test}

        results_df, predictions, _ = compare_feature_sets(
            feature_sets, labels, classifiers=get_classifiers(mode="cpu"),
        )

        assert set(results_df["mode"]) == {"flat", "section", "hybrid"}
        assert set(predictions.keys()) == {"flat", "section", "hybrid"}
        assert results_df["used_validation"].all()


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_classification_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_random_predictions_low(self):
        y_true = np.array([0, 1, 2] * 10)
        y_pred = np.array([0] * 30)  # always predict 0
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] < 0.5
        assert metrics["macro_f1"] < 0.5

    def test_all_keys_present(self):
        y = np.array([0, 1])
        metrics = compute_classification_metrics(y, y)
        expected_keys = {"accuracy", "macro_f1", "balanced_accuracy",
                         "macro_precision", "macro_recall"}
        assert set(metrics.keys()) == expected_keys
