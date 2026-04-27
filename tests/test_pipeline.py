import pytest
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

DATA_PATH = "heart.csv"

@pytest.fixture(scope="module")
def data():
    return prepare_data(DATA_PATH)

@pytest.fixture(scope="module")
def trained_model(data):
    X_train, X_test, y_train, y_test = data
    return train_model(X_train, y_train)

# ── Tests prepare_data ─────────────────────────────────────
class TestPrepareData:
    def test_returns_four_elements(self, data):
        assert len(data) == 4

    def test_train_size(self, data):
        X_train, X_test, y_train, y_test = data
        assert len(X_train) > 0

    def test_test_size(self, data):
        X_train, X_test, y_train, y_test = data
        assert len(X_test) > 0

    def test_correct_split_ratio(self, data):
        X_train, X_test, y_train, y_test = data
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_features_count(self, data):
        X_train, X_test, y_train, y_test = data
        assert X_train.shape[1] == 13

    def test_scaler_saved(self):
        assert os.path.exists("scaler.pkl")

    def test_labels_binary(self, data):
        X_train, X_test, y_train, y_test = data
        assert set(y_train.unique()).issubset({0, 1})

# ── Tests train_model ──────────────────────────────────────
class TestTrainModel:
    def test_returns_mlp_classifier(self, trained_model):
        assert isinstance(trained_model, MLPClassifier)

    def test_model_is_fitted(self, trained_model):
        assert hasattr(trained_model, "coefs_")

    def test_model_can_predict(self, trained_model, data):
        X_train, X_test, y_train, y_test = data
        preds = trained_model.predict(X_test[:5])
        assert len(preds) == 5

    def test_predictions_binary(self, trained_model, data):
        X_train, X_test, y_train, y_test = data
        preds = trained_model.predict(X_test)
        assert set(preds).issubset({0, 1})

# ── Tests evaluate_model ───────────────────────────────────
class TestEvaluateModel:
    def test_returns_dict(self, trained_model, data):
        X_train, X_test, y_train, y_test = data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert isinstance(metrics, dict)

    def test_accuracy_key(self, trained_model, data):
        X_train, X_test, y_train, y_test = data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert "accuracy" in metrics

    def test_f1_key(self, trained_model, data):
        X_train, X_test, y_train, y_test = data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert "f1_score" in metrics

    def test_accuracy_above_threshold(self, trained_model, data):
        X_train, X_test, y_train, y_test = data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert metrics["accuracy"] >= 0.70

    def test_auc_above_threshold(self, trained_model, data):
        X_train, X_test, y_train, y_test = data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert metrics["auc_roc"] >= 0.75

# ── Tests save/load model ──────────────────────────────────
class TestSaveLoadModel:
    def test_save_creates_file(self, trained_model):
        save_model(trained_model)
        assert os.path.exists("heart_nn_model.pkl")

    def test_load_returns_model(self):
        model, scaler = load_model()
        assert model is not None
        assert scaler is not None

    def test_loaded_model_predicts(self):
        model, scaler = load_model()
        sample = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)
        assert pred[0] in [0, 1]
