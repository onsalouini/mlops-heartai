import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

client = TestClient(app)

# ── Tests route GET / ──────────────────────────────────────
class TestHome:
    def test_status_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_json(self):
        r = client.get("/")
        assert r.headers["content-type"] == "application/json"

    def test_message_key(self):
        r = client.get("/")
        assert "message" in r.json()

# ── Tests route GET /health ────────────────────────────────
class TestHealth:
    def test_status_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

# ── Tests route POST /predict ──────────────────────────────
class TestPredict:
    VALID = {"features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]}
    VALID2 = {"features": [37,1,2,130,250,0,1,187,0,3.5,0,0,2]}

    def test_status_200(self):
        r = client.post("/predict", json=self.VALID)
        assert r.status_code == 200

    def test_prediction_key(self):
        r = client.post("/predict", json=self.VALID)
        assert "prediction" in r.json()

    def test_label_key(self):
        r = client.post("/predict", json=self.VALID)
        assert "label" in r.json()

    def test_prediction_binary(self):
        r = client.post("/predict", json=self.VALID)
        assert r.json()["prediction"] in [0, 1]

    def test_known_case_positive(self):
        r = client.post("/predict", json=self.VALID)
        assert r.json()["prediction"] == 1

    def test_second_sample(self):
        r = client.post("/predict", json=self.VALID2)
        assert r.status_code == 200

    def test_wrong_features_count(self):
        r = client.post("/predict", json={"features": [1, 2, 3]})
        assert r.status_code in [400, 422]

    def test_empty_features(self):
        r = client.post("/predict", json={"features": []})
        assert r.status_code in [400, 422]

    def test_missing_body(self):
        r = client.post("/predict", json={})
        assert r.status_code == 422
