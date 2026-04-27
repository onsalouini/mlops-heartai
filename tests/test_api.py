import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app, raise_server_exceptions=False)

class TestHome:
    def test_status_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_message_key(self):
        r = client.get("/")
        assert "message" in r.json()

class TestHealth:
    def test_status_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

class TestPredict:
    VALID = {"features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]}

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

    def test_missing_body(self):
        r = client.post("/predict", json={})
        assert r.status_code == 422
