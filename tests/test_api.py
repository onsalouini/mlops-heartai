import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

class TestHome:
    def test_status_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_message_key(self, client):
        r = client.get("/")
        assert "message" in r.json()

class TestHealth:
    def test_status_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

class TestPredict:
    VALID = {"features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]}

    def test_status_200(self, client):
        r = client.post("/predict", json=self.VALID)
        assert r.status_code == 200

    def test_prediction_key(self, client):
        r = client.post("/predict", json=self.VALID)
        assert "prediction" in r.json()

    def test_label_key(self, client):
        r = client.post("/predict", json=self.VALID)
        assert "label" in r.json()

    def test_prediction_binary(self, client):
        r = client.post("/predict", json=self.VALID)
        assert r.json()["prediction"] in [0, 1]

    def test_missing_body(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422
