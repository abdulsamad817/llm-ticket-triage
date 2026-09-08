"""
Tests the API layer with the model mocked out, so these tests run fast and
don't require a GPU or downloaded weights — CI can run these on every push.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.inference.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_empty_message():
    response = client.post("/predict", json={"message": ""})
    assert response.status_code == 422


@patch("src.inference.api.get_model")
def test_predict_returns_structured_fields_on_success(mock_get_model):
    mock_model = mock_get_model.return_value
    mock_model.predict.return_value = {
        "intent": "billing_dispute",
        "sentiment": "negative",
        "urgency": "high",
        "category": "billing",
        "summary": "Customer charged twice",
    }

    response = client.post("/predict", json={"message": "I was charged twice for my order"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "billing_dispute"
    assert body["urgency"] == "high"


@patch("src.inference.api.get_model")
def test_predict_returns_422_on_model_error(mock_get_model):
    mock_model = mock_get_model.return_value
    mock_model.predict.return_value = {"error": "Model did not return valid JSON.", "raw_output": "garbage"}

    response = client.post("/predict", json={"message": "some message"})
    assert response.status_code == 422
    assert "error" in response.json()["detail"]