"""Automated test suite for FastAPI Prediction Service and Inference Module."""

import math
import pytest
from fastapi.testclient import TestClient

from app import app
from predict import FEATURE_COLUMNS, load_model, predict

client = TestClient(app)


@pytest.fixture(scope="module")
def valid_cust1_payload():
    """Benchmark sample matching CUST00001 (Need Attention under v3 learned model)."""
    return {
        "customer_id": "CUST00001",
        "attendance_frequency_7d": 0.0,
        "missed_session_count_7d": 0.0,
        "reschedule_frequency_7d": 0.0,
        "remaining_package_sessions": 10.0,
        "days_until_package_expiry": 30.0,
        "days_since_last_visit": 11.0,
        "reminder_count_7d": 0.0,
        "reminder_response_count_7d": 0.0,
    }


@pytest.fixture(scope="module")
def valid_at_risk_payload():
    """Benchmark sample matching CUST00005 (At Risk under v3 learned model: 0 visits in 7d, 7d absent, 1 missed, 2 reschedules)."""
    return {
        "customer_id": "CUST00005",
        "attendance_frequency_7d": 0.0,
        "missed_session_count_7d": 1.0,
        "reschedule_frequency_7d": 2.0,
        "remaining_package_sessions": 7.0,
        "days_until_package_expiry": 23.0,
        "days_since_last_visit": 7.0,
        "reminder_count_7d": 1.0,
        "reminder_response_count_7d": 0.0,
    }


@pytest.fixture(scope="module")
def valid_active_payload():
    """Benchmark sample matching CUST00007 (Active under v3 learned model: 2 visits in 7d, 0d absence)."""
    return {
        "customer_id": "CUST00007",
        "attendance_frequency_7d": 2.0,
        "missed_session_count_7d": 0.0,
        "reschedule_frequency_7d": 1.0,
        "remaining_package_sessions": 6.0,
        "days_until_package_expiry": 21.0,
        "days_since_last_visit": 0.0,
        "reminder_count_7d": 0.0,
        "reminder_response_count_7d": 0.0,
    }


# Health check tests
def test_api_health_endpoint():
    """Verifies GET /api/health returns HTTP 200, status: 'ok', and model_version."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "random-forest-8fitur-v1"


def test_compat_health_endpoint():
    """Verifies GET /health compatibility route returns HTTP 200, status: 'ok', and model_version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "random-forest-8fitur-v1"


# Prediction endpoint tests
def test_api_predict_cust1(valid_cust1_payload):
    """Verifies POST /api/predict on benchmark sample CUST00001."""
    response = client.post("/api/predict", json=valid_cust1_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "CUST00001"
    assert data["label"] == "Need Attention"
    assert data["segmen"] == "Need Attention"
    assert data["segmen_kode"] == 2
    assert isinstance(data["probabilities"], dict)
    assert "Active" in data["probabilities"]
    assert "At Risk" in data["probabilities"]
    assert "Need Attention" in data["probabilities"]
    assert isinstance(data["top_factors"], list)
    assert len(data["top_factors"]) > 0


def test_compat_predict_cust1(valid_cust1_payload):
    """Verifies POST /predict compatibility route for Express backend."""
    response = client.post("/predict", json=valid_cust1_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Need Attention"
    assert data["segmen_kode"] == 2


def test_predict_at_risk_sample(valid_at_risk_payload):
    """Verifies prediction on an At Risk customer profile."""
    response = client.post("/api/predict", json=valid_at_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "At Risk"
    assert data["segmen_kode"] == 1
    assert data["probabilities"]["At Risk"] > 0.5


def test_predict_active_sample(valid_active_payload):
    """Verifies prediction on an Active customer profile."""
    response = client.post("/api/predict", json=valid_active_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Active"
    assert data["segmen_kode"] == 0
    assert data["probabilities"]["Active"] > 0.5


def test_predict_probability_sum(valid_cust1_payload):
    """Verifies that output probabilities sum to 1.0."""
    response = client.post("/api/predict", json=valid_cust1_payload)
    assert response.status_code == 200
    data = response.json()
    prob_sum = sum(data["probabilities"].values())
    assert math.isclose(prob_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3)


# Validation failure tests (HTTP 422)
def test_predict_missing_feature():
    """Verifies that omitting a required feature triggers HTTP 422 Unprocessable Entity."""
    incomplete_payload = {
        "customer_id": "CUST_INCOMPLETE",
        # Missing attendance_frequency_7d
        "missed_session_count_7d": 0.0,
        "reschedule_frequency_7d": 0.0,
        "remaining_package_sessions": 10.0,
        "days_until_package_expiry": 30.0,
        "days_since_last_visit": 11.0,
        "reminder_count_7d": 0.0,
        "reminder_response_count_7d": 0.0,
    }
    response = client.post("/api/predict", json=incomplete_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_invalid_data_type():
    """Verifies that passing a string for a numeric feature triggers HTTP 422."""
    invalid_payload = {
        "customer_id": "CUST_BAD_TYPE",
        "attendance_frequency_7d": "not-a-number",
        "missed_session_count_7d": 0.0,
        "reschedule_frequency_7d": 0.0,
        "remaining_package_sessions": 10.0,
        "days_until_package_expiry": 30.0,
        "days_since_last_visit": 11.0,
        "reminder_count_7d": 0.0,
        "reminder_response_count_7d": 0.0,
    }
    response = client.post("/api/predict", json=invalid_payload)
    assert response.status_code == 422


# Direct inference module tests
def test_direct_predict_with_sequence(valid_cust1_payload):
    """Verifies predict() function directly accepts a sequence of 8 floats."""
    feature_seq = [valid_cust1_payload[col] for col in FEATURE_COLUMNS]
    res = predict(feature_seq)
    assert res["label"] == "Need Attention"
    assert res["segmen_kode"] == 2
    assert len(res["top_factors"]) >= 1


def test_direct_predict_invalid_feature_count():
    """Verifies predict() rejects sequences with incorrect length."""
    with pytest.raises(ValueError, match="Expected exactly 8 features"):
        predict([1.0, 2.0, 3.0])
