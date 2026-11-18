"""Automated test suite verifying model artifacts, schema, and training outputs."""

import json
from pathlib import Path
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from predict import FEATURE_COLUMNS, LABEL_MAPPING


@pytest.fixture(scope="module")
def models_dir():
    """Returns the models directory path."""
    return Path(__file__).resolve().parent.parent / "models"


@pytest.fixture(scope="module")
def model_path(models_dir):
    """Returns the path to the trained Random Forest model (v3 default, fallback v2/v1)."""
    v3 = models_dir / "random_forest_8fitur_v3.joblib"
    if v3.is_file():
        return v3
    v2 = models_dir / "random_forest_8fitur_v2.joblib"
    return v2 if v2.is_file() else models_dir / "random_forest_8fitur.joblib"


@pytest.fixture(scope="module")
def meta_path(models_dir):
    """Returns the path to the model metadata JSON (v3 default, fallback v2/v1)."""
    v3 = models_dir / "model_meta_v3.json"
    if v3.is_file():
        return v3
    v2 = models_dir / "model_meta_v2.json"
    return v2 if v2.is_file() else models_dir / "model_meta.json"


def test_model_artifact_exists_and_loads(model_path):
    """Verifies the Random Forest .joblib artifact exists and is loadable."""
    assert model_path.is_file(), f"Model file missing: {model_path}"
    model = joblib.load(model_path)
    assert isinstance(model, RandomForestClassifier)


def test_model_expects_exact_8_features(model_path):
    """Verifies that the model is trained on exactly 8 features in canonical order."""
    model = joblib.load(model_path)
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == 8

    # Verify feature names if saved in model
    if hasattr(model, "feature_names_in_"):
        assert list(model.feature_names_in_) == FEATURE_COLUMNS


def test_metadata_file_validity(meta_path):
    """Verifies that model_meta.json exists and contains required keys and valid types."""
    assert meta_path.is_file(), f"Metadata file missing: {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "model_name" in meta
    assert meta["model_name"] == "RandomForestClassifier"
    assert "scikit_learn_version" in meta
    assert "features" in meta
    assert meta["features"]["count"] == 8
    assert meta["features"]["order"] == FEATURE_COLUMNS

    assert "label_mapping" in meta
    # Keys in JSON are strings
    loaded_labels = {int(k): v for k, v in meta["label_mapping"].items()}
    assert loaded_labels == LABEL_MAPPING

    assert "dataset" in meta
    assert meta["dataset"]["total_rows"] == 6229
    assert meta["dataset"]["train_rows"] == 4360
    assert meta["dataset"]["test_rows"] == 1869

    assert "evaluation" in meta
    rf_eval = meta["evaluation"]["random_forest"]
    assert 0.70 <= rf_eval["test_accuracy"] <= 1.0
    assert "cv_5fold" in rf_eval
    assert rf_eval["cv_5fold"]["mean"] >= 0.70


def test_feature_order_integrity():
    """Verifies canonical 8-feature order against contract specification."""
    expected_order = [
        "attendance_frequency_7d",
        "missed_session_count_7d",
        "reschedule_frequency_7d",
        "remaining_package_sessions",
        "days_until_package_expiry",
        "days_since_last_visit",
        "reminder_count_7d",
        "reminder_response_count_7d",
    ]
    assert FEATURE_COLUMNS == expected_order
