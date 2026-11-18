"""Inference Module for Gymnasium Customer Retention Prediction.

Provides core model loading and prediction functions:
- load_model(): Loads the trained 8-feature RandomForest model and metadata.
- predict(): Predicts customer retention segment, class probabilities, and top factors.
- Rules-based explainability: Explains predictions using domain heuristics combined
  with feature importances (NO heavy/slow SHAP dependency).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd

load_dotenv()

FEATURE_COLUMNS: List[str] = [
    "attendance_frequency_7d",
    "missed_session_count_7d",
    "reschedule_frequency_7d",
    "remaining_package_sessions",
    "days_until_package_expiry",
    "days_since_last_visit",
    "reminder_count_7d",
    "reminder_response_count_7d",
]

LABEL_MAPPING: Dict[int, str] = {
    0: "Active",
    1: "At Risk",
    2: "Need Attention",
}

_MODEL_CACHE: Dict[str, Any] = {}


def get_default_model_paths() -> Tuple[Path, Path]:
    """Resolves default model and metadata paths from config or v3 defaults."""
    base_dir = Path(__file__).resolve().parent
    env_model = os.getenv("MODEL_PATH")
    if env_model:
        model_path = Path(env_model)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
    else:
        v3_path = base_dir / "models" / "random_forest_8fitur_v3.joblib"
        if v3_path.is_file():
            model_path = v3_path
        else:
            model_path = base_dir / "models" / "random_forest_8fitur.joblib"

    if "_v3" in model_path.name:
        meta_path = model_path.parent / "model_meta_v3.json"
    elif "_v2" in model_path.name:
        meta_path = model_path.parent / "model_meta_v2.json"
    else:
        meta_path = model_path.parent / "model_meta.json"

    return model_path, meta_path


def load_model(
    model_path: Union[str, Path, None] = None,
    meta_path: Union[str, Path, None] = None,
    force_reload: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Loads and caches the Random Forest model and associated metadata."""
    default_model, default_meta = get_default_model_paths()
    m_path = Path(model_path) if model_path else default_model
    meta_p = Path(meta_path) if meta_path else default_meta

    cache_key = str(m_path.resolve())
    if not force_reload and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]["model"], _MODEL_CACHE[cache_key]["meta"]

    if not m_path.is_file():
        raise FileNotFoundError(f"Model file not found at: {m_path}")

    model = joblib.load(m_path)

    metadata: Dict[str, Any] = {}
    if meta_p.is_file():
        with open(meta_p, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    _MODEL_CACHE[cache_key] = {"model": model, "meta": metadata}
    return model, metadata


def extract_top_factors(
    features: Dict[str, float],
    predicted_label: str = "",
    feature_importances: Dict[str, float] | None = None,
    max_factors: int = 3,
) -> List[str]:
    """Generates top factors explaining prediction directly from trained model feature importances.
    
    Strictly learned and data-driven: ranks features by model importance and formats
    the customer's actual input values in natural Indonesian sentences.
    """
    sentence_templates = {
        "remaining_package_sessions": "Sisa sesi paket: {v} sesi",
        "days_until_package_expiry": "Masa aktif paket: {v} hari tersisa",
        "attendance_frequency_7d": "Frekuensi kehadiran 7 hari: {v} sesi",
        "days_since_last_visit": "Kunjungan terakhir: {v} hari yang lalu",
        "missed_session_count_7d": "Sesi terlewat dalam 7 hari: {v} sesi",
        "reschedule_frequency_7d": "Frekuensi reschedule 7 hari: {v} kali",
        "reminder_count_7d": "Notifikasi pengingat terkirim: {v} pesan",
        "reminder_response_count_7d": "Respons pengingat 7 hari: {v} kali",
    }

    importances = feature_importances or {
        "remaining_package_sessions": 0.2345,
        "days_until_package_expiry": 0.2088,
        "reminder_count_7d": 0.1621,
        "missed_session_count_7d": 0.1360,
        "days_since_last_visit": 0.0934,
        "attendance_frequency_7d": 0.0916,
        "reminder_response_count_7d": 0.0570,
        "reschedule_frequency_7d": 0.0167,
    }

    sorted_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    factors: List[str] = []
    for feat_name, _ in sorted_features:
        if feat_name in features:
            val = features[feat_name]
            tpl = sentence_templates.get(feat_name, f"{feat_name}: {{v}}")
            factors.append(tpl.format(v=int(val)))
            if len(factors) >= max_factors:
                break
    return factors

    return factors[:4]


def predict(
    features: Union[Dict[str, float], Sequence[float]],
    model_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Performs inference on input features and returns structured response.
    
    Args:
        features: Either a dict mapping feature names to values, or a sequence
                  of 8 numerical values in exact canonical feature order.
        model_path: Optional custom path to model artifact.
        
    Returns:
        Dict with keys: label, segmen, segmen_kode, probabilities, top_factors.
    """
    model, metadata = load_model(model_path=model_path)

    # Normalize input features into canonical dict and feature vector
    if isinstance(features, dict):
        missing = [col for col in FEATURE_COLUMNS if col not in features]
        if missing:
            raise ValueError(f"Input features missing required keys: {missing}")
        feature_dict = {col: float(features[col]) for col in FEATURE_COLUMNS}
        feature_vector = np.array([[feature_dict[col] for col in FEATURE_COLUMNS]])
    elif isinstance(features, (list, tuple, np.ndarray)):
        if len(features) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Expected exactly {len(FEATURE_COLUMNS)} features, received {len(features)}"
            )
        feature_dict = {
            col: float(val) for col, val in zip(FEATURE_COLUMNS, features)
        }
        feature_vector = np.array([list(features)], dtype=float)
    else:
        raise TypeError(f"Unsupported features type: {type(features)}")

    # Create DataFrame with canonical feature column names to preserve feature alignment
    feature_df = pd.DataFrame(feature_vector, columns=FEATURE_COLUMNS)

    # Predict class code and probability distribution
    pred_code = int(model.predict(feature_df)[0])
    label_name = LABEL_MAPPING.get(pred_code, f"Cluster_{pred_code}")

    proba_array = model.predict_proba(feature_df)[0]
    classes = getattr(model, "classes_", [0, 1, 2])
    probabilities: Dict[str, float] = {}
    for cls_code, prob in zip(classes, proba_array):
        cls_name = LABEL_MAPPING.get(int(cls_code), str(cls_code))
        probabilities[cls_name] = round(float(prob), 4)

    # Extract top factors
    importances = (
        metadata.get("features", {}).get("importances") if metadata else None
    )
    top_factors = extract_top_factors(
        feature_dict, predicted_label=label_name, feature_importances=importances
    )

    return {
        "label": label_name,
        "segmen": label_name,
        "segmen_kode": pred_code,
        "probabilities": probabilities,
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    # Test sample CUST00001 (Need Attention in dataset)
    sample_cust1 = {
        "attendance_frequency_7d": 0.0,
        "missed_session_count_7d": 0.0,
        "reschedule_frequency_7d": 0.0,
        "remaining_package_sessions": 10.0,
        "days_until_package_expiry": 30.0,
        "days_since_last_visit": 11.0,
        "reminder_count_7d": 0.0,
        "reminder_response_count_7d": 0.0,
    }
    result = predict(sample_cust1)
    print("Inference Test Result (CUST00001):")
    print(json.dumps(result, indent=2))
