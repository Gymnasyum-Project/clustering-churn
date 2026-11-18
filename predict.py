"""Inference Module for Gymnasium Customer Retention Prediction.

Provides core model loading and prediction functions:
- load_model(): Loads the trained 8-feature RandomForest model and metadata.
- predict(): Predicts customer retention segment, class probabilities, and top factors.
- Rules-based explainability: Explains predictions using domain heuristics combined
  with feature importances (NO heavy/slow SHAP dependency).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd

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
    """Resolves default model and metadata paths."""
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "random_forest_8fitur.joblib"
    meta_path = base_dir / "models" / "model_meta.json"
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
    predicted_label: str,
    feature_importances: Dict[str, float] | None = None,
) -> List[str]:
    """Generates transparent, domain-grounded top factors explaining the prediction.
    
    Uses domain heuristics aligned with customer retention risk factors:
    - Attendance volume and frequency
    - Days since last gym visit
    - Package expiry proximity and remaining sessions
    - Missed/rescheduled sessions and reminder engagement
    Falls back to feature importance ordering if no specific threshold triggers.
    """
    factors: List[str] = []
    att = features.get("attendance_frequency_7d", 0.0)
    missed = features.get("missed_session_count_7d", 0.0)
    resched = features.get("reschedule_frequency_7d", 0.0)
    rem_sess = features.get("remaining_package_sessions", 0.0)
    days_expiry = features.get("days_until_package_expiry", 0.0)
    days_last_visit = features.get("days_since_last_visit", 0.0)
    rem_count = features.get("reminder_count_7d", 0.0)
    rem_resp = features.get("reminder_response_count_7d", 0.0)

    if predicted_label == "Active":
        if att >= 2:
            factors.append(f"Frekuensi kehadiran 7 hari tinggi ({int(att)} sesi)")
        if days_last_visit <= 3:
            factors.append(f"Baru saja berkunjung ({int(days_last_visit)} hari lalu)")
        if missed == 0:
            factors.append("Tidak ada sesi yang terlewat dalam 7 hari terakhir")
        if rem_sess >= 6:
            factors.append(f"Sisa sesi paket masih mencukupi ({int(rem_sess)} sesi)")
        if days_expiry >= 20:
            factors.append(f"Masa aktif paket masih panjang ({int(days_expiry)} hari)")

    elif predicted_label == "At Risk":
        if att == 0:
            factors.append("Tidak ada kehadiran dalam 7 hari terakhir (0 sesi)")
        elif att <= 1:
            factors.append(f"Kehadiran sangat rendah ({int(att)} sesi dalam 7 hari)")
        elif att >= 3:
            factors.append(f"Frekuensi kehadiran tercatat {int(att)} sesi dalam 7 hari")
        if days_last_visit >= 7:
            factors.append(f"Sudah lama tidak berkunjung ({int(days_last_visit)} hari)")
        if days_expiry <= 14:
            factors.append(f"Masa berlaku paket hampir habis ({int(days_expiry)} hari tersisa)")
        elif days_expiry >= 30:
            factors.append(f"Masa berlaku paket aktif tersisa {int(days_expiry)} hari")
        if rem_sess <= 3:
            factors.append(f"Sisa sesi paket menipis ({int(rem_sess)} sesi)")
        elif rem_sess >= 7:
            factors.append(f"Sisa sesi paket keanggotaan ({int(rem_sess)} sesi)")
        if resched >= 1:
            factors.append(f"Frekuensi reschedule cukup sering ({int(resched)} kali dalam 7 hari)")
        if rem_count >= 1 and rem_resp == 0:
            factors.append(f"Tidak merespons {int(rem_count)} pesan pengingat")

    elif predicted_label == "Need Attention":
        if missed >= 1:
            factors.append(f"Terdapat sesi terlewat ({int(missed)} sesi dalam 7 hari)")
        if resched >= 1:
            factors.append(f"Frekuensi reschedule tinggi ({int(resched)} kali dalam 7 hari)")
        if att <= 1:
            factors.append(f"Kehadiran mingguan rendah ({int(att)} sesi)")
        if days_last_visit >= 5:
            factors.append(f"Jarak kunjungan terakhir cukup lama ({int(days_last_visit)} hari)")
        if rem_count >= 1 and rem_resp == 0:
            factors.append("Belum merespons pengingat terbaru")

    # Fallback to feature importance ordering formatted as natural Indonesian sentences
    if len(factors) < 2 and feature_importances:
        sentence_templates = {
            "remaining_package_sessions": "Sisa sesi paket: {v} sesi",
            "days_until_package_expiry": "Masa aktif paket: {v} hari tersisa",
            "attendance_frequency_7d": "Frekuensi kehadiran 7 hari: {v} sesi",
            "days_since_last_visit": "Kunjungan terakhir: {v} hari yang lalu",
            "reschedule_frequency_7d": "Frekuensi reschedule 7 hari: {v} kali",
            "missed_session_count_7d": "Sesi terlewat dalam 7 hari: {v} sesi",
            "reminder_count_7d": "Notifikasi pengingat terkirim: {v} pesan",
            "reminder_response_count_7d": "Respons pengingat 7 hari: {v} kali",
        }
        top_sorted = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        for feat_name, _ in top_sorted:
            val = features.get(feat_name, 0.0)
            tpl = sentence_templates.get(feat_name, "{feat_name}: {v}")
            candidate = tpl.format(v=int(val), feat_name=feat_name)
            if candidate not in factors:
                factors.append(candidate)
            if len(factors) >= 3:
                break

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
