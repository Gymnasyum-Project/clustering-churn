"""Training Module for Gymnasium Customer Retention Classification.

Converts and refactors logic from Klasifikasi.ipynb with the CRITICAL FEATURE FIX:
- Restricts training inputs from 13 features to EXACTLY the 8 backend features
  consumed by Express backend's feature.service.js.
- Input Features (in exact order):
    1. attendance_frequency_7d
    2. missed_session_count_7d
    3. reschedule_frequency_7d
    4. remaining_package_sessions
    5. days_until_package_expiry
    6. days_since_last_visit
    7. reminder_count_7d
    8. reminder_response_count_7d
- Stratified 70/30 train/test split (4,360 train, 1,869 test).
- Trains DecisionTreeClassifier and RandomForestClassifier.
- Evaluates per-class precision, recall, f1-score, accuracy, and 5-fold cross-validation.
- Saves final RandomForestClassifier to models/random_forest_8fitur.joblib.
- Saves comprehensive metadata and metrics to models/model_meta.json.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Canonical 8 backend features in exact contract order
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


def find_data_file(filename: str, search_dirs: Tuple[Path, ...] = ()) -> Path:
    """Locates an input dataset file in expected project directories."""
    default_dirs = (
        Path.cwd() / "data",
        Path.cwd(),
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "gymnasium-classification",
    )
    for directory in search_dirs + default_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate dataset file '{filename}' in expected locations.")


def load_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads dataset and extracts the 8 required backend features and target."""
    df = pd.read_csv(csv_path)
    
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required backend features: {missing_features}")
    
    if "Target" not in df.columns:
        raise ValueError("Dataset does not contain 'Target' column for training.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["Target"].copy()
    return X, y


def train_and_evaluate(
    input_csv: str = "data_clustering_kmeans8.csv",
    models_dir: Path | None = None,
    random_state: int = 42,
    test_size: float = 0.3,
    model_filename: str = "random_forest_8fitur_v3.joblib",
    meta_filename: str = "model_meta_v3.json",
) -> Dict[str, Any]:
    """Executes the training pipeline, comparisons, and serialization."""
    base_dir = Path(__file__).resolve().parent
    mod_dir = models_dir or (base_dir / "models")
    mod_dir.mkdir(parents=True, exist_ok=True)

    csv_file = find_data_file(input_csv)
    print(f"[Train] Loading training data from: {csv_file}")
    X, y = load_dataset(csv_file)
    total_samples = len(X)
    print(f"[Train] Total samples: {total_samples}")
    print(f"[Train] Input features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    # Stratified Train-Test Split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Train] Split sizes -> Train: {len(X_train)}, Test: {len(X_test)}")

    # 1. Decision Tree Classifier
    print("\n" + "=" * 50)
    print("[Train] Training DecisionTreeClassifier(random_state=42)...")
    dt_model = DecisionTreeClassifier(random_state=random_state)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_acc = float(accuracy_score(y_test, y_pred_dt))
    dt_report = classification_report(
        y_test, y_pred_dt, target_names=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())], output_dict=True
    )
    print(f"Decision Tree Accuracy: {dt_acc:.4f}")
    print("Decision Tree Performance:")
    print(classification_report(y_test, y_pred_dt, target_names=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())]))

    # 2. Random Forest Classifier
    print("=" * 50)
    print("[Train] Training RandomForestClassifier(random_state=42, n_estimators=100)...")
    rf_model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_report = classification_report(
        y_test, y_pred_rf, target_names=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())], output_dict=True
    )
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print("Random Forest Performance:")
    print(classification_report(y_test, y_pred_rf, target_names=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())]))

    # 3. 5-Fold Cross Validation for Random Forest
    print("=" * 50)
    print("[Train] Running 5-Fold Cross Validation on Random Forest (scoring='accuracy')...")
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="accuracy")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"CV Accuracy Scores: {[round(float(s), 4) for s in cv_scores]}")
    print(f"CV Mean Accuracy:   {cv_mean:.4f} ({cv_mean * 100:.2f}%)")
    print(f"CV Std Dev:         {cv_std:.4f}")

    # Feature Importances
    importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURE_COLUMNS, rf_model.feature_importances_)
    }
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    print("\n[Train] Random Forest Feature Importances:")
    for feat, imp in sorted_importances.items():
        print(f"  - {feat}: {imp:.4f}")

    # Confusion Matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()
    print("\n[Train] Random Forest Confusion Matrix:")
    print(np.array(cm_rf))

    # 4. Save Final Model (.joblib)
    model_save_path = mod_dir / model_filename
    joblib.dump(rf_model, model_save_path)
    print(f"\n[Train] Saved Random Forest model: {model_save_path}")

    # 5. Save Model Metadata (.json)
    metadata: Dict[str, Any] = {
        "model_name": "RandomForestClassifier",
        "model_file": model_save_path.name,
        "scikit_learn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "features": {
            "count": len(FEATURE_COLUMNS),
            "order": FEATURE_COLUMNS,
            "importances": sorted_importances,
        },
        "label_mapping": LABEL_MAPPING,
        "dataset": {
            "total_rows": total_samples,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": test_size,
        },
        "evaluation": {
            "random_forest": {
                "test_accuracy": rf_acc,
                "cv_5fold": {
                    "scores": [round(float(s), 4) for s in cv_scores],
                    "mean": round(cv_mean, 4),
                    "std": round(cv_std, 4),
                },
                "classification_report": rf_report,
                "confusion_matrix": cm_rf,
            },
            "decision_tree": {
                "test_accuracy": dt_acc,
                "classification_report": dt_report,
            },
        },
        "methodology_limitation_warning": (
            "Target labels originate from unsupervised K-Means clustering on the same dataset. "
            "Classifier metrics reflect boundary learning fidelity of K-Means partitions rather than "
            "empirically observed ground-truth customer churn."
        ),
    }

    meta_save_path = mod_dir / meta_filename
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Train] Saved Model Metadata: {meta_save_path}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
