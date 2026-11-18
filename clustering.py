"""Clustering Module for Gymnasium Customer Retention.

Reproduces and operationalizes the logic from Clustering.ipynb:
1. Loads raw synthetic customer retention data (12,000 rows).
2. Performs null & duplicate checks on raw data with customer_id.
3. Drops customer_id for feature processing.
4. Applies sequential IQR outlier filtering on numerical columns:
   Q1 - 1.5*IQR to Q3 + 1.5*IQR per column, reducing 12,000 to exactly 6,229 rows.
5. Standardizes features with StandardScaler.
6. Trains KMeans clustering (k=3, random_state=42).
7. Computes silhouette score (~0.19, indicating weak cluster separation).
8. Maps cluster labels: 0 -> 'Active', 1 -> 'At Risk', 2 -> 'Need Attention'.
9. Exports data_clustering.csv, data_clustering_inverse.csv, and model_clustering.joblib.
"""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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


def clean_and_filter_outliers(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Cleans raw data and applies sequential IQR outlier filtering.
    
    Preserves exact Colab notebook sequence:
    - Drop nulls and duplicates with customer_id present (preserves rows with identical features).
    - Extract customer_id and isolate numerical features.
    - Sequentially compute Q1, Q3, and IQR per numerical feature and filter.
    
    Returns:
        df_filtered: Cleaned DataFrame with 6,229 filtered rows (unscaled).
        customer_ids: Corresponding customer_id Series for surviving rows.
        numerical_cols: List of numerical feature names.
    """
    df = df_raw.copy()

    # Step 1: Null and duplicate checks on raw data (as done in notebook cells 12-15)
    df = df.dropna().drop_duplicates()

    # Step 2: Extract customer_id column before dropping
    id_cols = [c for c in df.columns if "id" in c.lower()]
    customer_ids = df[id_cols[0]] if id_cols else pd.Series(range(len(df)), name="customer_id")

    # Step 3: Identify numerical features (excluding id)
    numerical_cols = [c for c in df.columns if "id" not in c.lower()]
    filtered_df = df[numerical_cols].copy()

    # Step 4: Sequential IQR outlier removal (notebook cell 18)
    for col in numerical_cols:
        q1 = filtered_df[col].quantile(0.25)
        q3 = filtered_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_df = filtered_df[
            (filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)
        ]

    surviving_ids = customer_ids.loc[filtered_df.index]
    return filtered_df, surviving_ids, numerical_cols


def run_clustering(
    input_csv: str = "synthetic_customer_retention.csv",
    output_dir: Path | None = None,
    models_dir: Path | None = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """Executes the complete clustering pipeline and saves outputs."""
    base_dir = Path(__file__).resolve().parent
    out_dir = output_dir or (base_dir / "data")
    mod_dir = models_dir or (base_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_data_file(input_csv)
    print(f"[Clustering] Loading raw data from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    total_raw_rows = len(raw_df)
    print(f"[Clustering] Total raw rows: {total_raw_rows}")

    filtered_df, surviving_ids, numerical_cols = clean_and_filter_outliers(raw_df)
    surviving_rows = len(filtered_df)
    dropped_rows = total_raw_rows - surviving_rows
    print(f"[Clustering] Surviving rows after sequential IQR: {surviving_rows}")
    print(f"[Clustering] Outliers dropped: {dropped_rows} ({dropped_rows / total_raw_rows * 100:.2f}%)")

    # Check for zero-variance columns (e.g. has_ever_visited becomes constant 1.0)
    for col in numerical_cols:
        std = filtered_df[col].std()
        if std == 0.0 or np.isclose(std, 0.0):
            print(f"[Clustering Warning] Feature '{col}' has zero variance (std=0.0, constant value={filtered_df[col].iloc[0]}).")

    # Standard scaling
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(filtered_df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=numerical_cols, index=filtered_df.index)

    # KMeans clustering (k=3, random_state=42, n_init='auto' as in original notebook)
    print("[Clustering] Fitting KMeans(n_clusters=3, random_state=42)...")
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(scaled_df)

    # Compute silhouette score
    sil_score = float(silhouette_score(scaled_df, labels))
    print(f"[Clustering] Silhouette score: {sil_score:.4f} (weak cluster separation)")

    # Prepare scaled output dataframe
    scaled_output = scaled_df.copy()
    scaled_output["Target"] = labels
    scaled_output["Nama_Segmen"] = scaled_output["Target"].map(LABEL_MAPPING)

    # Prepare unscaled inverse dataframe
    inverse_output = filtered_df.copy()
    inverse_output.insert(0, "customer_id", surviving_ids)
    inverse_output["Target"] = labels
    inverse_output["Nama_Segmen"] = inverse_output["Target"].map(LABEL_MAPPING)

    # Save CSVs to data/
    scaled_csv_path = out_dir / "data_clustering.csv"
    inverse_csv_path = out_dir / "data_clustering_inverse.csv"
    scaled_output.to_csv(scaled_csv_path, index=False)
    inverse_output.to_csv(inverse_csv_path, index=False)
    print(f"[Clustering] Saved scaled data: {scaled_csv_path}")
    print(f"[Clustering] Saved unscaled inverse data: {inverse_csv_path}")

    # Save KMeans model to .joblib
    model_path = mod_dir / "model_clustering.joblib"
    joblib.dump(kmeans, model_path)
    print(f"[Clustering] Saved KMeans model: {model_path}")

    # Distribution summary
    distribution = inverse_output["Nama_Segmen"].value_counts().to_dict()
    print("[Clustering] Cluster Distribution:")
    for seg_name, count in distribution.items():
        print(f"  - {seg_name}: {count} ({count / surviving_rows * 100:.2f}%)")

    return {
        "raw_rows": total_raw_rows,
        "filtered_rows": surviving_rows,
        "dropped_outliers": dropped_rows,
        "silhouette_score": sil_score,
        "distribution": distribution,
        "scaled_csv": str(scaled_csv_path),
        "inverse_csv": str(inverse_csv_path),
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    run_clustering()
