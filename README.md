# Percobaan-gymnasium — Customer Retention Risk Inference API

Machine Learning service for predicting gym customer retention risk categories (`Active`, `At Risk`, `Need Attention`). This service converts exploratory notebooks into modular Python scripts and exposes a production-ready FastAPI inference API matching the contract expected by the Express backend, alongside an interactive Streamlit testing web interface.

---

## 1. Architecture & Background

The Gymnasium retention system predicts member churn risk using behavioral patterns.
- **Backend Flow**: The Express backend (`src/services/feature.service.js`) constructs customer features and issues a `POST /predict` HTTP call to this model service.
- **Prediction Output**:
  - `label` / `segmen`: `Active` (0), `At Risk` (1), or `Need Attention` (2).
  - `probabilities`: Calibrated probability distribution across the three classes.
  - `top_factors`: Domain-grounded explainability factors highlighting why the model predicted that segment.

### The 8-Feature Contract Fix
The original Colab notebook (`Klasifikasi.ipynb`) was trained on 13 features. However, the Express backend extracts and sends **only 8 features**. This service trains and serves exclusively on the canonical 8 backend features in exact order:
1. `attendance_frequency_7d` (Sessions attended in the last 7 days)
2. `missed_session_count_7d` (Sessions booked but missed in the last 7 days)
3. `reschedule_frequency_7d` (Sessions rescheduled in the last 7 days)
4. `remaining_package_sessions` (Remaining sessions in current package)
5. `days_until_package_expiry` (Days before membership package expires)
6. `days_since_last_visit` (Days elapsed since the member's last visit)
7. `reminder_count_7d` (WhatsApp/Notification reminders sent)
8. `reminder_response_count_7d` (Reminders responded to by member)

---

## 2. Directory Structure

```
H:\Project\gymnasium\Percobaan_gymnasium\
├── archive/
│   ├── Clustering.ipynb             # Preserved original Clustering notebook
│   ├── Klasifikasi.ipynb            # Preserved original Classification notebook
│   └── model_clustering.h5          # Preserved original KMeans model
├── data/
│   ├── synthetic_customer_retention.csv  # Raw synthetic dataset (12,000 rows)
│   ├── data_clustering.csv          # Scaled features + Target + Nama_Segmen
│   └── data_clustering_inverse.csv  # Unscaled features + customer_id + Target + Nama_Segmen
├── models/
│   ├── random_forest_8fitur.joblib  # Trained production Random Forest model
│   ├── model_meta.json              # Model metrics, feature importances, and config
│   └── model_clustering.joblib      # KMeans clustering model artifact
├── tests/
│   ├── test_predict.py              # FastAPI endpoint and inference test suite
│   └── test_train.py                # Model file, metadata, and feature order test suite
├── clustering.py                    # Modular clustering pipeline script
├── train.py                         # 8-feature classification training script
├── predict.py                       # Core inference and domain explainability module
├── app.py                           # FastAPI inference service application
├── screenshots/
│   └── ui-final.png                 # Streamlit test application interface screenshot
├── streamlit_test_app.py            # Streamlit test web application (Cloud entry point)
├── requirements.txt                 # Dependencies including FastAPI and Streamlit
├── requirements-test.txt            # Test-only dependencies
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git exclusion rules with model exception
└── README.md                        # Documentation
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Setup Steps
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env
```

---

## 4. Pipeline Execution

### Step 1: Run Clustering Pipeline
Loads `synthetic_customer_retention.csv`, performs sequential IQR outlier removal (12,000 -> 6,229 rows), standardizes features, fits KMeans ($k=3$), and exports datasets:
```bash
python clustering.py
```

### Step 2: Run Training Pipeline
Loads `data_clustering_inverse.csv`, isolates the 8 canonical backend features, executes stratified 70/30 train/test split, trains DecisionTree and RandomForest classifiers, evaluates metrics with 5-fold cross-validation, and saves `models/random_forest_8fitur.joblib` and `models/model_meta.json`:
```bash
python train.py
```

---

## 5. Running the Services

### A. FastAPI Inference Service (Production API)
Start the FastAPI application with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://127.0.0.1:8000/docs`

### B. Streamlit Test App (Test UI)
Aplikasi Streamlit menyediakan web interface interaktif untuk menguji model inferensi secara langsung dengan preset sampel (CUST00001, CUST00005, CUST00009) maupun input kustom.

Menjalankan secara lokal:
```bash
streamlit run streamlit_test_app.py
```

### C. Deploy ke Streamlit Cloud
Proyek disiapkan untuk dideploy ke **Streamlit Cloud** dari repositori GitHub **PRIVATE**:
- **Repo Name**: `Percobaan-gymnasium` (**PRIVATE**)
- **Main file path: `streamlit_test_app.py`**
- **Python Version**: `3.11.9` (versi pengujian lokal)
- **FastAPI Service Note**: Berkas backend inferensi FastAPI (`app.py`, `predict.py`, `train.py`, `clustering.py`) tetap tersimpan di dalam repositori yang sama untuk integrasi frontend mendatang, namun **bukan** merupakan berkas yang dijalankan oleh Streamlit Cloud.

#### Solusi 3 Blocker Deployment:
1. **Model Artifact Tracking**: `.gitignore` mempertahankan aturan umum pengabaian `models/*.joblib`, dengan pengecualian eksplisit `!models/random_forest_8fitur.joblib`. Berkas model (16.81 MB) berhasil terlacak ke git dan siap dimuat di cloud.
2. **Root Dependencies**: Streamlit Cloud membaca berkas `requirements.txt` di root repository. Entri `streamlit>=1.30.0` telah ditambahkan ke root `requirements.txt` tanpa menghapus atau mengubah dependensi FastAPI.
3. **Target Entry Point**: Entry point cloud ditentukan pada **`streamlit_test_app.py`** (bukan `app.py`).

---

## 6. API Reference

### Health Check
- **Endpoints**: `GET /api/health` and `GET /health`
- **Response**:
```json
{
  "status": "ok",
  "model_version": "random-forest-8fitur-v1"
}
```

### Predict Customer Retention Risk
- **Endpoints**: `POST /api/predict` and `POST /predict`
- **Request Body**:
```json
{
  "customer_id": "CUST00001",
  "attendance_frequency_7d": 0,
  "missed_session_count_7d": 0,
  "reschedule_frequency_7d": 0,
  "remaining_package_sessions": 10,
  "days_until_package_expiry": 30,
  "days_since_last_visit": 11,
  "reminder_count_7d": 0,
  "reminder_response_count_7d": 0
}
```

- **Response Body**:
```json
{
  "customer_id": "CUST00001",
  "label": "Need Attention",
  "segmen": "Need Attention",
  "segmen_kode": 2,
  "probabilities": {
    "Active": 0.0,
    "At Risk": 0.01,
    "Need Attention": 0.99
  },
  "top_factors": [
    "Kehadiran mingguan rendah (0 sesi)",
    "Jarak kunjungan terakhir cukup lama (11 hari)"
  ]
}
```

### Missing Feature Validation (HTTP 422)
If any of the 8 required numerical features is omitted, FastAPI automatically rejects the request with HTTP 422 Unprocessable Entity detailing the missing field.

---

## 7. Automated Tests

Execute the automated test suite with pytest:
```bash
pytest -v
```

Test coverage includes:
- `tests/test_predict.py`:
  - Health endpoint verification (`GET /api/health`, `GET /health`)
  - Prediction accuracy on benchmark samples (`CUST00001`, At Risk sample)
  - Missing field validation rejection (HTTP 422)
  - Probability distribution integrity (summing to $1.0 \pm 0.001$)
  - Non-empty, human-readable `top_factors`
- `tests/test_train.py`:
  - Artifact existence (`random_forest_8fitur.joblib`, `model_meta.json`)
  - Feature ordering and count validation (exactly 8 features)
  - Label mapping consistency (`Active`, `At Risk`, `Need Attention`)

---

## 8. Honest Limitations & Methodology Notes

1. **Circular Classifier Ground-Truth**: The `Target` labels (0, 1, 2) were generated via unsupervised KMeans clustering on the same dataset. Classifier accuracy measures how faithfully the Random Forest replicates KMeans decision boundaries, not ground-truth empirical churn observations.
2. **Feature Reduction Effect**: In the original notebook with all 13 features, Random Forest achieved 98% accuracy. Restricting the feature set to the 8 backend features yields **75.66% test accuracy** and **76.51% 5-fold cross-validation accuracy**, representing a realistic trade-off for system alignment.
3. **Outlier Filtering Data Loss**: Sequential IQR outlier removal eliminates **5,771 out of 12,000 rows (48.09%)**, retaining 6,229 rows.
4. **Zero-Variance Feature**: After sequential IQR outlier filtering, the feature `has_ever_visited` becomes constant $1.0$ ($\sigma = 0.0$). This feature was excluded from the 8 backend features.
5. **No Slow SHAP Overhead**: Explainability uses deterministic domain heuristics paired with feature importances, ensuring sub-10ms response latency without heavy SHAP dependencies.
