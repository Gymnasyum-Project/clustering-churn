from typing import Dict
import streamlit as st

from predict import FEATURE_COLUMNS, predict

st.set_page_config(
    page_title="Percobaan-gymnasium — Test Model Prediksi",
    page_icon="🏋️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 0.6rem;
    }
    .info-badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .meta-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.22);
    }
    .badge-active {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.25rem;
        color: #16a34a;
        background-color: rgba(34, 197, 94, 0.14);
        border: 1.5px solid #22c55e;
    }
    .badge-need-attention {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.25rem;
        color: #d97706;
        background-color: rgba(245, 158, 11, 0.14);
        border: 1.5px solid #f59e0b;
    }
    .badge-at-risk {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.25rem;
        color: #dc2626;
        background-color: rgba(239, 68, 68, 0.14);
        border: 1.5px solid #ef4444;
    }
    .status-match {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        color: #16a34a;
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid #22c55e;
    }
    .status-mismatch {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        color: #dc2626;
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
    }
    .status-custom {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #2563eb;
        background-color: rgba(37, 99, 235, 0.12);
        border: 1px solid #3b82f6;
    }
    .factor-card {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        background: rgba(128, 128, 128, 0.08);
        border-left: 4px solid #2563eb;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🏋️ Percobaan-gymnasium — Test Model Prediksi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Uji model Random Forest 8 fitur: Active / Need Attention / At Risk — sebelum integrasi frontend.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-badge-bar">
        <span class="meta-chip">🤖 Model: Random Forest (100 Estimators)</span>
        <span class="meta-chip">🎯 Akurasi Uji: 76.14% (8 Fitur Backend)</span>
        <span class="meta-chip">🔄 5-Fold CV: 76.37% (±0.73%)</span>
        <span class="meta-chip">🏷️ Target: K-Means Clustering (Silhouette 0.19 — Weak)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

PRESETS: Dict[str, Dict[str, object]] = {
    "CUST00001 (Need Attention)": {
        "true_label": "Need Attention",
        "true_code": 2,
        "values": {
            "attendance_frequency_7d": 0.0,
            "missed_session_count_7d": 0.0,
            "reschedule_frequency_7d": 0.0,
            "remaining_package_sessions": 10.0,
            "days_until_package_expiry": 30.0,
            "days_since_last_visit": 11.0,
            "reminder_count_7d": 0.0,
            "reminder_response_count_7d": 0.0,
        },
    },
    "CUST00005 (Active)": {
        "true_label": "Active",
        "true_code": 0,
        "values": {
            "attendance_frequency_7d": 0.0,
            "missed_session_count_7d": 1.0,
            "reschedule_frequency_7d": 2.0,
            "remaining_package_sessions": 7.0,
            "days_until_package_expiry": 23.0,
            "days_since_last_visit": 7.0,
            "reminder_count_7d": 1.0,
            "reminder_response_count_7d": 0.0,
        },
    },
    "CUST00009 (At Risk)": {
        "true_label": "At Risk",
        "true_code": 1,
        "values": {
            "attendance_frequency_7d": 4.0,
            "missed_session_count_7d": 0.0,
            "reschedule_frequency_7d": 1.0,
            "remaining_package_sessions": 9.0,
            "days_until_package_expiry": 36.0,
            "days_since_last_visit": 2.0,
            "reminder_count_7d": 0.0,
            "reminder_response_count_7d": 0.0,
        },
    },
}

with st.container(border=True):
    st.markdown("### 1. Pilih Preset Sampel Benchmark")
    preset_cols = st.columns([3, 2])
    with preset_cols[0]:
        preset_choice = st.selectbox(
            "Pilih preset pelanggan terverifikasi atau input kustom:",
            options=["Custom Input"] + list(PRESETS.keys()),
            index=1,
            label_visibility="collapsed",
        )

    default_vals = PRESETS["CUST00001 (Need Attention)"]["values"]
    selected_true_label = None
    selected_true_code = None

    if preset_choice in PRESETS:
        preset_info = PRESETS[preset_choice]
        default_vals = preset_info["values"]
        selected_true_label = preset_info["true_label"]
        selected_true_code = preset_info["true_code"]
        with preset_cols[1]:
            badge_class = (
                "badge-active"
                if selected_true_label == "Active"
                else (
                    "badge-need-attention"
                    if selected_true_label == "Need Attention"
                    else "badge-at-risk"
                )
            )
            st.markdown(
                f'<div style="text-align: right; padding-top: 4px;">'
                f'<span style="font-size: 0.88rem; opacity: 0.8; margin-right: 8px;">Target Asli Dataset:</span>'
                f'<span class="{badge_class}" style="font-size: 0.95rem; padding: 0.25rem 0.8rem;">'
                f"{selected_true_label} (Kode {selected_true_code})</span></div>",
                unsafe_allow_html=True,
            )
    else:
        with preset_cols[1]:
            st.markdown(
                '<div style="text-align: right; padding-top: 8px;">'
                '<span class="status-custom">Mode Input Kustom (Manual)</span></div>',
                unsafe_allow_html=True,
            )

with st.container(border=True):
    st.markdown("### 2. Parameter 8 Fitur Backend")
    st.caption("Nilai input yang dikirimkan oleh Express backend (`src/services/feature.service.js`) ke model inferensi:")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        att_7d = st.number_input(
            "1. attendance_frequency_7d — Kehadiran 7 hari terakhir (sesi)",
            min_value=0.0,
            max_value=30.0,
            value=float(default_vals["attendance_frequency_7d"]),
            step=1.0,
            help="Jumlah sesi gym yang dihadiri dalam 7 hari terakhir",
        )
    with row1_c2:
        days_expiry = st.number_input(
            "5. days_until_package_expiry — Sisa masa aktif paket (hari)",
            min_value=0.0,
            max_value=365.0,
            value=float(default_vals["days_until_package_expiry"]),
            step=1.0,
            help="Jumlah hari sebelum paket keanggotaan kedaluwarsa",
        )

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        missed_7d = st.number_input(
            "2. missed_session_count_7d — Sesi terlewat 7 hari terakhir (sesi)",
            min_value=0.0,
            max_value=30.0,
            value=float(default_vals["missed_session_count_7d"]),
            step=1.0,
            help="Jumlah sesi yang sudah dibooking tetapi tidak dihadiri",
        )
    with row2_c2:
        days_visit = st.number_input(
            "6. days_since_last_visit — Hari sejak kunjungan terakhir (hari)",
            min_value=0.0,
            max_value=365.0,
            value=float(default_vals["days_since_last_visit"]),
            step=1.0,
            help="Jumlah hari yang berlalu sejak kunjungan terakhir member ke gym",
        )

    row3_c1, row3_c2 = st.columns(2)
    with row3_c1:
        resched_7d = st.number_input(
            "3. reschedule_frequency_7d — Frekuensi reschedule 7 hari (kali)",
            min_value=0.0,
            max_value=30.0,
            value=float(default_vals["reschedule_frequency_7d"]),
            step=1.0,
            help="Berapa kali member mengajukan penjadwalan ulang sesi dalam 7 hari",
        )
    with row3_c2:
        rem_count = st.number_input(
            "7. reminder_count_7d — Notifikasi pengingat terkirim (pesan)",
            min_value=0.0,
            max_value=30.0,
            value=float(default_vals["reminder_count_7d"]),
            step=1.0,
            help="Jumlah pesan pengingat yang dikirim oleh sistem dalam 7 hari",
        )

    row4_c1, row4_c2 = st.columns(2)
    with row4_c1:
        rem_sess = st.number_input(
            "4. remaining_package_sessions — Sisa kuota sesi paket aktif (sesi)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_vals["remaining_package_sessions"]),
            step=1.0,
            help="Sisa kuota sesi latihan pada paket aktif",
        )
    with row4_c2:
        rem_resp = st.number_input(
            "8. reminder_response_count_7d — Respons pengingat member (kali)",
            min_value=0.0,
            max_value=30.0,
            value=float(default_vals["reminder_response_count_7d"]),
            step=1.0,
            help="Jumlah respons positif dari member terhadap pengingat",
        )

input_features = {
    "attendance_frequency_7d": att_7d,
    "missed_session_count_7d": missed_7d,
    "reschedule_frequency_7d": resched_7d,
    "remaining_package_sessions": rem_sess,
    "days_until_package_expiry": days_expiry,
    "days_since_last_visit": days_visit,
    "reminder_count_7d": rem_count,
    "reminder_response_count_7d": rem_resp,
}

st.write("")
btn_pred = st.button("🔮 Prediksi Segmen Pelanggan", type="primary", use_container_width=True)

if btn_pred:
    with st.spinner("Menjalankan inferensi model Random Forest 8-fitur..."):
        result = predict(input_features)

    st.write("")
    with st.container(border=True):
        st.markdown("### 3. Hasil Prediksi & Evaluasi Segmen")

        pred_label = result["label"]
        pred_code = result["segmen_kode"]

        if pred_label == "Active":
            badge_html = f'<span class="badge-active">🟢 Active (Kode {pred_code})</span>'
        elif pred_label == "Need Attention":
            badge_html = f'<span class="badge-need-attention">🟡 Need Attention (Kode {pred_code})</span>'
        else:
            badge_html = f'<span class="badge-at-risk">🔴 At Risk (Kode {pred_code})</span>'

        if selected_true_label:
            if pred_label == selected_true_label:
                status_html = '<span class="status-match">✅ Cocok (MATCH)</span>'
            else:
                status_html = '<span class="status-mismatch">❌ Tidak Cocok (MISMATCH)</span>'
        else:
            status_html = '<span class="status-custom">Mode Input Kustom</span>'

        header_cols = st.columns([1, 1])
        with header_cols[0]:
            st.markdown(
                f'<div style="margin-bottom: 1rem;"><div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 4px;">Hasil Prediksi Model:</div>'
                f"{badge_html}</div>",
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.markdown(
                f'<div style="margin-bottom: 1rem;"><div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 4px;">Status Validasi Ground-Truth:</div>'
                f"{status_html}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        prob_col, factor_col = st.columns([1, 1])

        with prob_col:
            st.markdown("#### 📊 Distribusi Probabilitas Kelas")
            st.caption("Urutan terstandarisasi: Active / Need Attention / At Risk")
            probs = result["probabilities"]
            ordered_keys = ["Active", "Need Attention", "At Risk"]

            for cls_name in ordered_keys:
                p_val = probs.get(cls_name, 0.0)
                is_highest = cls_name == pred_label
                pct_str = f"{p_val * 100:.2f}%"

                if is_highest:
                    st.markdown(
                        f'<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">'
                        f'<strong>⭐ {cls_name} (Tertinggi)</strong>'
                        f'<strong style="color: #2563eb;">{pct_str}</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">'
                        f"<span>{cls_name}</span>"
                        f'<span style="opacity: 0.8;">{pct_str}</span></div>',
                        unsafe_allow_html=True,
                    )
                st.progress(float(p_val))
                st.write("")

        with factor_col:
            st.markdown("#### 💡 Faktor Utama Penentu (Maksimal 3)")
            st.caption("Penjelasan domain berbasis heuristik aturan & bobot fitur:")
            top_factors = result.get("top_factors", [])[:3]

            if top_factors:
                for idx, factor in enumerate(top_factors, 1):
                    clean_factor = factor
                    if ":" in clean_factor and not clean_factor.endswith("sesi") and not clean_factor.endswith("hari"):
                        parts = clean_factor.split(":")
                        key_name = parts[0].strip()
                        val_str = parts[1].strip() if len(parts) > 1 else ""
                        translations = {
                            "remaining_package_sessions": f"Sisa sesi paket: {val_str} sesi",
                            "days_until_package_expiry": f"Masa aktif paket: {val_str} hari tersisa",
                            "attendance_frequency_7d": f"Frekuensi kehadiran 7 hari: {val_str} sesi",
                            "days_since_last_visit": f"Kunjungan terakhir: {val_str} hari lalu",
                            "reschedule_frequency_7d": f"Frekuensi reschedule 7 hari: {val_str} kali",
                            "missed_session_count_7d": f"Sesi terlewat 7 hari: {val_str} sesi",
                            "reminder_count_7d": f"Notifikasi pengingat terkirim: {val_str} pesan",
                            "reminder_response_count_7d": f"Respons pengingat 7 hari: {val_str} kali",
                        }
                        clean_factor = translations.get(key_name, clean_factor)

                    st.markdown(
                        f'<div class="factor-card"><strong>{idx}.</strong> {clean_factor}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Tidak ada faktor dominan spesifik yang terpicu.")
