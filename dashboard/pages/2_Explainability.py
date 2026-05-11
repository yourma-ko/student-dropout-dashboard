import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Explainability — SHAP",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).parent.parent
PLOTS_DIR = BASE_DIR / "plots"

GENDER_MAP = {'F': 0, 'M': 1}
HIGHEST_EDUCATION_MAP = {
    'A Level or Equivalent': 0, 'HE Qualification': 1,
    'Lower Than A Level': 2, 'No Formal quals': 3,
    'Post Graduate Qualification': 4,
}
IMD_BAND_MAP = {
    '0-10%': 0, '10-20': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4,
    '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8, '90-100%': 9,
}

FIXED_CODE_MODULE       = 2
FIXED_CODE_PRESENTATION = 3
FIXED_AGE_BAND          = 0
FIXED_DISABILITY        = 0

PRESET_BAD = dict(
    total_clicks=50.0, avg_clicks=2.0, activity_count=15.0, active_days=5.0,
    avg_score=20.0, num_assessments=1.0,
    gender="M", highest_education="Lower Than A Level",
    imd_band="10-20", num_of_prev_attempts=2,
    studied_credits=120, date_registration=5,
)
PRESET_GOOD = dict(
    total_clicks=2200.0, avg_clicks=4.0, activity_count=550.0, active_days=100.0,
    avg_score=88.0, num_assessments=7.0,
    gender="F", highest_education="HE Qualification",
    imd_band="80-90%", num_of_prev_attempts=0,
    studied_credits=60, date_registration=-120,
)


@st.cache_resource
def load_artifacts():
    model         = joblib.load(BASE_DIR / "models" / "model.pkl")
    scaler        = joblib.load(BASE_DIR / "models" / "scaler.pkl")
    feature_names = joblib.load(BASE_DIR / "models" / "feature_names.pkl")
    median_values = joblib.load(BASE_DIR / "models" / "median_values.pkl")
    return model, scaler, feature_names, median_values


@st.cache_resource
def get_explainer(_model):
    # CalibratedClassifierCV wraps 5 base estimators; use the first for TreeExplainer
    base_xgb = _model.calibrated_classifiers_[0].estimator
    return shap.TreeExplainer(base_xgb)


model, scaler, feature_names, median_values = load_artifacts()
explainer = get_explainer(model)

# ─────────────────────────────────────────────
st.title("Model Explainability — SHAP Analysis")
st.markdown(
    "**SHAP** (SHapley Additive exPlanations) assigns each feature a contribution value "
    "for a specific prediction. Positive SHAP = pushes toward dropout; "
    "negative SHAP = pushes away from dropout."
)

# ─────────────────────────────────────────────
# Section 1: Global SHAP
# ─────────────────────────────────────────────
st.divider()
st.subheader("Global Feature Importance")
st.caption("Mean |SHAP value| across 500 test-set students — computed in the research notebook")

col_img, col_tbl = st.columns([3, 2])

with col_img:
    shap_img = PLOTS_DIR / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
    else:
        st.warning("plots/shap_summary.png not found. Re-run the notebook to generate it.")

with col_tbl:
    st.markdown("**Feature descriptions**")
    feat_desc = pd.DataFrame({
        "Feature":     [
            "avg_score", "total_clicks", "activity_count",
            "active_days", "num_assessments", "total_score",
            "avg_clicks", "num_of_prev_attempts",
            "studied_credits", "date_registration",
        ],
        "Description": [
            "Mean score across all assessments",
            "Total VLE platform interactions",
            "Number of distinct VLE sessions",
            "Days with at least one login",
            "Assessments submitted",
            "Sum of all assessment scores",
            "Mean clicks per VLE session",
            "Times module previously attempted",
            "Total credits being studied",
            "Days from course start (negative = before)",
        ],
        "Risk direction": [
            "Low ↑ risk", "Low ↑ risk", "Low ↑ risk",
            "Low ↑ risk", "Low ↑ risk", "Low ↑ risk",
            "Mixed",      "High ↑ risk",
            "High ↑ risk", "Late ↑ risk",
        ],
    })
    st.dataframe(feat_desc, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Section 2: Local SHAP for a student profile
# ─────────────────────────────────────────────
st.divider()
st.subheader("Individual Student Explanation")
st.markdown("Choose a profile to see **which features drive** the model's prediction.")

profile = st.radio(
    "Student profile",
    ["High-Risk Student", "Low-Risk Student"],
    horizontal=True,
)

p = PRESET_BAD if profile == "High-Risk Student" else PRESET_GOOD
total_score = p["avg_score"] * p["num_assessments"]

input_dict = {
    "code_module":          FIXED_CODE_MODULE,
    "code_presentation":    FIXED_CODE_PRESENTATION,
    "id_student":           median_values["id_student"],
    "gender":               GENDER_MAP[p["gender"]],
    "region":               median_values["region"],
    "highest_education":    HIGHEST_EDUCATION_MAP[p["highest_education"]],
    "imd_band":             IMD_BAND_MAP[p["imd_band"]],
    "age_band":             float(FIXED_AGE_BAND),
    "num_of_prev_attempts": float(p["num_of_prev_attempts"]),
    "studied_credits":      float(p["studied_credits"]),
    "disability":           float(FIXED_DISABILITY),
    "total_clicks":         float(p["total_clicks"]),
    "avg_clicks":           float(p["avg_clicks"]),
    "activity_count":       float(p["activity_count"]),
    "active_days":          float(p["active_days"]),
    "avg_score":            float(p["avg_score"]),
    "total_score":          float(total_score),
    "num_assessments":      float(p["num_assessments"]),
    "date_registration":    float(p["date_registration"]),
}

input_df    = pd.DataFrame([input_dict])[feature_names]
scaled      = scaler.transform(input_df)
prob        = model.predict_proba(scaled)[0][1]
shap_vals   = explainer.shap_values(scaled)

risk_color = "#dc3545" if prob > 0.5 else ("#fd7e14" if prob > 0.25 else "#28a745")
risk_label = "HIGH RISK" if prob > 0.5 else ("MEDIUM RISK" if prob > 0.25 else "LOW RISK")

col_info, col_chart = st.columns([1, 2])

with col_info:
    st.markdown(f"**Dropout probability: {prob*100:.1f}%**")
    st.markdown(
        f"<span style='color:{risk_color};font-size:22px;font-weight:bold;'>{risk_label}</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    shap_series = pd.Series(shap_vals[0], index=feature_names)
    top_idx     = shap_series.abs().nlargest(8).index
    shap_df     = pd.DataFrame({
        "Feature":    top_idx,
        "SHAP":       shap_series[top_idx].round(4),
        "Direction":  ["↑ dropout" if v > 0 else "↓ dropout"
                       for v in shap_series[top_idx]],
    }).reset_index(drop=True)

    st.markdown("**Top 8 contributing features:**")
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

with col_chart:
    shap_series_sorted = shap_series.abs().nlargest(10)
    top_features       = shap_series_sorted.index[::-1]
    top_values         = shap_series[top_features]

    colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.barh(range(len(top_features)), top_values.values, color=colors, height=0.6)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, color="white", fontsize=10)
    ax.axvline(0, color="white", alpha=0.4, linewidth=1)
    ax.set_xlabel("SHAP value  (red = ↑ dropout risk · green = ↓ dropout risk)",
                  color="white", fontsize=9)
    ax.set_title("Feature Contributions to This Prediction", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_alpha(0.2)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.caption(
    "Note: SHAP values are computed from the underlying XGBoost model "
    "(pre-calibration) — directions and rankings are stable."
)
