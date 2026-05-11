import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# LABEL ENCODER MAPPINGS (reproduced from training)
# sklearn LabelEncoder sorts categories alphabetically
# ---------------------------------------------------

GENDER_MAP = {'F': 0, 'M': 1}
HIGHEST_EDUCATION_MAP = {
    'A Level or Equivalent': 0, 'HE Qualification': 1,
    'Lower Than A Level': 2, 'No Formal quals': 3,
    'Post Graduate Qualification': 4
}
IMD_BAND_MAP = {
    '0-10%': 0, '10-20': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4,
    '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8, '90-100%': 9
}

# Fixed internal values for fields not shown in the form:
#   code_module=2  → CCC (highest historical dropout rate: 44.5%)
#   code_presentation=3 → 2014J (34% dropout rate)
#   age_band, disability, region → dataset medians
FIXED_CODE_MODULE       = 2
FIXED_CODE_PRESENTATION = 3
FIXED_AGE_BAND          = 0
FIXED_DISABILITY        = 0

# Demo presets -------------------------------------------------------
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

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    model         = joblib.load(BASE_DIR / "models" / "model.pkl")
    scaler        = joblib.load(BASE_DIR / "models" / "scaler.pkl")
    feature_names = joblib.load(BASE_DIR / "models" / "feature_names.pkl")
    median_values = joblib.load(BASE_DIR / "models" / "median_values.pkl")
    return model, scaler, feature_names, median_values

model, scaler, feature_names, median_values = load_artifacts()

# ---------------------------------------------------
# SESSION STATE — preset loader
# ---------------------------------------------------

if "preset" not in st.session_state:
    st.session_state.preset = None

def load_preset(name: str):
    st.session_state.preset = name

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Prediction", "Model Metrics", "About Project"]
)

# ---------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------

if page == "Overview":

    st.title("Student Dropout Prediction System")
    st.markdown("""
    ### Explainable Machine Learning System for Predicting Student Dropout Risk

    This dashboard demonstrates a machine learning-based predictive system
    developed using the Open University Learning Analytics Dataset (OULAD).

    The system predicts the probability of student dropout using:
    - learning activity,
    - assessment performance,
    - demographic information,
    - and engagement metrics.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Model", "XGBoost (Calibrated)")
    col2.metric("F1-Score", "0.7909")
    col3.metric("ROC-AUC", "0.9423")

    st.divider()
    st.subheader("System Pipeline")
    st.markdown("""
    ```text
    Student Data
        ↓
    Feature Engineering
        ↓
    Scaled Features (StandardScaler)
        ↓
    Calibrated XGBoost Model
        ↓
    Dropout Risk Prediction
    ```
    """)

    st.subheader("Dataset Information")
    st.dataframe(pd.DataFrame({
        "Property": ["Dataset", "Records", "Features", "Target", "Task Type"],
        "Value":    ["OULAD", "~32,000", "19", "Dropout / Non-dropout",
                     "Binary Classification"],
    }), use_container_width=True)

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------

elif page == "Prediction":

    st.title("Student Dropout Risk Prediction")
    st.caption(
        "Module CCC · Presentation 2014J  |  "
        "Risk thresholds calibrated to OULAD base dropout rate of 31%"
    )

    # ── Preset buttons ────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        st.button("Load Bad Student",  on_click=load_preset, args=("bad",),
                  use_container_width=True)
    with col_b:
        st.button("Load Good Student", on_click=load_preset, args=("good",),
                  use_container_width=True)

    # Resolve active preset values
    p = PRESET_BAD if st.session_state.preset == "bad" else (
        PRESET_GOOD if st.session_state.preset == "good" else None
    )

    def v(key, default):
        return p[key] if p and key in p else default

    st.divider()

    # ================================================================
    # SECTION 1 — Learning Activity
    # ================================================================
    st.subheader("Learning Activity")

    c1, c2 = st.columns(2)
    with c1:
        total_clicks = st.number_input(
            "Total Clicks",
            min_value=0.0, value=v("total_clicks", 602.0),
            help="Total VLE interactions. Median: 602"
        )
        activity_count = st.number_input(
            "Activity Count",
            min_value=0.0, value=v("activity_count", 201.0),
            help="Number of distinct VLE sessions. Median: 201"
        )
    with c2:
        active_days = st.number_input(
            "Active Days",
            min_value=0.0, value=v("active_days", 40.0),
            help="Days with at least one VLE interaction. Median: 40"
        )
        avg_clicks = st.number_input(
            "Avg Clicks per Session",
            min_value=0.0, value=v("avg_clicks", 2.9),
            help="Mean clicks per VLE session. Median: 2.9"
        )

    # ================================================================
    # SECTION 2 — Assessment Performance
    # ================================================================
    st.subheader("Assessment Performance")

    c1, c2 = st.columns(2)
    with c1:
        num_assessments = st.number_input(
            "Assessments Submitted",
            min_value=0.0, value=v("num_assessments", 5.0),
            help="Number of submitted assessments. Median: 5"
        )
    with c2:
        avg_score = st.number_input(
            "Average Score",
            min_value=0.0, max_value=100.0, value=v("avg_score", 70.0),
            help="Mean score across assessments (0–100). Median: 70.6"
        )

    # total_score is always consistent: avg × count
    total_score = avg_score * num_assessments

    # ================================================================
    # SECTION 3 — Student Background
    # ================================================================
    st.subheader("Student Background")

    c1, c2 = st.columns(2)
    with c1:
        gender_options = list(GENDER_MAP.keys())
        gender = st.selectbox(
            "Gender",
            gender_options,
            index=gender_options.index(v("gender", "M"))
        )

        edu_options = list(HIGHEST_EDUCATION_MAP.keys())
        highest_education = st.selectbox(
            "Highest Education",
            edu_options,
            index=edu_options.index(v("highest_education", "HE Qualification")),
            help="Highest qualification before this course"
        )

        num_of_prev_attempts = st.number_input(
            "Previous Attempts",
            min_value=0, max_value=6,
            value=int(v("num_of_prev_attempts", 0)), step=1,
            help="Times previously attempted this module. Median: 0"
        )

    with c2:
        imd_options = list(IMD_BAND_MAP.keys())
        imd_band = st.selectbox(
            "IMD Band",
            imd_options,
            index=imd_options.index(v("imd_band", "50-60%")),
            help=(
                "Index of Multiple Deprivation — socioeconomic area indicator. "
                "0-10% = most deprived · 90-100% = least deprived"
            )
        )

        studied_credits = st.number_input(
            "Studied Credits",
            min_value=30, max_value=655,
            value=int(v("studied_credits", 60)), step=5,
            help="Total credits being studied. Median: 60"
        )

        date_registration = st.number_input(
            "Registration Date",
            value=int(v("date_registration", -57)),
            help="Days relative to course start. Negative = before start. Median: −57"
        )

    # ================================================================
    # BUILD FEATURE VECTOR
    # ================================================================

    input_dict = {
        "code_module":          FIXED_CODE_MODULE,
        "code_presentation":    FIXED_CODE_PRESENTATION,
        "id_student":           median_values["id_student"],
        "gender":               GENDER_MAP[gender],
        "region":               median_values["region"],
        "highest_education":    HIGHEST_EDUCATION_MAP[highest_education],
        "imd_band":             IMD_BAND_MAP[imd_band],
        "age_band":             float(FIXED_AGE_BAND),
        "num_of_prev_attempts": float(num_of_prev_attempts),
        "studied_credits":      float(studied_credits),
        "disability":           float(FIXED_DISABILITY),
        "total_clicks":         float(total_clicks),
        "avg_clicks":           float(avg_clicks),
        "activity_count":       float(activity_count),
        "active_days":          float(active_days),
        "avg_score":            float(avg_score),
        "total_score":          float(total_score),
        "num_assessments":      float(num_assessments),
        "date_registration":    float(date_registration),
    }

    input_df = pd.DataFrame([input_dict])[feature_names]

    # ================================================================
    # PREDICT
    # ================================================================

    st.divider()

    if st.button("Predict Dropout Risk", type="primary", use_container_width=True):

        scaled_input = scaler.transform(input_df)
        probability  = model.predict_proba(scaled_input)[0][1]
        prob_pct     = probability * 100

        # Thresholds calibrated to OULAD base dropout rate (31%):
        #   LOW    < 25%  — below base rate
        #   MEDIUM 25–50% — near / moderately above base rate
        #   HIGH   > 50%  — majority probability of dropout
        if prob_pct < 25:
            risk_level  = "LOW RISK"
            risk_color  = "#28a745"
            gauge_color = "green"
        elif prob_pct < 50:
            risk_level  = "MEDIUM RISK"
            risk_color  = "#fd7e14"
            gauge_color = "orange"
        else:
            risk_level  = "HIGH RISK"
            risk_color  = "#dc3545"
            gauge_color = "red"

        st.subheader("Prediction Result")

        col_text, col_gauge = st.columns([1, 2])

        with col_text:
            st.markdown("**Dropout Probability**")
            st.markdown(f"## {prob_pct:.2f}%")
            st.markdown(
                f"**Risk Level:**<br>"
                f"<span style='color:{risk_color}; font-size:26px; "
                f"font-weight:bold;'>{risk_level}</span>",
                unsafe_allow_html=True
            )
            st.caption(
                f"OULAD base dropout rate: **31%**  ·  "
                f"This student: **{prob_pct:.1f}%**"
            )

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 44}},
                title={"text": "Dropout Risk", "font": {"size": 16}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickvals": [0, 25, 50, 75, 100],
                    },
                    "bar":  {"color": gauge_color, "thickness": 0.28},
                    "steps": [
                        {"range": [0,  25], "color": "#d4edda"},
                        {"range": [25, 50], "color": "#fff3cd"},
                        {"range": [50, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line":      {"color": gauge_color, "width": 4},
                        "thickness": 0.8,
                        "value":     prob_pct,
                    },
                }
            ))
            fig.update_layout(
                height=280,
                margin=dict(t=50, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Key Input Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Assessments", f"{num_assessments:.0f}")
        c2.metric("Avg Score",   f"{avg_score:.1f}")
        c3.metric("Active Days", f"{active_days:.0f}")
        c4.metric("Total Clicks", f"{total_clicks:.0f}")

# ---------------------------------------------------
# MODEL METRICS PAGE
# ---------------------------------------------------

elif page == "Model Metrics":

    st.title("Model Evaluation")

    metrics_df = pd.DataFrame({
        "Model":   ["Gradient Boosting", "SVM", "Random Forest",
                    "XGBoost", "Logistic Regression", "KNN"],
        "F1":      [0.7886, 0.7832, 0.7801, 0.7783, 0.7748, 0.7078],
        "ROC-AUC": [0.9362, 0.9155, 0.9364, 0.9377, 0.9179, 0.8835],
    })
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Final Selected Model")
    st.success(
        "XGBoost (Calibrated)\n\n"
        "Accuracy:    0.8659\n"
        "Precision:   0.7692\n"
        "Recall:      0.8139\n"
        "F1-score:    0.7909\n"
        "ROC-AUC:     0.9423\n"
        "Brier Score: 0.0892"
    )

# ---------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------

elif page == "About Project":

    st.title("About the Project")
    st.markdown("""
    ### Development of a Predictive Model for Student Dropout Risk Based on Learning Analytics

    This project was developed as a diploma research project at Astana IT University.

    **Technologies used:**
    - Python · Scikit-learn · XGBoost · SHAP
    - Streamlit · Plotly

    The system demonstrates how machine learning can support
    educational decision-making and early intervention strategies.
    """)
