import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Model Metrics",
    page_icon="📊",
    layout="wide",
)

BASE_DIR  = Path(__file__).parent.parent
PLOTS_DIR = BASE_DIR / "plots"

st.title("Model Evaluation & Comparison")
st.caption("OULAD dataset · 32,593 students · 80/20 stratified train/test split")

# ─────────────────────────────────────────────
# Full comparison table
# ─────────────────────────────────────────────
st.subheader("All Models — Test Set Metrics")

comparison = pd.DataFrame({
    "Model": [
        "Baseline — Logistic Regression",
        "Random Forest (default)",
        "Gradient Boosting (default)",
        "XGBoost (default)",
        "SVM (default)",
        "KNN (default)",
        "XGBoost (Optuna tuned)",
        "Gradient Boosting (Optuna tuned)",
        "Gradient Boosting (calibrated)",
        "XGBoost (calibrated) ★",
    ],
    "Accuracy":  [0.8400, 0.8619, 0.8642, 0.8595, 0.8540, 0.8225,
                  0.8664, 0.8652, 0.8653, 0.8659],
    "Precision": [0.6900, 0.7744, 0.7658, 0.7653, 0.7285, 0.7267,
                  0.7703, 0.7669, 0.7685, 0.7692],
    "Recall":    [0.8833, 0.7858, 0.8129, 0.7917, 0.8469, 0.6898,
                  0.8139, 0.8149, 0.8124, 0.8139],
    "F1":        [0.7748, 0.7801, 0.7886, 0.7783, 0.7832, 0.7078,
                  0.7915, 0.7902, 0.7899, 0.7909],
    "ROC-AUC":   [0.9179, 0.9364, 0.9362, 0.9377, 0.9155, 0.8835,
                  0.9418, 0.9412, 0.9406, 0.9423],
    "Brier ↓":   [0.1188, 0.0938, 0.0927, 0.0940, 0.1041, 0.1242,
                  0.0897, 0.0897, 0.0899, 0.0892],
})

FINAL_ROW = 9  # 0-indexed row of XGBoost (calibrated) ★

def _highlight(row):
    style = "background-color: #1a472a; font-weight: bold"
    return [style if row.name == FINAL_ROW else "" for _ in row]

st.dataframe(
    comparison.style.apply(_highlight, axis=1).format(
        {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
         "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "Brier ↓": "{:.4f}"}
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("★ Final selected model  ·  Brier Score: lower = better probability calibration")

# ─────────────────────────────────────────────
# Final model highlight
# ─────────────────────────────────────────────
st.divider()
st.subheader("Final Model: XGBoost (Calibrated)")
st.markdown(
    "Selected automatically by composite score **F1 + ROC-AUC − Brier Score** "
    "among all tuned and calibrated candidates."
)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy",  "0.8659")
c2.metric("F1-Score",  "0.7909")
c3.metric("ROC-AUC",  "0.9423")

c4, c5, c6 = st.columns(3)
c4.metric("Precision", "0.7692")
c5.metric("Recall",    "0.8139")
c6.metric("Brier Score", "0.0892", help="Lower is better — measures probability calibration quality")

# ─────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────
st.divider()
st.subheader("Cross-Validation Results — 5-Fold Stratified")
st.markdown("CV was run on the **training set** (26,074 records) to avoid data leakage.")

cv_df = pd.DataFrame({
    "Model": [
        "Baseline (Logistic Regression)",
        "XGBoost (Optuna — best trial)",
        "Gradient Boosting (Optuna — best trial)",
    ],
    "CV F1": ["0.7712 ± 0.0081", "0.7895", "0.7878"],
    "CV ROC-AUC": ["0.9172 ± 0.0046", "—", "—"],
    "Note": [
        "5-fold CV with mean ± std",
        "Best objective value across 30 Optuna trials",
        "Best objective value across 20 Optuna trials",
    ],
})
st.dataframe(cv_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Calibration improvement
# ─────────────────────────────────────────────
st.divider()
st.subheader("Calibration Effect (Brier Score)")
st.markdown(
    "After applying `CalibratedClassifierCV(method='isotonic', cv=5)` "
    "the Brier score improves, indicating better-calibrated probabilities."
)

cal_df = pd.DataFrame({
    "Model": ["XGBoost", "Gradient Boosting"],
    "Brier before calibration": [0.0897, 0.0897],
    "Brier after calibration":  [0.0892, 0.0899],
    "Improvement": ["✓ −0.0005", "✗ +0.0002"],
})
st.dataframe(cal_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Performance plots
# ─────────────────────────────────────────────
st.divider()
st.subheader("Performance Plots")

col_roc, col_cal = st.columns(2)

with col_roc:
    st.markdown("**ROC Curve** — AUC = 0.9423")
    roc_path = PLOTS_DIR / "roc_curve.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
    else:
        st.info("plots/roc_curve.png not found — re-run the notebook to generate it.")

with col_cal:
    st.markdown("**Calibration Curve** — before vs. after isotonic calibration")
    cal_path = PLOTS_DIR / "calibration_curve.png"
    if cal_path.exists():
        st.image(str(cal_path), use_container_width=True)
    else:
        st.info("plots/calibration_curve.png not found — re-run the notebook to generate it.")

# ─────────────────────────────────────────────
# Dataset info
# ─────────────────────────────────────────────
st.divider()
st.subheader("Dataset & Pipeline Summary")

col_ds, col_pipe = st.columns(2)

with col_ds:
    st.dataframe(pd.DataFrame({
        "Property": [
            "Dataset", "Source", "Total Records", "Features used",
            "Target (positive class)", "Base dropout rate",
            "Train size", "Test size", "Split strategy",
        ],
        "Value": [
            "OULAD", "Open University Learning Analytics Dataset",
            "32,593", "19",
            "Withdrawn (dropout)", "31.16%",
            "26,074 (80%)", "6,519 (20%)",
            "Stratified by target label",
        ],
    }), use_container_width=True, hide_index=True)

with col_pipe:
    st.markdown("**Modelling pipeline steps:**")
    st.markdown("""
    1. Load 7 raw OULAD tables → aggregate to 1 dataset (1 row per student × course)
    2. Label encode categorical features
    3. Stratified 80/20 train/test split
    4. StandardScaler fit on train, applied to both splits
    5. 5-fold stratified cross-validation (training set only)
    6. Train 6 models with default hyperparameters
    7. Hyperparameter tuning with **Optuna** (XGBoost: 30 trials, GB: 20 trials)
    8. Probability calibration with **CalibratedClassifierCV** (isotonic, cv=5)
    9. Final model selection by composite score (F1 + ROC-AUC − Brier)
    10. SHAP explainability on 500-sample test subset
    """)
