"""
Streamlit Demo App
==================
Interactive credit scoring demo with SHAP explanations.

Run with: streamlit run demo/streamlit_app.py

Provides a visual interface to:
1. Input features and get a credit risk score
2. View SHAP-based feature attributions
3. See the Model Card
4. Compare baseline vs challenger models
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import SHAPExplainer

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Alt-Credit-Score Demo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Alt-Credit-Score — Explainable Credit Scoring Demo")
st.markdown("""
This demo showcases an **explainable credit scoring engine** trained on the
UCI German Credit Dataset augmented with **synthetic alternative-data features**
(telco, e-wallet, utility payment signals).

**STARS AI Governance Framework** (BSP) alignment:
- **Transparency**: SHAP explanations per prediction
- **Accountability**: Auto-generated Model Card
- **Responsibility**: Fairness audit with CI gate
- **Sustainability**: Borderline scores route to human review
- **Security**: No real PII — all alternative data is synthetic
""")

# ============================================================
# Load models
# ============================================================
@st.cache_resource
def load_models():
    challenger_path = PROJECT_ROOT / "models" / "challenger_xgboost.pkl"
    baseline_path = PROJECT_ROOT / "models" / "baseline_logreg.pkl"
    importance_path = PROJECT_ROOT / "models" / "challenger_feature_importance.json"

    challenger = joblib.load(challenger_path) if challenger_path.exists() else None
    baseline = joblib.load(baseline_path) if baseline_path.exists() else None

    feature_names = []
    if importance_path.exists():
        import json
        with open(importance_path) as f:
            importances = json.load(f)
        feature_names = [item["feature"] for item in importances]

    return challenger, baseline, feature_names


challenger_model, baseline_model, feature_names = load_models()

if challenger_model is None:
    st.error("Models not found. Please train models first by running:")
    st.code("python -m src.train_baseline && python -m src.train_challenger")
    st.stop()

# ============================================================
# Sidebar: Configuration
# ============================================================
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.radio(
    "Model",
    options=["Challenger (XGBoost)", "Baseline (Logistic Regression)"],
)

# Borderline thresholds
borderline_low = st.sidebar.slider("Borderline Low", 0.0, 0.5, 0.40)
borderline_high = st.sidebar.slider("Borderline High", 0.5, 1.0, 0.60)

# ============================================================
# Tab: Score a Sample
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔢 Score a Sample", "📈 SHAP Explanation", "📋 Model Card"])

with tab1:
    st.subheader("Score a Random Sample")

    col1, col2 = st.columns(2)
    with col1:
        sample_idx = st.slider("Sample Index", 0, 100, 0)

    if st.button("Generate Score"):
        # Generate a random sample for demonstration
        np.random.seed(sample_idx)
        sample = np.random.randn(len(feature_names))

        # Scale to reasonable ranges
        sample = (sample + 2) * 5  # Shift to positive, scale

        # Score with selected model
        if model_choice == "Baseline (Logistic Regression)" and baseline_model is not None:
            model = baseline_model
            model_name = "baseline"
        else:
            model = challenger_model
            model_name = "challenger"

        X_sample = sample.reshape(1, -1)
        default_prob = float(model.predict_proba(X_sample)[:, 1][0])

        # Risk classification
        if default_prob >= borderline_high:
            risk_level = "🔴 HIGH RISK"
            color = "red"
        elif default_prob <= borderline_low:
            risk_level = "🟢 LOW RISK"
            color = "green"
        else:
            risk_level = "🟡 BORDERLINE"
            color = "orange"

        st.markdown(f"### Default Probability: **{default_prob:.4f}**")
        st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)

        if borderline_low < default_prob < borderline_high:
            st.warning("⚠️ **Human Review Required** — Score is in the borderline zone. "
                      "This prediction should be reviewed by a human under the STARS sustainability pillar.")

        # Feature table
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": [round(float(v), 2) for v in sample],
        })
        st.dataframe(feature_df, use_container_width=True)

        # Store for explanation tab
        st.session_state["current_sample"] = sample
        st.session_state["current_prob"] = default_prob
        st.session_state["current_model_name"] = model_name

with tab2:
    st.subheader("SHAP Feature Attribution")

    if "current_sample" not in st.session_state:
        st.info("Generate a score first in the 'Score a Sample' tab.")
    else:
        sample = st.session_state["current_sample"]
        X_sample = sample.reshape(1, -1)

        # Compute SHAP
        model_type = "tree" if st.session_state.get("current_model_name") == "challenger" else "linear"
        explainer = SHAPExplainer(challenger_model if model_type == "tree" else baseline_model,
                                  feature_names, model_type=model_type)
        background = np.random.randn(50, len(feature_names))
        explainer.fit(background, sample_size=20)

        explanation = explainer.explain_single(X_sample, return_dict=True)

        st.markdown(f"**Base value:** {explanation['base_value']:.4f}")
        st.markdown(f"**SHAP prediction:** {explanation['prediction_shap']:.4f}")

        # SHAP values chart
        attributions = explanation["attributions"][:15]
        shap_df = pd.DataFrame(attributions)
        shap_df = shap_df.sort_values("shap_value", key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["red" if v > 0 else "green" for v in shap_df["shap_value"]]
        ax.barh(shap_df["feature"], shap_df["shap_value"], color=colors)
        ax.set_xlabel("SHAP Value (positive = increases risk)")
        ax.set_title("Top Feature Attributions")
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(shap_df, use_container_width=True)

with tab3:
    st.subheader("Model Card")
    model_card_path = PROJECT_ROOT / "MODEL_CARD.md"
    if model_card_path.exists():
        with open(model_card_path) as f:
            st.markdown(f.read())
    else:
        st.warning("Model Card not found. Run the model card generator first.")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("""
⚠️ **DISCLAIMER:** This is a demonstration project. All alternative-data features
are synthetically generated. No real personal data was used or accessed.
This model is NOT intended for production credit decisions.

Built aligned to BSP's STARS AI Governance Framework.
""")
