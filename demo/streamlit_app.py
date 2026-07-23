"""
Alt-Credit-Score — Interactive API Playground
=============================================
A step-by-step web guide that demonstrates every API endpoint live.

Run with: streamlit run demo/streamlit_app.py
(Server must be running: uvicorn api.main:app --reload --port 8000)
"""

import sys
import json
from pathlib import Path
from urllib import request, error
from urllib.error import URLError, HTTPError
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Alt-Credit-Score API Playground",
    page_icon="🎯",
    layout="wide",
)

# ============================================================
# Helper
# ============================================================
API_BASE = "http://localhost:8000"
API_KEY = "demo-key-change-in-production"


def api_get(path: str, auth: bool = True) -> dict | None:
    """Make a GET request to the API."""
    url = f"{API_BASE}{path}"
    headers = {"X-API-Key": API_KEY} if auth else {}
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"_error": str(e)}


def api_post(path: str, data: dict) -> dict | None:
    """Make a POST request to the API."""
    url = f"{API_BASE}{path}"
    body = json.dumps(data).encode("utf-8")
    req = request.Request(
        url, data=body,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"_error": str(e)}


def show_code_block(label: str, code: str, lang: str = "bash"):
    """Display a copy-paste code block."""
    st.markdown(f"**{label}**")
    st.code(code, language=lang)


def show_curl_for_score(features, model):
    """Generate the exact curl command for a scoring request."""
    feat_preview = json.dumps(features[:5]) + ', ... (' + str(len(features)) + ' total)'
    return (
        f'curl -X POST {API_BASE}/score \\\n'
        f'  -H "X-API-Key: {API_KEY}" \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f'  -d \'{{"features": [{feat_preview}], "model": "{model}"}}\''
    )


def show_curl_for_explain(features, top_n):
    """Generate the exact curl command for an explain request."""
    feat_preview = json.dumps(features[:3]) + ', ... (' + str(len(features)) + ' total)'
    return (
        f'curl -X POST {API_BASE}/explain \\\n'
        f'  -H "X-API-Key: {API_KEY}" \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f'  -d \'{{"features": [{feat_preview}], "top_n": {top_n}}}\''
    )


# ============================================================
# Applicant Profiles (pre-built sample data)
# ============================================================
PROFILES = {
    "🏦 Maria — Stable Professional (Low Risk)": {
        "synthetic_telco_topup_freq": 18,
        "synthetic_telco_avg_data_usage_gb": 15,
        "synthetic_telco_sim_tenure_months": 72,
        "synthetic_telco_avg_call_minutes": 450,
        "synthetic_ewallet_tx_count": 25,
        "synthetic_ewallet_avg_tx_amount_usd": 45,
        "synthetic_utility_bill_on_time_rate": 0.95,
        "synthetic_rent_payment_regularity": 0.90,
        "synthetic_social_credit_inquiries": 1,
        "synthetic_income_volatility": 0.15,
    },
    "🚗 Juan — Gig Worker (Borderline)": {
        "synthetic_telco_topup_freq": 8,
        "synthetic_telco_avg_data_usage_gb": 5,
        "synthetic_telco_sim_tenure_months": 18,
        "synthetic_telco_avg_call_minutes": 120,
        "synthetic_ewallet_tx_count": 8,
        "synthetic_ewallet_avg_tx_amount_usd": 15,
        "synthetic_utility_bill_on_time_rate": 0.60,
        "synthetic_rent_payment_regularity": 0.50,
        "synthetic_social_credit_inquiries": 5,
        "synthetic_income_volatility": 0.65,
    },
    "⚠️ Pedro — High Risk": {
        "synthetic_telco_topup_freq": 2,
        "synthetic_telco_avg_data_usage_gb": 0.5,
        "synthetic_telco_sim_tenure_months": 3,
        "synthetic_telco_avg_call_minutes": 20,
        "synthetic_ewallet_tx_count": 0,
        "synthetic_ewallet_avg_tx_amount_usd": 0,
        "synthetic_utility_bill_on_time_rate": 0.20,
        "synthetic_rent_payment_regularity": 0.10,
        "synthetic_social_credit_inquiries": 12,
        "synthetic_income_volatility": 1.20,
    },
}


def build_feature_vector(feature_names: list, profile: dict) -> list:
    """Build a full feature vector from the model's expected feature names."""
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    vector = [0.0] * len(feature_names)
    for name, val in profile.items():
        if name in name_to_idx:
            vector[name_to_idx[name]] = val
    return vector


# ============================================================
# App State
# ============================================================
if "step" not in st.session_state:
    st.session_state.step = 0
if "feature_names" not in st.session_state:
    st.session_state.feature_names = []
if "last_features" not in st.session_state:
    st.session_state.last_features = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None
if "model_card" not in st.session_state:
    st.session_state.model_card = None


# ============================================================
# Header
# ============================================================
st.title("🎯 Alt-Credit-Score — Interactive API Playground")
st.markdown("""
This is a **step-by-step guide** that calls the API **live** and shows you exactly how to use it.
Each step demonstrates one API endpoint with the actual request and response.

> **Prerequisite:** Start the API server first: `uvicorn api.main:app --reload --port 8000`
""")

# ============================================================
# Navigation: Step tabs
# ============================================================
step_names = [
    "1️⃣ Health Check",
    "2️⃣ Feature Names",
    "3️⃣ Score Applicant",
    "4️⃣ Explain (SHAP)",
    "5️⃣ Model Card",
    "6️⃣ All Code Samples",
]
tabs = st.tabs(step_names)

# ============================================================
# STEP 1: Health Check
# ============================================================
with tabs[0]:
    st.header("Step 1: Health Check — GET /health")
    st.markdown("Before doing anything, verify the API server is running.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        show_code_block("Copy-paste this command:", f'curl {API_BASE}/health')

    with col_b:
        if st.button("🔍 Call /health", type="primary", use_container_width=True):
            result = api_get("/health", auth=False)
            if result and "_error" not in result:
                st.success(f"✅ Server is UP — {result.get('status', 'healthy')}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Model Loaded", "✅" if result.get("model_loaded") else "❌")
                c2.metric("Features", result.get("n_features", "?"))
                c3.metric("Baseline", "✅" if result.get("baseline_loaded") else "❌")
                st.json(result, expanded=False)
            else:
                st.error(f"❌ Cannot connect to {API_BASE}. Make sure the server is running!")
                st.code("uvicorn api.main:app --reload --port 8000")

    st.divider()
    st.markdown("**What you learned:** `/health` needs no auth. Returns server status and feature count.")

# ============================================================
# STEP 2: Feature Names
# ============================================================
with tabs[1]:
    st.header("Step 2: Feature Names — GET /feature-names")
    st.markdown("Learn what features the model expects, and in what order.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        show_code_block("Copy-paste this command:",
                        f'curl -H "X-API-Key: {API_KEY}" {API_BASE}/feature-names')

    with col_b:
        if st.button("📋 Call /feature-names", type="primary", use_container_width=True):
            result = api_get("/feature-names")
            if result and "_error" not in result:
                names = result.get("feature_names", [])
                st.session_state.feature_names = names
                st.success(f"✅ Model expects **{len(names)} features**")

                # Show synthetic features highlighted
                synth = [n for n in names if n.startswith("synthetic_")]
                trad = [n for n in names if not n.startswith("synthetic_")]
                c1, c2 = st.columns(2)
                c1.metric("Synthetic Alt-Data", len(synth))
                c2.metric("Traditional (German Credit)", len(trad))

                # Feature table
                df = pd.DataFrame({"Index": range(len(names)), "Feature": names})
                df["Type"] = df["Feature"].apply(lambda x: "🟢 Synthetic" if x.startswith("synthetic_") else "🔵 Traditional")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error("❌ Cannot connect. Is the API server running?")

    st.divider()
    st.markdown("**What you learned:** Features are numbered by index. When you call `/score`, send values in this order.")

# ============================================================
# STEP 3: Score an Applicant
# ============================================================
with tabs[2]:
    st.header("Step 3: Score an Applicant — POST /score")
    st.markdown("Send applicant data and get a credit risk assessment.")

    if not st.session_state.feature_names:
        st.warning("⚠️ Complete Step 2 first to load feature names!")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Choose an Applicant")
            selected_profile = st.selectbox(
                "Pre-built profiles:",
                list(PROFILES.keys()),
                help="Each profile sets realistic values for the 10 synthetic features",
            )

            st.subheader("Choose Model")
            model = st.radio("Model:", ["challenger", "baseline"], horizontal=True)

            if st.button("🚀 Score This Applicant", type="primary", use_container_width=True):
                profile = PROFILES[selected_profile]
                features = build_feature_vector(st.session_state.feature_names, profile)
                st.session_state.last_features = features

                result = api_post("/score", {"features": features, "model": model})
                st.session_state.last_result = result

            st.divider()
            show_code_block("Python SDK equivalent:",
                f'''from api_client import CreditScoreClient

client = CreditScoreClient(api_key="{API_KEY}")
result = client.score(features, model="{model}")
print(result)''', lang="python")

        with col_right:
            if st.session_state.last_result:
                r = st.session_state.last_result
                if "_error" in r:
                    st.error(f"API Error: {r['_error']}")
                else:
                    # Big score display
                    prob = r.get("score", 0)
                    risk = r.get("risk_level", "unknown")
                    credit_score = round((1 - prob) * 850, 1)

                    if risk == "low_risk":
                        emoji, color = "✅", "green"
                    elif risk == "high_risk":
                        emoji, color = "🚫", "red"
                    else:
                        emoji, color = "⚠️", "orange"

                    st.markdown(f"### {emoji} Risk: **{risk.replace('_', ' ').title()}**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Default Probability", f"{prob:.1%}")
                    c2.metric("Est. Credit Score", f"~{credit_score}")
                    c3.metric("Human Review", "Yes" if r.get("human_review_required") else "No")

                    if r.get("human_review_required"):
                        st.warning("⚠️ **Human Review Required** — This score is in the borderline zone. A human underwriter should review this case.")

                    st.subheader("API Response (JSON)")
                    st.json(r, expanded=False)

                    st.subheader("Request Details")
                    st.code(show_curl_for_score(st.session_state.last_features, model))
            else:
                st.info("Select an applicant and click **Score** to see the result here.")

    st.divider()
    st.markdown("**What you learned:** `POST /score` sends a feature array + model name. Returns probability, risk level, and auto-decision flag.")

# ============================================================
# STEP 4: SHAP Explanation
# ============================================================
with tabs[3]:
    st.header("Step 4: Explain — POST /explain")
    st.markdown("Get a SHAP breakdown showing **why** the model gave that score.")

    if not st.session_state.last_features:
        st.warning("⚠️ Score an applicant in Step 3 first!")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            top_n = st.slider("Top N features to explain:", 3, 20, 10)

            if st.button("🔬 Get SHAP Explanation", type="primary", use_container_width=True):
                result = api_post("/explain", {
                    "features": st.session_state.last_features,
                    "top_n": top_n,
                })
                st.session_state.last_explanation = result

            st.divider()
            show_code_block("Python SDK equivalent:",
                f'''from api_client import CreditScoreClient

client = CreditScoreClient(api_key="{API_KEY}")
exp = client.explain(features, top_n={top_n})
exp.print_attribution_table()''', lang="python")

        with col_right:
            if st.session_state.last_explanation:
                exp = st.session_state.last_explanation
                if "_error" in exp:
                    st.error(f"API Error: {exp['_error']}")
                else:
                    st.markdown(f"**Base Value:** {exp.get('base_value', 0):.4f}  |  **SHAP Prediction:** {exp.get('prediction_shap', 0):.4f}")

                    attrs = exp.get("attributions", [])
                    if attrs:
                        df = pd.DataFrame(attrs)
                        # Sort by absolute SHAP value
                        df["_abs"] = df["shap_value"].abs()
                        df = df.sort_values("_abs", ascending=True)

                        fig = px.bar(
                            df, y="feature", x="shap_value",
                            color="shap_value",
                            color_continuous_scale=["green", "yellow", "red"],
                            title="Feature Impact on Default Risk",
                            labels={"shap_value": "SHAP Value", "feature": ""},
                        )
                        fig.update_layout(height=max(400, len(attrs) * 50),
                                         xaxis_title="Positive = increases risk | Negative = decreases risk")
                        st.plotly_chart(fig, use_container_width=True)

                        # Attribution table
                        display = df[["feature", "shap_value", "feature_value", "direction"]].reset_index(drop=True)
                        display.columns = ["Feature", "SHAP Value", "Feature Value", "Direction"]
                        st.dataframe(display, use_container_width=True, hide_index=True)
                    else:
                        st.info("No attributions returned.")

                    st.subheader("Request")
                    st.code(show_curl_for_explain(st.session_state.last_features, top_n))
            else:
                st.info("Click **Get SHAP Explanation** to see the breakdown.")

    st.divider()
    st.markdown("**What you learned:** `POST /explain` returns which features pushed the score up (red) or down (green) with exact impact values.")

# ============================================================
# STEP 5: Model Card
# ============================================================
with tabs[4]:
    st.header("Step 5: Model Card — GET /model-card")
    st.markdown("Get the auto-generated documentation for the current model.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        show_code_block("Copy-paste this command:", f'curl {API_BASE}/model-card')

        st.markdown("")
        if st.button("📄 Call /model-card", type="primary", use_container_width=True):
            result = api_get("/model-card", auth=False)
            st.session_state.model_card = result

    with col_b:
        if st.session_state.model_card:
            mc = st.session_state.model_card
            if "_error" in mc:
                st.error(f"API Error: {mc['_error']}")
            else:
                st.success("✅ Model Card retrieved")
                card_text = mc.get("model_card", "")
                st.markdown(card_text)
        else:
            st.info("Click to load the Model Card.")

    st.divider()
    st.markdown("**What you learned:** `/model-card` is public (no auth needed). Returns training data provenance, metrics, and limitations.")

# ============================================================
# STEP 6: All Code Samples
# ============================================================
with tabs[5]:
    st.header("Step 6: Complete Code Samples")
    st.markdown("Everything you need to integrate this API into your own application.")

    tab_a, tab_b, tab_c = st.tabs(["Python SDK", "curl / Bash", "PowerShell"])

    with tab_a:
        st.subheader("Python SDK")
        st.code('''
# Install: pip install requests (or use the bundled api_client)

from api_client import CreditScoreClient

# 1. Connect
client = CreditScoreClient(
    base_url="http://localhost:8000",
    api_key="demo-key-change-in-production",
)

# 2. Health check
health = client.health()
print(health)

# 3. Get feature names
feature_names = client.get_feature_names()
print(f"Model expects {len(feature_names)} features")

# 4. Build feature vector
features = [0.0] * len(feature_names)
name_to_idx = {name: i for i, name in enumerate(feature_names)}
features[name_to_idx["synthetic_telco_topup_freq"]] = 18.0
features[name_to_idx["synthetic_ewallet_tx_count"]] = 25.0
features[name_to_idx["synthetic_utility_bill_on_time_rate"]] = 0.95
features[name_to_idx["synthetic_income_volatility"]] = 0.15

# 5. Score
result = client.score(features, model="challenger")
print(result)

# 6. Explain
explanation = client.explain(features, top_n=10)
explanation.print_attribution_table()

# 7. Model card
print(client.get_model_card())
''', language="python")

    with tab_b:
        st.subheader("curl / Bash")
        st.code('''
#!/bin/bash
BASE_URL="http://localhost:8000"
API_KEY="demo-key-change-in-production"

# Health check
curl $BASE_URL/health

# Feature names
curl -H "X-API-Key: $API_KEY" $BASE_URL/feature-names

# Score (replace ... with all 80 feature values)
curl -X POST $BASE_URL/score \\
  -H "X-API-Key: $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"features": [18, 0, ..., 0.95, ...], "model": "challenger"}'

# Explain
curl -X POST $BASE_URL/explain \\
  -H "X-API-Key: $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"features": [18, 0, ...], "top_n": 10}'

# Model card
curl $BASE_URL/model-card
''', language="bash")

    with tab_c:
        st.subheader("PowerShell")
        st.code('''
$BaseUrl = "http://localhost:8000"
$Headers = @{ "X-API-Key" = "demo-key-change-in-production" }

# Health
Invoke-RestMethod -Uri "$BaseUrl/health"

# Feature names
Invoke-RestMethod -Uri "$BaseUrl/feature-names" -Headers $Headers

# Score
$body = @{ features = @(18.0, 0.0, ...); model = "challenger" } | ConvertTo-Json
Invoke-RestMethod -Uri "$BaseUrl/score" -Method Post -Headers $Headers -Body $body

# Model card
Invoke-RestMethod -Uri "$BaseUrl/model-card"
''', language="powershell")

# ============================================================
# API Reference Table
# ============================================================
st.divider()
st.header("📖 API Quick Reference")
st.dataframe(pd.DataFrame([
    {"Endpoint": "GET /health", "Auth": "No", "Description": "Server health & model status"},
    {"Endpoint": "GET /feature-names", "Auth": "Yes", "Description": "List expected features in order"},
    {"Endpoint": "POST /score", "Auth": "Yes", "Description": "Score an applicant (body: features[], model)"},
    {"Endpoint": "POST /explain", "Auth": "Yes", "Description": "SHAP explanation (body: features[], top_n)"},
    {"Endpoint": "GET /model-card", "Auth": "No", "Description": "Model documentation (markdown)"},
]), use_container_width=True, hide_index=True)

st.divider()
st.caption("""
⚠️ **DISCLAIMER:** All alternative-data features are synthetically generated for demonstration.
No real personal data was used. This model is NOT intended for production credit decisions.
Built aligned to BSP's STARS AI Governance Framework.
""")