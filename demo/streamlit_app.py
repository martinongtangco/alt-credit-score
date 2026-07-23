"""
Alt-Credit-Score — Learn the API, Step by Step
================================================
A friendly, educational web guide that teaches you how credit scoring
APIs work — from zero knowledge to making your first prediction.

Run: streamlit run demo/streamlit_app.py
(Server: python -m uvicorn api.main:app --reload --port 8000)
"""

import sys
import json
from pathlib import Path
from urllib import request, error

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = "http://localhost:8000"
API_KEY = "demo-key-change-in-production"

# ============================================================
# Helpers
# ============================================================
def api_get(path: str, auth: bool = True) -> dict | None:
    url = f"{API_BASE}{path}"
    headers = {"X-API-Key": API_KEY} if auth else {}
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"_error": str(e)}


def api_post(path: str, data: dict) -> dict | None:
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


def build_feature_vector(feature_names: list, profile: dict) -> list:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    vector = [0.0] * len(feature_names)
    for name, val in profile.items():
        if name in name_to_idx:
            vector[name_to_idx[name]] = val
    return vector


PROFILES = {
    "Maria — Steady Office Worker (Low Risk)": {
        "story": (
            "Maria is 32 years old and works at a bank. She has had the same phone "
            "number for 6+ years, uses mobile data regularly, keeps her e-wallet active "
            "with daily transactions, and always pays her electricity bill on time. "
            "She has only checked her credit once — she's confident she's in good shape."
        ),
        "features": {
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
    },
    "Juan — Freelance Driver (Uncertain)": {
        "story": (
            "Juan is 25 and drives for a ride-hailing app. His income varies month to "
            "month — some weeks are great, some are lean. He has his current phone "
            "number for about 1.5 years, uses moderate data, and sometimes forgets to "
            "pay bills on time. He's applied for a few small loans recently."
        ),
        "features": {
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
    },
    "Pedro — Struggling (High Risk)": {
        "story": (
            "Pedro is 22 and hasn't found stable work. He barely uses his phone — low "
            "data, few calls, no e-wallet at all. He got a new SIM card just 3 months "
            "ago and has a history of late bill payments. Multiple lenders have recently "
            "checked his credit, and his income bounces all over the place."
        ),
        "features": {
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
    },
}

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Credit Scoring API — Learn by Doing", page_icon="🎓", layout="wide")

# ============================================================
# Session state
# ============================================================
for key, default in [("feature_names", []), ("last_features", []), ("last_result", None),
                      ("last_explanation", None), ("model_card", None), ("steps_done", set())]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# WELCOME / LANDING PAGE
# ============================================================
st.title("🎓 How Does a Credit Scoring API Work? Let Me Show You.")

st.markdown("""
### You don't need to be a data scientist to understand this.

Imagine you're a bank manager deciding whether to lend someone money. You'd look at:
- **Do they have a stable job?** → steady income = lower risk
- **Do they pay their bills on time?** → responsible person = lower risk
- **Have many other lenders checked their credit?** → desperate = higher risk

But what about people who **never had a bank account**? No loan history, no credit card — "thin file" customers.
Traditional banks can't evaluate them.

**Our API solves this problem.** Instead of bank records, it looks at **alternative data**:

| Instead of... | We look at... | What it tells us |
|---|---|---|
| Bank statements | 📱 How often they top up their phone | Regular habits = reliable person |
| Credit history | 💳 E-wallet transaction frequency | Financial activity = financially active |
| Employment records | ⏰ How long they've had the same phone number | Stability = stable person |
| Loan repayment record | 💡 On-time utility bill payments | Responsibility = responsible payer |

### This tutorial will walk you through 5 steps:
""")

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown("**Step 1**  \n🏥  \nIs the server alive?")
c2.markdown("**Step 2**  \n📋  \nWhat data does it need?")
c3.markdown("**Step 3**  \n🎯  \nScore a real person")
c4.markdown("**Step 4**  \n🔬  \nWhy that score?")
c5.markdown("**Step 5**  \n📄  \nModel documentation")

st.markdown("---")

# ============================================================
# STEP TABS
# ============================================================
tabs = st.tabs([
    "Step 1: Is the Server Alive?",
    "Step 2: What Data Does It Need?",
    "Step 3: Score a Real Person",
    "Step 4: Why That Score?",
    "Step 5: The Model's Report Card",
    "All Code Samples",
])

# ============================================================
# STEP 1: Health Check
# ============================================================
with tabs[0]:
    st.header("Step 1: Is the Server Alive? 🏥")

    st.markdown("""
### Before we do anything, we need to check the server is running.

Think of it like knocking on someone's door before asking them a question.
You knock first to make sure they're home.

**The "knock" in API terms is called a *Health Check*.** You send a simple request and the server
answers with "I'm alive!" or stays silent (which means something's wrong).
""")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.container(border=True):
            st.markdown("#### 🔑 How You'd Do This Yourself")
            st.markdown("**In your terminal:**")
            st.code(f"curl {API_BASE}/health", language="bash")
            st.markdown("**In Python:**")
            st.code("""import requests
r = requests.get("http://localhost:8000/health")
print(r.json())
""", language="python")
            st.markdown("**Notice:** No API key needed! Health checks are public — anyone should be able to check if a service is up.")

    with col_right:
        with st.container(border=True):
            st.markdown("#### 👆 Try It Right Now")
            if st.button("🔨 Knock on the door! (Call /health)", type="primary", use_container_width=True):
                result = api_get("/health", auth=False)

                if result and "_error" not in result:
                    st.success("✅ The server answered! It's alive and ready.")

                    st.markdown("**What the server told us:**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AI Model Loaded", "✅ Yes" if result.get("model_loaded") else "❌ No")
                    c2.metric("Backup Model", "✅ Yes" if result.get("baseline_loaded") else "❌ No")
                    c3.metric("Number of Features", result.get("n_features", "?"))

                    st.markdown("**The raw response (this is what your code receives):**")
                    st.json(result, expanded=False)

                    st.info("💡 **What 'n_features' means:** The AI model looks at 80 different signals (features) to make its decision. We'll see what they all are in Step 2.")

                    st.session_state.steps_done.add(1)
                else:
                    st.error("❌ No response. The server isn't running.")
                    st.markdown("**Start it in a terminal:**")
                    st.code("python -m uvicorn api.main:app --reload --port 8000")

    st.divider()
    st.info("🎓 **Key takeaway:** Every API has a `/health` endpoint. Always check it first. If health fails, nothing else will work.")

# ============================================================
# STEP 2: Feature Names
# ============================================================
with tabs[1]:
    st.header("Step 2: What Data Does the Model Need? 📋")

    st.markdown("""
### Great question! Before we ask the AI to score someone, we need to know: **what information does it want?**

Imagine a doctor asking for a blood test. They don't just say "give me your blood."
They say: "I need cholesterol, blood sugar, white blood cell count..." — a specific list.

Our AI model is the same. It needs **80 specific numbers** in a specific order.

Let's ask the API what those numbers are.
""")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.container(border=True):
            st.markdown("#### 🔑 How You'd Do This Yourself")
            st.code(f"""curl -H "X-API-Key: {API_KEY}" \\
  {API_BASE}/feature-names
""", language="bash")
            st.markdown("**In Python:**")
            st.code("""from api_client import CreditScoreClient
client = CreditScoreClient(api_key="demo-key-change-in-production")
names = client.get_feature_names()
print(f"The model needs {len(names)} features")
""", language="python")
            st.warning("⚠️ **This endpoint needs an API key!** Unlike the health check, this one is protected. The key goes in the `X-API-Key` header.")

    with col_right:
        with st.container(border=True):
            if st.button("📋 Ask the API: 'What features do you need?'", type="primary", use_container_width=True):
                result = api_get("/feature-names")

                if result and "_error" not in result:
                    names = result["feature_names"]
                    st.session_state.feature_names = names
                    st.success(f"✅ Got it! The model needs **{len(names)} numbers**.")

                    synth = [n for n in names if n.startswith("synthetic_")]
                    trad = [n for n in names if not n.startswith("synthetic_")]

                    st.markdown("#### Two Types of Features")
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**🟢 Alternative Data: {len(synth)} features**  \n*These are the creative signals — phone habits, e-wallet usage, bill payment patterns.*")
                    c2.markdown(f"**🔵 Traditional Data: {len(trad)} features**  \n*Classic credit data — account type, credit history, employment — from the German Credit dataset.*")

                    st.markdown("#### The 10 Most Important Alternative Features")
                    important_features = {
                        "Phone top-ups per month": "How often they add credit",
                        "Phone data usage (GB)": "Active user = stable person",
                        "SIM card age (months)": "Longer = more stable",
                        "Monthly call minutes": "Social connections matter",
                        "E-wallet transactions/month": "Financially active",
                        "Avg e-wallet transaction ($)": "Transaction size matters",
                        "Utility bill on-time rate": "0.95 = almost always on time",
                        "Rent payment regularity": "0.90 = reliable renter",
                        "Credit inquiries count": "Too many = desperate",
                        "Income volatility": "Low = stable income",
                    }
                    for feat, meaning in important_features.items():
                        st.caption(f"→ `{feat}` = {meaning}")

                    st.markdown("#### All 80 Features (scroll down)")
                    df = pd.DataFrame({
                        "#": range(len(names)),
                        "Feature Name": names,
                        "Type": ["🟢 Alt Data" if n.startswith("synthetic_") else "🔵 Traditional" for n in names],
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True, height=300)

                    st.session_state.steps_done.add(2)
                else:
                    st.error("❌ Cannot connect. Is the API running?")

    st.divider()
    st.info("🎓 **Key takeaway:** Every ML model needs specific inputs in a specific order. The `/feature-names` endpoint tells you exactly what to send. Save this list — you'll need it for Step 3.")

# ============================================================
# STEP 3: Score an Applicant
# ============================================================
with tabs[2]:
    st.header("Step 3: Score a Real Person 🎯")

    st.markdown("""
### Now the exciting part! Let's actually evaluate a person's creditworthiness.

We have **3 people** with different backgrounds. Click through each one to see how the AI scores them.

The model will give each person a **probability of default** — basically, "How likely is this person to NOT pay back a loan?"

- **0% to 40%** = 🟢 Safe to lend (Low Risk)
- **40% to 60%** = 🟡 Needs a human to decide (Borderline)
- **60% to 100%** = 🔴 Too risky (High Risk)
""")

    if not st.session_state.feature_names:
        st.warning("⚠️ **Go back to Step 2 first** — we need the feature names before we can score anyone!")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("#### 👤 Pick a Person")
            selected = st.selectbox("Choose:", list(PROFILES.keys()))
            profile = PROFILES[selected]

            st.markdown("**Their story:**")
            st.info(profile["story"])

            st.markdown("**What the AI sees about them:**")
            feat_df = pd.DataFrame(
                list(profile["features"].items()),
                columns=["Feature", "Value"]
            )
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            st.markdown("#### 🤖 Which AI Model?")
            model = st.radio("Choose:", [
                ("Challenger — XGBoost (more accurate)", "challenger"),
                ("Baseline — Logistic Regression (simpler)", "baseline"),
            ], horizontal=True, format_func=lambda x: x[0])

            if st.button("🎯 Score This Person!", type="primary", use_container_width=True):
                features = build_feature_vector(st.session_state.feature_names, profile["features"])
                st.session_state.last_features = features
                result = api_post("/score", {"features": features, "model": model[1]})
                st.session_state.last_result = result
                st.session_state.last_model = model[1]

            st.markdown("---")
            st.markdown("#### 🔑 How You'd Do This in Code")
            st.code(f"""# Send the 80-feature array to the API
import requests

response = requests.post(
    "{API_BASE}/score",
    headers={{
        "X-API-Key": "{API_KEY}",
        "Content-Type": "application/json"
    }},
    json={{
        "features": features,  # your 80 numbers
        "model": "{model[1]}"
    }}
)
print(response.json())
""", language="python")

        with col_right:
            if st.session_state.last_result:
                r = st.session_state.last_result

                if "_error" in r:
                    st.error(f"API Error: {r['_error']}")
                else:
                    prob = r["score"]
                    risk = r["risk_level"]
                    credit = round((1 - prob) * 850, 1)

                    # Big result card
                    if risk == "low_risk":
                        emoji, label, bg = "✅", "LOW RISK — Recommend Approve", "#d4edda"
                    elif risk == "high_risk":
                        emoji, label, bg = "🚫", "HIGH RISK — Recommend Decline", "#f8d7da"
                    else:
                        emoji, label, bg = "⚠️", "BORDERLINE — Human Review Needed", "#fff3cd"

                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {bg}; text-align: center;">
                        <h2>{emoji} {label.upper()}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Chance They Default", f"{prob:.1%}")
                    c2.metric("Estimated Credit Score", f"~{credit}")
                    c3.metric("Needs Human Review?", "Yes" if r["human_review_required"] else "No")

                    # Plain English interpretation
                    st.markdown("#### 📖 What This Means in Plain English")
                    if risk == "low_risk":
                        st.success(f"The AI thinks this person has only a **{prob:.1%} chance** of not paying back. That's very low. The recommendation is to **approve the loan automatically**.")
                    elif risk == "high_risk":
                        st.error(f"The AI sees a **{prob:.1%} chance** of default. That's too high to lend money safely. The recommendation is to **decline the application**.")
                    else:
                        st.warning(f"The score is **{prob:.1%}** — right in the gray area. The AI isn't confident enough to decide. A **human underwriter should review this case personally**.")

                    st.markdown("#### Raw API Response")
                    st.json(r, expanded=False)

            else:
                st.info("👈 Select a person and click **Score This Person!** to see the result.")

    st.divider()
    st.info("🎓 **Key takeaway:** The `/score` endpoint takes 80 numbers and returns a probability. Low = safe borrower. High = risky. Middle = needs a human. That's the entire credit scoring engine in one API call.")

# ============================================================
# STEP 4: SHAP Explanation
# ============================================================
with tabs[3]:
    st.header("Step 4: But WHY That Score? 🔬")

    st.markdown("""
### This is the most important step.

Imagine a doctor tells you "you have a 70% chance of heart disease." Your immediate reaction is:
> **"But WHY? What's causing it?"**

You wouldn't accept a number without an explanation. The same applies to credit scoring.

**SHAP** (SHapley Additive exPlanations) is a method that breaks down each prediction and shows:
- Which features **pushed the score up** ⬆️ (more risky)
- Which features **pushed the score down** ⬇️ (less risky)
- By **how much** each feature mattered

Think of it like a tug-of-war. The base value is the starting line. Each feature pulls the score left or right.
""")

    if not st.session_state.last_features:
        st.warning("⚠️ **Score someone in Step 3 first!** We need a person to explain.")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            top_n = st.slider("How many features to show?", 3, 20, 10,
                             help="More features = more detail but also more noise")

            if st.button("🔬 Ask 'WHY?' (Call /explain)", type="primary", use_container_width=True):
                result = api_post("/explain", {
                    "features": st.session_state.last_features,
                    "top_n": top_n,
                })
                st.session_state.last_explanation = result

            st.markdown("---")
            st.markdown("#### 🔑 How You'd Do This in Code")
            st.code(f"""import requests

response = requests.post(
    "{API_BASE}/explain",
    headers={{
        "X-API-Key": "{API_KEY}",
        "Content-Type": "application/json"
    }},
    json={{
        "features": features,
        "top_n": {top_n}
    }}
)
data = response.json()
for attr in data["attributions"]:
    print(f"{{attr['feature']}}: {{attr['shap_value']}}")
""", language="python")

        with col_right:
            if st.session_state.last_explanation:
                exp = st.session_state.last_explanation

                if "_error" in exp:
                    st.error(f"API Error: {exp['_error']}")
                else:
                    attrs = exp.get("attributions", [])
                    if not attrs:
                        st.info("No attributions returned.")
                    else:
                        st.markdown(f"**Starting point (average default rate):** {exp['base_value']:.4f}  →  **Final prediction:** {exp['prediction_shap']:.4f}")

                        # Bar chart
                        df = pd.DataFrame(attrs)
                        df["_abs"] = df["shap_value"].abs()
                        df = df.sort_values("_abs", ascending=True)
                        df["Impact"] = df["shap_value"].apply(
                            lambda x: "⬆️ Increases Risk" if x > 0.001 else ("⬇️ Decreases Risk" if x < -0.001 else "→ Neutral")
                        )

                        fig = px.bar(
                            df, y="feature", x="shap_value",
                            color="shap_value",
                            color_continuous_scale=["green", "yellow", "red"],
                            title="Tug-of-War: What Pushed This Score Up or Down?",
                        )
                        fig.update_layout(height=max(400, len(attrs) * 55))
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("#### 📖 Reading This Chart")
                        st.markdown("""
- **Green bars (left/negative)** = This feature made the person look **safer** ⬇️
- **Red bars (right/positive)** = This feature made the person look **riskier** ⬆️
- **Longer bars** = More influence on the final decision
- **Short bars** = Little influence
""")

                        # Plain English table
                        st.markdown("#### What Each Feature Means")
                        for _, row in df.iterrows():
                            feat = row["feature"]
                            val = row["shap_value"]
                            fv = row["feature_value"]
                            direction = "⬆️ riskier" if val > 0.001 else ("⬇️ safer" if val < -0.001 else "→ neutral")
                            st.caption(f"**{feat}** (value: {fv:.1f}) → pushed score {direction} by {abs(val):.4f}")

            else:
                st.info("👈 Click **Ask 'WHY?'** to see the SHAP breakdown.")

    st.divider()
    st.info("🎓 **Key takeaway:** A credit score without an explanation is just a black box. SHAP opens the box and shows you exactly why each person got their score. This is required by law in many countries!")

# ============================================================
# STEP 5: Model Card
# ============================================================
with tabs[4]:
    st.header("Step 5: The Model's Report Card 📄")

    st.markdown("""
### Every responsible AI system comes with documentation — called a **Model Card**.

Think of it like the nutrition label on food packaging. Before you trust a model with real decisions,
you should know:

- **What data was it trained on?** → Is the data relevant to my users?
- **How accurate is it?** → Can I trust the predictions?
- **What are its known weaknesses?** → Where might it fail?
- **Who built it and when?** → Is it recent? Who's accountable?
- **What ethical safeguards exist?** → Does it discriminate?

Our API serves the Model Card as a public endpoint — anyone can inspect it.
""")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 🔑 How You'd Do This")
        st.code(f"""# No API key needed — it's public!
curl {API_BASE}/model-card

# Or in Python:
client = CreditScoreClient(api_key="...")
print(client.get_model_card())
""", language="python")

        st.markdown("")
        if st.button("📄 Load the Model Card", type="primary", use_container_width=True):
            result = api_get("/model-card", auth=False)
            st.session_state.model_card = result

    with col_right:
        if st.session_state.model_card:
            mc = st.session_state.model_card
            if "_error" in mc:
                st.error(f"API Error: {mc['_error']}")
            else:
                st.success("✅ Model Card loaded!")
                st.markdown(mc["model_card"])
                st.session_state.steps_done.add(5)
        else:
            st.info("👈 Click **Load the Model Card** to see the documentation.")

    st.divider()
    st.info("🎓 **Key takeaway:** Always read the Model Card before using an AI system in production. If there's no Model Card, that's a red flag.")

# ============================================================
# STEP 6: All Code Samples
# ============================================================
with tabs[5]:
    st.header("All Code Samples — Copy & Go 📦")
    st.markdown("Here are complete, working examples in 3 languages. Copy any of these to start integrating the API into your application.")

    tab_py, tab_curl, tab_ps = st.tabs(["Python SDK", "curl / Bash", "PowerShell"])

    with tab_py:
        st.code('''
# ============================================================
# Complete Python Example — Copy & Paste
# ============================================================

from api_client import CreditScoreClient

# 1️⃣ Connect to the API
client = CreditScoreClient(
    base_url="http://localhost:8000",
    api_key="demo-key-change-in-production",
)

# 2️⃣ Check the server is alive
health = client.health()
print(health)
# API Health: healthy
#   Challenger Model: ✅ loaded
#   Baseline Model:   ✅ loaded
#   Features:         80

# 3️⃣ Get the feature names (what the model expects)
feature_names = client.get_feature_names()
print(f"Model expects {len(feature_names)} features")

# 4️⃣ Build a feature vector for a new applicant
features = [0.0] * len(feature_names)

# Use a dictionary to map names → indices
name_to_idx = {name: i for i, name in enumerate(feature_names)}

# Set the synthetic features for our applicant
features[name_to_idx["synthetic_telco_topup_freq"]] = 18.0
features[name_to_idx["synthetic_telco_avg_data_usage_gb"]] = 15.0
features[name_to_idx["synthetic_telco_sim_tenure_months"]] = 72.0
features[name_to_idx["synthetic_ewallet_tx_count"]] = 25.0
features[name_to_idx["synthetic_utility_bill_on_time_rate"]] = 0.95
features[name_to_idx["synthetic_social_credit_inquiries"]] = 1.0
features[name_to_idx["synthetic_income_volatility"]] = 0.15

# 5️⃣ Get a credit risk score
result = client.score(features, model="challenger")
print(result)
# ==================================================
#   CREDIT RISK ASSESSMENT
# ==================================================
#   ✅ Risk Level:        LOW RISK
#   📊 Default Probability: 12.3%
#   📈 Est. Credit Score:   ~745.5
#   ✅ Recommendation:       AUTO-APPROVE
# ==================================================

# 6️⃣ Get the "why" (SHAP explanation)
explanation = client.explain(features, top_n=10)
explanation.print_attribution_table()

# 7️⃣ Read the model documentation
print(client.get_model_card())
''', language="python")

    with tab_curl:
        st.code(f'''#!/bin/bash
# ============================================================
# Complete curl / Bash Example
# ============================================================

BASE_URL="{API_BASE}"
API_KEY="{API_KEY}"

# Step 1: Health check (no auth needed)
echo "=== Health Check ==="
curl $BASE_URL/health

# Step 2: Get feature names
echo ""
echo "=== Feature Names ==="
curl -H "X-API-Key: $API_KEY" $BASE_URL/feature-names

# Step 3: Score an applicant
echo ""
echo "=== Score ==="
curl -X POST $BASE_URL/score \\
  -H "X-API-Key: $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"features": [18, 15, 72, 450, 25, 45, 0.95, 0.90, 1, 0.15, 0, 0, ...], "model": "challenger"}}'

# Step 4: Get SHAP explanation
echo ""
echo "=== Explain ==="
curl -X POST $BASE_URL/explain \\
  -H "X-API-Key: $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"features": [18, 15, 72, ...], "top_n": 10}}'

# Step 5: Model card (no auth needed)
echo ""
echo "=== Model Card ==="
curl $BASE_URL/model-card
''', language="bash")

    with tab_ps:
        st.code(f'''# ============================================================
# Complete PowerShell Example
# ============================================================

$BaseUrl = "{API_BASE}"
$Headers = @{{ "X-API-Key" = "{API_KEY}" }}

# Step 1: Health
Invoke-RestMethod -Uri "$BaseUrl/health"

# Step 2: Feature names
Invoke-RestMethod -Uri "$BaseUrl/feature-names" -Headers $Headers

# Step 3: Score
$body = @{{
    features = @(18.0, 15.0, 72.0, 450.0, 25.0, 45.0, 0.95, 0.90, 1.0, 0.15, 0.0...)
    model    = "challenger"
}} | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "$BaseUrl/score" -Method Post -Headers $Headers -Body $body

# Step 4: Explain
$body = @{{ features = @(18.0, ...); top_n = 10 }} | ConvertTo-Json
Invoke-RestMethod -Uri "$BaseUrl/explain" -Method Post -Headers $Headers -Body $body

# Step 5: Model card
Invoke-RestMethod -Uri "$BaseUrl/model-card"
''', language="powershell")

# ============================================================
# Footer: API Reference
# ============================================================
st.divider()
st.markdown("### 📖 API Quick Reference")

df = pd.DataFrame([
    {"Endpoint": "`GET /health`", "Needs API Key?": "❌ No", "What It Does": "Checks if the server is running"},
    {"Endpoint": "`GET /feature-names`", "Needs API Key?": "✅ Yes", "What It Does": "Lists all features the model expects (in order)"},
    {"Endpoint": "`POST /score`", "Needs API Key?": "✅ Yes", "What It Does": "Takes 80 feature values → returns credit risk score"},
    {"Endpoint": "`POST /explain`", "Needs API Key?": "✅ Yes", "What It Does": "Takes 80 feature values → shows WHY that score (SHAP)"},
    {"Endpoint": "`GET /model-card`", "Needs API Key?": "❌ No", "What It Does": "Returns model documentation (training data, accuracy, limitations)"},
])
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("""
⚠️ **DISCLAIMER:** All alternative-data features are synthetic (computer-generated). No real personal data was used.
This model is for EDUCATIONAL PURPOSES only and is NOT designed for production credit decisions.
Built aligned to BSP's STARS AI Governance Framework.
""")