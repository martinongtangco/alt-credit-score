# alt-credit-score

> **Explainable Alternative-Data Credit Scoring Engine** — built around BSP's STARS AI Governance Framework

[![CI](https://github.com/martinongtangco/alt-credit-score/actions/workflows/ci.yml/badge.svg)](https://github.com/martinongtangco/alt-credit-score/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## ⚠️ Data Governance Notice

> **All alternative-data features in this project are synthetically generated for demonstration purposes. No real telco, e-wallet, or personal data was used or accessed.**

This project uses the [UCI German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (public, anonymized) augmented with a synthetic alternative-data layer that simulates the *type* of signals used by alternative credit bureaus — without using any real user data.

---

## Overview

A working, open-source credit scoring engine that demonstrates:

| Capability | How |
|---|---|
| **Alternative-Data Scoring** | Synthetic telco/e-wallet/utility features with controlled statistical relationships to credit risk |
| **Explainability** | Per-prediction SHAP feature attributions via API endpoint |
| **Fairness Auditing** | Demographic parity & equal opportunity checks across synthetic demographic segments |
| **Model Cards** | Auto-generated documentation of training data, metrics, limitations |
| **Human-in-the-Loop** | Borderline scores automatically flagged for human review |
| **Two-Model Comparison** | Logistic regression (interpretable) vs XGBoost (accurate) — with SHAP bridging the gap |

## STARS AI Governance Framework Mapping

This project is explicitly architected around the [BSP STARS AI Governance Framework](https://www.bsp.gov.ph/):

| STARS Pillar | What This Repo Does |
|---|---|
| **S**ustainability | Borderline scores (configurable confidence band) route to a `"human_review_required": true` flag instead of auto-decisioning |
| **T**ransparency | Per-prediction SHAP explanations exposed via `/explain` API endpoint; global feature-importance report generated per model version |
| **A**ccountability | Auto-generated Model Card (`MODEL_CARD.md`) documenting training data provenance, performance metrics, known limitations, and intended use — regenerated on every model retrain |
| **R**esponsibility (Social Fairness) | Fairness audit module computing demographic parity and equal-opportunity difference across synthetic demographic segments (income bracket, region, age band); CI build fails if fairness metrics drift past threshold |
| **S**ecurity | No real PII ever enters the repo or model; synthetic-data policy enforced and documented; API requires authentication |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/martinongtangco/alt-credit-score.git
cd alt-credit-score
pip install -r requirements.txt

# 2. Train models
python -m src.train_baseline
python -m src.train_challenger

# 3. Run explainability analysis
python -m src.explainability

# 4. Generate model card
python -m src.model_card

# 5. Run tests (including fairness gate)
pytest tests/ -v

# 6. Start API server
uvicorn api.main:app --reload --port 8000

# 7. Open Streamlit demo
streamlit run demo/streamlit_app.py
```

## Project Structure

```
alt-credit-score/
├── data/
│   ├── raw/                      # downloaded public dataset (cached)
│   └── synthetic_generator.py    # synthetic alt-data feature generator
├── src/
│   ├── data_pipeline.py          # load data, join base + synthetic features
│   ├── train_baseline.py         # logistic regression / scorecard model
│   ├── train_challenger.py       # XGBoost gradient boosting model
│   ├── explainability.py         # SHAP wrapper for both models
│   ├── fairness_audit.py         # fairness metrics + CI gate
│   └── model_card.py             # auto-generates MODEL_CARD.md
├── api/
│   └── main.py                   # FastAPI: /score, /explain, /model-card
├── demo/
│   └── streamlit_app.py          # interactive scoring + SHAP UI
├── models/                       # saved model artifacts (git-ignored)
├── tests/
│   └── test_fairness_gate.py     # CI fairness gate test
├── .github/workflows/ci.yml      # lint, test, fairness-gate on PR
├── MODEL_CARD.md                 # auto-generated example
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Health check |
| `/score` | POST | Yes | Get credit risk score |
| `/explain` | POST | Yes | Get SHAP explanation |
| `/model-card` | GET | No | Get Model Card (markdown) |
| `/feature-names` | GET | Yes | List expected features |

### Example: Score Request

```bash
curl -X POST http://localhost:8000/score \
  -H "X-API-Key: demo-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, ...], "model": "challenger"}'
```

### Example: Explain Request

```bash
curl -X POST http://localhost:8000/explain \
  -H "X-API-Key: demo-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, ...], "top_n": 10}'
```

## Synthetic Feature Documentation

All synthetic features are prefixed with `synthetic_` and documented below:

| Feature | Description | Risk Direction |
|---|---|---|
| `synthetic_telco_topup_freq` | Monthly mobile top-up count | Higher = lower risk |
| `synthetic_telco_avg_data_usage_gb` | Monthly data usage (GB) | Higher = lower risk |
| `synthetic_telco_sim_tenure_months` | SIM tenure (months) | Higher = lower risk |
| `synthetic_telco_avg_call_minutes` | Monthly call minutes | Higher = lower risk |
| `synthetic_ewallet_tx_count` | Monthly e-wallet transactions | Higher = lower risk |
| `synthetic_ewallet_avg_tx_amount_usd` | Avg e-wallet tx amount | Higher = lower risk |
| `synthetic_utility_bill_on_time_rate` | On-time utility bill rate | Higher = lower risk |
| `synthetic_rent_payment_regularity` | On-time rent payment rate | Higher = lower risk |
| `synthetic_social_credit_inquiries` | Informal credit inquiry count | Higher = higher risk |
| `synthetic_income_volatility` | Income CV (std/mean) | Higher = higher risk |

## Model Comparison

| Metric | Baseline (Logistic Regression) | Challenger (XGBoost) |
|---|---|---|
| Interpretability | Native (scorecard) | Requires SHAP |
| Accuracy | Good | Better |
| Explainability | Coefficients + scorecard | SHAP values |
| Best for | Regulatory explainability | Predictive performance |

## License

MIT — see [LICENSE](LICENSE) for details.

## Disclaimer

This is a **demonstration project** trained on public data with synthetically
generated alternative-data features. It is **NOT** designed for production
credit decisions and has **NOT** been validated for regulatory compliance
in any jurisdiction.
