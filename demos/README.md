# Alt-Credit-Score API - Complete Consumer Guide

> **Stop reading code. Start using the API.** Step-by-step instructions with copy-paste examples in Python, curl, and PowerShell.

---

## Quick Start

### Step 1: Start the API Server
```bash
uvicorn api.main:app --reload --port 8000
```

### Step 2: Run a Demo
```bash
# Python (recommended):
python demos/demo_python_client.py

# PowerShell (Windows):
.\demos\demo_powershell.ps1

# curl (Linux/Mac/Git Bash):
bash demos/demo_curl.sh
```

---

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/feature-names` | GET | Yes | List expected features |
| `/score` | POST | Yes | Score an applicant |
| `/explain` | POST | Yes | SHAP explanation |
| `/model-card` | GET | No | Model documentation |

**Auth header:** `X-API-Key: demo-key-change-in-production`

---

## Each Endpoint with Examples

### 1. Health Check
```bash
curl http://localhost:8000/health
```
Response: `{ "status": "healthy", "model_loaded": true, "n_features": 70 }`

### 2. Feature Names
```bash
curl -H "X-API-Key: demo-key-change-in-production" http://localhost:8000/feature-names
```

### 3. Score an Applicant
```bash
curl -X POST http://localhost:8000/score \
  -H "X-API-Key: demo-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"features": [18.0, 0.0, ...], "model": "challenger"}'
```
Response:
```json
{
    "score": 0.1523,
    "risk_level": "low_risk",
    "human_review_required": false,
    "model_used": "challenger"
}
```

### 4. SHAP Explanation
```bash
curl -X POST http://localhost:8000/explain \
  -H "X-API-Key: demo-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"features": [18.0, 0.0, ...], "top_n": 10}'
```

### 5. Model Card
```bash
curl http://localhost:8000/model-card
```

---

## Building Feature Vectors

The model expects 70 values. Key synthetic features:

| Index | Feature | Good | Bad |
|-------|---------|------|-----|
| 0 | `synthetic_telco_topup_freq` | 15+ | < 5 |
| 51 | `synthetic_telco_avg_data_usage_gb` | 10+ | < 2 |
| 52 | `synthetic_telco_sim_tenure_months` | 36+ | < 6 |
| 53 | `synthetic_telco_avg_call_minutes` | 200+ | < 50 |
| 54 | `synthetic_ewallet_tx_count` | 15+ | 0 |
| 55 | `synthetic_ewallet_avg_tx_amount_usd` | 30+ | 0 |
| 56 | `synthetic_utility_bill_on_time_rate` | 0.90+ | < 0.40 |
| 57 | `synthetic_rent_payment_regularity` | 0.85+ | < 0.30 |
| 58 | `synthetic_social_credit_inquiries` | < 3 | 8+ |
| 59 | `synthetic_income_volatility` | < 0.30 | > 0.80 |

---

## Risk Thresholds

| Score | Risk Level | Action |
|-------|-----------|--------|
| 0.00 - 0.40 | `low_risk` | Auto-approve |
| 0.40 - 0.60 | `borderline` | Human review |
| 0.60 - 1.00 | `high_risk` | Auto-decline |

---

## Python SDK

```python
from api_client import CreditScoreClient

client = CreditScoreClient(api_key="demo-key-change-in-production")

# Health check
print(client.health())

# Get feature names
names = client.get_feature_names()

# Score
result = client.score(features, model="challenger")
print(result)

# Explain
explanation = client.explain(features, top_n=10)
explanation.print_attribution_table()
```

---

## Files

| File | Purpose |
|------|---------|
| `demo_python_client.py` | Full walkthrough (3 applicants) |
| `demo_curl.sh` | Bash/curl demo |
| `demo_powershell.ps1` | Windows PowerShell demo |
| `../api_client/` | Python SDK |