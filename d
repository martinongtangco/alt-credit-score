# Alt-Credit-Score API - Complete Consumer Guide

> **Stop reading code. Start using the API.** This guide shows you exactly how to consume every endpoint, with copy-paste examples in Python, curl, and PowerShell.

---

## Quick Start (5 minutes)

### Step 1: Start the API Server
```bash
uvicorn api.main:app --reload --port 8000
```

### Step 2: Run a Demo
```bash
# Python (recommended - walks through all endpoints):
python demos/demo_python_client.py

# Or PowerShell (Windows):
.\demos\demo_powershell.ps1

# Or curl (Linux/Mac/Git Bash):
bash demos/demo_curl.sh
```

---

## API Endpoints

| Endpoint | Method | Auth | What It Does |
|----------|--------|------|-------------|
| `/health` | GET | No | Is the server running? |
| `/feature-names` | GET | Yes | What features does the model expect? |
| `/score` | POST | Yes | Score an applicant's credit risk |
| `/explain` | POST | Yes | Why that score? (SHAP explanation) |
| `/model-card` | GET | No | Model documentation |

**Auth:** Add header `X-API-Key: demo-key-change-in-production`

---

## The 3-Step Scoring Workflow

```
1. GET /feature-names  → Learn what features the model needs
         ↓
2. POST /score         → Send features, get risk assessment
         ↓
3. POST /explain       → Get SHAP breakdown (why this score?)
```

---

## Step-by-Step: Each Endpoint

### 1. Health Check (GET /health)

**curl:**
```bash
curl http://localhost:8000/health
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

**Response:**
```json
{ "status": "healthy", "model_loaded": true, "n_features": 70 }
```

---

### 2. Get Feature Names (GET /feature-names)

**curl:**
```bash
curl -H "X-API-Key: demo-key-change-in-production" http://localhost:8000/feature-names
```

**Response:**
```json
{ "feature_names": ["synthetic_telco_topup_freq", "duration_months", ...], "n_features": 70 }
```

---

### 3. Score an Applicant (POST /score)

**curl:**
```bash
curl -X POST http://localhost:8000/score \
  -H "X-API-Key: demo-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"features": [18.0, 0.0, ...], "model": "challenger"}'
```

**PowerShell:**
```powershell
$headers = @{ "X-API-Key" = "demo-key-change-in-production"; "Content-Type" = "application/json" }
$body = @{ features = @(18.0, 0.0, ...); model = "challenger" } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://localhost:8000/score" -Method Post -Headers $headers -Body $body
```

**Response:**
```json
{
    "score": 0.1523,
    "risk_level": "low_risk",
    "human_review_required": false,
    "model_used": "challenger",
    "timestamp": "2026-07-23T08:05:00"
}
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | `float[]` | Yes | 70 feature values in model order |
| `model` | `string` | No | `"challenger"` (default) or `"baseline"` |

---

### 4. Get SHAP Explanation (POST /explain)

**curl:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "X-API-Key: demo-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"features": [18.0, 0.0, ...], "top_n": 10}'
```

**Response:**
```json
{
    "base_value": 0.3021,
    "prediction_shap": 0.1850,
    "attributions": [
        { "feature": "synthetic_telco_topup_freq", "shap_value": -0.0523, "feature_value": 18.0, "direction": "decreased" }
    ],
    "model_used": "challenger",
    "timestamp": "2026-07-23T08:05:00"
}
```

**Reading SHAP values:**
- **Negative SHAP** = feature reduced the risk score ✅
- **Positive SHAP** = feature increased the risk score ⚠️

---

### 5. Get Model Card (GET /model-card)

```bash
curl http://localhost:8000/model-card
```

Returns the model's documentation in Markdown format.

---

## Building Feature Vectors

The model expects 70 values. Key synthetic features and their indices:

| Index | Feature | Good Value | Bad Value |
|-------|---------|-----------|-----------|
| 0 | `synthetic_telco_topup_freq` | 15+ | < 5 |
| 51 | `synthetic_telco_avg_data_usage_gb` | 10+ GB | < 2 GB |
| 52 | `synthetic_telco_sim_tenure_months` | 36+ months | < 6 months |
| 53 | `synthetic_telco_avg_call_minutes` | 200+ | < 50 |
| 54 | `synthetic_ewallet_tx_count` | 15+ | 0 |
| 55 | `synthetic_ewallet_avg_tx_amount_usd` | $30+ | $0 |
| 56 | `synthetic_utility_bill_on_time_rate` | 0.90+ | < 0.40 |
| 57 | `synthetic_rent_payment_regularity` | 0.85+ | < 0.30 |
| 58 | `synthetic_social_credit_inquiries` | < 3 | 8+ |
| 59 | `synthetic_income_volatility` | < 0.30 | > 0.80 |

**Example:**
```python
vector = [0.0] * 70
vector[0]  = 18.0    # high telco topup - reliable
vector[51] = 15.0    # high data usage - active
vector[52] = 72.0    # long SIM tenure - stable
vector[56] = 0.95    # bills on time - responsible
vector[58] = 1.0     # few inquiries - not desperate
vector[59] = 0.15    # stable income
```

---

## Understanding Responses

### Risk Thresholds

| Score Range | Risk Level | Action |
|-------------|-----------|--------|
| 0.00 - 0.40 | `low_risk` | Auto-approve |
| 0.40 - 0.60 | `borderline` | Human review required |
| 0.60 - 1.00 | `high_risk` | Auto-decline |

### Credit Score Mapping
```python
credit_score = round((1 - default_probability) * 850, 1)
```

---

## Python SDK

```python
from api_client import CreditScoreClient

client = CreditScoreClient(base_url="http://localhost:8000", api_key="demo-key-change-in-production")

# Check health
print(client.health())

# Get feature names
names = client.get_feature_names()

# Score an applicant
result = client.score(my_features, model="challenger")
print(result)

# Get explanation
explanation = client.explain(my_features, top_n=10)
explanation.print_attribution_table()

# Get model card
print(client.get_model_card()[:500])
```

---

## Integration Examples

### Node.js
```javascript
const response = await fetch('http://localhost:8000/score', {
    method: 'POST',
    headers: { 'X-API-Key': 'demo-key-change-in-production', 'Content-Type': 'application/json' },
    body: JSON.stringify({ features: [18.0, 0.0, ...], model: 'challenger' })
});
const result = await response.json();
console.log(result.risk_level); // "low_risk"
```

### Error Handling (Python)
```python
from api_client.models import APIError

try:
    result = client.score(features)
except APIError as e:
    print(f"API Error {e.status_code}: {e}")
```

---

## Files

| File | Purpose |
|------|---------|
| `demos/README.md` | This guide |
| `demos/demo_python_client.py` | Full Python walkthrough (3 applicants) |
| `demos/demo_curl.sh` | Bash/curl demo |
| `demos/demo_powershell.ps1` | Windows PowerShell demo |
| `api_client/client.py` | Python SDK |
| `api_client/models.py` | Response data models |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Start server: `uvicorn api.main:app --reload` |
| Model not found | Train models: `python -m src.train_challenger` |
| Wrong feature count | Check with `GET /feature-names` |
| Invalid API key | Verify `X-API-Key` header |