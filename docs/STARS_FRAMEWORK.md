# STARS AI Governance Framework Mapping

> Complete mapping of BSP's STARS AI Governance Framework to project implementation.

The Bangko Senteng ng Pilipinas (BSP) STARS framework provides five pillars for responsible AI governance. This document details how each pillar is implemented in the alt-credit-score project.

---

## Framework Overview

```
┌────────────────────────────────────────────────────────────┐
│                  STARS AI Governance                       │
│                                                            │
│   S  Sustainability  ── Human-in-the-loop review gates     │
│   T  Transparency    ── Explainable predictions (SHAP)     │
│   A  Accountability  ── Model Cards + audit trails         │
│   R  Responsibility  ── Fairness auditing + CI gates       │
│   S  Security       ── No PII + synthetic data policy      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## S — Sustainability

> **Principle:** AI systems should incorporate human oversight for decisions that fall outside confident prediction ranges, preventing automated harm in uncertain situations.

### Implementation

**Borderline Human Review Gate**

Scores that fall within a configurable confidence band are automatically flagged for human review rather than being auto-decisioned.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | Confidence band check after prediction |
| **Default Low Threshold** | 0.40 (40% default probability) |
| **Default High Threshold** | 0.60 (60% default probability) |
| **Configurable Via** | `BORDERLINE_LOW` / `BORDERLINE_HIGH` environment variables |
| **Response Flag** | `human_review_required: true` in API response |

**Risk Classification Logic:**

```
default_prob >= BORDERLINE_HIGH  →  "high_risk"       (auto-reject)
default_prob <= BORDERLINE_LOW   →  "low_risk"        (auto-approve)
BORDERLINE_LOW < prob < HIGH     →  "borderline"      (HUMAN REVIEW)
```

**Code Locations:**

| File | Function / Class |
|------|-----------------|
| `api/main.py` | `/score` endpoint — lines 188-197 |
| `demo/streamlit_app.py` | Borderline warning — lines 140-142 |

**API Response Example:**
```json
{
  "score": 0.51,
  "risk_level": "borderline",
  "human_review_required": true,
  "model_used": "challenger",
  "timestamp": "2026-07-22T19:00:00"
}
```

---

## T — Transparency

> **Principle:** AI systems must provide clear, understandable explanations for their decisions, both at the global (model-level) and local (prediction-level) granularity.

### Implementation

**Per-Prediction SHAP Explanations**

Every prediction can be accompanied by a SHAP-based feature attribution breakdown showing which features drove the score up or down, and by how much.

| Aspect | Detail |
|--------|--------|
| **Method** | SHAP (SHapley Additive exPlanations) |
| **Tree Model Backend** | `shap.TreeExplainer` (exact, fast) |
| **Linear Model Backend** | `shap.LinearExplainer` (exact) |
| **API Endpoint** | `POST /explain` |
| **Streamlit Tab** | "SHAP Explanation" |

**Global Feature Importance**

Mean absolute SHAP values across the entire test set provide a global view of which features most influence model predictions.

**Code Locations:**

| File | Function / Class |
|------|-----------------|
| `src/explainability.py` | `SHAPExplainer` class |
| `src/explainability.py` | `SHAPExplainer.explain_single()` — per-prediction |
| `src/explainability.py` | `SHAPExplainer.global_importance()` — global |
| `api/main.py` | `/explain` endpoint |
| `demo/streamlit_app.py` | SHAP bar chart visualization |

**Explain Response Format:**
```json
{
  "base_value": 0.3000,
  "prediction_shap": 0.3421,
  "attributions": [
    {
      "feature": "synthetic_ewallet_tx_count",
      "shap_value": 0.0823,
      "feature_value": 15.0,
      "direction": "increases_risk"
    },
    {
      "feature": "synthetic_utility_bill_on_time_rate",
      "shap_value": -0.0651,
      "feature_value": 0.92,
      "direction": "decreases_risk"
    }
  ],
  "model_used": "challenger",
  "timestamp": "2026-07-22T19:00:00"
}
```

**Scorecard Interpretability (Baseline Model)**

The baseline logistic regression model includes a `ScorecardTransformer` that converts coefficients into a FICO-style points-based scorecard, providing native interpretability without SHAP.

| File | Class |
|------|-------|
| `src/train_baseline.py` | `ScorecardTransformer` |

---

## A — Accountability

> **Principle:** AI systems must maintain clear documentation of their design, training data, performance characteristics, and known limitations. This documentation must be regenerated on every model retrain.

### Implementation

**Auto-Generated Model Cards**

A Model Card is automatically generated after every model training run, documenting:

| Section | Content |
|---------|---------|
| Model Details | Type, version, framework, training date |
| Training Data | Source, sample sizes, feature counts, default rates |
| Synthetic Feature List | All `synthetic_` prefixed features documented |
| Performance Metrics | ROC-AUC, accuracy, precision, recall, F1 |
| Intended Use | Demonstration only, not for production |
| Known Limitations | 5 standard limitations explicitly listed |
| Ethical Considerations | PII policy, fairness, human review |
| STARS Alignment | All five pillars documented |
| Training Parameters | Hyperparameters used |
| Fairness Results | Per-attribute audit outcomes |

**Code Locations:**

| File | Function |
|------|----------|
| `src/model_card.py` | `generate_model_card()` |
| `src/model_card.py` | `save_model_card()` |
| `api/main.py` | `GET /model-card` endpoint |
| `demo/streamlit_app.py` | "Model Card" tab |

**Model Card Lifecycle:**

```
Model Retrained
       │
       ▼
generate_model_card()
       │
       ▼
save_model_card()  ──▶  MODEL_CARD.md
       │
       ▼
Served via GET /model-card
Rendered in Streamlit
```

**Artifact Versioning:**

Each model artifact includes a `trained_at` ISO timestamp, enabling version tracking:

| Artifact | Location | Content |
|----------|----------|---------|
| Baseline model | `models/baseline_logreg.pkl` | Serialized sklearn pipeline |
| Baseline scorecard | `models/baseline_scorecard.json` | PDO weights per feature |
| Challenger model | `models/challenger_xgboost.pkl` | Serialized XGBoost pipeline |
| Feature importances | `models/challenger_feature_importance.json` | Ranked feature list |

---

## R — Responsibility (Social Fairness)

> **Principle:** AI systems must be audited for fairness across demographic segments. Models that exhibit unacceptable bias must be prevented from deployment.

### Implementation

**Fairness Audit Module**

Computes fairness metrics across three synthetic demographic segments:

| Sensitive Attribute | Groups | Purpose |
|--------------------|--------|---------|
| `synthetic_income_bracket` | low, medium, high | Income-based fairness |
| `synthetic_region` | region_A, region_B, region_C | Geographic fairness |
| `synthetic_age_band` | young, middle, senior | Age-based fairness |

**Metrics Computed:**

| Metric | Formula | Gate Threshold |
|--------|---------|---------------|
| Demographic Parity Difference | max(pred_rates) - min(pred_rates) | ≤ 0.10 (audit), ≤ 0.15 (CI) |
| Equal Opportunity Difference | max(TPRs) - min(TPRs) | ≤ 0.10 (audit), ≤ 0.15 (CI) |
| ROC-AUC Disparity | max(AUCs) - min(AUCs) | Informational |

**CI Gate (Build-Failing Enforcement):**

The `tests/test_fairness_gate.py` test suite is designed to **fail the CI build** if fairness thresholds are exceeded, preventing biased models from being merged.

**Code Locations:**

| File | Function / Class |
|------|-----------------|
| `src/fairness_audit.py` | `FairnessAuditor` class |
| `src/fairness_audit.py` | `FairnessAuditor.audit()` |
| `src/fairness_audit.py` | `run_fairness_audit()` |
| `tests/test_fairness_gate.py` | `test_fairness_gate_passes()` |

**Audit Output Structure:**
```json
{
  "overall_pass": true,
  "sensitive_attributes": {
    "synthetic_income_bracket": {
      "status": "passed",
      "demographic_parity_difference": 0.05,
      "equal_opportunity_difference": 0.03,
      "passes": {
        "demographic_parity": true,
        "equal_opportunity": true
      }
    }
  }
}
```

**Fairlearn Integration:**

When the `fairlearn` package is available, an additional `MetricFrame` report is generated via `FairnessAuditor.get_fairlearn_report()`. This provides an independent fairness assessment using Fairlearn's established methodology.

---

## S — Security

> **Principle:** AI systems must never process real personally identifiable information (PII). Data handling policies must be explicitly documented and enforced.

### Implementation

**Synthetic-Only Data Policy**

All alternative-data features are synthetically generated. No real telco, e-wallet, or personal data enters the repository or models.

| Guarantee | Mechanism |
|-----------|-----------|
| No real PII | Synthetic generator creates all alt-data features |
| Reproducible | Fixed random seed (42) ensures deterministic output |
| Clearly labeled | All synthetic features use `synthetic_` prefix |
| Documented | Data governance notice in README.md |

**PII Detection in CI:**

The test suite includes an automated check that scans all feature names for PII-related keywords. The build fails if any are found.

| PII Keywords Checked | — |
|---------------------|---|
| `ssn` | `social_security` |
| `national_id` | `phone_number` |
| `email` | `address` |
| `name` | `bank_account` |

**API Authentication:**

All scoring and explanation endpoints require API key authentication via the `X-API-Key` header.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | API key header (`X-API-Key`) |
| **Default Key** | `demo-key-change-in-production` |
| **Configurable Via** | `API_KEY` environment variable |
| **Unauthenticated Endpoints** | `/health`, `/model-card` (public) |

**Code Locations:**

| File | Function / Mechanism |
|------|---------------------|
| `data/synthetic_generator.py` | `SyntheticAltDataGenerator` — all features synthetic |
| `tests/test_fairness_gate.py` | `test_no_real_pii_in_features()` — PII keyword scan |
| `api/main.py` | `get_api_key()` — API key authentication |
| `README.md` | Data Governance Notice section |

---

## Pillar-to-Component Matrix

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Pillar              │ api/    │ demo/   │ src/        │ data/   │ tests/ │
├─────────────────────┼─────────┼─────────┼─────────────┼─────────┼────────┤
│ Sustainability      │  /score │  UI     │  —          │  —      │  —     │
│ Transparency        │ /explain│  UI     │ explain.    │  —      │  —     │
│ Accountability      │/model-c │  UI     │ model_card  │  —      │  —     │
│ Responsibility      │  —      │  —      │ fairness    │  synth  │ fairness│
│ Security            │  auth   │  —      │  —          │ synth   │ pii    │
└──────────────────────────────────────────────────────────────────────────┘

Legend:
  /score      = /score endpoint borderline gate
  /explain    = /explain endpoint SHAP attributions
  /model-c    = /model-card endpoint
  UI          = Streamlit visualization
  explain.    = explainability.py
  model_card  = model_card.py
  fairness    = fairness_audit.py
  synth       = synthetic_generator.py (no real data)
  fairness    = test_fairness_gate.py
  auth        = API key authentication
  pii         = PII keyword detection test
```

---

## Compliance Checklist

Use this checklist when evaluating the project against STARS requirements:

- [ ] **Sustainability:** Borderline scores are flagged with `human_review_required: true`
- [ ] **Sustainability:** Borderline thresholds are configurable via environment variables
- [ ] **Transparency:** Per-prediction SHAP explanations available via `/explain` endpoint
- [ ] **Transparency:** Global feature importance report generated per model version
- [ ] **Transparency:** Baseline model provides native scorecard interpretability
- [ ] **Accountability:** Model Card auto-generated on every retrain
- [ ] **Accountability:** Model Card documents training data, metrics, limitations
- [ ] **Accountability:** Model artifacts include training timestamp
- [ ] **Responsibility:** Fairness audit computes demographic parity difference
- [ ] **Responsibility:** Fairness audit computes equal opportunity difference
- [ ] **Responsibility:** CI build fails when fairness thresholds are exceeded
- [ ] **Responsibility:** Fairlearn MetricFrame report available when fairlearn is installed
- [ ] **Security:** No real PII in any feature, dataset, or model artifact
- [ ] **Security:** All synthetic features clearly prefixed with `synthetic_`
- [ ] **Security:** API endpoints require authentication
- [ ] **Security:** CI test scans feature names for PII keywords