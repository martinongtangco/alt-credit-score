# Architecture Overview

> alt-credit-score — Explainable Alternative-Data Credit Scoring Engine

## System Context

```
┌──────────────────────────────────────────────────────────────────┐
│                         External Users                           │
│                    (Analysts, Auditors, Devs)                    │
└──────────────┬─────────────────────────────────┬─────────────────┘
               │                                 │
               ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │   Streamlit Demo    │           │     FastAPI Server  │
    │   (Port: 8501)      │           │   (Port: 8000)      │
    │                     │           │                     │
    │  /score             │           │  /score             │
    │  /explain (SHAP)    │           │  /explain (SHAP)    │
    │  /model-card        │           │  /model-card        │
    │  /feature-names     │           │  /feature-names     │
    │  /health            │           │  /health            │
    └─────────┬───────────┘           └─────────┬───────────┘
              │                                 │
              └─────────────┬───────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────┐
            │       Core ML Pipeline          │
            │                                 │
            │  ┌───────────────┐              │
            │  │ Data Pipeline │              │
            │  └───────┬───────┘              │
            │          │                      │
            │  ┌───────┴───────┐              │
            │  │   Models      │              │
            │  │  (Baseline /  │              │
            │  │  Challenger)  │              │
            │  └───────┬───────┘              │
            │          │                      │
            │  ┌───────┴───────┐              │
            │  │ Governance    │              │
            │  │ (SHAP, Fair-  │              │
            │  │  ness, Cards) │              │
            │  └───────────────┘              │
            └─────────────────────────────────┘
```

## Architectural Style

The project follows a **layered architecture** with three distinct layers:

| Layer | Responsibility | Modules |
|-------|---------------|---------|
| **Presentation** | User interaction, API interfaces | `api/`, `demo/` |
| **Business Logic** | ML training, explainability, fairness auditing | `src/` |
| **Data** | Data loading, preprocessing, synthetic generation | `data/` |

Communication flows **top-down**: Presentation → Business Logic → Data. Models are persisted as artifacts in `models/`.

## Component Inventory

| Component | Path | Role |
|-----------|------|------|
| **FastAPI Server** | `api/main.py` | REST API for scoring, explanation, and governance endpoints |
| **Streamlit Demo** | `demo/streamlit_app.py` | Interactive web UI for scoring and SHAP visualization |
| **Data Pipeline** | `src/data_pipeline.py` | Loads, preprocesses, and prepares feature matrices |
| **Synthetic Generator** | `data/synthetic_generator.py` | Generates synthetic alternative-data features |
| **Baseline Trainer** | `src/train_baseline.py` | Trains logistic regression + scorecard transformer |
| **Challenger Trainer** | `src/train_challenger.py` | Trains XGBoost gradient boosting model |
| **SHAP Explainer** | `src/explainability.py` | Per-prediction and global SHAP feature attribution |
| **Fairness Auditor** | `src/fairness_audit.py` | Demographic parity and equal opportunity auditing |
| **Model Card Generator** | `src/model_card.py` | Auto-generates Model Card markdown documentation |
| **Fairness Gate Tests** | `tests/test_fairness_gate.py` | CI tests that block fairness-regressed models |

## Data Flow

```
┌─────────────────┐     ┌───────────────────────┐     ┌─────────────────┐
│ UCI German      │────▶│  Data Pipeline        │────▶│ Feature Matrix  │
│ Credit Dataset  │     │  (load, encode, join) │     │ (X, y, names)   │
└─────────────────┘     └───────────────────────┘     └────────┬────────┘
                                                                │
                                          ┌─────────────────────┤
                                          │                     ▼
┌─────────────────┐     ┌───────────────────────┐     ┌─────────────────┐
│ Synthetic       │────▶│  Synthetic Generator  │────▶│ Alt-Data        │
│ Parameters      │     │  (telco, ewallet,     │     │ Features        │
│ (seed, noise)   │     │   utility, demo)      │     │ (13 columns)    │
└─────────────────┘     └───────────────────────┘     └─────────────────┘
```

### Training Flow

```
Feature Matrix (X, y)
         │
         ▼
  ┌──────────────┐
  │ Train/Test   │  (80/20 split, stratified)
  │ Split        │
  └──────┬───────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│Baseline│ │Challenger│
│LogReg  │ │  XGBoost │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌────────────────────────┐
│  SHAP Explainability   │  ← per-prediction + global attribution
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Fairness Audit        │  ← demographic parity, equal opportunity
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Model Card Generation │  ← auto-documentation
└────────────────────────┘
```

### Inference Flow

```
HTTP POST /score or /explain
         │
         ▼
┌────────────────┐
│ API Key Auth   │──fail──▶ 401 Unauthorized
└────────┬───────┘
         │ pass
         ▼
┌────────────────┐
│ Validate Input │──fail──▶ 400 Bad Request
│ (feature count)│
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Load Model     │  (lazy load at startup)
│ from disk      │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Predict        │  → default probability
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Risk Classify  │  → low_risk / borderline / high_risk
└────────┬───────┘
         │
         ▼
┌─────────────────────────┐
│ Human Review Gate       │  → borderline → human_review_required=true
│ (Sustainability pillar) │
└─────────────────────────┘
```

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core implementation |
| **ML Framework** | scikit-learn, XGBoost | Model training |
| **Explainability** | SHAP | Feature attribution |
| **Fairness** | Fairlearn | Bias auditing |
| **API** | FastAPI + Uvicorn | REST API server |
| **Demo UI** | Streamlit | Interactive web application |
| **Data** | pandas, numpy | Data manipulation |
| **Visualization** | matplotlib, seaborn | Charts and plots |
| **Testing** | pytest | Unit + fairness gate tests |
| **Serialization** | joblib | Model persistence |

## Directory Structure

```
alt-credit-score/
├── api/                          # Presentation Layer — REST API
│   └── main.py                   # FastAPI server with scoring endpoints
├── demo/                         # Presentation Layer — Web UI
│   └── streamlit_app.py          # Interactive Streamlit dashboard
├── src/                          # Business Logic Layer
│   ├── data_pipeline.py          # Data loading, preprocessing, feature engineering
│   ├── train_baseline.py         # Logistic regression + scorecard transformer
│   ├── train_challenger.py       # XGBoost gradient boosting
│   ├── explainability.py         # SHAP wrapper for both model types
│   ├── fairness_audit.py         # Fairness metrics across demographic segments
│   └── model_card.py             # Auto-generated model documentation
├── data/                         # Data Layer
│   └── synthetic_generator.py    # Synthetic alternative-data feature generator
├── models/                       # Model artifacts (git-ignored)
│   ├── baseline_logreg.pkl       # Serialized baseline model
│   ├── baseline_scorecard.json   # Scorecard configuration
│   ├── challenger_xgboost.pkl    # Serialized challenger model
│   └── challenger_feature_importance.json  # Feature importance ranking
├── notebooks/                    # Jupyter notebooks for exploration
│   └── 01_eda_and_baseline.ipynb
├── tests/                        # Test suite
│   └── test_fairness_gate.py     # CI fairness gate + accuracy tests
├── docs/                         # Architecture documentation
│   ├── ARCHITECTURE.md           # This file
│   ├── COMPONENTS.md             # Detailed component specifications
│   └── STARS_FRAMEWORK.md        # STARS governance framework mapping
├── MODEL_CARD.md                 # Auto-generated model documentation
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and quick start
```

## Design Decisions

### Two-Model Architecture (Baseline vs Challenger)

The project maintains two parallel models to demonstrate the accuracy-interpretability trade-off:

| Aspect | Baseline (Logistic Regression) | Challenger (XGBoost) |
|--------|-------------------------------|---------------------|
| **Interpretability** | Native via scorecard points | Requires SHAP |
| **Accuracy** | Good | Better |
| **Decision Boundary** | Linear | Non-linear (ensemble trees) |
| **Best For** | Regulatory explainability | Predictive performance |
| **Scorecard** | PDO-based point system | N/A (probabilities only) |

SHAP serves as the "bridge" that makes the challenger model interpretable, narrowing the gap between the two approaches.

### Synthetic Data Strategy

All alternative-data features are synthetically generated rather than sourced from real systems. This decision ensures:

1. **No PII exposure** — the repository can remain fully open-source
2. **Reproducibility** — fixed random seed produces identical features
3. **Controllable signal strength** — noise level and correlation parameters are explicit
4. **Demonstration readiness** — no data pipeline to real systems required

### Borderline Human Review Gate

Scores falling within a configurable confidence band (default: 0.40–0.60) are flagged with `human_review_required: true` rather than auto-decisioned. This implements the "Sustainability" pillar of the STARS framework by ensuring uncertain predictions receive human oversight.

### Configuration via Environment Variables

The API server is configured via environment variables, enabling different configurations across environments without code changes:

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_KEY` | `demo-key-change-in-production` | API authentication key |
| `MODEL_PATH` | `models/challenger_xgboost.pkl` | Challenger model path |
| `BASELINE_MODEL_PATH` | `models/baseline_logreg.pkl` | Baseline model path |
| `BORDERLINE_LOW` | `0.40` | Lower bound for human review zone |
| `BORDERLINE_HIGH` | `0.60` | Upper bound for human review zone |

## Cross-Cutting Concerns

### Governance (STARS Framework)

Every module is tagged with its STARS pillar alignment. See [STARS_FRAMEWORK.md](./STARS_FRAMEWORK.md) for the complete mapping.

### Fairness as a First-Class Concern

Fairness is not an afterthought — it is enforced as a CI gate. The `tests/test_fairness_gate.py` test suite will **fail the build** if:

- Demographic parity difference exceeds threshold (default: 0.15)
- Equal opportunity difference exceeds threshold (default: 0.15)
- Model ROC-AUC falls below minimum (0.70)
- Any feature name contains PII keywords

### Reproducibility

- Fixed random seeds (default: 42) throughout the pipeline
- Synthetic generator is deterministic given the same seed
- Model artifacts are serialized with joblib for exact reconstruction