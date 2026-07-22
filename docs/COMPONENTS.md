# Component Specifications

> Detailed specifications for each component in the alt-credit-score system.

---

## 1. FastAPI Server (`api/main.py`)

### Purpose

REST API server that exposes credit scoring, explainability, and governance endpoints. Serves as the primary programmatic interface for production integration.

### Dependencies

- **Internal:** `src.explainability.SHAPExplainer`
- **External:** fastapi, uvicorn, joblib, numpy, pydantic

### Endpoints

| Endpoint | Method | Auth | Request | Response |
|----------|--------|------|---------|----------|
| `/health` | GET | None | — | Server health status |
| `/score` | POST | API Key | `ScoreRequest` | `ScoreResponse` |
| `/explain` | POST | API Key | `ExplainRequest` | `ExplainResponse` |
| `/model-card` | GET | None | — | Model Card markdown |
| `/feature-names` | GET | API Key | — | Feature name list |

### Request/Response Schemas

#### ScoreRequest
```json
{
  "features": [1.0, 2.0, ...],    // float[], must match feature count
  "model": "challenger"           // "challenger" | "baseline"
}
```

#### ScoreResponse
```json
{
  "score": 0.3421,                       // default probability [0, 1]
  "risk_level": "low_risk",              // "low_risk" | "borderline" | "high_risk"
  "human_review_required": false,        // true when score is in borderline zone
  "model_used": "challenger",
  "timestamp": "2026-07-22T19:00:00"
}
```

#### ExplainRequest
```json
{
  "features": [1.0, 2.0, ...],    // float[], must match feature count
  "top_n": 10                     // number of top SHAP attributions to return
}
```

#### ExplainResponse
```json
{
  "base_value": 0.3,
  "prediction_shap": 0.3421,
  "attributions": [
    {
      "feature": "synthetic_ewallet_tx_count",
      "shap_value": 0.0823,
      "feature_value": 15.0,
      "direction": "increases_risk"
    }
  ],
  "model_used": "challenger",
  "timestamp": "2026-07-22T19:00:00"
}
```

### Lifecycle

1. **Startup:** `load_models()` is called on `/startup` event
2. Model files are loaded via joblib into global state
3. Feature names are read from `challenger_feature_importance.json`
4. Model card content is cached from `MODEL_CARD.md`
5. SHAP explainer is created lazily on first `/explain` request

### Error Responses

| Status Code | Condition |
|-------------|-----------|
| 400 | Feature count mismatch |
| 401 | Invalid or missing API key |
| 503 | Model not loaded (startup failure) |

---

## 2. Streamlit Demo (`demo/streamlit_app.py`)

### Purpose

Interactive web application for demonstrating credit scoring with visual SHAP explanations. Targeted at non-technical stakeholders and regulatory auditors.

### Dependencies

- **Internal:** `src.explainability.SHAPExplainer`
- **External:** streamlit, joblib, numpy, pandas, matplotlib

### Interface

Three-tab layout:

| Tab | Function |
|-----|----------|
| **Score a Sample** | Generate random samples, select model, view risk classification |
| **SHAP Explanation** | Visualize per-feature SHAP attributions as horizontal bar chart |
| **Model Card** | Render auto-generated Model Card markdown |

### Sidebar Controls

- **Model selector:** Challenger (XGBoost) vs Baseline (Logistic Regression)
- **Borderline thresholds:** Adjustable sliders for borderline low/high bounds

### Caching

Uses `@st.cache_resource` for model loading to avoid reloading on every interaction.

---

## 3. Data Pipeline (`src/data_pipeline.py`)

### Purpose

Single entry point for loading, preprocessing, and preparing the feature matrix used by all models.

### Dependencies

- **Internal:** `data.synthetic_generator.SyntheticAltDataGenerator` (optional)
- **External:** pandas, numpy, scikit-learn

### Key Functions

#### `load_german_credit_data(data_path) -> DataFrame`

Loads the UCI German Credit Dataset. Downloads and caches if not present locally. Recodes target so that `default=1` means risky borrower.

**Input:** Optional path to local CSV file
**Output:** DataFrame with 20 standardized columns + `default` target

#### `prepare_features(df, include_synthetic, seed) -> (X, y, feature_names, metadata)`

Prepares the final feature matrix:

1. One-hot encodes 13 categorical columns from the base dataset
2. Retains 7 numeric columns directly
3. Optionally generates and joins synthetic alternative-data features
4. Encodes remaining string columns (synthetic demographics)
5. Converts all columns to numeric, drops rows with NaN

**Output:** NumPy feature matrix, target array, feature name list, feature metadata dict

#### `load_and_prepare(include_synthetic, test_size, seed, data_path) -> dict`

Full pipeline: load → prepare → train/test split.

**Output keys:** `X_train`, `X_test`, `y_train`, `y_test`, `feature_names`, `feature_metadata`, `n_features`, `n_train`, `n_test`, `default_rate_train`, `default_rate_test`

### Feature Categories

| Category | Count (approx) | Examples |
|----------|----------------|----------|
| **Base numeric** | 7 | `duration_months`, `amount`, `age_years` |
| **Base categorical (one-hot)** | ~40 | `status_checking_account_0`, `purpose_1` |
| **Synthetic numeric** | 10 | `synthetic_telco_topup_freq`, `synthetic_ewallet_tx_count` |
| **Synthetic demographic** | 3 (one-hot → ~7) | `synthetic_income_bracket_low` |

---

## 4. Synthetic Data Generator (`data/synthetic_generator.py`)

### Purpose

Generates realistic alternative-data features with controlled statistical relationships to credit risk. All features are synthetic — no real personal data is used.

### Class: `SyntheticAltDataGenerator`

#### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `noise_level` | 0.15 | Controls signal-to-noise ratio |

#### `generate(n_samples, default_rates) -> DataFrame`

Generates synthetic features for `n_samples` individuals. Uses `default_rates` as a latent risk signal to create correlated features.

**Feature Generation Strategy:**

Each feature's target mean is a linear function of the default rate, with Gaussian noise added:

```
target_mean = base + (correlation_direction * default_rates * scale)
values = target_mean + Gaussian(0, noise_level * scale)
values = clip(values, min, max)
```

#### Generated Features

**Telco Features (4):**

| Feature | Correlation with Default | Mechanism |
|---------|-------------------------|-----------|
| `synthetic_telco_topup_freq` | -0.35 | Responsible users top-up more regularly |
| `synthetic_telco_avg_data_usage_gb` | -0.25 | Higher usage = more financially engaged |
| `synthetic_telco_sim_tenure_months` | -0.30 | Longer tenure = more stable |
| `synthetic_telco_avg_call_minutes` | -0.15 | Moderate signal of stability |

**E-Wallet Features (2):**

| Feature | Correlation with Default | Mechanism |
|---------|-------------------------|-----------|
| `synthetic_ewallet_tx_count` | -0.40 | Strong signal of financial activity |
| `synthetic_ewallet_avg_tx_amount_usd` | -0.20 | Average transaction size |

**Utility Payment Features (2):**

| Feature | Correlation with Default | Mechanism |
|---------|-------------------------|-----------|
| `synthetic_utility_bill_on_time_rate` | -0.45 | Very strong signal |
| `synthetic_rent_payment_regularity` | -0.38 | Payment discipline proxy |

**Risk Indicators (2):**

| Feature | Correlation with Default | Mechanism |
|---------|-------------------------|-----------|
| `synthetic_social_credit_inquiries` | +0.28 | More inquiries = higher risk |
| `synthetic_income_volatility` | +0.32 | Income instability = higher risk |

**Synthetic Demographics (3, for fairness auditing):**

| Feature | Type | Purpose |
|---------|------|---------|
| `synthetic_income_bracket` | categorical (low/medium/high) | Fairness testing proxy |
| `synthetic_region` | categorical (region_A/B/C) | Fairness testing proxy |
| `synthetic_age_band` | categorical (young/middle/senior) | Fairness testing proxy |

Demographic features are **independent** of default risk (random assignment) but may correlate with other features, enabling realistic fairness auditing.

#### `get_feature_metadata() -> dict`

Returns metadata for all generated features (description, unit, risk direction, target correlation).

#### `generate_documentation_df() -> DataFrame`

Returns a summary DataFrame suitable for documentation.

---

## 5. Baseline Trainer (`src/train_baseline.py`)

### Purpose

Trains a logistic regression model and transforms coefficients into a points-based scorecard (FICO-style).

### Dependencies

- **External:** scikit-learn, numpy, pandas, joblib

### Class: `ScorecardTransformer`

Transforms logistic regression coefficients into a human-readable scorecard using the PDO (Points-to-Double-Odds) method:

```
score = base_score - (pdo / log(2)) * log(odds / odds_at_base)
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_score` | 650 | Score at reference odds |
| `pdo` | 50 | Points for each odds doubling |
| `odds_at_base` | 2.0 | Odds ratio at base_score |

#### Methods

| Method | Purpose |
|--------|---------|
| `fit(model, feature_names, scaler)` | Extract coefficients, compute PDO weights |
| `transform(X)` | Convert feature matrix to scorecard points |
| `get_scorecard_table()` | Return DataFrame with coefficient and weight per feature |
| `to_dict()` | Export scorecard configuration for JSON serialization |

### Function: `train_baseline(X_train, X_test, y_train, y_test, feature_names, max_iter) -> dict`

**Pipeline:**
1. `StandardScaler` → `LogisticRegression(C=1.0, class_weight="balanced")`
2. Evaluate: ROC-AUC, accuracy, precision, recall, F1, confusion matrix
3. Fit `ScorecardTransformer` to extract human-readable scorecard
4. Save model (`baseline_logreg.pkl`) and scorecard (`baseline_scorecard.json`)

**Output dict keys:** `model`, `scorecard`, `metrics`, `model_type`, `model_path`, `scorecard_path`, `feature_names`, `n_features`, `trained_at`

---

## 6. Challenger Trainer (`src/train_challenger.py`)

### Purpose

Trains an XGBoost gradient boosting model on the full feature set.

### Dependencies

- **External:** scikit-learn, xgboost, numpy, pandas, joblib

### Function: `train_challenger(X_train, X_test, y_train, y_test, feature_names, n_estimators, max_depth, learning_rate) -> dict`

**Pipeline:**
1. `StandardScaler` → `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)`
2. Evaluate: ROC-AUC, accuracy, precision, recall, F1, confusion matrix
3. Extract native XGBoost feature importances
4. Save model (`challenger_xgboost.pkl`) and importances (`challenger_feature_importance.json`)

**Default Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `eval_metric` | logloss |
| `random_state` | 42 |

**Output dict keys:** `model`, `metrics`, `feature_importances`, `model_type`, `model_path`, `feature_names`, `n_features`, `n_estimators`, `max_depth`, `learning_rate`, `trained_at`

---

## 7. SHAP Explainer (`src/explainability.py`)

### Purpose

Provides per-prediction and global feature attribution using SHAP (SHapley Additive exPlanations). Supports both tree-based and linear models.

### Dependencies

- **External:** shap, scikit-learn, numpy, pandas, joblib

### Class: `SHAPExplainer`

#### Constructor

```python
SHAPExplainer(model, feature_names, model_type="tree")
```

| Parameter | Options | Description |
|-----------|---------|-------------|
| `model` | — | Trained sklearn pipeline or raw model |
| `feature_names` | — | List of feature name strings |
| `model_type` | `"tree"`, `"linear"`, other | Determines which SHAP backend to use |

#### SHAP Backend Selection

| Model Type | SHAP Explainer | Why |
|------------|---------------|-----|
| `tree` | `shap.TreeExplainer` | Exact, fast for tree ensembles |
| `linear` | `shap.LinearExplainer` | Exact for linear models |
| other | `shap.KernelExplainer` | Model-agnostic fallback (slow) |

#### Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `fit(X_background, sample_size)` | Background data | self | Prepare explainer with reference distribution |
| `explain(X)` | Feature matrix | SHAP values | Compute attributions for all samples |
| `explain_single(x, return_dict)` | Single sample | Dict with attributions | Per-prediction explanation |
| `global_importance(X)` | Feature matrix | DataFrame | Mean \|SHAP\| across all samples |
| `summary_plot_data(X)` | Feature matrix | Dict | Raw data for SHAP summary plots |

#### `explain_single` Output Format

```json
{
  "base_value": 0.3,
  "prediction_shap": 0.3421,
  "attributions": [
    {
      "feature": "synthetic_ewallet_tx_count",
      "shap_value": 0.0823,
      "feature_value": 15.0,
      "direction": "increases_risk"
    }
  ],
  "n_features": 57
}
```

### Function: `compute_explanations(model, X_test, feature_names, model_type, sample_size) -> dict`

Convenience function that creates an explainer, fits it, computes global importance, and generates a sample explanation.

---

## 8. Fairness Auditor (`src/fairness_audit.py`)

### Purpose

Audits model predictions for fairness across demographic segments by computing demographic parity and equal opportunity differences.

### Dependencies

- **External:** scikit-learn, numpy, pandas, fairlearn (optional)

### Class: `FairnessAuditor`

#### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensitive_feature_names` | 3 synthetic demographics | Columns to audit across |
| `demographic_parity_threshold` | 0.10 | Maximum allowed DP difference |
| `equal_opportunity_threshold` | 0.10 | Maximum allowed EO difference |

#### `audit(y_true, y_pred, y_proba, sensitive_attributes) -> dict`

Computes fairness metrics for each sensitive attribute:

**Per-Group Metrics:**
- Sample count
- Actual default rate
- Predicted default rate
- ROC-AUC
- Accuracy

**Cross-Group Metrics:**

| Metric | Formula | Threshold |
|--------|---------|-----------|
| **Demographic Parity Difference** | max(pred_rates) - min(pred_rates) | ≤ 0.10 |
| **Equal Opportunity Difference** | max(TPRs) - min(TPRs) | ≤ 0.10 |
| **ROC-AUC Disparity** | max(AUCs) - min(AUCs) | Informational |

**Output structure:**
```json
{
  "audited_at": "2026-07-22T19:00:00",
  "overall_pass": true,
  "thresholds": {
    "demographic_parity": 0.10,
    "equal_opportunity": 0.10
  },
  "sensitive_attributes": {
    "synthetic_income_bracket": {
      "status": "passed",
      "demographic_parity_difference": 0.05,
      "equal_opportunity_difference": 0.03,
      "roc_auc_disparity": 0.02,
      "passes": {
        "demographic_parity": true,
        "equal_opportunity": true
      },
      "groups": { ... }
    }
  }
}
```

#### `get_fairlearn_report(y_true, y_pred, sensitive_attributes) -> dict | None`

Generates a Fairlearn MetricFrame report if the `fairlearn` package is available. Returns `None` otherwise.

### Function: `run_fairness_audit(model, X_test, y_test, feature_names, sensitive_data, model_type) -> dict`

Convenience function that runs predictions through the model and performs the full fairness audit.

---

## 9. Model Card Generator (`src/model_card.py`)

### Purpose

Auto-generates a Model Card (markdown) documenting training data provenance, performance metrics, known limitations, and intended use.

### Function: `generate_model_card(...) -> str`

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_type` | str | `"baseline_logistic_regression"` or `"challenger_xgboost"` |
| `metrics` | dict | ROC-AUC, accuracy, precision, recall, F1 |
| `feature_names` | list | All feature names used in training |
| `n_train` | int | Training set size |
| `n_test` | int | Test set size |
| `n_features` | int | Total feature count |
| `default_rate_train` | float | Default rate in training set |
| `default_rate_test` | float | Default rate in test set |
| `fairness_results` | dict | Optional fairness audit results |
| `training_params` | dict | Optional hyperparameters |
| `data_source` | str | Training data source description |
| `trained_at` | str | ISO timestamp |

#### Generated Sections

1. Model Details (type, version, framework)
2. Training Data (source, sizes, feature breakdown)
3. Synthetic Feature List
4. Performance Metrics
5. Intended Use
6. Known Limitations (5 standard limitations)
7. Ethical Considerations
8. STARS Framework Alignment
9. Training Parameters
10. Fairness Audit Results (if provided)

### Function: `save_model_card(card_content, output_path) -> str`

Saves the generated markdown to file. Default path is `MODEL_CARD.md` in the project root.

---

## 10. Fairness Gate Tests (`tests/test_fairness_gate.py`)

### Purpose

Pytest test suite that enforces fairness and quality gates in CI. Build fails if thresholds are violated.

### Tests

| Test | Assertion | Threshold |
|------|-----------|-----------|
| `test_fairness_gate_passes` | Demographic parity ≤ threshold | 0.15 |
| `test_fairness_gate_passes` | Equal opportunity ≤ threshold | 0.15 |
| `test_model_has_reasonable_accuracy` | ROC-AUC ≥ minimum | 0.70 |
| `test_synthetic_features_included` | At least 5 synthetic features | ≥ 5 |
| `test_no_real_pii_in_features` | No PII keywords in feature names | 0 matches |

### PII Keywords Checked

`ssn`, `social_security`, `national_id`, `phone_number`, `email`, `address`, `name`, `bank_account`

### Fixtures

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `data` | module | Load and prepare data once |
| `baseline_result` | module | Train baseline model once |
| `challenger_result` | module | Train challenger model once |

---

## Component Interaction Diagram

```
                    ┌───────────────────┐
                    │  External Client  │
                    │  (HTTP / Streamlit)│
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │  /score    │ │  /explain  │ │ /model-card│
       └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
             │              │              │
             ▼              ▼              ▼
       ┌─────────────────────────────────────────┐
       │           FastAPI Server                │
       │                                         │
       │  ┌───────────────────────────────────┐  │
       │  │  Load model from models/          │  │
       │  │  (joblib deserialization)         │  │
       │  └───────────────────────────────────┘  │
       └────────┬────────────────────────┬───────┘
                │                        │
                ▼                        ▼
       ┌────────────────┐      ┌─────────────────────┐
       │ model.predict  │      │ SHAPExplainer       │
       │  .predict_proba│      │  .explain_single()  │
       └────────────────┘      └─────────────────────┘
```

## Model Artifact Lifecycle

```
  Training                      Serving
┌────────┐    joblib.dump    ┌────────┐
│ Python │ ────────────────▶ │  .pkl  │ ─── joblib.load ──▶ In-memory Model
│ Object │                   │  File  │
└────────┘                   └────────┘
     │                           │
     │    json.dump              │
     │ ──────────────────────▶   │
┌────────┐                   ┌────────┐
│ Metrics│                   │ .json  │ ─── json.load ──▶ Feature names
│ Dict   │                   │ File   │
└────────┘                   └────────┘