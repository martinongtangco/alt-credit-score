"""
Model Card Generator
====================
Auto-generates a Model Card (markdown) documenting training data provenance,
performance metrics, known limitations, and intended use.

Regenerated on every model retrain — checked in as MODEL_CARD.md.

STARS Alignment:
- Accountability: Auto-generated Model Card documenting training data,
  performance metrics, known limitations, and intended use.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


PROJECT_ROOT = Path(__file__).parent.parent


def generate_model_card(
    model_type: str,
    metrics: Dict[str, Any],
    feature_names: list,
    n_train: int,
    n_test: int,
    n_features: int,
    default_rate_train: float,
    default_rate_test: float,
    fairness_results: Optional[Dict[str, Any]] = None,
    training_params: Optional[Dict[str, Any]] = None,
    data_source: str = "UCI German Credit Data + Synthetic Alternative Data",
    trained_at: Optional[str] = None,
) -> str:
    """
    Generate a Model Card as a markdown string.

    Args:
        model_type: "baseline_logistic_regression" or "challenger_xgboost".
        metrics: Dictionary with roc_auc, accuracy, precision, recall, f1.
        feature_names: List of feature names used in training.
        n_train, n_test: Training and test set sizes.
        n_features: Number of features.
        default_rate_train, default_rate_test: Default rates in each set.
        fairness_results: Optional fairness audit results.
        training_params: Optional hyperparameters.
        data_source: Description of training data source.
        trained_at: ISO timestamp of training.

    Returns:
        Markdown string for MODEL_CARD.md.
    """
    if trained_at is None:
        trained_at = datetime.now().isoformat()

    synthetic_features = [f for f in feature_names if f.startswith("synthetic_")]
    traditional_features = [f for f in feature_names if not f.startswith("synthetic_")]

    card = f"""# Model Card: {model_type.replace('_', ' ').title()}

> **Auto-generated** at {trained_at}
> **DO NOT EDIT MANUALLY** — this file is regenerated on every model retrain.

---

## 1. Model Details

| Property | Value |
|---|---|
| **Model Type** | {model_type} |
| **Version** | {trained_at[:19].replace(' ', 'T')} |
| **Training Date** | {trained_at} |
| **Author** | alt-credit-score project (open-source demo) |
| **Framework** | Python / scikit-learn / XGBoost |

## 2. Training Data

| Property | Value |
|---|---|
| **Data Source** | {data_source} |
| **Training Samples** | {n_train} |
| **Test Samples** | {n_test} |
| **Total Features** | {n_features} |
| **Traditional Features** | {len(traditional_features)} |
| **Synthetic Alt-Data Features** | {len(synthetic_features)} |
| **Default Rate (Train)** | {default_rate_train:.3f} |
| **Default Rate (Test)** | {default_rate_test:.3f} |

**IMPORTANT:** All alternative-data features are synthetically generated for
demonstration purposes. No real telco, e-wallet, or personal data was used or accessed.

### Synthetic Feature List
"""

    for feat in synthetic_features:
        card += f"- `{feat}`\n"

    card += f"""
## 3. Performance Metrics

| Metric | Value |
|---|---|
| **ROC-AUC** | {metrics.get('roc_auc', 'N/A'):.4f} |
| **Accuracy** | {metrics.get('accuracy', 'N/A'):.4f} |
| **Precision** | {metrics.get('precision', 'N/A'):.4f} |
| **Recall** | {metrics.get('recall', 'N/A'):.4f} |
| **F1 Score** | {metrics.get('f1', 'N/A'):.4f} |

## 4. Intended Use

- **Primary Use:** Demonstration of explainable credit scoring with alternative data.
- **Decision Type:** Informational only — NOT designed for production credit decisions.
- **Target Population:** Thin-file / unbanked consumers (demonstrated on UCI German Credit data).

## 5. Known Limitations

1. **Synthetic Data:** Alternative-data features are synthetically generated, not from real users.
2. **Single Dataset:** Trained only on UCI German Credit Data — may not generalize.
3. **No Temporal Validation:** No time-based train/test split.
4. **Proxy Demographics:** Fairness audits use synthetic demographic segments, not real protected attributes.
5. **Demo Scope:** This model has NOT been validated for regulatory compliance in any jurisdiction.

## 6. Ethical Considerations

- All synthetic features are clearly labeled with `synthetic_` prefix.
- No real PII enters the model or the repository.
- Fairness audits are automated and block CI on threshold violations.
- Borderline scores route to "human review required" instead of auto-decisioning.

## 7. STARS Framework Alignment

| STARS Pillar | Implementation |
|---|---|
| **S**ustainability | Borderline scores trigger human review flag |
| **T**ransparency | SHAP explainability per prediction |
| **A**ccountability | This auto-generated Model Card |
| **R**esponsibility | Fairness audit with CI gate |
| **S**ecurity | No real PII; synthetic data policy enforced |

## 8. Training Parameters
"""

    if training_params:
        for key, value in training_params.items():
            card += f"| {key} | {value} |\n"
    else:
        card += "| N/A | No custom parameters |\n"

    if fairness_results:
        card += f"""
## 9. Fairness Audit Results

**Overall Gate:** {'PASSED' if fairness_results.get('overall_pass') else 'FAILED'}

"""
        for attr_name, attr_result in fairness_results.get("sensitive_attributes", {}).items():
            if attr_result.get("status") == "skipped":
                continue
            card += f"### {attr_name}\n\n"
            card += f"- **Status:** {attr_result['status'].upper()}\n"
            card += f"- **Demographic Parity Difference:** {attr_result.get('demographic_parity_difference', 'N/A')}\n"
            card += f"- **Equal Opportunity Difference:** {attr_result.get('equal_opportunity_difference', 'N/A')}\n"
            card += f"- **ROC-AUC Disparity:** {attr_result.get('roc_auc_disparity', 'N/A')}\n\n"

    card += f"""---

*This Model Card was auto-generated by the alt-credit-score project.*
*See [README.md](README.md) for full project documentation.*
"""

    return card


def save_model_card(card_content: str, output_path: Optional[str] = None) -> str:
    """Save model card to a markdown file."""
    if output_path is None:
        output_path = PROJECT_ROOT / "MODEL_CARD.md"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    return str(output_path)


if __name__ == "__main__":
    # Example usage
    card = generate_model_card(
        model_type="challenger_xgboost",
        metrics={"roc_auc": 0.81, "accuracy": 0.75, "precision": 0.62, "recall": 0.55, "f1": 0.58},
        feature_names=["amount", "duration", "synthetic_telco_topup_freq", "synthetic_ewallet_tx_count"],
        n_train=1000,
        n_test=250,
        n_features=4,
        default_rate_train=0.30,
        default_rate_test=0.30,
        training_params={"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    )
    path = save_model_card(card)
    print(f"Model Card saved to: {path}")
