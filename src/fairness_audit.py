"""
Fairness Audit Module
=====================
Computes demographic parity and equal opportunity differences across
synthetic demographic segments using fairlearn.

This is the Responsibility (Social Fairness) component of STARS:
the build will FAIL in CI if fairness metrics drift past configured thresholds.

STARS Alignment:
- Responsibility: Fairness audit with CI gate on demographic parity
                  and equal opportunity difference.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from fairlearn.metrics import MetricFrame
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False


PROJECT_ROOT = Path(__file__).parent.parent


class FairnessAuditor:
    """
    Audit a credit scoring model for fairness across demographic segments.

    Uses synthetic demographic features (income_bracket, region, age_band)
    as proxy sensitive attributes. In a production system, these would be
    replaced with real protected attributes under proper governance.
    """

    def __init__(
        self,
        sensitive_feature_names: Optional[List[str]] = None,
        demographic_parity_threshold: float = 0.10,
        equal_opportunity_threshold: float = 0.10,
    ):
        """
        Args:
            sensitive_feature_names: Column names of sensitive attributes.
            demographic_parity_threshold: Max allowed demographic parity difference.
            equal_opportunity_threshold: Max allowed equal opportunity difference.
        """
        self.sensitive_feature_names = sensitive_feature_names or [
            "synthetic_income_bracket",
            "synthetic_region",
            "synthetic_age_band",
        ]
        self.demographic_parity_threshold = demographic_parity_threshold
        self.equal_opportunity_threshold = equal_opportunity_threshold

    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        sensitive_attributes: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run fairness audit across all sensitive attributes.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            sensitive_attributes: DataFrame with sensitive attribute columns.

        Returns:
            Dictionary with fairness metrics, pass/fail status per attribute.
        """
        results = {
            "audited_at": datetime.now().isoformat(),
            "sensitive_attributes": {},
            "overall_pass": True,
            "thresholds": {
                "demographic_parity": self.demographic_parity_threshold,
                "equal_opportunity": self.equal_opportunity_threshold,
            },
        }

        for attr_name in self.sensitive_feature_names:
            if attr_name not in sensitive_attributes.columns:
                results["sensitive_attributes"][attr_name] = {
                    "status": "skipped",
                    "reason": f"Column {attr_name} not found in data",
                }
                continue

            sensitive = sensitive_attributes[attr_name].values
            groups = np.unique(sensitive)

            if len(groups) < 2:
                results["sensitive_attributes"][attr_name] = {
                    "status": "skipped",
                    "reason": f"Only 1 group found in {attr_name}",
                }
                continue

            attr_result = self._compute_attr_metrics(
                y_true, y_pred, y_proba, sensitive, attr_name, groups
            )
            results["sensitive_attributes"][attr_name] = attr_result

            if not attr_result["passes"]["demographic_parity"] or \
               not attr_result["passes"]["equal_opportunity"]:
                results["overall_pass"] = False

        return results

    def _compute_attr_metrics(
        self, y_true, y_pred, y_proba, sensitive, attr_name, groups
    ) -> Dict[str, Any]:
        """Compute fairness metrics for a single sensitive attribute."""
        # Per-group metrics
        group_metrics = {}
        for group in groups:
            mask = sensitive == group
            if mask.sum() < 10:
                continue

            group_metrics[group] = {
                "n_samples": int(mask.sum()),
                "default_rate": float(y_true[mask].mean()),
                "predicted_default_rate": float(y_pred[mask].mean()),
                "roc_auc": float(roc_auc_score(y_true[mask], y_proba[mask]))
                    if len(np.unique(y_true[mask])) > 1 else 0.5,
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            }

        # Demographic parity difference: max difference in predicted positive rate
        pred_rates = [m["predicted_default_rate"] for m in group_metrics.values()]
        dp_diff = max(pred_rates) - min(pred_rates) if len(pred_rates) > 1 else 0.0

        # Equal opportunity difference: max difference in TPR
        tprs = []
        for group, metrics in group_metrics.items():
            mask = sensitive == group
            actual_positives = y_true[mask] == 1
            if actual_positives.sum() > 0:
                tpr = (y_pred[mask] == 1) & (y_true[mask] == 1)
                tprs.append(float(tpr.sum() / actual_positives.sum()))
        eo_diff = max(tprs) - min(tprs) if len(tprs) > 1 else 0.0

        # ROC-AUC disparity
        roc_aucs = [m["roc_auc"] for m in group_metrics.values() if m["roc_auc"] > 0]
        roc_auc_diff = max(roc_aucs) - min(roc_aucs) if len(roc_aucs) > 1 else 0.0

        dp_pass = dp_diff <= self.demographic_parity_threshold
        eo_pass = eo_diff <= self.equal_opportunity_threshold

        return {
            "status": "passed" if (dp_pass and eo_pass) else "failed",
            "groups": group_metrics,
            "demographic_parity_difference": round(dp_diff, 4),
            "equal_opportunity_difference": round(eo_diff, 4),
            "roc_auc_disparity": round(roc_auc_diff, 4),
            "passes": {
                "demographic_parity": dp_pass,
                "equal_opportunity": eo_pass,
            },
            "thresholds": {
                "demographic_parity": self.demographic_parity_threshold,
                "equal_opportunity": self.equal_opportunity_threshold,
            },
        }

    def get_fairlearn_report(
        self, y_true, y_pred, sensitive_attributes
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a fairlearn MetricFrame report if fairlearn is available.
        """
        if not FAIRLEARN_AVAILABLE:
            return None

        reports = {}
        for attr_name in self.sensitive_feature_names:
            if attr_name not in sensitive_attributes.columns:
                continue
            try:
                metric_frame = MetricFrame(
                    metrics={
                        "accuracy": accuracy_score,
                        "roc_auc": roc_auc_score,
                    },
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_attributes[attr_name],
                )
                reports[attr_name] = {
                    "by_group": metric_frame.by_group.to_dict(),
                    "overview": float(metric_frame.overall()),
                }
            except Exception as e:
                reports[attr_name] = {"error": str(e)}

        return reports


def run_fairness_audit(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    sensitive_data: pd.DataFrame,
    model_type: str = "tree",
) -> Dict[str, Any]:
    """
    Convenience function: run full fairness audit on a trained model.

    Args:
        model: Trained model pipeline.
        X_test: Test features.
        y_test: Test labels.
        feature_names: Feature names.
        sensitive_data: DataFrame with sensitive attributes (aligned with X_test).
        model_type: "tree" or "linear" (for info only).

    Returns:
        Fairness audit results dictionary.
    """
    print("\n" + "=" * 60)
    print("FAIRNESS AUDIT")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auditor = FairnessAuditor()
    results = auditor.audit(y_test, y_pred, y_proba, sensitive_data)

    print(f"\nOverall Fairness Gate: {'PASSED' if results['overall_pass'] else 'FAILED'}")
    print()

    for attr_name, attr_result in results["sensitive_attributes"].items():
        if attr_result.get("status") == "skipped":
            print(f"  {attr_name}: SKIPPED ({attr_result.get('reason', 'unknown')})")
        else:
            status = attr_result["status"].upper()
            dp = attr_result.get("demographic_parity_difference", "N/A")
            eo = attr_result.get("equal_opportunity_difference", "N/A")
            print(f"  {attr_name}: {status}")
            print(f"    Demographic Parity Diff: {dp}")
            print(f"    Equal Opportunity Diff:  {eo}")
            for group, metrics in attr_result.get("groups", {}).items():
                print(f"    Group '{group}' (n={metrics['n_samples']}): "
                      f"ROC-AUC={metrics['roc_auc']:.4f}, "
                      f"Accuracy={metrics['accuracy']:.4f}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_pipeline import load_german_credit_data, prepare_features
    from src.train_challenger import train_challenger
    from sklearn.model_selection import train_test_split

    df = load_german_credit_data()
    X, y, feature_names, _ = prepare_features(df, include_synthetic=True)

    # Extract sensitive attributes before splitting
    synthetic_df_cols = [c for c in feature_names if c.startswith("synthetic_")]
    sensitive_col_indices = [feature_names.index(c) for c in [
        "synthetic_income_bracket", "synthetic_region", "synthetic_age_band"
    ] if c in feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build sensitive attributes DataFrame for test set
    sensitive_data = pd.DataFrame()
    for i, col_idx in enumerate(sensitive_col_indices):
        # Map numeric values back to category labels (simple approach)
        sensitive_data[feature_names[col_idx]] = X_test[:, col_idx]

    result = train_challenger(X_train, X_test, y_train, y_test, feature_names)

    audit_results = run_fairness_audit(
        result["model"], X_test, y_test, feature_names, sensitive_data
    )
