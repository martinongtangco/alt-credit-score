"""
Explainability Module: SHAP-based Feature Attribution
=====================================================
Provides per-prediction explanations using SHAP (SHapley Additive exPlanations).

This is the core Transparency component of the STARS governance framework:
every prediction can be accompanied by a clear breakdown of which features
drove the score up or down, and by how much.

STARS Alignment:
- Transparency: Per-prediction SHAP explanations via `/explain` endpoint.
- Accountability: Global feature-importance report generated per model version.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import shap
import joblib


# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class SHAPExplainer:
    """
    Wrapper around SHAP that works with both tree-based models (XGBoost)
    and linear models (Logistic Regression).
    """

    def __init__(self, model, feature_names: list, model_type: str = "tree"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self._explainer = None

    def fit(self, X_background: np.ndarray, sample_size: int = 100):
        """Prepare the SHAP explainer using a background dataset."""
        if self.model_type == "tree":
            if hasattr(self.model, "named_steps"):
                tree_model = self.model.named_steps.get("xgboost") or self.model.steps[-1][1]
            else:
                tree_model = self.model
            indices = np.random.choice(
                len(X_background), size=min(sample_size, len(X_background)), replace=False
            )
            self._explainer = shap.TreeExplainer(tree_model)
        elif self.model_type == "linear":
            if hasattr(self.model, "named_steps"):
                linear_model = self.model.named_steps["logistic_regression"]
                scaler = self.model.named_steps["scaler"]
                background = scaler.transform(X_background[:sample_size])
            else:
                linear_model = self.model
                background = X_background[:sample_size]
            self._explainer = shap.LinearExplainer(
                linear_model, background, feature_dependence="independent"
            )
        else:
            indices = np.random.choice(
                len(X_background), size=min(100, len(X_background)), replace=False
            )
            background = X_background[indices]

            def model_predict(X):
                return self.model.predict_proba(X)[:, 1]

            self._explainer = shap.KernelExplainer(model_predict, background)
        return self

    def explain(self, X: np.ndarray):
        """Compute SHAP values for the given samples."""
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not fitted. Call .fit() first.")
        return self._explainer.shap_values(X)

    def explain_single(self, x: np.ndarray, return_dict: bool = True):
        """Explain a single prediction."""
        explanation = self.explain(x)
        if isinstance(explanation, list):
            shap_vals = explanation[1] if len(explanation) > 1 else explanation[0]
            base_val = 0
        else:
            shap_vals = explanation.values if hasattr(explanation, "values") else explanation
            base_val = explanation.base_values if hasattr(explanation, "base_values") else 0

        if hasattr(shap_vals, "shape") and len(shap_vals.shape) > 1:
            shap_vals = shap_vals[0] if shap_vals.shape[0] == 1 else shap_vals.flatten()

        if return_dict:
            attributions = []
            for i, name in enumerate(self.feature_names):
                if i < len(shap_vals):
                    attributions.append({
                        "feature": name,
                        "shap_value": float(shap_vals[i]),
                        "feature_value": float(x[0][i]) if hasattr(x, "shape") and x.ndim > 1 else float(x[i]),
                        "direction": "increases_risk" if shap_vals[i] > 0 else "decreases_risk",
                    })
            attributions.sort(key=lambda a: abs(a["shap_value"]), reverse=True)
            base_float = float(base_val[0]) if hasattr(base_val, "__len__") else float(base_val)
            pred_float = base_float + sum(float(s) for s in shap_vals)
            return {
                "base_value": base_float,
                "prediction_shap": pred_float,
                "attributions": attributions,
                "n_features": len(self.feature_names),
            }
        return explanation

    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Compute global feature importance from SHAP values across all samples."""
        explanation = self.explain(X)
        if isinstance(explanation, list):
            shap_vals = explanation[1] if len(explanation) > 1 else explanation[0]
        else:
            shap_vals = explanation.values if hasattr(explanation, "values") else explanation
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

    def summary_plot_data(self, X: np.ndarray) -> dict:
        """Return data suitable for a SHAP summary plot."""
        explanation = self.explain(X)
        if isinstance(explanation, list):
            shap_vals = explanation[1] if len(explanation) > 1 else explanation[0]
        else:
            shap_vals = explanation.values if hasattr(explanation, "values") else explanation
        return {
            "shap_values": shap_vals.tolist(),
            "feature_values": X.tolist(),
            "feature_names": self.feature_names,
        }


def compute_explanations(
    model, X_test: np.ndarray, feature_names: list,
    model_type: str = "tree", sample_size: int = 100,
):
    """Create explainer, fit it, and compute global importance."""
    print("\n" + "=" * 60)
    print("SHAP Explainability Analysis")
    print("=" * 60)

    explainer = SHAPExplainer(model, feature_names, model_type=model_type)
    explainer.fit(X_test, sample_size=sample_size)

    importance_df = explainer.global_importance(X_test)
    print(f"\nTop 15 Features (by mean |SHAP value|):")
    print(importance_df.head(15).to_string(index=False))

    sample_idx = 0
    single_explanation = explainer.explain_single(X_test[sample_idx:sample_idx + 1])
    print(f"\nSingle Prediction Explanation (sample {sample_idx}):")
    print(f"  Base value: {single_explanation['base_value']:.4f}")
    print(f"  SHAP prediction: {single_explanation['prediction_shap']:.4f}")
    print(f"  Top 5 drivers:")
    for attr in single_explanation["attributions"][:5]:
        print(f"    {attr['feature']}: {attr['shap_value']:+.4f} ({attr['direction']})")

    return {
        "explainer": explainer,
        "global_importance": importance_df.to_dict("records"),
        "sample_explanation": single_explanation,
        "model_type": model_type,
        "computed_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_pipeline import load_and_prepare
    from src.train_challenger import train_challenger

    data = load_and_prepare(include_synthetic=True)
    result = train_challenger(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        data["feature_names"],
    )
    shap_result = compute_explanations(
        result["model"], data["X_test"],
        data["feature_names"], model_type="tree",
    )
