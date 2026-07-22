"""
Challenger Model: XGBoost Gradient Boosting
============================================
Trains an XGBoost gradient boosting model on the full feature set
(including synthetic alternative-data features).

This model trades some interpretability for higher predictive accuracy.
Explainability is restored via SHAP (see explainability.py).

STARS Alignment:
- Transparency: SHAP values provide per-prediction explanations.
- Security: Trained only on synthetic + public data, no real PII.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import xgboost as xgb
import joblib


# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def train_challenger(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Train the challenger XGBoost model.

    Args:
        X_train, X_test, y_train, y_test: Train/test splits.
        feature_names: List of feature names.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.

    Returns:
        Dictionary with model, metrics, and metadata.
    """
    print("=" * 60)
    print("CHALLENGER MODEL: XGBoost Gradient Boosting")
    print("=" * 60)

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgboost", xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Train with early stopping using validation set
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\nTest Set Metrics:")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")

    # Feature importance from XGBoost
    xgb_model = pipeline.named_steps["xgboost"]
    importances = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\nTop 15 Features (XGBoost importance):")
    print(importance_df.head(15).to_string(index=False))

    # Save model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "challenger_xgboost.pkl"
    joblib.dump(pipeline, model_path)

    # Save feature importances
    importance_path = model_dir / "challenger_feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(
            importance_df[["feature", "importance"]].to_dict("records"),
            f,
            indent=2,
        )

    result = {
        "model": pipeline,
        "metrics": metrics,
        "feature_importances": importance_df.to_dict("records"),
        "model_type": "challenger_xgboost",
        "model_path": str(model_path),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "trained_at": datetime.now().isoformat(),
    }

    print(f"\nModel saved to: {model_path}")
    print(f"Feature importances saved to: {importance_path}")
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_pipeline import load_and_prepare

    data = load_and_prepare(include_synthetic=True)
    result = train_challenger(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        data["feature_names"],
    )