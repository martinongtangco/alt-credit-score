"""
Baseline Model: Logistic Regression Scorecard
==============================================
Trains a logistic regression model and transforms coefficients into
a points-based scorecard similar to traditional FICO-style scoring.

This model prioritizes interpretability over raw accuracy, making it
ideal for explaining decisions to non-technical stakeholders.

STARS Alignment:
- Transparency: Each feature has a clear point assignment.
- Responsibility: Linear model is inherently more auditable.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from sklearn.linear_model import LogisticRegression
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
import joblib


# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class ScorecardTransformer:
    """
    Transforms logistic regression coefficients into a points-based
    scorecard with a base score and point increments per feature.

    Uses the "points per logarithmic odds" (PDO) method:
    - base_score: the score corresponding to a reference odds ratio
    - pdo: the point increase for a factor-of-N increase in odds
    """

    def __init__(self, base_score: int = 650, pdo: int = 50, odds_at_base: float = 2.0):
        """
        Args:
            base_score: Score assigned to the reference odds.
            pdo: Points-to-Double-Odds — points added for each odds doubling.
            odds_at_base: The odds ratio that corresponds to base_score.
        """
        self.base_score = base_score
        self.pdo = pdo
        self.odds_at_base = odds_at_base

    def fit(self, model: Pipeline, feature_names: list, scaler: StandardScaler = None):
        """
        Fit the scorecard to a trained logistic regression pipeline.

        Args:
            model: Trained sklearn Pipeline with LogisticRegression as final step.
            feature_names: Names of features (before scaling).
            scaler: Optional StandardScaler used in the pipeline.
        """
        self._scaler = scaler
        lr = model.named_steps["logistic_regression"]
        coefs = lr.coef_[0]
        intercept = lr.intercept_[0]

        # Compute points for each feature value
        # Score = base_score - (pdo / log(2)) * log(odds / odds_at_base)
        # For each feature: points = -(pdo / log(2)) * coef * x
        self._feature_points = {}
        self._feature_names = feature_names

        for i, name in enumerate(feature_names):
            self._feature_points[name] = {
                "coefficient": float(coefs[i]),
                "pdo_weight": float(-(self.pdo / np.log(2)) * coefs[i]),
            }

        self._base_points = float(
            self.base_score - (self.pdo / np.log(2)) * np.log(1.0 / self.odds_at_base)
            + (self.pdo / np.log(2)) * intercept
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert feature matrix to scorecard points."""
        scaled = self._scaler.transform(X) if self._scaler else X
        scores = np.full(X.shape[0], self._base_points)
        for i in range(scaled.shape[1]):
            name = self._feature_names[i]
            weight = self._feature_points[name]["pdo_weight"]
            scores += scaled[:, i] * weight
        return scores

    def get_scorecard_table(self) -> pd.DataFrame:
        """Return a DataFrame showing the scorecard structure."""
        rows = []
        for name, info in self._feature_points.items():
            rows.append({
                "feature": name,
                "logistic_coef": round(info["coefficient"], 6),
                "pdo_weight": round(info["pdo_weight"], 4),
            })
        return pd.DataFrame(rows).sort_values("pdo_weight", key=abs, ascending=False)

    def to_dict(self) -> dict:
        """Export scorecard configuration for serialization."""
        return {
            "base_score": self.base_score,
            "pdo": self.pdo,
            "odds_at_base": self.odds_at_base,
            "base_points": self._base_points,
            "feature_points": self._feature_points,
        }


def train_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """
    Train the baseline logistic regression model.

    Args:
        X_train, X_test, y_train, y_test: Train/test splits.
        feature_names: List of feature names.
        max_iter: Max iterations for logistic regression solver.

    Returns:
        Dictionary with model, metrics, scorecard, and metadata.
    """
    print("=" * 60)
    print("BASELINE MODEL: Logistic Regression Scorecard")
    print("=" * 60)

    # Build pipeline: scale + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            max_iter=max_iter,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])

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

    # Build scorecard
    scaler = pipeline.named_steps["scaler"]
    scorecard = ScorecardTransformer(base_score=650, pdo=50)
    scorecard.fit(pipeline, feature_names, scaler)

    # Show top features by PDO weight
    table = scorecard.get_scorecard_table()
    print(f"\nTop 10 Scorecard Features (by PDO weight magnitude):")
    print(table.head(10).to_string(index=False))

    # Save model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_logreg.pkl"
    joblib.dump(pipeline, model_path)

    scorecard_path = model_dir / "baseline_scorecard.json"
    with open(scorecard_path, "w") as f:
        json.dump(scorecard.to_dict(), f, indent=2)

    result = {
        "model": pipeline,
        "scorecard": scorecard,
        "metrics": metrics,
        "model_type": "baseline_logistic_regression",
        "model_path": str(model_path),
        "scorecard_path": str(scorecard_path),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "trained_at": datetime.now().isoformat(),
    }

    print(f"\nModel saved to: {model_path}")
    print(f"Scorecard saved to: {scorecard_path}")
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_pipeline import load_and_prepare

    data = load_and_prepare(include_synthetic=True)
    result = train_baseline(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        data["feature_names"],
    )