"""
Fairness Gate Tests
===================
These tests ensure fairness metrics meet configured thresholds.
The CI pipeline will FAIL if these tests fail, preventing
fairness-regressed models from being merged.

STARS Alignment:
- Responsibility: CI-gated fairness checks prevent biased models
  from being deployed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import load_and_prepare
from src.train_baseline import train_baseline
from src.train_challenger import train_challenger
from src.fairness_audit import FairnessAuditor


# Fairness thresholds — adjust based on your tolerance
DEMOGRAPHIC_PARITY_THRESHOLD = 0.15
EQUAL_OPPORTUNITY_THRESHOLD = 0.15


@pytest.fixture(scope="module")
def data():
    """Load and prepare data once per test module."""
    return load_and_prepare(include_synthetic=True, seed=42)


@pytest.fixture(scope="module")
def baseline_result(data):
    """Train baseline model."""
    return train_baseline(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        data["feature_names"],
    )


@pytest.fixture(scope="module")
def challenger_result(data):
    """Train challenger model."""
    return train_challenger(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        data["feature_names"],
    )


def _build_sensitive_df(X_test, feature_names):
    """Build a sensitive attributes DataFrame from test features.

    Note: synthetic demographic columns are stored as numeric codes after
    one-hot encoding and to_numeric coercion. We use them as group labels.
    """
    sensitive_cols = [
        "synthetic_income_bracket",
        "synthetic_region",
        "synthetic_age_band",
    ]
    result = {}
    for col in sensitive_cols:
        if col in feature_names:
            idx = feature_names.index(col)
            result[col] = X_test[:, idx]
    return pd.DataFrame(result)


def test_fairness_gate_passes(data):
    """
    Main fairness gate: train a model and verify fairness metrics
    are within acceptable thresholds across all sensitive attributes.
    """
    # Train challenger model
    result = train_challenger(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        data["feature_names"],
    )

    model = result["model"]
    y_pred = model.predict(data["X_test"])
    y_proba = model.predict_proba(data["X_test"])[:, 1]

    sensitive_df = _build_sensitive_df(data["X_test"], data["feature_names"])

    auditor = FairnessAuditor(
        demographic_parity_threshold=DEMOGRAPHIC_PARITY_THRESHOLD,
        equal_opportunity_threshold=EQUAL_OPPORTUNITY_THRESHOLD,
    )

    audit_results = auditor.audit(
        data["y_test"], y_pred, y_proba, sensitive_df
    )

    # Check each attribute that wasn't skipped
    for attr_name, attr_result in audit_results["sensitive_attributes"].items():
        if attr_result.get("status") == "skipped":
            continue

        dp_diff = attr_result.get("demographic_parity_difference", 0)
        eo_diff = attr_result.get("equal_opportunity_difference", 0)

        assert dp_diff <= DEMOGRAPHIC_PARITY_THRESHOLD, (
            f"{attr_name}: Demographic parity difference {dp_diff:.4f} "
            f"exceeds threshold {DEMOGRAPHIC_PARITY_THRESHOLD}"
        )
        assert eo_diff <= EQUAL_OPPORTUNITY_THRESHOLD, (
            f"{attr_name}: Equal opportunity difference {eo_diff:.4f} "
            f"exceeds threshold {EQUAL_OPPORTUNITY_THRESHOLD}"
        )


def test_model_has_reasonable_accuracy(data, challenger_result):
    """Ensure the model achieves a minimum ROC-AUC."""
    auc = challenger_result["metrics"]["roc_auc"]
    assert auc >= 0.70, f"ROC-AUC {auc:.4f} below minimum threshold of 0.70"


def test_synthetic_features_included(data):
    """Verify synthetic features are present in the dataset."""
    synthetic = [f for f in data["feature_names"] if f.startswith("synthetic_")]
    assert len(synthetic) >= 5, (
        f"Expected at least 5 synthetic features, found {len(synthetic)}"
    )


def test_no_real_pii_in_features(data):
    """Ensure no column names suggest real PII."""
    pii_keywords = ["ssn", "social_security", "national_id", "phone_number",
                     "email", "address", "name", "bank_account"]
    for name in data["feature_names"]:
        for keyword in pii_keywords:
            assert keyword not in name.lower(), (
                f"Feature '{name}' contains PII keyword '{keyword}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
