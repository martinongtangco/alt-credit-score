#!/usr/bin/env python
"""
Demo: Using the Alt-Credit-Score API from Python
=================================================
This script demonstrates all 5 API endpoints using realistic applicant profiles.

Run this script AFTER starting the API server:
    uvicorn api.main:app --reload --port 8000

Then run:
    python demos/demo_python_client.py
"""

import sys
import json
from pathlib import Path

# Add project root to path so we can import api_client
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api_client import CreditScoreClient
from api_client.models import APIError


# ============================================================
# Three realistic applicant profiles
# ============================================================
APPLICANTS = {
    "Maria - Stable Professional (Low Risk)": {
        "description": (
            "32-year-old bank employee, stable job 5+ years, good telco habits, "
            "regular e-wallet user, pays bills on time."
        ),
        "features_override": {
            "synthetic_telco_topup_freq": 18,
            "synthetic_telco_avg_data_usage_gb": 15,
            "synthetic_telco_sim_tenure_months": 72,
            "synthetic_telco_avg_call_minutes": 450,
            "synthetic_ewallet_tx_count": 25,
            "synthetic_ewallet_avg_tx_amount_usd": 45,
            "synthetic_utility_bill_on_time_rate": 0.95,
            "synthetic_rent_payment_regularity": 0.90,
            "synthetic_social_credit_inquiries": 1,
            "synthetic_income_volatility": 0.15,
        },
    },
    "Juan - Gig Worker (Borderline)": {
        "description": (
            "25-year-old freelance driver, irregular income, moderate telco usage, "
            "occasional e-wallet use, sometimes late on bills."
        ),
        "features_override": {
            "synthetic_telco_topup_freq": 8,
            "synthetic_telco_avg_data_usage_gb": 5,
            "synthetic_telco_sim_tenure_months": 18,
            "synthetic_telco_avg_call_minutes": 120,
            "synthetic_ewallet_tx_count": 8,
            "synthetic_ewallet_avg_tx_amount_usd": 15,
            "synthetic_utility_bill_on_time_rate": 0.60,
            "synthetic_rent_payment_regularity": 0.50,
            "synthetic_social_credit_inquiries": 5,
            "synthetic_income_volatility": 0.65,
        },
    },
    "Pedro - High Risk Profile": {
        "description": (
            "22-year-old, no stable employment, very low telco activity, "
            "no e-wallet, multiple credit inquiries, very volatile income."
        ),
        "features_override": {
            "synthetic_telco_topup_freq": 2,
            "synthetic_telco_avg_data_usage_gb": 0.5,
            "synthetic_telco_sim_tenure_months": 3,
            "synthetic_telco_avg_call_minutes": 20,
            "synthetic_ewallet_tx_count": 0,
            "synthetic_ewallet_avg_tx_amount_usd": 0,
            "synthetic_utility_bill_on_time_rate": 0.20,
            "synthetic_rent_payment_regularity": 0.10,
            "synthetic_social_credit_inquiries": 12,
            "synthetic_income_volatility": 1.20,
        },
    },
}


def build_feature_vector(feature_names: list, overrides: dict) -> list:
    """Build a full feature vector from the model's expected feature names."""
    vector = []
    for name in feature_names:
        if name in overrides:
            vector.append(overrides[name])
        else:
            vector.append(0.0)
    return vector


def print_separator(char="=", width=70):
    print(char * width)


def main():
    # ============================================================
    # STEP 1: Connect to the API
    # ============================================================
    print_separator()
    print("  STEP 1: Connect to the Alt-Credit-Score API")
    print_separator()
    print()
    print('  Creating client with base_url="http://localhost:8000"')
    print('  and api_key="demo-key-change-in-production"')
    print()

    client = CreditScoreClient(
        base_url="http://localhost:8000",
        api_key="demo-key-change-in-production",
    )

    # ============================================================
    # STEP 2: Check API Health
    # ============================================================
    print_separator()
    print("  STEP 2: Check API Health (GET /health)")
    print_separator()
    print()
    print("  >>> health = client.health()")
    print()

    health = client.health()
    print(health)
    print()

    if not health.model_loaded:
        print("  ERROR: Model is not loaded! Make sure the API server is running.")
        print("  Start it with: uvicorn api.main:app --reload --port 8000")
        return

    # ============================================================
    # STEP 3: Get Feature Names
    # ============================================================
    print_separator()
    print("  STEP 3: Get Expected Feature Names (GET /feature-names)")
    print_separator()
    print()
    print("  >>> feature_names = client.get_feature_names()")
    print()

    feature_names = client.get_feature_names()
    print(f"  Model expects {len(feature_names)} features:")
    print()
    for i, name in enumerate(feature_names):
        print(f"    [{i:2d}] {name}")
    print()

    # ============================================================
    # STEP 4: Score Each Applicant
    # ============================================================
    print_separator()
    print("  STEP 4: Score Each Applicant (POST /score)")
    print_separator()
    print()

    for applicant_name, applicant_data in APPLICANTS.items():
        print(f"  Applicant: {applicant_name}")
        print(f"  Profile:   {applicant_data['description']}")
        print()

        features = build_feature_vector(feature_names, applicant_data["features_override"])

        print(f"  >>> result = client.score(features, model='challenger')")
        print()

        result = client.score(features, model="challenger")
        print(result)
        print()

        # Also compare with baseline model
        print(f"  --- Same applicant, using BASELINE model ---")
        print(f"  >>> result = client.score(features, model='baseline')")
        print()

        result_baseline = client.score(features, model="baseline")
        print(result_baseline)
        print()
        print_separator("-", 70)
        print()

    # ============================================================
    # STEP 5: Get SHAP Explanation for Borderline Case
    # ============================================================
    print_separator()
    print("  STEP 5: Get SHAP Explanation (POST /explain)")
    print_separator()
    print()
    print("  Getting explanation for Juan (the borderline case)...")
    print()

    juan_features = build_feature_vector(
        feature_names,
        APPLICANTS["Juan - Gig Worker (Borderline)"]["features_override"],
    )

    print("  >>> explanation = client.explain(juan_features, top_n=10)")
    print("  >>> explanation.print_attribution_table()")
    print()

    explanation = client.explain(juan_features, top_n=10)
    explanation.print_attribution_table()
    print()

    # ============================================================
    # STEP 6: Get Model Card
    # ============================================================
    print_separator()
    print("  STEP 6: Get Model Card (GET /model-card)")
    print_separator()
    print()
    print("  >>> card = client.get_model_card()")
    print()

    card = client.get_model_card()
    print(f"  Model Card ({len(card)} characters total):")
    print()
    print(card[:800])
    if len(card) > 800:
        print(f"\n  ... ({len(card) - 800} more characters)")
    print()

    # ============================================================
    # SUMMARY
    # ============================================================
    print_separator()
    print("  DEMO COMPLETE!")
    print_separator()
    print()
    print("  What we demonstrated:")
    print("    1. Connected to the API server")
    print("    2. Checked health status")
    print("    3. Retrieved feature names (the model's expected input)")
    print("    4. Scored 3 applicants with different risk profiles")
    print("    5. Compared Challenger (XGBoost) vs Baseline (LogReg) models")
    print("    6. Got SHAP explanation for the borderline case")
    print("    7. Retrieved the Model Card documentation")
    print()
    print("  See demos/README.md for curl and PowerShell equivalents.")
    print()


if __name__ == "__main__":
    main()