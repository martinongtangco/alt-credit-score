"""
Alt-Credit-Score Python Client SDK
===================================
A simple, easy-to-use Python client for the Alt-Credit-Score API.

Quick Start:
    from api_client import CreditScoreClient

    client = CreditScoreClient(base_url="http://localhost:8000", api_key="demo-key-change-in-production")

    # Get a risk score
    result = client.score(features, model="challenger")
    print(f"Risk: {result.risk_level} (score: {result.score})")

    # Get SHAP explanation
    explanation = client.explain(features, top_n=10)
    explanation.print_attribution_table()
"""

from api_client.client import CreditScoreClient
from api_client.models import ScoreResult, ExplainResult, HealthStatus

__all__ = ["CreditScoreClient", "ScoreResult", "ExplainResult", "HealthStatus"]
__version__ = "1.0.0"