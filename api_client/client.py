"""
CreditScoreClient - Python SDK for the Alt-Credit-Score API.
"""

import json
from typing import List, Optional
from urllib import request, error
from urllib.error import URLError, HTTPError

from api_client.models import (
    HealthStatus,
    ScoreResult,
    ExplainResult,
    FeatureAttribution,
    APIError,
)


class CreditScoreClient:
    """
    Client for the Alt-Credit-Score API.

    Usage:
        client = CreditScoreClient(
            base_url="http://localhost:8000",
            api_key="demo-key-change-in-production",
        )

        # Check if the API is running
        health = client.health()
        print(health)

        # Get feature names the model expects
        features = client.get_feature_names()
        print(f"Model expects {len(features)} features: {features}")

        # Score a applicant
        result = client.score(my_features, model="challenger")
        print(result)

        # Get explanation
        explanation = client.explain(my_features, top_n=10)
        explanation.print_attribution_table()
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "demo-key-change-in-production"):
        """
        Initialize the API client.

        Args:
            base_url: The base URL of the API server (e.g., "http://localhost:8000")
            api_key: The API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, content_type: bool = True) -> dict:
        """Build request headers."""
        headers = {"X-API-Key": self.api_key}
        if content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _get(self, path: str) -> dict:
        """Make a GET request and return the JSON response."""
        url = f"{self.base_url}{path}"
        req = request.Request(url, headers=self._headers())
        try:
            with request.urlopen(req) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            body = e.read().decode()
            raise APIError(f"GET {path} failed with {e.code}: {body}", e.code, body)

    def _post(self, path: str, data: dict) -> dict:
        """Make a POST request and return the JSON response."""
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        req = request.Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with request.urlopen(req) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            body = e.read().decode()
            raise APIError(f"POST {path} failed with {e.code}: {body}", e.code, body)

    def health(self) -> HealthStatus:
        """
        Check API health.

        Returns:
            HealthStatus with model loading status

        Example:
            health = client.health()
            print(health)
            # API Health: healthy
            #   Challenger Model: ✅ loaded
            #   Baseline Model:   ✅ loaded
            #   Features:         70
        """
        # /health does not require auth
        url = f"{self.base_url}/health"
        req = request.Request(url)
        with request.urlopen(req) as response:
            data = json.loads(response.read().decode())
        return HealthStatus(**data)

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names the model expects, in order.

        Returns:
            List of feature name strings

        Example:
            features = client.get_feature_names()
            print(f"Model expects {len(features)} features")
            print(f"First feature: {features[0]}")
        """
        data = self._get("/feature-names")
        return data["feature_names"]

    def score(self, features: List[float], model: str = "challenger") -> ScoreResult:
        """
        Get a credit risk score for an applicant.

        Args:
            features: List of feature values in the order the model expects.
                      Use client.get_feature_names() to see the expected order.
            model: Which model to use - "challenger" (XGBoost) or "baseline" (Logistic Regression)

        Returns:
            ScoreResult with risk score, level, and recommendation

        Example:
            # Get the feature names first
            feature_names = client.get_feature_names()

            # Build your feature vector (must match the order)
            my_features = [0.5] * len(feature_names)  # placeholder values

            # Score the applicant
            result = client.score(my_features, model="challenger")
            print(result)
            # ==================================================
            #   CREDIT RISK ASSESSMENT
            # ==================================================
            #   ✅ Risk Level:        LOW RISK
            #   📊 Default Probability: 15.2%
            #   📈 Est. Credit Score:   ~721
            #   🤖 Model Used:          challenger
            #   ✅ Recommendation:       AUTO-APPROVE
            # ==================================================
        """
        data = self._post("/score", {"features": features, "model": model})
        return ScoreResult(**data)

    def explain(self, features: List[float], top_n: int = 10) -> ExplainResult:
        """
        Get a SHAP-based explanation for a prediction.

        Args:
            features: List of feature values
            top_n: Number of top feature attributions to return

        Returns:
            ExplainResult with SHAP attributions

        Example:
            explanation = client.explain(my_features, top_n=5)
            explanation.print_attribution_table()
            # ======================================================================
            #   SHAP EXPLANATION - Why This Score?
            # ======================================================================
            #   Base value (avg default rate): 0.3021
            #   Final prediction:              0.2150
            #   Model:                         challenger
            #   Top Feature Attributions (SHAP values):
            #   ⬇️ synthetic_telco_topup_freq              value=15.2000  impact=-0.0523
            #   ⬇️ synthetic_ewallet_tx_count              value=8.5000   impact=-0.0312
            #   ...
            # ======================================================================
        """
        data = self._post("/explain", {"features": features, "top_n": top_n})

        attributions = []
        for attr in data.get("attributions", []):
            attributions.append(FeatureAttribution(
                feature=attr["feature"],
                shap_value=attr["shap_value"],
                feature_value=attr["feature_value"],
                direction=attr["direction"],
            ))

        return ExplainResult(
            base_value=data["base_value"],
            prediction_shap=data["prediction_shap"],
            attributions=attributions,
            model_used=data.get("model_used", "challenger"),
            timestamp=data.get("timestamp", ""),
        )

    def get_model_card(self) -> str:
        """
        Get the Model Card documentation (markdown format).

        Returns:
            The Model Card as a markdown string

        Example:
            card = client.get_model_card()
            print(card[:500])  # print first 500 chars
        """
        # /model-card does not require auth
        url = f"{self.base_url}/model-card"
        req = request.Request(url)
        with request.urlopen(req) as response:
            data = json.loads(response.read().decode())
        return data["model_card"]


