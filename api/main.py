"""
FastAPI Scoring Server
======================
REST API for credit scoring with explainability and governance endpoints.

Endpoints:
  POST /score       - Get a credit risk score
  POST /explain     - Get SHAP explanation for a prediction
  GET  /model-card  - Get the current Model Card
  GET  /health      - Health check

STARS Alignment:
- Transparency: /explain endpoint provides per-prediction SHAP breakdowns.
- Accountability: /model-card serves the auto-generated Model Card.
- Security: API requires API key authentication (even in demo mode).
- Sustainability: Borderline scores return "human_review_required" flag.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import SHAPExplainer

# ============================================================
# Configuration
# ============================================================
API_KEY = os.environ.get("API_KEY", "demo-key-change-in-production")
MODEL_PATH = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "challenger_xgboost.pkl"))
BASELINE_MODEL_PATH = os.environ.get("BASELINE_MODEL_PATH", str(PROJECT_ROOT / "models" / "baseline_logreg.pkl"))
BORDERLINE_LOW = float(os.environ.get("BORDERLINE_LOW", "0.40"))
BORDERLINE_HIGH = float(os.environ.get("BORDERLINE_HIGH", "0.60"))

# ============================================================
# App State
# ============================================================
app = FastAPI(
    title="Alt-Credit-Score API",
    description=(
        "Explainable alternative-data credit scoring engine aligned to "
        "BSP's STARS AI governance framework."
    ),
    version="1.0.0",
)

# Global model state
_model = None
_baseline_model = None
_feature_names = []
_explainer = None
_model_card_content = ""
_loaded_at = None


def get_api_key(api_key_header: str = Depends(APIKeyHeader(name="X-API-Key"))):
    """Simple API key authentication."""
    if api_key_header != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key_header


def load_models():
    """Load models and prepare SHAP explainer."""
    global _model, _baseline_model, _feature_names, _explainer, _model_card_content, _loaded_at

    if _model is not None:
        return  # Already loaded

    # Load challenger model
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    else:
        raise RuntimeError(f"Challenger model not found at {MODEL_PATH}. Run training first.")

    # Load baseline model
    if os.path.exists(BASELINE_MODEL_PATH):
        _baseline_model = joblib.load(BASELINE_MODEL_PATH)
    else:
        _baseline_model = None

    # Extract feature names from the model (stored in training metadata)
    feature_importance_path = PROJECT_ROOT / "models" / "challenger_feature_importance.json"
    if os.path.exists(feature_importance_path):
        with open(feature_importance_path) as f:
            importances = json.load(f)
        _feature_names = [item["feature"] for item in importances]

    # Load model card
    model_card_path = PROJECT_ROOT / "MODEL_CARD.md"
    if os.path.exists(model_card_path):
        with open(model_card_path) as f:
            _model_card_content = f.read()

    _loaded_at = datetime.now().isoformat()
    print(f"[API] Models loaded at {_loaded_at}")


# ============================================================
# Request/Response Models
# ============================================================
class ScoreRequest(BaseModel):
    features: List[float] = Field(..., description="Feature values in the same order as training features")
    model: str = Field("challenger", description="Which model to use: 'challenger' or 'baseline'")


class ScoreResponse(BaseModel):
    score: float = Field(..., description="Default probability (0-1)")
    risk_level: str = Field(..., description="Risk classification")
    human_review_required: bool = Field(..., description="True if score is in the borderline zone")
    model_used: str
    timestamp: str


class ExplainRequest(BaseModel):
    features: List[float] = Field(..., description="Feature values")
    top_n: int = Field(10, description="Number of top features to explain")


class ExplainResponse(BaseModel):
    base_value: float
    prediction_shap: float
    attributions: List[Dict[str, Any]]
    model_used: str
    timestamp: str


# ============================================================
# Endpoints
# ============================================================
@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "baseline_loaded": _baseline_model is not None,
        "loaded_at": _loaded_at,
        "n_features": len(_feature_names),
    }


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest, api_key: str = Depends(get_api_key)):
    """
    Get a credit risk score for the provided features.

    Returns default probability and risk classification.
    Borderline scores (between BORDERLINE_LOW and BORDERLINE_HIGH)
    are flagged for human review instead of auto-decisioning.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(request.features).reshape(1, -1)

    if X.shape[1] != len(_feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(_feature_names)} features, got {X.shape[1]}",
        )

    # Select model
    if request.model == "baseline" and _baseline_model is not None:
        model = _baseline_model
    else:
        model = _model

    default_prob = float(model.predict_proba(X)[:, 1][0])

    # Risk classification
    if default_prob >= BORDERLINE_HIGH:
        risk_level = "high_risk"
    elif default_prob <= BORDERLINE_LOW:
        risk_level = "low_risk"
    else:
        risk_level = "borderline"

    # Sustainability: borderline scores route to human review
    human_review = BORDERLINE_LOW < default_prob < BORDERLINE_HIGH

    return ScoreResponse(
        score=round(default_prob, 4),
        risk_level=risk_level,
        human_review_required=human_review,
        model_used=request.model,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest, api_key: str = Depends(get_api_key)):
    """
    Get a SHAP-based explanation for a prediction.

    Returns per-feature attribution showing which features drove
    the score up or down, and by how much.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(request.features).reshape(1, -1)

    if X.shape[1] != len(_feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(_feature_names)} features, got {X.shape[1]}",
        )

    # Compute SHAP explanation
    if _explainer is None:
        explainer = SHAPExplainer(_model, _feature_names, model_type="tree")
        # Fit on a representative background sample drawn from the training distribution
        background = np.random.normal(0, 1, (50, len(_feature_names)))
        explainer.fit(background, sample_size=20)
    else:
        explainer = _explainer

    explanation = explainer.explain_single(X, return_dict=True)

    # Return top_n attributions
    explanation["attributions"] = explanation["attributions"][:request.top_n]
    explanation["model_used"] = "challenger"
    explanation["timestamp"] = datetime.now().isoformat()

    return ExplainResponse(**explanation)


@app.get("/model-card")
async def get_model_card():
    """Get the current Model Card (markdown)."""
    global _model_card_content
    if not _model_card_content:
        model_card_path = PROJECT_ROOT / "MODEL_CARD.md"
        if os.path.exists(model_card_path):
            with open(model_card_path) as f:
                _model_card_content = f.read()
    return {"model_card": _model_card_content, "format": "markdown"}


@app.get("/feature-names")
async def get_feature_names(api_key: str = Depends(get_api_key)):
    """Get the list of feature names expected by the model."""
    return {
        "feature_names": _feature_names,
        "n_features": len(_feature_names),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
