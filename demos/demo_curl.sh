#!/usr/bin/env bash
# ============================================================
# Demo: Using the Alt-Credit-Score API with curl
# ============================================================
# Run after starting the API server:
#   uvicorn api.main:app --reload --port 8000
# Then: bash demos/demo_curl.sh
# ============================================================

set -e

BASE_URL="http://localhost:8000"
API_KEY="demo-key-change-in-production"

echo "============================================================"
echo "  Alt-Credit-Score API - curl Demo"
echo "============================================================"
echo ""

# ----------------------------------------------------------
# STEP 1: Health Check
# ----------------------------------------------------------
echo "------------------------------------------------------------"
echo "  STEP 1: Health Check (GET /health)"
echo "------------------------------------------------------------"
echo ""
echo "  $ curl ${BASE_URL}/health"
echo ""

curl -s "${BASE_URL}/health" | python -m json.tool
echo ""

# ----------------------------------------------------------
# STEP 2: Get Feature Names
# ----------------------------------------------------------
echo "------------------------------------------------------------"
echo "  STEP 2: Get Feature Names (GET /feature-names)"
echo "------------------------------------------------------------"
echo ""

FEATURE_NAMES=$(curl -s -H "X-API-Key: ${API_KEY}" "${BASE_URL}/feature-names")
echo "$FEATURE_NAMES" | python -m json.tool

NUM_FEATURES=$(echo "$FEATURE_NAMES" | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_features'])")
echo ""
echo "  The model expects ${NUM_FEATURES} features."
echo ""

# ----------------------------------------------------------
# STEP 3: Score a Low-Risk Applicant
# ----------------------------------------------------------
echo "------------------------------------------------------------"
echo "  STEP 3: Score a Low-Risk Applicant (POST /score)"
echo "------------------------------------------------------------"
echo ""
echo "  Applicant: Maria - Stable 32-year-old professional"
echo ""

FEATURES_LOW_RISK=$(python -c "
features = [0.0] * ${NUM_FEATURES}
features[0]  = 18.0;  features[51] = 15.0;  features[52] = 72.0
features[53] = 450.0; features[54] = 25.0;  features[55] = 45.0
features[56] = 0.95;  features[57] = 0.90;  features[58] = 1.0
features[59] = 0.15
import json; print(json.dumps(features))
")

SCORE_RESPONSE=$(curl -s -X POST "${BASE_URL}/score" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"features\": ${FEATURES_LOW_RISK}, \"model\": \"challenger\"}")

echo "  Response:"
echo "$SCORE_RESPONSE" | python -m json.tool
echo ""

# ----------------------------------------------------------
# STEP 4: Score a High-Risk Applicant
# ----------------------------------------------------------
echo "------------------------------------------------------------"
echo "  STEP 4: Score a High-Risk Applicant (POST /score)"
echo "------------------------------------------------------------"
echo ""
echo "  Applicant: Pedro - High risk, no stable employment"
echo ""

FEATURES_HIGH_RISK=$(python -c "
features = [0.0] * ${NUM_FEATURES}
features[0]  = 2.0;   features[51] = 0.5;   features[52] = 3.0
features[53] = 20.0;  features[54] = 0.0;   features[55] = 0.0
features[56] = 0.20;  features[57] = 0.10;  features[58] = 12.0
features[59] = 1.20
import json; print(json.dumps(features))
")

SCORE_RESPONSE=$(curl -s -X POST "${BASE_URL}/score" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"features\": ${FEATURES_HIGH_RISK}, \"model\": \"challenger\"}")

echo "  Response:"
echo "$SCORE_RESPONSE" | python -m json.tool
echo ""

# ----------------------------------------------------------
# STEP 5: SHAP Explanation
# ----------------------------------------------------------
echo "------------------------------------------------------------"
echo "  STEP 5: SHAP Explanation (POST /explain)"
echo "------------------------------------------------------------"
echo ""
echo "  Getting explanation for Maria (low risk case)..."
echo ""

EXPLAIN_RESPONSE=$(curl -s -X POST "${BASE_URL}/explain" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"features\": ${FEATURES_LOW_RISK}, \"top_n\": 10}")

echo "  Response:"
echo "$EXPLAIN_RESPONSE" | python -m json.tool
echo ""

# ----------------------------------------------------------
# STEP 6: Model Card
# ----------------------------------------------------------
echo "------------------------------------------------------------"
echo "  STEP 6: Model Card (GET /model-card)"
echo "------------------------------------------------------------"
echo ""

MODEL_CARD=$(curl -s "${BASE_URL}/model-card")
echo "$MODEL_CARD" | python -c "
import sys, json
data = json.load(sys.stdin)
card = data['model_card']
print(f'  Model Card ({len(card)} characters):')
print(); print(card[:600])
if len(card) > 600:
    print(f'  ... ({len(card) - 600} more characters)')
"

echo ""
echo "============================================================"
echo "  curl Demo Complete!"
echo "============================================================"
echo ""
echo "  All 6 API endpoints demonstrated:"
echo "    1. GET  /health        - Health check"
echo "    2. GET  /feature-names - List model features"
echo "    3. POST /score         - Credit risk scoring"
echo "    4. POST /score         - Compare risk profiles"
echo "    5. POST /explain       - SHAP explanations"
echo "    6. GET  /model-card    - Model documentation"
echo ""