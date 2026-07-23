# ============================================================
# Demo: Using the Alt-Credit-Score API with PowerShell
# ============================================================
# Run: .\demos\demo_powershell.ps1
# (After starting: uvicorn api.main:app --reload --port 8000)
# ============================================================

$ErrorActionPreference = "Stop"
$BaseUrl = "http://localhost:8000"
$ApiKey = "demo-key-change-in-production"
$AuthHeaders = @{ "X-API-Key" = $ApiKey; "Content-Type" = "application/json" }

function Write-Step($s, $t) { Write-Host ""; Write-Host "============================================================"; Write-Host "  STEP $s: $t"; Write-Host "============================================================"; Write-Host "" }

# STEP 1: Health
Write-Step 1 "Health Check (GET /health)"
$health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
$health | ConvertTo-Json -Depth 5
Write-Host "  Status: $($health.status) | Model loaded: $($health.model_loaded)" -ForegroundColor Green

# STEP 2: Feature Names
Write-Step 2 "Get Feature Names (GET /feature-names)"
$featureData = Invoke-RestMethod -Uri "$BaseUrl/feature-names" -Method Get -Headers $AuthHeaders
Write-Host "  Model expects $($featureData.n_features) features" -ForegroundColor Cyan
for ($i = 0; $i -lt $featureData.feature_names.Count; $i++) {
    Write-Host "    [$($i.ToString().PadLeft(2))] $($featureData.feature_names[$i])"
}

# STEP 3: Score Low Risk
Write-Step 3 "Score Low-Risk Applicant (POST /score)"
Write-Host "  Maria - Stable 32yo professional, good telco habits" -ForegroundColor Cyan
$numFeatures = $featureData.n_features
$fLow = New-Object 'double[]' $numFeatures
$fLow[0]=18; $fLow[51]=15; $fLow[52]=72; $fLow[53]=450; $fLow[54]=25; $fLow[55]=45
$fLow[56]=0.95; $fLow[57]=0.90; $fLow[58]=1; $fLow[59]=0.15
$scoreLow = Invoke-RestMethod -Uri "$BaseUrl/score" -Method Post -Headers $AuthHeaders `
    -Body (@{ features = $fLow; model = "challenger" } | ConvertTo-Json -Compress)
$scoreLow | ConvertTo-Json -Depth 5
Write-Host "  >> $($scoreLow.risk_level) (probability: $($scoreLow.score))" -ForegroundColor Green

# STEP 4: Score High Risk
Write-Step 4 "Score High-Risk Applicant (POST /score)"
Write-Host "  Pedro - 22yo, no stable employment" -ForegroundColor Cyan
$fHigh = New-Object 'double[]' $numFeatures
$fHigh[0]=2; $fHigh[51]=0.5; $fHigh[52]=3; $fHigh[53]=20; $fHigh[54]=0; $fHigh[55]=0
$fHigh[56]=0.20; $fHigh[57]=0.10; $fHigh[58]=12; $fHigh[59]=1.20
$scoreHigh = Invoke-RestMethod -Uri "$BaseUrl/score" -Method Post -Headers $AuthHeaders `
    -Body (@{ features = $fHigh; model = "challenger" } | ConvertTo-Json -Compress)
$scoreHigh | ConvertTo-Json -Depth 5
Write-Host "  >> $($scoreHigh.risk_level) (probability: $($scoreHigh.score))" -ForegroundColor Red

# STEP 5: Compare Models
Write-Step 5 "Compare Challenger vs Baseline"
$scoreBase = Invoke-RestMethod -Uri "$BaseUrl/score" -Method Post -Headers $AuthHeaders `
    -Body (@{ features = $fLow; model = "baseline" } | ConvertTo-Json -Compress)
Write-Host "  Challenger: prob=$($scoreLow.score) level=$($scoreLow.risk_level)" -ForegroundColor Cyan
Write-Host "  Baseline:   prob=$($scoreBase.score) level=$($scoreBase.risk_level)" -ForegroundColor Cyan

# STEP 6: SHAP Explanation
Write-Step 6 "SHAP Explanation (POST /explain)"
$explain = Invoke-RestMethod -Uri "$BaseUrl/explain" -Method Post -Headers $AuthHeaders `
    -Body (@{ features = $fLow; top_n = 10 } | ConvertTo-Json -Compress)
Write-Host "  Base: $($explain.base_value) | Prediction: $($explain.prediction_shap)" -ForegroundColor Cyan
foreach ($a in $explain.attributions) {
    $s = if ($a.shap_value -gt 0) { "+" } else { "" }
    Write-Host "    $($a.feature.PadRight(45)) => $($s)$($a.shap_value.ToString('0.0000')) ($($a.direction))"
}
Write-Host "  (Positive = increases risk | Negative = decreases risk)" -ForegroundColor Gray

# STEP 7: Model Card
Write-Step 7 "Model Card (GET /model-card)"
$card = (Invoke-RestMethod -Uri "$BaseUrl/model-card" -Method Get).model_card
Write-Host "  Model Card ($($card.Length) chars):" -ForegroundColor Cyan
if ($card.Length -gt 500) { Write-Host $card.Substring(0, 500); Write-Host "  ... (truncated)" -ForegroundColor Gray }
else { Write-Host $card }

# Summary
Write-Host ""; Write-Host "============================================================"
Write-Host "  DEMO COMPLETE! All API endpoints demonstrated."
Write-Host "============================================================"
Write-Host ""
Write-Host "  1. GET /health       2. GET /feature-names"
Write-Host "  3. POST /score       4. POST /explain"
Write-Host "  5. GET /model-card"
Write-Host ""