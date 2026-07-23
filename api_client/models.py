"""
Data models for the Alt-Credit-Score API client.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class APIError(Exception):
    """Raised when the API returns an error response."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


@dataclass
class HealthStatus:
    """Response from GET /health"""
    status: str
    model_loaded: bool
    baseline_loaded: bool
    loaded_at: Optional[str]
    n_features: int

    def __str__(self):
        loaded = "✅" if self.model_loaded else "❌"
        baseline = "✅" if self.baseline_loaded else "❌"
        return (
            f"API Health: {self.status}\n"
            f"  Challenger Model: {loaded} loaded\n"
            f"  Baseline Model:   {baseline} loaded\n"
            f"  Features:         {self.n_features}\n"
            f"  Loaded at:        {self.loaded_at or 'N/A'}"
        )


@dataclass
class ScoreResult:
    """Response from POST /score"""
    score: float
    risk_level: str
    human_review_required: bool
    model_used: str
    timestamp: str

    def credit_decision_summary(self) -> str:
        """Human-readable credit decision summary."""
        credit_score = round((1 - self.score) * 850, 1)  # Map to ~FICO scale for display
        emoji = {"low_risk": "✅", "high_risk": "🚫", "borderline": "⚠️"}.get(self.risk_level, "❓")

        lines = [
            "=" * 50,
            f"  CREDIT RISK ASSESSMENT",
            "=" * 50,
            f"  {emoji} Risk Level:        {self.risk_level.upper().replace('_', ' ')}",
            f"  📊 Default Probability: {self.score:.1%}",
            f"  📈 Est. Credit Score:   ~{credit_score}",
            f"  🤖 Model Used:          {self.model_used}",
            f"  🕐 Timestamp:           {self.timestamp}",
        ]

        if self.human_review_required:
            lines.append(f"  ⚠️  Action Required:     HUMAN REVIEW REQUIRED")
            lines.append(f"     This application falls in the borderline zone.")
            lines.append(f"     A human underwriter should review this case.")
        elif self.risk_level == "low_risk":
            lines.append(f"  ✅ Recommendation:       AUTO-APPROVE")
        else:
            lines.append(f"  🚫 Recommendation:       AUTO-DECLINE")

        lines.append("=" * 50)
        return "\n".join(lines)

    def __str__(self):
        return self.credit_decision_summary()


@dataclass
class FeatureAttribution:
    """Single feature attribution from SHAP explanation."""
    feature: str
    shap_value: float
    feature_value: float
    direction: str  # "increased" or "decreased" risk

    def __str__(self):
        arrow = "⬆️" if self.direction == "increased" else "⬇️"
        impact = f"+{self.shap_value:.4f}" if self.shap_value > 0 else f"{self.shap_value:.4f}"
        return f"  {arrow} {self.feature:<45} value={self.feature_value:.4f}  impact={impact}"


@dataclass
class ExplainResult:
    """Response from POST /explain"""
    base_value: float
    prediction_shap: float
    attributions: List[FeatureAttribution]
    model_used: str
    timestamp: str

    def print_attribution_table(self) -> str:
        """Print a formatted SHAP attribution table."""
        lines = [
            "",
            "=" * 70,
            "  SHAP EXPLANATION - Why This Score?",
            "=" * 70,
            f"  Base value (avg default rate): {self.base_value:.4f}",
            f"  Final prediction:              {self.prediction_shap:.4f}",
            f"  Model:                         {self.model_used}",
            "-" * 70,
            "  Top Feature Attributions (SHAP values):",
            "-" * 70,
        ]

        for attr in self.attributions:
            lines.append(str(attr))

        lines.append("=" * 70)
        lines.append(
            "  Positive SHAP = increases default risk | "
            "Negative SHAP = decreases default risk"
        )
        lines.append("=" * 70)

        output = "\n".join(lines)
        print(output)
        return output

    def __str__(self):
        return self.print_attribution_table()