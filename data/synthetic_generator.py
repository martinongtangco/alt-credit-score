"""
Synthetic Alternative-Data Feature Generator
=============================================
Generates realistic telco/e-wallet-style alternative data features with
statistical relationships to credit risk. ALL features are synthetic —
no real personal data is used or accessed.

Each feature is designed to mirror the *type* of signal used by alternative
credit bureaus (CIBI, FinScore, etc.) without using any real user data.
"""

import numpy as np
import pandas as pd
from typing import Optional


class SyntheticAltDataGenerator:
    """
    Generates synthetic alternative-data features with controlled
    statistical relationships to a target variable (default risk).

    Features are labeled with `synthetic_` prefix for unambiguous tracking.
    """

    def __init__(
        self,
        seed: int = 42,
        noise_level: float = 0.15,
    ):
        """
        Args:
            seed: Random seed for reproducibility.
            noise_level: Standard deviation of noise added to features
                         (controls how strong the signal-to-noise ratio is).
        """
        self.rng = np.random.RandomState(seed)
        self.noise_level = noise_level
        self._feature_metadata = {
            "synthetic_telco_topup_freq": {
                "description": "Monthly count of mobile phone top-ups (synthetic)",
                "unit": "count/month",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.35,
            },
            "synthetic_telco_avg_data_usage_gb": {
                "description": "Average monthly mobile data usage in GB (synthetic)",
                "unit": "GB/month",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.25,
            },
            "synthetic_telco_sim_tenure_months": {
                "description": "Number of months with the same SIM/provider (synthetic)",
                "unit": "months",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.30,
            },
            "synthetic_telco_avg_call_minutes": {
                "description": "Average monthly call minutes (synthetic)",
                "unit": "minutes/month",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.15,
            },
            "synthetic_ewallet_tx_count": {
                "description": "Monthly e-wallet transaction count (synthetic)",
                "unit": "count/month",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.40,
            },
            "synthetic_ewallet_avg_tx_amount_usd": {
                "description": "Average e-wallet transaction amount in USD (synthetic)",
                "unit": "USD",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.20,
            },
            "synthetic_utility_bill_on_time_rate": {
                "description": "Fraction of utility bills paid on time in last 12 months (synthetic)",
                "unit": "rate [0-1]",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.45,
            },
            "synthetic_rent_payment_regularity": {
                "description": "Fraction of rent payments made on schedule (synthetic)",
                "unit": "rate [0-1]",
                "direction": "higher = lower risk",
                "correlation_with_default": -0.38,
            },
            "synthetic_social_credit_inquiries": {
                "description": "Number of informal credit inquiries (peer lending, BNPL apps) (synthetic)",
                "unit": "count/month",
                "direction": "higher = higher risk",
                "correlation_with_default": 0.28,
            },
            "synthetic_income_volatility": {
                "description": "Coefficient of variation of monthly income deposits (synthetic)",
                "unit": "CV (std/mean)",
                "direction": "higher = higher risk",
                "correlation_with_default": 0.32,
            },
            # Synthetic demographic segments for fairness auditing
            "synthetic_income_bracket": {
                "description": "Synthetic income bracket for fairness testing (synthetic)",
                "unit": "categorical: low/medium/high",
                "direction": "n/a (demographic proxy)",
                "correlation_with_default": 0.0,
            },
            "synthetic_region": {
                "description": "Synthetic geographic region for fairness testing (synthetic)",
                "unit": "categorical: region_A/region_B/region_C",
                "direction": "n/a (demographic proxy)",
                "correlation_with_default": 0.0,
            },
            "synthetic_age_band": {
                "description": "Synthetic age band for fairness testing (synthetic)",
                "unit": "categorical: young/middle/senior",
                "direction": "n/a (demographic proxy)",
                "correlation_with_default": 0.0,
            },
        }

    def generate(self, n_samples: int, default_rates: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate synthetic alternative-data features for n_samples individuals.

        Args:
            n_samples: Number of rows to generate.
            default_rates: Optional array of default probabilities (from a baseline
                          model) to use as the latent risk signal. If None, random
                          default probabilities are used.

        Returns:
            DataFrame with synthetic feature columns.
        """
        if default_rates is None:
            # Uniform random risk if no baseline provided
            default_rates = self.rng.uniform(0.05, 0.6, n_samples)

        rows = []

        # --- TELCO FEATURES ---
        # Top-up frequency: responsible users top up more regularly
        target_mean = 15 + (-1.0 * default_rates) * 20  # range ~5 to 35
        rows.append(self._add_noise("synthetic_telco_topup_freq", target_mean, 2, 40))

        # Data usage: higher usage = more engaged financially
        target_mean = 5 + (-1.0 * default_rates) * 8
        rows.append(self._add_noise("synthetic_telco_avg_data_usage_gb", target_mean, 0.5, 20))

        # SIM tenure: longer tenure = more stable
        target_mean = 12 + (-1.0 * default_rates) * 36
        rows.append(self._add_noise("synthetic_telco_sim_tenure_months", target_mean, 6, 60))

        # Call minutes: moderate signal
        target_mean = 100 + (-1.0 * default_rates) * 200
        rows.append(self._add_noise("synthetic_telco_avg_call_minutes", target_mean, 20, 600))

        # --- E-WALLET FEATURES ---
        # Transaction count: strong signal of financial activity
        target_mean = 10 + (-1.0 * default_rates) * 30
        rows.append(self._add_noise("synthetic_ewallet_tx_count", target_mean, 2, 50))

        # Average transaction amount
        target_mean = 5 + (-1.0 * default_rates) * 20
        rows.append(self._add_noise("synthetic_ewallet_avg_tx_amount_usd", target_mean, 1, 50))

        # --- UTILITY & BILL PAYMENT ---
        # On-time utility bill rate: very strong signal
        target_mean = np.clip(0.9 - default_rates * 0.7, 0.0, 1.0)
        rows.append(self._add_noise("synthetic_utility_bill_on_time_rate", target_mean, 0.05, 1.0, clip=(0, 1)))

        # Rent payment regularity
        target_mean = np.clip(0.85 - default_rates * 0.6, 0.0, 1.0)
        rows.append(self._add_noise("synthetic_rent_payment_regularity", target_mean, 0.05, 1.0, clip=(0, 1)))

        # --- RISK INDICATORS (positive correlation with default) ---
        # Social credit inquiries: more inquiries = higher risk
        target_mean = 1 + default_rates * 8
        rows.append(self._add_noise("synthetic_social_credit_inquiries", target_mean, 0.5, 15))

        # Income volatility
        target_mean = 0.1 + default_rates * 0.5
        rows.append(self._add_noise("synthetic_income_volatility", target_mean, 0.05, 2.0, clip=(0, 2)))

        # --- SYNTHETIC DEMOGRAPHICS (for fairness auditing) ---
        # These are independent of default risk but may correlate with features
        income_brackets = self.rng.choice(["low", "medium", "high"], n_samples, p=[0.3, 0.45, 0.25])
        regions = self.rng.choice(["region_A", "region_B", "region_C"], n_samples, p=[0.4, 0.35, 0.25])
        age_bands = self.rng.choice(["young", "middle", "senior"], n_samples, p=[0.35, 0.45, 0.20])

        df = pd.DataFrame({r[0]: r[1] for r in rows})
        df["synthetic_income_bracket"] = income_brackets
        df["synthetic_region"] = regions
        df["synthetic_age_band"] = age_bands

        return df

    def _add_noise(
        self,
        name: str,
        target: np.ndarray,
        scale: float,
        max_val: float,
        clip: Optional[tuple] = None,
    ) -> tuple:
        """Add controlled noise to a target signal."""
        values = target + self.rng.normal(0, scale * self.noise_level / 0.15, size=target.shape)
        if clip:
            values = np.clip(values, clip[0], clip[1])
        else:
            values = np.clip(values, 0, max_val)
        return (name, values)

    def get_feature_metadata(self) -> dict:
        """Return metadata for all generated features."""
        return self._feature_metadata.copy()

    def generate_documentation_df(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all synthetic features for documentation."""
        rows = []
        for col, meta in self._feature_metadata.items():
            rows.append({
                "feature_name": col,
                "description": meta["description"],
                "unit": meta["unit"],
                "risk_direction": meta["direction"],
                "target_correlation_with_default": meta["correlation_with_default"],
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick demo: generate 100 samples and print summary
    gen = SyntheticAltDataGenerator(seed=42)
    df = gen.generate(n_samples=100)
    print(f"Generated {df.shape[0]} samples with {df.shape[1]} synthetic features")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nFeature summary:")
    print(df.describe())
    print("\nFeature documentation:")
    print(gen.generate_documentation_df().to_string(index=False))