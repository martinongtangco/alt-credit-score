"""
Data Pipeline
=============
Loads the UCI German Credit Dataset (or Kaggle alternative), performs
preprocessing, and joins with synthetic alternative-data features.

This module is the single entry point for preparing the feature matrix
used by both the baseline and challenger models.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


# Project root is parent of src/
PROJECT_ROOT = Path(__file__).parent.parent


def load_german_credit_data(
    data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the UCI German Credit Dataset.

    The dataset is downloaded from the UCI ML repository if not already present.
    Column names are mapped to readable names and the target variable is
    recoded to 0 (good credit) / 1 (default risk).

    Args:
        data_path: Optional path to a local CSV. If None, downloads from UCI.

    Returns:
        DataFrame with standardized column names and target = 1 for risky borrowers.
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path, sep=";")
    else:
        # Download from UCI
        import urllib.request
        url = (
            "https://archive.ics.uci.edu/static/public/"
            "144/statlog+german+credit+data.zip"
        )
        cache_dir = PROJECT_ROOT / "data" / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / "german_credit.zip"

        if not zip_path.exists():
            print(f"Downloading German Credit Dataset from UCI...")
            urllib.request.urlretrieve(url, zip_path)

            # Extract
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(cache_dir)

        csv_file = cache_dir / "statlog german credit data" / "german.data"
        if not csv_file.exists():
            csv_file = cache_dir / "german.data"

        # The UCI dataset has no header; use known column structure
        # The target (default) is the LAST column (index 20)
        column_names = [
            "status_checking_account", "duration_months", "credit_history",
            "purpose", "amount", "saving_account", "employment_duration",
            "installment_rate", "personal_status_sex", "other_debtors",
            "residence_duration", "property", "age_years", "installment_plan",
            "housing", "existing_credits", "occupation", "dependents",
            "telephone", "foreign_worker", "default"
        ]
        df = pd.read_csv(csv_file, header=None, sep=r"\s+", names=column_names)

    # Recode target: in German Credit, 1 = good, 2 = risky
    df["default"] = (df["default"] == 2).astype(int)

    return df


def _encode_categorical(df: pd.DataFrame, cat_columns: List[str]) -> pd.DataFrame:
    """One-hot encode categorical columns with prefix for clarity."""
    df = df.copy()
    for col in cat_columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
    return df


def prepare_features(
    df: pd.DataFrame,
    include_synthetic: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Prepare the final feature matrix from raw German Credit data.

    Args:
        df: Raw DataFrame from load_german_credit_data().
        include_synthetic: If True, generate and join synthetic alt-data features.
        seed: Random seed for synthetic data generator.

    Returns:
        (X, y, feature_names) tuple ready for modeling.
    """
    df = df.copy()

    # Identify categorical vs numeric columns in the base dataset
    categorical_cols = [
        "status_checking_account",
        "credit_history",
        "purpose",
        "saving_account",
        "employment_duration",
        "personal_status_sex",
        "other_debtors",
        "property",
        "installment_plan",
        "housing",
        "occupation",
        "telephone",
        "foreign_worker",
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Numeric features to keep directly
    numeric_cols = [
        "duration_months",
        "amount",
        "installment_rate",
        "residence_duration",
        "age_years",
        "existing_credits",
        "dependents",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Encode categoricals
    df = _encode_categorical(df, categorical_cols)

    # Separate target
    y = df["default"]
    df = df.drop(columns=["default"])

    # Add synthetic alternative-data features
    if include_synthetic:
        # Import here to avoid hard dependency if synthetic generator unavailable
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from data.synthetic_generator import SyntheticAltDataGenerator

        # Generate synthetic features with independent risk signal
        # Using ground-truth labels would create data leakage (features
        # become direct functions of the target). Instead, use a random
        # risk profile so the synthetic features have a realistic, weaker
        # signal-to-noise relationship.
        alt_gen = SyntheticAltDataGenerator(seed=seed)
        risk_signal = alt_gen.rng.uniform(0.1, 0.9, len(df))
        synthetic_df = alt_gen.generate(n_samples=len(df), default_rates=risk_signal)

        # Join synthetic features
        df = pd.concat([df.reset_index(drop=True), synthetic_df.reset_index(drop=True)], axis=1)

        # Store metadata for documentation
        _feature_metadata = alt_gen.get_feature_metadata()
    else:
        _feature_metadata = {}

    # Encode remaining string/object columns (synthetic demographics, etc.)
    string_cols = [
        c for c in df.columns
        if df[c].dtype in ("object", "string") or pd.api.types.is_string_dtype(df[c])
    ]
    if string_cols:
        df = pd.get_dummies(df, columns=string_cols, dtype=int)

    # Convert bool columns (from one-hot encoding) to int before to_numeric
    # pd.to_numeric coerces bools to NaN with errors='coerce'
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Ensure all remaining columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaN (from encoding failures)
    valid_mask = ~df.isna().any(axis=1)
    df = df[valid_mask].reset_index(drop=True)
    y = y[valid_mask.values].reset_index(drop=True)

    feature_names = list(df.columns)
    X = df.values.astype(np.float64)

    return X, y.values, feature_names, _feature_metadata


def load_and_prepare(
    include_synthetic: bool = True,
    test_size: float = 0.2,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> dict:
    """
    Full pipeline: load data, prepare features, split into train/test.

    Args:
        include_synthetic: Include synthetic alt-data features.
        test_size: Fraction for test set.
        seed: Random seed.
        data_path: Optional local data path.

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_names,
        and feature_metadata.
    """
    from sklearn.model_selection import train_test_split

    df = load_german_credit_data(data_path=data_path)
    X, y, feature_names, feature_metadata = prepare_features(
        df, include_synthetic=include_synthetic, seed=seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "feature_metadata": feature_metadata,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
    }


if __name__ == "__main__":
    result = load_and_prepare(include_synthetic=True)
    print(f"Train: {result['n_train']} samples, {result['n_features']} features")
    print(f"Test:  {result['n_test']} samples")
    print(f"Default rate — Train: {result['default_rate_train']:.3f}, "
          f"Test: {result['default_rate_test']:.3f}")
    print(f"\nFeature names ({len(result['feature_names'])}):")
    for i, name in enumerate(result["feature_names"]):
        synthetic_marker = " [SYNTHETIC]" if "synthetic_" in name else ""
        print(f"  {i+1:3d}. {name}{synthetic_marker}")