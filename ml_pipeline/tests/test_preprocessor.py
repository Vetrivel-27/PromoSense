"""Unit tests for Feature Segregation and Tabular Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.models.preprocessor import (
    GOVERNANCE_COLS,
    FeatureSegregator,
    TabularPreprocessor,
)


@pytest.fixture
def sample_feature_data():
    df_feat = pd.read_parquet(ml_settings.PROCESSED_DATA_DIR / "train_features.parquet").head(500)
    df_out = pd.read_parquet(ml_settings.PROCESSED_DATA_DIR / "train_outcomes.parquet").head(500)
    return df_feat, df_out


def test_feature_segregator_isolates_governance_columns(sample_feature_data):
    feat, out = sample_feature_data
    X, T, y, meta = FeatureSegregator.extract_components(feat, out)

    # Invariant 1: No governance or ID columns in predictor matrix X
    for col in GOVERNANCE_COLS:
        assert col not in X.columns, f"Leaked governance column '{col}' into predictor matrix X!"

    # Invariant 2: X has exactly 39 predictors
    assert X.shape[1] == 39

    # Invariant 3: T is binary
    assert set(T.unique()).issubset({0, 1})

    # Invariant 4: y is non-null and matches row count
    assert y is not None
    assert len(y) == len(X)
    assert not y.isna().any()

    # Invariant 5: Metadata table contains keys and clean control flag
    assert "household_key" in meta.columns
    assert "campaign_id" in meta.columns
    assert "is_strict_clean_control" in meta.columns


def test_tabular_preprocessor_linear_mode(sample_feature_data):
    feat, _ = sample_feature_data
    X, _, _, _ = FeatureSegregator.extract_components(feat)

    prep = TabularPreprocessor(mode="linear")
    X_trans = prep.fit_transform(X)

    # Invariant: Output is numeric 2D array with zero NaNs
    assert isinstance(X_trans, np.ndarray)
    assert not np.isnan(X_trans).any()
    assert X_trans.shape[0] == len(X)
    assert X_trans.shape[1] > 0


def test_tabular_preprocessor_tree_mode(sample_feature_data):
    feat, _ = sample_feature_data
    X, _, _, _ = FeatureSegregator.extract_components(feat)

    prep = TabularPreprocessor(mode="tree")
    X_trans = prep.fit_transform(X)

    # Invariant: Categorical columns have pandas 'category' dtype
    assert isinstance(X_trans, pd.DataFrame)
    for cat_col in prep.cat_cols:
        assert isinstance(X_trans[cat_col].dtype, pd.CategoricalDtype)
    # Zero nulls in transformed tree format
    assert not X_trans[prep.num_cols].isna().any().any()
