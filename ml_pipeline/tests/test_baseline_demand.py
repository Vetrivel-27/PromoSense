"""Unit tests for Baseline Demand Estimators."""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.models.baseline_demand import (
    LightGBMDemandModel,
    MeanDemandBaseline,
    ProratedHistoricalBaseline,
    RidgeDemandModel,
    compute_regression_metrics,
)
from ml_pipeline.src.models.preprocessor import FeatureSegregator


@pytest.fixture
def baseline_training_data():
    feat = pd.read_parquet(ml_settings.PROCESSED_DATA_DIR / "train_features.parquet").head(1000)
    out = pd.read_parquet(ml_settings.PROCESSED_DATA_DIR / "train_outcomes.parquet").head(1000)
    X, T, y, meta = FeatureSegregator.extract_components(feat, out)
    # Filter strictly to untreated controls
    mask_0 = T == 0
    return X[mask_0].reset_index(drop=True), y[mask_0].reset_index(drop=True)


def test_regression_metrics_computation():
    y_true = np.array([10.0, 20.0, 30.0, 0.0])
    y_pred = np.array([12.0, 18.0, 35.0, 0.0])
    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] <= 1.0
    assert "dollar_bias" in metrics
    assert "pct_bias" in metrics


def test_prorated_baseline(baseline_training_data):
    X_0, y_0 = baseline_training_data
    model = ProratedHistoricalBaseline()
    model.fit(X_0, y_0)
    preds = model.predict(X_0)

    assert len(preds) == len(X_0)
    assert (preds >= 0).all()
    assert np.isfinite(preds).all()


def test_mean_baseline(baseline_training_data):
    X_0, y_0 = baseline_training_data
    model = MeanDemandBaseline()
    model.fit(X_0, y_0)
    preds = model.predict(X_0)

    assert len(preds) == len(X_0)
    assert np.allclose(preds, np.mean(y_0))


def test_ridge_demand_model(baseline_training_data):
    X_0, y_0 = baseline_training_data
    model = RidgeDemandModel(alpha=100.0)
    model.fit(X_0, y_0)
    preds = model.predict(X_0)

    assert len(preds) == len(X_0)
    assert (preds >= 0).all()
    assert np.isfinite(preds).all()


def test_lightgbm_demand_model(baseline_training_data):
    X_0, y_0 = baseline_training_data
    model = LightGBMDemandModel(n_estimators=30, learning_rate=0.05, n_jobs=2)
    model.fit(X_0, y_0)
    preds = model.predict(X_0)

    assert len(preds) == len(X_0)
    assert (preds >= 0).all()
    assert np.isfinite(preds).all()
    assert len(model.feature_importances_) == 39
