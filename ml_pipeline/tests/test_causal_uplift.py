"""Unit tests for Propensity Modeling, Causal Uplift Meta-Learners, and Metrics."""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.src.causal.meta_learners import SLearner, TLearner, XLearner
from ml_pipeline.src.causal.propensity import (
    PropensityScoreModel,
    compute_standardized_mean_differences,
)
from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.evaluate.uplift_metrics import (
    compute_auuc,
    compute_qini_score,
    compute_top_decile_lift,
    compute_uplift_curve,
    evaluate_uplift_model,
)
from ml_pipeline.src.models.preprocessor import FeatureSegregator


@pytest.fixture
def causal_sample_data():
    feat = pd.read_parquet(ml_settings.PROCESSED_DATA_DIR / "train_features.parquet").head(1000)
    out = pd.read_parquet(ml_settings.PROCESSED_DATA_DIR / "train_outcomes.parquet").head(1000)
    X, T, y, meta = FeatureSegregator.extract_components(feat, out)
    return X, T, y


def test_propensity_score_model(causal_sample_data):
    X, T, _ = causal_sample_data
    prop_model = PropensityScoreModel(estimator_type="lightgbm", n_jobs=2)
    prop_model.fit(X, T)

    e_scores = prop_model.predict_proba(X)
    assert len(e_scores) == len(X)
    assert (e_scores >= 0.0).all() and (e_scores <= 1.0).all()

    # Discrimination metrics
    disc = prop_model.evaluate_discrimination(X, T)
    assert 0.5 <= disc["roc_auc"] <= 1.0
    assert 0.0 <= disc["brier_score"] <= 1.0

    # Overlap diagnostics
    overlap = prop_model.compute_overlap_diagnostics(X, T)
    assert 0.0 <= overlap["pct_in_trim_bounds_05_95"] <= 100.0


def test_standardized_mean_differences(causal_sample_data):
    X, T, _ = causal_sample_data
    e_dummy = np.full(len(X), 0.1)
    smd_df = compute_standardized_mean_differences(
        X, T, propensity_scores=e_dummy, features_to_check=["hh_spend_90d"]
    )
    assert len(smd_df) == 1
    assert "raw_smd" in smd_df.columns
    assert "ipw_smd" in smd_df.columns
    assert np.isfinite(smd_df["raw_smd"].iloc[0])


def test_s_learner_cate_prediction(causal_sample_data):
    X, T, y = causal_sample_data
    learner = SLearner(base_estimator="ridge")
    learner.fit(X, T, y)

    cate = learner.predict_cate(X)
    assert len(cate) == len(X)
    assert np.isfinite(cate).all()


def test_t_learner_cate_prediction(causal_sample_data):
    X, T, y = causal_sample_data
    learner = TLearner(base_estimator="lightgbm", n_jobs=2)
    learner.fit(X, T, y)

    cate = learner.predict_cate(X)
    assert len(cate) == len(X)
    assert np.isfinite(cate).all()


def test_x_learner_cate_prediction(causal_sample_data):
    X, T, y = causal_sample_data
    learner = XLearner(base_estimator="lightgbm", n_jobs=2)
    learner.fit(X, T, y)

    cate = learner.predict_cate(X)
    assert len(cate) == len(X)
    assert np.isfinite(cate).all()


def test_uplift_evaluation_metrics():
    np.random.seed(42)
    n = 200
    y = np.random.exponential(scale=30.0, size=n)
    t = np.random.binomial(n=1, p=0.2, size=n)
    score = np.random.normal(loc=0.0, scale=1.0, size=n)

    curve_df = compute_uplift_curve(y, t, score, n_bins=10)
    assert len(curve_df) == 11
    assert curve_df.iloc[0]["n_cum"] == 0
    assert curve_df.iloc[-1]["n_cum"] == n

    qini = compute_qini_score(curve_df)
    auuc = compute_auuc(curve_df)
    assert np.isfinite(qini)
    assert np.isfinite(auuc)

    top_decile = compute_top_decile_lift(y, t, score)
    assert "top_decile_lift" in top_decile
    assert "top_decile_lift_multiplier" in top_decile

    eval_dict = evaluate_uplift_model(y, t, score, model_name="test_model")
    assert eval_dict["model_name"] == "test_model"
    assert "qini_score" in eval_dict
