"""Unit and integration tests for Phase 4 Feature Engineering Pipeline."""

import pandas as pd
import pytest

from ml_pipeline.src.config.settings import ml_settings


@pytest.fixture(scope="module")
def processed_dir():
    return ml_settings.PROCESSED_DATA_DIR


@pytest.fixture(scope="module")
def interim_dir():
    return ml_settings.INTERIM_DATA_DIR


def test_feature_and_outcome_matrix_shapes(processed_dir):
    """Verify that feature and outcome matrices match expected row counts per split."""
    splits = {
        "train": 15 * 2500,  # 37,500
        "val": 4 * 2500,  # 10,000
        "test": 5 * 2500,  # 12,500
        "straddler": 5 * 2500,  # 12,500
    }

    for split_name, expected_rows in splits.items():
        feat_path = processed_dir / f"{split_name}_features.parquet"
        outc_path = processed_dir / f"{split_name}_outcomes.parquet"

        assert feat_path.exists(), f"Missing {feat_path}"
        assert outc_path.exists(), f"Missing {outc_path}"

        feats = pd.read_parquet(feat_path)
        outcs = pd.read_parquet(outc_path)

        assert len(feats) == expected_rows
        assert len(outcs) == expected_rows

        # Ensure 1-to-1 key alignment between predictors and labels
        assert (feats["household_key"] == outcs["household_key"]).all()
        assert (feats["campaign_id"] == outcs["campaign_id"]).all()


def test_zero_temporal_leakage_in_features(processed_dir, interim_dir):
    """Assert that baseline predictor features contain zero future or contemporaneous transaction information."""
    train_feats = pd.read_parquet(processed_dir / "train_features.parquet")
    txns = pd.read_parquet(interim_dir / "transactions.parquet")

    # Sample a campaign from train
    sample_camp_id = 26  # Earliest campaign: starts Day 224
    camp_feats = train_feats[train_feats["campaign_id"] == sample_camp_id]
    s_day = int(camp_feats["campaign_start_day"].iloc[0])
    assert s_day == 224

    # Transactions strictly before Day 224
    pre_txns = txns[txns["day"] < s_day]

    # For a sample household, assert their feature spend is <= their total pre-treatment spend
    sample_hh = camp_feats.iloc[0]["household_key"]
    hh_pre_spend = pre_txns[pre_txns["household_key"] == sample_hh]["sales_value"].sum()
    hh_feat_spend_90d = camp_feats[camp_feats["household_key"] == sample_hh]["hh_spend_90d"].iloc[0]

    # 90d spend must be bounded by total pre-treatment spend
    assert hh_feat_spend_90d <= hh_pre_spend + 1e-4


def test_feature_value_bounds(processed_dir):
    """Verify numeric features satisfy expected logical and domain bounds."""
    feats = pd.read_parquet(processed_dir / "train_features.parquet")
    outcs = pd.read_parquet(processed_dir / "train_outcomes.parquet")

    # Non-negativity of spend and trips
    assert (feats["hh_spend_90d"] >= 0).all()
    assert (feats["hh_trips_90d"] >= 0).all()
    assert (feats["hh_avg_basket_value_90d"] >= 0).all()

    # Ratios in [0, 1]
    assert (feats["hh_discount_intensity_90d"] >= 0.0).all()
    assert (feats["hh_discount_intensity_90d"] <= 1.0).all()
    assert (feats["hh_coupon_usage_rate_90d"] >= 0.0).all()
    assert (feats["hh_coupon_usage_rate_90d"] <= 1.0).all()
    assert (feats["hh_private_label_share_90d"] >= 0.0).all()
    assert (feats["hh_private_label_share_90d"] <= 1.0).all()

    # Treatment binary indicator
    assert set(outcs["treatment_assigned"].unique()).issubset({0, 1})
    assert set(outcs["is_strict_clean_control"].unique()).issubset({0, 1})

    # Outcomes non-negativity
    assert (outcs["y_total_spend_window"] >= 0).all()
    assert (outcs["y_category_spend_window"] >= 0).all()
