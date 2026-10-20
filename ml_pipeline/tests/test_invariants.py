"""Invariant Test Suite for PromoSense Data & Causal Pipeline.

Enforces fundamental invariants:
1. Transaction classification reconciliation & mutual exclusivity.
2. Positive shelf price on valid purchases and NaN on unpriced entries.
3. Entity primary key uniqueness.
4. Empirical fiscal calendar alignment (observed Days 1-711 vs extrapolated Days 712+).
5. Coupon fan-out prevention and duplicate triple tracking.
6. Time-gated split non-overlap (zero outcome window leakage).
7. Treatment and control pool exact reconciliation (T + C = 2,500).
8. Consistent inclusive right-censoring duration on Campaign 24 (53 days observed).
"""

import pandas as pd
import pytest

from ml_pipeline.src.config.settings import ml_settings


@pytest.fixture(scope="module")
def interim_dir():
    return ml_settings.INTERIM_DATA_DIR


@pytest.fixture(scope="module")
def processed_dir():
    return ml_settings.PROCESSED_DATA_DIR


def test_transaction_classification_reconciliation(interim_dir):
    """Assert all 2,595,732 rows are classified into 6 mutually exclusive, exhaustive categories."""
    df = pd.read_parquet(
        interim_dir / "transactions.parquet",
        columns=["row_classification"],
    )
    total_rows = len(df)
    assert total_rows == 2595732

    expected_categories = {
        "valid_purchase",
        "void_or_zero_entry",
        "promotional_freebie",
        "coupon_clearing_line",
        "zero_sales_unexplained",
        "fee_or_service_charge",
    }
    actual_categories = set(df["row_classification"].unique())
    assert actual_categories == expected_categories

    counts = df["row_classification"].value_counts()
    assert counts.sum() == total_rows
    assert counts["valid_purchase"] == 2576815
    assert counts["void_or_zero_entry"] == 14399
    assert counts["promotional_freebie"] == 3749
    assert counts["coupon_clearing_line"] == 444
    assert counts["zero_sales_unexplained"] == 258
    assert counts["fee_or_service_charge"] == 67


def test_shelf_price_valid_purchases_positive(interim_dir):
    """Assert shelf_price is strictly positive for valid purchases and NaN for unpriced entries."""
    df = pd.read_parquet(
        interim_dir / "transactions.parquet",
        columns=["row_classification", "shelf_price", "sales_value", "quantity"],
    )

    valid_mask = df["row_classification"] == "valid_purchase"
    valid_prices = df.loc[valid_mask, "shelf_price"]
    assert valid_prices.notna().all()
    assert (valid_prices > 0.0).all()

    freebie_mask = df["row_classification"] == "promotional_freebie"
    freebie_prices = df.loc[freebie_mask, "shelf_price"]
    assert freebie_prices.notna().all()
    assert (freebie_prices > 0.0).all()

    unpriced_classes = [
        "void_or_zero_entry",
        "coupon_clearing_line",
        "zero_sales_unexplained",
        "fee_or_service_charge",
    ]
    for cls in unpriced_classes:
        unpriced_prices = df.loc[df["row_classification"] == cls, "shelf_price"]
        assert unpriced_prices.isna().all()


def test_entity_primary_key_uniqueness(interim_dir):
    """Assert all core entities possess strictly unique primary keys."""
    hhs = pd.read_parquet(interim_dir / "households.parquet", columns=["household_key"])
    assert hhs["household_key"].is_unique
    assert len(hhs) == 2500

    prods = pd.read_parquet(interim_dir / "products.parquet", columns=["product_id"])
    assert prods["product_id"].is_unique
    assert len(prods) == 92353

    camps = pd.read_parquet(interim_dir / "campaigns.parquet", columns=["campaign_id"])
    assert camps["campaign_id"].is_unique
    assert len(camps) == 30

    stores = pd.read_parquet(interim_dir / "stores.parquet", columns=["store_id"])
    assert stores["store_id"].is_unique
    assert len(stores) == 582


def test_empirical_fiscal_calendar_alignment(interim_dir):
    """Assert campaign weeks are derived from observed transaction fiscal calendar, not naive division."""
    camps = pd.read_parquet(interim_dir / "campaigns.parquet")
    txns = pd.read_parquet(interim_dir / "transactions.parquet", columns=["day", "week_no"])

    day_to_week = dict(zip(txns["day"], txns["week_no"], strict=False))

    for _, c in camps.iterrows():
        s_day = int(c["start_day"])
        e_day = int(c["end_day"])
        if s_day in day_to_week:
            assert c["start_week"] == day_to_week[s_day]
        if e_day in day_to_week:
            assert c["end_week"] == day_to_week[e_day]


def test_fiscal_calendar_observed_vs_extrapolated_boundary(interim_dir):
    """Assert Days 1-711 are observed and Days > 711 are marked extrapolated."""
    camps = pd.read_parquet(interim_dir / "campaigns.parquet")

    # All start days are <= 711, so start_week_basis must be observed
    assert (camps["start_week_basis"] == "observed").all()

    # Only Campaign 24 ends after Day 711
    censored = camps[camps["is_right_censored"]]
    assert len(censored) == 1
    assert censored.iloc[0]["campaign_id"] == 24
    assert censored.iloc[0]["end_day"] == 719
    assert censored.iloc[0]["end_week_basis"] == "extrapolated"

    # Inclusive duration check: 711 - 659 + 1 = 53 days
    assert censored.iloc[0]["observed_duration_days"] == 53
    assert censored.iloc[0]["duration_days"] == 61

    # Uncensored campaigns
    uncensored = camps[~camps["is_right_censored"]]
    assert (uncensored["end_week_basis"] == "observed").all()
    assert (uncensored["duration_days"] == uncensored["observed_duration_days"]).all()


def test_coupon_mapping_cardinality_and_fanout_prevention(interim_dir):
    """Assert coupon UPC linkages exist, track raw duplicates, and verify non-multiplying joins."""
    coupons = pd.read_parquet(interim_dir / "coupons.parquet")
    assert len(coupons) == 124548
    assert coupons["coupon_upc"].nunique() == 1135
    assert coupons["product_id"].nunique() == 44133

    # Document raw source duplicate invariant: exactly 5,164 duplicate rows exist in raw Dunnhumby coupon table
    raw_duplicates = coupons.duplicated().sum()
    assert raw_duplicates == 5164

    # Deduplicated mappings must have unique triples
    deduped = coupons.drop_duplicates(subset=["coupon_upc", "product_id", "campaign_id"])
    assert len(deduped) == 124548 - 5164

    # Fanout guard test: Joining a transaction line item on (product_id, campaign_id) using deduplicated coupons
    # must NOT cause uncontrolled row multiplication
    sample_prod_id = deduped.iloc[0]["product_id"]
    sample_camp_id = deduped.iloc[0]["campaign_id"]
    matching_coupons = deduped[
        (deduped["product_id"] == sample_prod_id) & (deduped["campaign_id"] == sample_camp_id)
    ]
    # In retail coupons, a product within a campaign maps to at most a small handful of legitimate coupons
    assert 1 <= len(matching_coupons) <= 10

    # Analytical Join Measure Preservation Test:
    # When annotating transactions with campaign coupon eligibility, pre-aggregating coupons to product level
    # guarantees exact row count and measure preservation (zero double-counting of sales or quantities).
    sample_txns = pd.read_parquet(
        interim_dir / "transactions.parquet",
        columns=["household_key", "basket_id", "product_id", "quantity", "sales_value"],
    ).head(1000)

    # Scoped eligibility flag per product for sample campaign
    camp_eligible_prods = deduped[deduped["campaign_id"] == sample_camp_id][
        ["product_id"]
    ].drop_duplicates()
    camp_eligible_prods["is_coupon_eligible"] = True

    annotated_txns = sample_txns.merge(camp_eligible_prods, on="product_id", how="left")
    annotated_txns["is_coupon_eligible"] = annotated_txns["is_coupon_eligible"].fillna(False)

    # Invariants: Exact row count preservation and exact sales/qty sum preservation
    assert len(annotated_txns) == len(sample_txns)
    assert annotated_txns["sales_value"].sum() == pytest.approx(sample_txns["sales_value"].sum())
    assert annotated_txns["quantity"].sum() == pytest.approx(sample_txns["quantity"].sum())


def test_time_gated_split_boundary_leakage_prevention(processed_dir):
    """Assert that time-gated splits enforce zero outcome-window overlap into future evaluation periods."""
    audit_path = processed_dir / "campaign_eligibility_audit.parquet"
    assert audit_path.exists()
    audit_df = pd.read_parquet(audit_path)

    train_camps = audit_df[audit_df["time_gated_split"] == "TRAIN_TIME_GATED"]
    val_camps = audit_df[audit_df["time_gated_split"] == "VAL_TIME_GATED"]
    test_camps = audit_df[audit_df["time_gated_split"] == "TEST_TIME_GATED"]
    straddlers = audit_df[audit_df["time_gated_split"] == "STRADDLER_SEPARATE_EVAL"]
    censored = audit_df[audit_df["time_gated_split"] == "EXCLUDED_RIGHT_CENSORED"]

    assert len(train_camps) == 15
    assert len(val_camps) == 4
    assert len(test_camps) == 5
    assert len(straddlers) == 5
    assert len(censored) == 1

    # Invariant: Every training campaign outcome window completes by Day 500
    assert (train_camps["end_day"] <= 500).all()

    # Invariant: Every validation campaign starts on or after Day 501 and completes by Day 610
    assert (val_camps["start_day"] >= 501).all()
    assert (val_camps["end_day"] <= 610).all()

    # Invariant: Every test campaign starts on or after Day 611 and completes by Day 711
    assert (test_camps["start_day"] >= 611).all()
    assert (test_camps["end_day"] <= 711).all()

    # Absolute non-overlap assertion:
    max_train_outcome_day = train_camps["end_day"].max()
    min_val_treatment_day = val_camps["start_day"].min()
    assert max_train_outcome_day < min_val_treatment_day

    max_val_outcome_day = val_camps["end_day"].max()
    min_test_treatment_day = test_camps["start_day"].min()
    assert max_val_outcome_day < min_test_treatment_day


def test_causal_control_pool_exact_reconciliation(processed_dir):
    """Assert that for every campaign, treated + control_pool equals exactly 2,500 total households."""
    audit_path = processed_dir / "campaign_eligibility_audit.parquet"
    audit_df = pd.read_parquet(audit_path)

    for _, row in audit_df.iterrows():
        n_treated = row["treated_count"]
        n_control = row["eligible_control_pool_count"]
        n_clean = row["strict_clean_control_count"]
        n_exposed = row["concurrent_exposed_controls_count"]

        # Total population reconciliation
        assert n_treated + n_control == 2500

        # Control decomposition reconciliation
        assert n_clean + n_exposed == n_control
        assert n_clean <= n_control
        assert n_clean >= 0
