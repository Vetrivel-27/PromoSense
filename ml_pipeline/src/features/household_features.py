"""Household behavioral and sensitivity feature engineering module.

Extracts point-in-time customer predictors strictly prior to a target campaign's start date:
- Multi-horizon spending and trip frequencies (30d, 90d, 180d)
- Average basket value and basket units
- Price and discount sensitivity (discount intensity, coupon redemption rate, promo share)
- Private label brand preference
"""

import numpy as np
import pandas as pd


def compute_household_features_at_cutoff(
    transactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    all_household_keys: list[int],
    cutoff_day: int,
) -> pd.DataFrame:
    """Compute point-in-time behavioral features for all households.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Full transactions dataset.
    products_df : pd.DataFrame
        Product master containing brand_type and departments.
    all_household_keys : list[int]
        List of all 2,500 active households to guarantee 100% population coverage.
    cutoff_day : int
        Campaign start day. Only transactions strictly with day < cutoff_day are used.

    Returns
    -------
    pd.DataFrame
        Indexed by household_key with all engineered features.
    """
    # Strict temporal barrier: zero future or contemporaneous leakage
    historical_txns = transactions_df[
        (transactions_df["day"] < cutoff_day)
        & (transactions_df["row_classification"].isin(["valid_purchase", "promotional_freebie"]))
    ].copy()

    # Pre-merge product brand type for private-label calculation
    if "brand_type" not in historical_txns.columns:
        prod_brands = products_df[["product_id", "brand_type"]]
        historical_txns = historical_txns.merge(prod_brands, on="product_id", how="left")

    base_hh_df = pd.DataFrame({"household_key": all_household_keys})

    if historical_txns.empty:
        # If no history exists prior to cutoff (e.g. Day 1), return initialized zeros
        return _initialize_empty_household_features(base_hh_df)

    # 1. 30-day Horizon
    txns_30d = historical_txns[historical_txns["day"] >= (cutoff_day - 30)]
    f_30d = (
        txns_30d.groupby("household_key")
        .agg(
            hh_spend_30d=("sales_value", "sum"),
            hh_trips_30d=("basket_id", "nunique"),
        )
        .reset_index()
    )

    # 2. 90-day Horizon (Primary behavioral window)
    txns_90d = historical_txns[historical_txns["day"] >= (cutoff_day - 90)]

    # Basket level aggregation for basket statistics
    basket_90d = (
        txns_90d.groupby(["household_key", "basket_id"])
        .agg(
            basket_spend=("sales_value", "sum"),
            basket_units=("quantity", "sum"),
        )
        .reset_index()
    )
    basket_stats_90d = (
        basket_90d.groupby("household_key")
        .agg(
            hh_avg_basket_value_90d=("basket_spend", "mean"),
            hh_avg_basket_units_90d=("basket_units", "mean"),
        )
        .reset_index()
    )

    # Line item level aggregation
    line_stats_90d = (
        txns_90d.groupby("household_key")
        .agg(
            hh_spend_90d=("sales_value", "sum"),
            hh_units_90d=("quantity", "sum"),
            hh_trips_90d=("basket_id", "nunique"),
            hh_unique_products_90d=("product_id", "nunique"),
            hh_total_retail_disc_90d=("retail_disc", lambda x: float(np.abs(x).sum())),
            hh_coupon_used_count_90d=("coupon_used", "sum"),
            hh_promo_item_count_90d=("in_store_promoted", "sum"),
            hh_total_line_items_90d=("product_id", "count"),
        )
        .reset_index()
    )

    # Private label spend in 90d
    private_spend_90d = (
        txns_90d[txns_90d["brand_type"] == "Private"]
        .groupby("household_key")["sales_value"]
        .sum()
        .reset_index()
        .rename(columns={"sales_value": "hh_private_label_spend_90d"})
    )

    # 3. 180-day Horizon (Long-term baseline)
    txns_180d = historical_txns[historical_txns["day"] >= (cutoff_day - 180)]
    f_180d = (
        txns_180d.groupby("household_key")
        .agg(
            hh_spend_180d=("sales_value", "sum"),
            hh_trips_180d=("basket_id", "nunique"),
        )
        .reset_index()
    )

    # Merge all horizons onto base population
    features = (
        base_hh_df.merge(f_30d, on="household_key", how="left")
        .merge(line_stats_90d, on="household_key", how="left")
        .merge(basket_stats_90d, on="household_key", how="left")
        .merge(private_spend_90d, on="household_key", how="left")
        .merge(f_180d, on="household_key", how="left")
    )

    # Fill inactivity nulls
    features["hh_spend_30d"] = features["hh_spend_30d"].fillna(0.0)
    features["hh_trips_30d"] = features["hh_trips_30d"].fillna(0)
    features["hh_spend_90d"] = features["hh_spend_90d"].fillna(0.0)
    features["hh_units_90d"] = features["hh_units_90d"].fillna(0.0)
    features["hh_trips_90d"] = features["hh_trips_90d"].fillna(0)
    features["hh_spend_180d"] = features["hh_spend_180d"].fillna(0.0)
    features["hh_trips_180d"] = features["hh_trips_180d"].fillna(0)
    features["hh_unique_products_90d"] = features["hh_unique_products_90d"].fillna(0)
    features["hh_avg_basket_value_90d"] = features["hh_avg_basket_value_90d"].fillna(0.0)
    features["hh_avg_basket_units_90d"] = features["hh_avg_basket_units_90d"].fillna(0.0)

    # Derived Sensitivity Ratios
    # Discount intensity: retail_disc / gross_sales
    gross_spend_90d = features["hh_spend_90d"] + features["hh_total_retail_disc_90d"].fillna(0.0)
    features["hh_discount_intensity_90d"] = np.where(
        gross_spend_90d > 0,
        features["hh_total_retail_disc_90d"].fillna(0.0) / gross_spend_90d,
        0.0,
    )

    # Coupon usage rate (per line item)
    lines_90d = features["hh_total_line_items_90d"].fillna(0)
    features["hh_coupon_usage_rate_90d"] = np.where(
        lines_90d > 0,
        features["hh_coupon_used_count_90d"].fillna(0.0) / lines_90d,
        0.0,
    )

    # Promotional item share
    features["hh_promotional_item_share_90d"] = np.where(
        lines_90d > 0,
        features["hh_promo_item_count_90d"].fillna(0.0) / lines_90d,
        0.0,
    )

    # Private label share
    features["hh_private_label_share_90d"] = np.where(
        features["hh_spend_90d"] > 0,
        features["hh_private_label_spend_90d"].fillna(0.0) / features["hh_spend_90d"],
        0.0,
    )

    # Drop intermediate raw sum columns
    cols_to_drop = [
        "hh_total_retail_disc_90d",
        "hh_coupon_used_count_90d",
        "hh_promo_item_count_90d",
        "hh_total_line_items_90d",
        "hh_private_label_spend_90d",
    ]
    features = features.drop(columns=cols_to_drop)

    return features


def _initialize_empty_household_features(base_hh_df: pd.DataFrame) -> pd.DataFrame:
    features = base_hh_df.copy()
    num_cols = [
        "hh_spend_30d",
        "hh_trips_30d",
        "hh_spend_90d",
        "hh_units_90d",
        "hh_trips_90d",
        "hh_unique_products_90d",
        "hh_avg_basket_value_90d",
        "hh_avg_basket_units_90d",
        "hh_spend_180d",
        "hh_trips_180d",
        "hh_discount_intensity_90d",
        "hh_coupon_usage_rate_90d",
        "hh_promotional_item_share_90d",
        "hh_private_label_share_90d",
    ]
    for col in num_cols:
        features[col] = 0.0
    return features
