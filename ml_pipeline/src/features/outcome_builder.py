"""Outcome label generation module.

Constructs target outcome measures strictly during a campaign's active promotional window
[start_day, end_day], completely isolated from baseline predictor features.
"""

import pandas as pd


def compute_campaign_outcomes(
    campaign_id: int,
    start_day: int,
    end_day: int,
    transactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    coupons_df: pd.DataFrame,
    all_household_keys: list[int],
    target_treated_hh: set[int],
    clean_control_hh: set[int],
) -> pd.DataFrame:
    """Compute realized purchase outcomes during the target campaign window.

    Parameters
    ----------
    campaign_id : int
        Target campaign ID.
    start_day : int
        Campaign window start.
    end_day : int
        Campaign window end.
    transactions_df : pd.DataFrame
        Transactions dataset.
    products_df : pd.DataFrame
        Product master containing commodity.
    coupons_df : pd.DataFrame
        Coupons table.
    all_household_keys : list[int]
        Full 2,500 household population.
    target_treated_hh : set[int]
        Assigned household keys (T=1).
    clean_control_hh : set[int]
        Strict clean control keys.

    Returns
    -------
    pd.DataFrame
        Table with household_key, treatment_assigned, is_strict_clean_control, and realized outcome metrics.
    """
    base_df = pd.DataFrame({"household_key": all_household_keys})
    base_df["campaign_id"] = campaign_id
    base_df["treatment_assigned"] = base_df["household_key"].isin(target_treated_hh).astype(int)
    base_df["is_strict_clean_control"] = base_df["household_key"].isin(clean_control_hh).astype(int)

    # Window transactions (strictly start_day <= day <= end_day)
    window_txns = transactions_df[
        (transactions_df["day"] >= start_day)
        & (transactions_df["day"] <= end_day)
        & (transactions_df["row_classification"].isin(["valid_purchase", "promotional_freebie"]))
    ]

    if window_txns.empty:
        base_df["y_category_spend_window"] = 0.0
        base_df["y_category_units_window"] = 0.0
        base_df["y_total_spend_window"] = 0.0
        base_df["y_total_trips_window"] = 0
        return base_df

    # 1. Total store-wide spend & trips in window
    total_outcomes = (
        window_txns.groupby("household_key")
        .agg(
            y_total_spend_window=("sales_value", "sum"),
            y_total_trips_window=("basket_id", "nunique"),
        )
        .reset_index()
    )

    # 2. Promoted category spend & units in window
    camp_coupons = coupons_df[coupons_df["campaign_id"] == campaign_id]
    camp_prod_ids = set(camp_coupons["product_id"])
    target_commodities = set(
        products_df[products_df["product_id"].isin(camp_prod_ids)]["commodity"]
    )

    if target_commodities:
        if "commodity" not in window_txns.columns:
            window_txns = window_txns.merge(
                products_df[["product_id", "commodity"]], on="product_id", how="left"
            )
        cat_txns = window_txns[window_txns["commodity"].isin(target_commodities)]
        cat_outcomes = (
            cat_txns.groupby("household_key")
            .agg(
                y_category_spend_window=("sales_value", "sum"),
                y_category_units_window=("quantity", "sum"),
            )
            .reset_index()
        )
    else:
        cat_outcomes = pd.DataFrame(
            columns=["household_key", "y_category_spend_window", "y_category_units_window"]
        )

    outcomes = base_df.merge(total_outcomes, on="household_key", how="left").merge(
        cat_outcomes, on="household_key", how="left"
    )

    outcomes["y_total_spend_window"] = outcomes["y_total_spend_window"].fillna(0.0)
    outcomes["y_total_trips_window"] = outcomes["y_total_trips_window"].fillna(0)
    outcomes["y_category_spend_window"] = outcomes["y_category_spend_window"].fillna(0.0)
    outcomes["y_category_units_window"] = outcomes["y_category_units_window"].fillna(0.0)

    return outcomes
