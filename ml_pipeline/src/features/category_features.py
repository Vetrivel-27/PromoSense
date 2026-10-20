"""Category and product preference feature module.

Extracts household affinity and spending patterns in target campaign categories
strictly prior to campaign start.
"""

import pandas as pd


def compute_category_preference_features(
    campaign_id: int,
    cutoff_day: int,
    transactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    coupons_df: pd.DataFrame,
    all_household_keys: list[int],
) -> pd.DataFrame:
    """Compute household baseline engagement with the categories promoted in the campaign.

    Parameters
    ----------
    campaign_id : int
        Target campaign ID.
    cutoff_day : int
        Campaign start day.
    transactions_df : pd.DataFrame
        Historical transactions dataset.
    products_df : pd.DataFrame
        Product master containing commodity and department.
    coupons_df : pd.DataFrame
        Coupons table with product_id and campaign_id.
    all_household_keys : list[int]
        Full population of 2,500 households.

    Returns
    -------
    pd.DataFrame
        Table with household_key and category affinity features.
    """
    base_df = pd.DataFrame({"household_key": all_household_keys})

    # Find products and commodities associated with the target campaign
    camp_coupons = coupons_df[coupons_df["campaign_id"] == campaign_id]
    camp_prod_ids = set(camp_coupons["product_id"])

    # If campaign has coupons, find the primary commodities
    target_commodities = set(
        products_df[products_df["product_id"].isin(camp_prod_ids)]["commodity"]
    )

    # Historical transactions strictly prior to cutoff
    hist_txns = transactions_df[
        (transactions_df["day"] < cutoff_day)
        & (transactions_df["row_classification"].isin(["valid_purchase", "promotional_freebie"]))
    ]

    if hist_txns.empty or not target_commodities:
        base_df["hh_target_category_spend_90d"] = 0.0
        base_df["hh_target_category_units_90d"] = 0.0
        base_df["hh_target_category_trips_90d"] = 0
        base_df["hh_has_purchased_target_category_180d"] = 0
        return base_df

    # Merge commodity onto transactions if missing
    if "commodity" not in hist_txns.columns:
        hist_txns = hist_txns.merge(
            products_df[["product_id", "commodity"]], on="product_id", how="left"
        )

    # 1. 90-day target category activity
    txns_90d = hist_txns[
        (hist_txns["day"] >= (cutoff_day - 90)) & (hist_txns["commodity"].isin(target_commodities))
    ]

    f_90d = (
        txns_90d.groupby("household_key")
        .agg(
            hh_target_category_spend_90d=("sales_value", "sum"),
            hh_target_category_units_90d=("quantity", "sum"),
            hh_target_category_trips_90d=("basket_id", "nunique"),
        )
        .reset_index()
    )

    # 2. 180-day loyalty indicator
    txns_180d = hist_txns[
        (hist_txns["day"] >= (cutoff_day - 180)) & (hist_txns["commodity"].isin(target_commodities))
    ]
    hh_with_cat_180d = set(txns_180d["household_key"])

    features = base_df.merge(f_90d, on="household_key", how="left")
    features["hh_target_category_spend_90d"] = features["hh_target_category_spend_90d"].fillna(0.0)
    features["hh_target_category_units_90d"] = features["hh_target_category_units_90d"].fillna(0.0)
    features["hh_target_category_trips_90d"] = features["hh_target_category_trips_90d"].fillna(0)
    features["hh_has_purchased_target_category_180d"] = (
        features["household_key"].isin(hh_with_cat_180d).astype(int)
    )

    return features
