"""Campaign history and concurrent promotional exposure feature module.

Extracts:
1. Past campaign exposure and redemption affinity strictly prior to campaign start.
2. Concurrent campaign exposure active during the target campaign's treatment window.
3. Clean control status indicator.
"""

import pandas as pd


def compute_campaign_exposure_features(
    campaign_id: int,
    start_day: int,
    end_day: int,
    all_campaigns_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    redemptions_df: pd.DataFrame,
    all_household_keys: list[int],
) -> pd.DataFrame:
    """Compute past and concurrent campaign features for all households for a target campaign.

    Parameters
    ----------
    campaign_id : int
        Target campaign ID.
    start_day : int
        Target campaign start day.
    end_day : int
        Target campaign end day.
    all_campaigns_df : pd.DataFrame
        Table of all campaigns with start_day, end_day, campaign_type.
    assignments_df : pd.DataFrame
        Table of household_key, campaign_id, campaign_type.
    redemptions_df : pd.DataFrame
        Table of household_key, day, campaign_id.
    all_household_keys : list[int]
        Full population of 2,500 households.

    Returns
    -------
    pd.DataFrame
        Table with household_key and campaign-specific exposure features.
    """
    base_df = pd.DataFrame({"household_key": all_household_keys})

    # 1. Past Campaigns Assigned: strictly campaigns whose end_day < start_day
    past_camps = all_campaigns_df[all_campaigns_df["end_day"] < start_day]["campaign_id"]
    past_assigns = assignments_df[assignments_df["campaign_id"].isin(past_camps)]
    past_assign_counts = (
        past_assigns.groupby("household_key")["campaign_id"]
        .nunique()
        .reset_index()
        .rename(columns={"campaign_id": "hh_past_campaigns_assigned_count"})
    )

    # 2. Past Coupon Redemptions: strictly redemptions with day < start_day
    past_redemptions = redemptions_df[redemptions_df["day"] < start_day]
    past_redemption_counts = (
        past_redemptions.groupby("household_key")["day"]
        .count()
        .reset_index()
        .rename(columns={"day": "hh_past_coupons_redeemed_count"})
    )

    # 3. Concurrent Campaigns: other campaigns active during [start_day, end_day]
    concurrent_camps_df = all_campaigns_df[
        (all_campaigns_df["campaign_id"] != campaign_id)
        & ~((all_campaigns_df["end_day"] < start_day) | (all_campaigns_df["start_day"] > end_day))
    ]

    concurrent_ids = concurrent_camps_df["campaign_id"].tolist()
    concurrent_assigns = assignments_df[assignments_df["campaign_id"].isin(concurrent_ids)]

    concurrent_counts = (
        concurrent_assigns.groupby("household_key")["campaign_id"]
        .nunique()
        .reset_index()
        .rename(columns={"campaign_id": "hh_concurrent_campaigns_count"})
    )

    # Breakdown by campaign type of concurrent exposures
    type_exposures = (
        concurrent_assigns.groupby(["household_key", "campaign_type"])["campaign_id"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    type_cols_map = {}
    for col in type_exposures.columns:
        if col != "household_key":
            type_cols_map[col] = f"hh_concurrent_{col}_active"
    type_exposures = type_exposures.rename(columns=type_cols_map)

    # 4. Target Treatment Assignment
    target_treated_hh = set(
        assignments_df[assignments_df["campaign_id"] == campaign_id]["household_key"]
    )
    base_df["treatment_assigned"] = base_df["household_key"].isin(target_treated_hh).astype(int)

    # Merge all
    features = (
        base_df.merge(past_assign_counts, on="household_key", how="left")
        .merge(past_redemption_counts, on="household_key", how="left")
        .merge(concurrent_counts, on="household_key", how="left")
        .merge(type_exposures, on="household_key", how="left")
    )

    features["hh_past_campaigns_assigned_count"] = features[
        "hh_past_campaigns_assigned_count"
    ].fillna(0)
    features["hh_past_coupons_redeemed_count"] = features["hh_past_coupons_redeemed_count"].fillna(
        0
    )
    features["hh_concurrent_campaigns_count"] = features["hh_concurrent_campaigns_count"].fillna(0)

    for col in type_cols_map.values():
        if col in features.columns:
            features[col] = (features[col].fillna(0) > 0).astype(int)
        else:
            features[col] = 0

    # Ensure canonical types exist
    for c_type in ["TypeA", "TypeB", "TypeC"]:
        col_name = f"hh_concurrent_{c_type}_active"
        if col_name not in features.columns:
            features[col_name] = 0

    # Strict clean control indicator (T=0 and concurrent=0)
    features["is_strict_clean_control"] = (
        (features["treatment_assigned"] == 0) & (features["hh_concurrent_campaigns_count"] == 0)
    ).astype(int)

    features["campaign_id"] = campaign_id

    return features
