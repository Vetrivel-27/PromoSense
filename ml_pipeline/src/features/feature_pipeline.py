"""Feature Engineering Pipeline Orchestrator.

Orchestrates point-in-time feature generation and separate outcome label generation
for all benchmark evaluation splits:
- TRAIN_TIME_GATED (15 campaigns)
- VAL_TIME_GATED (4 campaigns)
- TEST_TIME_GATED (5 campaigns)
- STRADDLER_SEPARATE_EVAL (5 campaigns)

Guarantees 100% temporal isolation:
- All predictors are computed strictly at Cutoff = start_day - 1.
- Outcome labels are stored in separate datasets.
"""

from pathlib import Path

import pandas as pd

from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.features.campaign_features import compute_campaign_exposure_features
from ml_pipeline.src.features.category_features import compute_category_preference_features
from ml_pipeline.src.features.household_features import compute_household_features_at_cutoff
from ml_pipeline.src.features.outcome_builder import compute_campaign_outcomes
from ml_pipeline.src.features.temporal_features import compute_temporal_features


class FeaturePipeline:
    """Orchestrates end-to-end point-in-time feature extraction."""

    def __init__(
        self,
        interim_dir: Path = ml_settings.INTERIM_DATA_DIR,
        processed_dir: Path = ml_settings.PROCESSED_DATA_DIR,
    ):
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("Loading interim datasets for Feature Pipeline...")
        txns = pd.read_parquet(self.interim_dir / "transactions.parquet")
        prods = pd.read_parquet(self.interim_dir / "products.parquet")
        camps = pd.read_parquet(self.interim_dir / "campaigns.parquet")
        assigns = pd.read_parquet(self.interim_dir / "campaign_assignments.parquet")
        coupons = pd.read_parquet(self.interim_dir / "coupons.parquet").drop_duplicates(
            subset=["coupon_upc", "product_id", "campaign_id"]
        )
        redemptions = pd.read_parquet(self.interim_dir / "coupon_redemptions.parquet")
        households = pd.read_parquet(self.interim_dir / "households.parquet")
        audit_df = pd.read_parquet(self.processed_dir / "campaign_eligibility_audit.parquet")

        all_hh_keys = sorted(households["household_key"].tolist())
        print(
            f"Loaded {len(all_hh_keys)} households, {len(camps)} campaigns, {len(txns)} transactions."
        )

        splits = {
            "train": audit_df[audit_df["time_gated_split"] == "TRAIN_TIME_GATED"],
            "val": audit_df[audit_df["time_gated_split"] == "VAL_TIME_GATED"],
            "test": audit_df[audit_df["time_gated_split"] == "TEST_TIME_GATED"],
            "straddler": audit_df[audit_df["time_gated_split"] == "STRADDLER_SEPARATE_EVAL"],
        }

        for split_name, split_camps in splits.items():
            print(f"\nProcessing {split_name.upper()} split ({len(split_camps)} campaigns)...")
            split_features_list = []
            split_outcomes_list = []

            for _, camp_row in split_camps.iterrows():
                c_id = int(camp_row["campaign_id"])
                s_day = int(camp_row["start_day"])
                e_day = int(camp_row["end_day"])
                print(
                    f"  Campaign {c_id:>2} (Days {s_day}..{e_day}): extracting features at cutoff Day {s_day - 1}..."
                )

                # 1. Household historical predictors (t < s_day)
                hh_feats = compute_household_features_at_cutoff(
                    transactions_df=txns,
                    products_df=prods,
                    all_household_keys=all_hh_keys,
                    cutoff_day=s_day,
                )

                # 2. Category preference predictors (t < s_day)
                cat_feats = compute_category_preference_features(
                    campaign_id=c_id,
                    cutoff_day=s_day,
                    transactions_df=txns,
                    products_df=prods,
                    coupons_df=coupons,
                    all_household_keys=all_hh_keys,
                )

                # 3. Campaign exposure & concurrent exposure
                camp_feats = compute_campaign_exposure_features(
                    campaign_id=c_id,
                    start_day=s_day,
                    end_day=e_day,
                    all_campaigns_df=camps,
                    assignments_df=assigns,
                    redemptions_df=redemptions,
                    all_household_keys=all_hh_keys,
                )

                # 4. Temporal & calendar predictors
                temp_dict = compute_temporal_features(camp_row)

                # Merge all predictors for this campaign
                merged_feats = hh_feats.merge(cat_feats, on="household_key").merge(
                    camp_feats, on="household_key"
                )
                for k, v in temp_dict.items():
                    merged_feats[k] = v

                # Merge household demographics if available
                demo_cols = [
                    "household_key",
                    "age_band",
                    "marital_status",
                    "income_band",
                    "homeowner_status",
                    "household_comp",
                    "household_size",
                    "kid_category",
                    "has_demographics",
                ]
                merged_feats = merged_feats.merge(
                    households[demo_cols], on="household_key", how="left"
                )

                split_features_list.append(merged_feats)

                # 5. Build detached outcome labels
                treated_hh = set(assigns[assigns["campaign_id"] == c_id]["household_key"])
                clean_controls = set(
                    camp_feats[camp_feats["is_strict_clean_control"] == 1]["household_key"]
                )
                camp_outcomes = compute_campaign_outcomes(
                    campaign_id=c_id,
                    start_day=s_day,
                    end_day=e_day,
                    transactions_df=txns,
                    products_df=prods,
                    coupons_df=coupons,
                    all_household_keys=all_hh_keys,
                    target_treated_hh=treated_hh,
                    clean_control_hh=clean_controls,
                )
                split_outcomes_list.append(camp_outcomes)

            # Combine and save split datasets
            full_features_df = pd.concat(split_features_list, ignore_index=True)
            full_outcomes_df = pd.concat(split_outcomes_list, ignore_index=True)

            feat_out_path = self.processed_dir / f"{split_name}_features.parquet"
            outc_out_path = self.processed_dir / f"{split_name}_outcomes.parquet"

            full_features_df.to_parquet(feat_out_path, index=False)
            full_outcomes_df.to_parquet(outc_out_path, index=False)

            print(f"Saved {split_name.upper()}:")
            print(
                f"  Features: {feat_out_path} ({len(full_features_df):,} rows, {len(full_features_df.columns)} columns)"
            )
            print(
                f"  Outcomes: {outc_out_path} ({len(full_outcomes_df):,} rows, {len(full_outcomes_df.columns)} columns)"
            )


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
