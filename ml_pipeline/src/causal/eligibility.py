"""Causal treatment & control eligibility module.

Calculates campaign-level treatment and control populations under both:
1. Strict Clean Controls: subset of unassigned households receiving ZERO campaigns during the window.
2. Concurrent-Adjusted Controls: full unassigned cohort with concurrent campaign exposure
   explicitly controlled in the covariate vector.

Also implements the Time-Gated Split Protocol:
- TRAIN_TIME_GATED (15 campaigns, end_day <= 500)
- VAL_TIME_GATED (4 campaigns, start_day >= 501 and end_day <= 610)
- TEST_TIME_GATED (5 campaigns, start_day >= 611 and end_day <= 711)
- STRADDLER (5 campaigns: 11, 12, 15, 18, 19 - cross split boundaries)
- EXCLUDED_RIGHT_CENSORED (1 campaign: 24 - observed duration truncated to 53 days at Day 711)

Important Methodological Guardrail:
Observational adjustment (IPW/propensity) balances OBSERVED covariates and concurrent exposures,
but does NOT guarantee elimination of unmeasured confounding.
"""

from pathlib import Path

import pandas as pd

from ml_pipeline.src.config.settings import ml_settings


def audit_campaign_eligibility(
    data_dir: Path = ml_settings.INTERIM_DATA_DIR,
) -> pd.DataFrame:
    camp_desc = pd.read_parquet(data_dir / "campaigns.parquet")
    camp_table = pd.read_parquet(data_dir / "campaign_assignments.parquet")
    households = pd.read_parquet(data_dir / "households.parquet")

    all_hh_keys = set(households["household_key"])
    total_analysis_population_size = len(all_hh_keys)

    # Sort campaigns chronologically by start_day, then end_day
    camp_desc = camp_desc.sort_values(["start_day", "end_day"]).reset_index(drop=True)
    camp_desc["chronological_rank"] = range(1, len(camp_desc) + 1)

    records = []

    for _, c in camp_desc.iterrows():
        c_id = int(c["campaign_id"])
        c_type = c["campaign_type"]
        s_day = int(c["start_day"])
        e_day = int(c["end_day"])
        s_week = int(c["start_week"])
        e_week = int(c["end_week"])
        duration = int(c["duration_days"])
        obs_end = int(c["observed_end_day"])
        obs_duration = int(c["observed_duration_days"])
        is_censored = bool(c["is_right_censored"])

        # 1. Treated cohort (T = 1)
        treated_hh = set(camp_table[camp_table["campaign_id"] == c_id]["household_key"])
        n_treated = len(treated_hh)

        # 2. Eligible control pool (T = 0)
        control_pool_hh = all_hh_keys - treated_hh
        n_control_pool = len(control_pool_hh)

        # 3. Identify overlapping campaigns running during [s_day, e_day]
        overlapping_camps = []
        for _, other_c in camp_desc.iterrows():
            other_id = int(other_c["campaign_id"])
            if other_id == c_id:
                continue
            o_s = int(other_c["start_day"])
            o_e = int(other_c["end_day"])
            if not (e_day < o_s or s_day > o_e):
                overlapping_camps.append(other_id)

        # 4. Partition the control pool into Strict Clean vs Concurrent-Exposed
        concurrently_exposed_hh = set(
            camp_table[
                (camp_table["campaign_id"].isin(overlapping_camps))
                & (camp_table["household_key"].isin(control_pool_hh))
            ]["household_key"]
        )
        n_concurrent_exposed = len(concurrently_exposed_hh)

        clean_controls_hh = control_pool_hh - concurrently_exposed_hh
        n_strict_clean = len(clean_controls_hh)

        # 5. Time-Gated Split Protocol (Leak-Free Windows)
        if is_censored:
            eval_protocol = "EXCLUDED_RIGHT_CENSORED"
        elif e_day <= 500:
            eval_protocol = "TRAIN_TIME_GATED"
        elif s_day >= 501 and e_day <= 610:
            eval_protocol = "VAL_TIME_GATED"
        elif s_day >= 611 and e_day <= 711:
            eval_protocol = "TEST_TIME_GATED"
        else:
            eval_protocol = "STRADDLER_SEPARATE_EVAL"

        records.append(
            {
                "chronological_rank": c["chronological_rank"],
                "campaign_id": c_id,
                "campaign_type": c_type,
                "start_day": s_day,
                "end_day": e_day,
                "start_week": s_week,
                "end_week": e_week,
                "duration_days": duration,
                "observed_end_day": obs_end,
                "observed_duration_days": obs_duration,
                "is_right_censored": is_censored,
                "total_analysis_population": total_analysis_population_size,
                "treated_count": n_treated,
                "eligible_control_pool_count": n_control_pool,
                "concurrent_overlapping_campaigns_count": len(overlapping_camps),
                "concurrent_exposed_controls_count": n_concurrent_exposed,
                "strict_clean_control_count": n_strict_clean,
                "control_contamination_pct": round(n_concurrent_exposed / n_control_pool * 100, 2),
                "time_gated_split": eval_protocol,
            }
        )

    audit_df = pd.DataFrame(records)
    out_path = ml_settings.PROCESSED_DATA_DIR / "campaign_eligibility_audit.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_parquet(out_path, index=False)
    print(f"Campaign eligibility audit saved to {out_path} ({len(audit_df)} campaigns)")
    return audit_df


if __name__ == "__main__":
    df = audit_campaign_eligibility()
    print("\n--- CAMPAIGN ELIGIBILITY & TIME-GATED EVALUATION SPLIT ---")
    cols = [
        "chronological_rank",
        "campaign_id",
        "campaign_type",
        "start_day",
        "end_day",
        "duration_days",
        "observed_duration_days",
        "treated_count",
        "eligible_control_pool_count",
        "strict_clean_control_count",
        "control_contamination_pct",
        "time_gated_split",
    ]
    print(df[cols].to_string(index=False))
