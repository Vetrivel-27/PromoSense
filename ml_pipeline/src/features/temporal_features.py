"""Temporal and calendar feature engineering module.

Generates leakage-free temporal predictors based on the empirical retailer fiscal calendar:
- Scheduled campaign duration
- Start week and harmonic annual seasonality components (sin / cos)
- Fiscal quarter indicators
- Overall timeline trend
"""

import numpy as np
import pandas as pd


def compute_temporal_features(campaign_row: pd.Series) -> dict[str, float | int]:
    """Compute temporal predictors for a campaign given its schedule.

    Parameters
    ----------
    campaign_row : pd.Series
        Must contain 'start_day', 'end_day', 'start_week', 'duration_days'.

    Returns
    -------
    dict
        Dictionary of temporal feature names and values.
    """
    start_week = int(campaign_row["start_week"])
    duration_days = int(campaign_row["duration_days"])
    start_day = int(campaign_row["start_day"])

    # Harmonic annual seasonality (52-week cycle)
    # Week 1..52 cycle
    week_in_year = ((start_week - 1) % 52) + 1
    rad = 2.0 * np.pi * week_in_year / 52.0
    seasonality_sin = float(np.sin(rad))
    seasonality_cos = float(np.cos(rad))

    # Fiscal Quarter (13 weeks per quarter)
    fiscal_quarter = int(((week_in_year - 1) // 13) + 1)

    return {
        "campaign_start_day": start_day,
        "campaign_duration_days": duration_days,
        "campaign_start_week": start_week,
        "seasonality_sin_52w": seasonality_sin,
        "seasonality_cos_52w": seasonality_cos,
        "fiscal_quarter": fiscal_quarter,
        "timeline_day_trend": start_day / 711.0,  # Normalized trend (0 to 1 over study)
    }
