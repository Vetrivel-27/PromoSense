"""Causal Uplift Evaluation Metrics Module.

Computes evaluation metrics specifically designed for uplift modeling
where individual counterfactual ground truth is unobservable:
1. Qini Curves and Qini Scores (Area between model Qini curve and random benchmark).
2. Cumulative Gain / Uplift Curves.
3. Area Under the Uplift Curve (AUUC).
4. Top-Decile Incremental Lift.
5. Targeting Decile Table with treated vs control outcome breakdowns.
"""

from typing import Any

import numpy as np
import pandas as pd


def compute_uplift_curve(
    y_true: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    uplift_pred: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Computes decile-by-decile cumulative gain and Qini values.

    Parameters
    ----------
    y_true : array-like
        Observed continuous or count outcomes (e.g. sales, units).
    treatment : array-like
        Binary treatment indicator T in {0, 1}.
    uplift_pred : array-like
        Predicted CATE tau(X).
    n_bins : int
        Number of quantile bins (default: 10 for deciles).

    Returns
    -------
    pd.DataFrame
        Table with cumulative population fractions, treated/control counts,
        mean outcomes, cumulative gain, and Qini values.
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=float),
            "t": np.asarray(treatment, dtype=int),
            "score": np.asarray(uplift_pred, dtype=float),
        }
    )

    # Sort descending by predicted uplift score (descending = highest predicted lift first)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    n_total = len(df)
    bin_size = n_total / n_bins

    records = []
    # Starting point (0 population)
    records.append(
        {
            "decile": 0,
            "population_fraction": 0.0,
            "n_cum": 0,
            "n_treated_cum": 0,
            "n_control_cum": 0,
            "y_treated_cum": 0.0,
            "y_control_cum": 0.0,
            "mean_treated_cum": 0.0,
            "mean_control_cum": 0.0,
            "cum_gain": 0.0,
            "qini_val": 0.0,
            "random_qini": 0.0,
        }
    )

    total_treated = int(df["t"].sum())
    total_control = n_total - total_treated
    total_y_treated = float(df[df["t"] == 1]["y"].sum())
    total_y_control = float(df[df["t"] == 0]["y"].sum())
    total_qini_end = (
        total_y_treated - (total_y_control * (total_treated / total_control))
        if total_control > 0
        else 0.0
    )

    for i in range(1, n_bins + 1):
        cutoff_idx = int(round(i * bin_size))
        subset = df.iloc[:cutoff_idx]

        n_cum = len(subset)
        pop_frac = n_cum / n_total

        t_sub = subset[subset["t"] == 1]
        c_sub = subset[subset["t"] == 0]

        n_t = len(t_sub)
        n_c = len(c_sub)

        y_t = float(t_sub["y"].sum())
        y_c = float(c_sub["y"].sum())

        mean_t = y_t / n_t if n_t > 0 else 0.0
        mean_c = y_c / n_c if n_c > 0 else 0.0

        # Cumulative gain: (mean_t - mean_c) * n_cum
        cum_gain = (mean_t - mean_c) * n_cum if (n_t > 0 and n_c > 0) else 0.0

        # Qini curve: y_t - (y_c * (n_t / n_c))
        qini_val = y_t - (y_c * (n_t / n_c)) if n_c > 0 else y_t

        # Random benchmark at this population fraction
        random_qini = total_qini_end * pop_frac

        records.append(
            {
                "decile": i,
                "population_fraction": pop_frac,
                "n_cum": n_cum,
                "n_treated_cum": n_t,
                "n_control_cum": n_c,
                "y_treated_cum": y_t,
                "y_control_cum": y_c,
                "mean_treated_cum": mean_t,
                "mean_control_cum": mean_c,
                "cum_gain": cum_gain,
                "qini_val": qini_val,
                "random_qini": random_qini,
            }
        )

    return pd.DataFrame(records)


def compute_qini_score(curve_df: pd.DataFrame) -> float:
    """Computes the Qini score: Area between model Qini curve and random baseline.

    Uses trapezoidal numerical integration.
    Positive value indicates outperformance over random targeting.
    """
    x = curve_df["population_fraction"].to_numpy(dtype=float)
    y_model = curve_df["qini_val"].to_numpy(dtype=float)
    y_rand = curve_df["random_qini"].to_numpy(dtype=float)

    area_model = float(np.trapezoid(y_model, x))
    area_rand = float(np.trapezoid(y_rand, x))

    return area_model - area_rand


def compute_auuc(curve_df: pd.DataFrame) -> float:
    """Computes Area Under the Uplift (Cumulative Gain) Curve."""
    x = curve_df["population_fraction"].to_numpy(dtype=float)
    y_gain = curve_df["cum_gain"].to_numpy(dtype=float)

    return float(np.trapezoid(y_gain, x))


def compute_top_decile_lift(
    y_true: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    uplift_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Computes incremental lift and treatment effect specifically in the top 10% predicted uplift decile.

    METHODOLOGICAL GUARDRAIL:
    In observational data with confounded treatment assignment, this represents an
    associational subgroup difference (mean(Y|T=1) - mean(Y|T=0)), NOT an unbiased
    causal treatment effect, because treated units in the top decile systematically
    differ in baseline engagement from untreated units in that same decile.
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=float),
            "t": np.asarray(treatment, dtype=int),
            "score": np.asarray(uplift_pred, dtype=float),
        }
    )
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    top_10_pct = df.iloc[: int(round(0.10 * len(df)))]

    t_group = top_10_pct[top_10_pct["t"] == 1]
    c_group = top_10_pct[top_10_pct["t"] == 0]

    mean_t = float(t_group["y"].mean()) if len(t_group) > 0 else 0.0
    mean_c = float(c_group["y"].mean()) if len(c_group) > 0 else 0.0
    lift = mean_t - mean_c

    # Overall ATE for comparison
    overall_mean_t = float(df[df["t"] == 1]["y"].mean())
    overall_mean_c = float(df[df["t"] == 0]["y"].mean())
    overall_ate = overall_mean_t - overall_mean_c

    lift_multiplier = (lift / overall_ate) if overall_ate > 0 else 1.0

    return {
        "top_decile_treated_mean": mean_t,
        "top_decile_control_mean": mean_c,
        "top_decile_lift": lift,
        "overall_cohort_ate": overall_ate,
        "top_decile_lift_multiplier": lift_multiplier,
    }


def evaluate_uplift_model(
    y_true: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    uplift_pred: pd.Series | np.ndarray,
    model_name: str = "model",
) -> dict[str, Any]:
    """Comprehensive evaluation package for an uplift model."""
    curve_df = compute_uplift_curve(y_true, treatment, uplift_pred, n_bins=10)
    qini_score = compute_qini_score(curve_df)
    auuc = compute_auuc(curve_df)
    top_decile = compute_top_decile_lift(y_true, treatment, uplift_pred)

    return {
        "model_name": model_name,
        "qini_score": qini_score,
        "auuc": auuc,
        "top_decile_lift": top_decile["top_decile_lift"],
        "overall_cohort_ate": top_decile["overall_cohort_ate"],
        "top_decile_multiplier": top_decile["top_decile_lift_multiplier"],
        "curve_df": curve_df,
    }
