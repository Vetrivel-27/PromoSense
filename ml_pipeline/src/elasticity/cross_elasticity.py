"""Cross-Price Elasticity and Counterfactual Cannibalization Modeling.

Methodological Guardrails:
1. Two-Product Econometric Formulation:
   Joint estimation of competing products within identical sub_commodities.
   Controls for display placement, mailer circulars, time trends, and seasonality.
2. Counterfactual Cannibalization Accounting:
   Directly computes counterfactual revenue changes:
     Delta R_A = R_A(Promo) - R_A(Base)
     Delta R_B = R_B(Promo on A) - R_B(Base)
3. Explicit Ratio Definitions and Guards:
   - Revenue-Offset Ratio: -Delta R_B / Delta R_A (fraction of Product A's incremental revenue offset by lost revenue on Product B).
   - Unit Substitution Ratio: -Delta Q_B / Delta Q_A (fraction of Product A's volume gain drawn from Product B).
   - Handles non-positive or poorly supported denominators explicitly.
4. Causal Distinction:
   Labeled as model-implied observational substitution, distinguishing observed cross-price correlations from causal market expansion.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class CrossPriceElasticityModel:
    """Joint Log-Log OLS for Substitute Product Pairs."""

    def __init__(self, include_controls: bool = True) -> None:
        self.include_controls = include_controls
        self.pair_info: dict[str, Any] = {}
        self.model_A: dict[str, Any] = {}
        self.model_B: dict[str, Any] = {}

    def _fit_single_equation(
        self,
        y: np.ndarray,
        log_p_own: np.ndarray,
        log_p_cross: np.ndarray,
        df_controls: pd.DataFrame,
        suffix: str = "",
    ) -> dict[str, Any]:
        """Fits single OLS equation for one product."""
        feature_names = ["intercept", "log_own_price", "log_cross_price"]
        cols = [np.ones(len(y)), log_p_own, log_p_cross]

        if self.include_controls:
            control_cols = [
                "display_share",
                "mailer_share",
                "week_trend",
                "sin_52w",
                "cos_52w",
                "sin_26w",
                "cos_26w",
            ]
            for c in control_cols:
                suffixed = f"{c}_{suffix}" if suffix else c
                actual_col = suffixed if suffixed in df_controls.columns else c
                feature_names.append(c)
                cols.append(df_controls[actual_col].to_numpy(dtype=float))

        X = np.column_stack(cols)
        n, k = X.shape

        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        df_e = max(n - k, 1)
        resids = y - X @ beta
        sigma2 = np.sum(resids**2) / df_e

        inv_xtx = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diagonal(sigma2 * inv_xtx), 1e-12))
        t_stat = beta / se
        p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_e))

        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(resids**2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "beta": beta,
            "se": se,
            "t_stat": t_stat,
            "p_val": p_val,
            "feature_names": feature_names,
            "own_elasticity": float(beta[1]),
            "own_se": float(se[1]),
            "cross_elasticity": float(beta[2]),
            "cross_se": float(se[2]),
            "r2": round(r2, 4),
        }

    def fit(
        self,
        panel_df: pd.DataFrame,
        prod_A_id: int,
        prod_B_id: int,
    ) -> "CrossPriceElasticityModel":
        """Fits joint cross-price model for substitute pair A and B."""
        df_A = panel_df[panel_df["product_id"] == prod_A_id].copy()
        df_B = panel_df[panel_df["product_id"] == prod_B_id].copy()

        # Merge by week_no
        merged = df_A.merge(
            df_B,
            on="week_no",
            suffixes=("_A", "_B"),
            how="inner",
        ).sort_values("week_no")

        if len(merged) < 20:
            raise ValueError(
                f"Insufficient overlapping weeks ({len(merged)}) between {prod_A_id} and {prod_B_id}"
            )

        self.pair_info = {
            "prod_A_id": prod_A_id,
            "prod_B_id": prod_B_id,
            "overlapping_weeks": len(merged),
        }

        # Equation A: log(Q_A) on log(P_A), log(P_B), controls
        y_A = merged["log_quantity_A"].to_numpy(dtype=float)
        p_A = merged["log_paid_price_A"].to_numpy(dtype=float)
        p_B = merged["log_paid_price_B"].to_numpy(dtype=float)
        self.model_A = self._fit_single_equation(y_A, p_A, p_B, merged, suffix="A")

        # Equation B: log(Q_B) on log(P_B), log(P_A), controls
        y_B = merged["log_quantity_B"].to_numpy(dtype=float)
        self.model_B = self._fit_single_equation(y_B, p_B, p_A, merged, suffix="B")

        return self

    def predict_pair(
        self,
        log_price_A: np.ndarray,
        log_price_B: np.ndarray,
        df_controls: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predicts log quantities for pair A and B given price paths and controls."""
        if not self.model_A or not self.model_B:
            raise ValueError("Model must be fitted before predict.")

        # Matrix for A
        cols_A = [np.ones(len(log_price_A)), log_price_A, log_price_B]
        if self.include_controls:
            for c in [
                "display_share",
                "mailer_share",
                "week_trend",
                "sin_52w",
                "cos_52w",
                "sin_26w",
                "cos_26w",
            ]:
                col_name = f"{c}_A" if f"{c}_A" in df_controls.columns else c
                cols_A.append(df_controls[col_name].to_numpy(dtype=float))
        X_A = np.column_stack(cols_A)
        pred_log_qA = X_A @ self.model_A["beta"]

        # Matrix for B (own price is B, cross is A)
        cols_B = [np.ones(len(log_price_B)), log_price_B, log_price_A]
        if self.include_controls:
            for c in [
                "display_share",
                "mailer_share",
                "week_trend",
                "sin_52w",
                "cos_52w",
                "sin_26w",
                "cos_26w",
            ]:
                col_name = f"{c}_B" if f"{c}_B" in df_controls.columns else c
                cols_B.append(df_controls[col_name].to_numpy(dtype=float))
        X_B = np.column_stack(cols_B)
        pred_log_qB = X_B @ self.model_B["beta"]

        return pred_log_qA, pred_log_qB


def simulate_counterfactual_cannibalization(
    pair_model: CrossPriceElasticityModel,
    eval_df_merged: pd.DataFrame,
    discount_pct: float = 0.20,
    with_display: bool = True,
    baseline_mode: str = "zero_merch",
) -> dict[str, Any]:
    """Simulates counterfactual promotion scenario and computes cannibalization metrics.

    Scenarios & Baselines:
    - baseline_mode="zero_merch": Extrapolative bound comparing zero-display baseline to with_display promo.
    - baseline_mode="observed_merch": Realistic status quo preserving observed historical merchandising,
      isolating pure price cuts or incremental display shifts.
    """
    n_weeks = len(eval_df_merged)
    if n_weeks == 0:
        raise ValueError("Evaluation dataframe has 0 observations.")

    # 1. Baseline Counterfactual (No promotion on either SKU)
    pA_shelf = eval_df_merged["shelf_price_A"].to_numpy(dtype=float)
    pB_shelf = eval_df_merged["shelf_price_B"].to_numpy(dtype=float)
    log_pA_base = np.log(np.maximum(pA_shelf, 1e-4))
    log_pB_base = np.log(np.maximum(pB_shelf, 1e-4))

    base_controls = eval_df_merged.copy()
    if baseline_mode == "zero_merch":
        base_controls["display_share"] = 0.0
        base_controls["display_share_A"] = 0.0
        base_controls["display_share_B"] = 0.0
        base_controls["mailer_share"] = 0.0
        base_controls["mailer_share_A"] = 0.0
        base_controls["mailer_share_B"] = 0.0
    elif baseline_mode == "observed_merch":
        # Retain observed historical merchandising in baseline
        pass
    else:
        raise ValueError(
            f"Unknown baseline_mode: {baseline_mode}. Use 'zero_merch' or 'observed_merch'."
        )

    pred_log_qA_base, pred_log_qB_base = pair_model.predict_pair(
        log_pA_base, log_pB_base, base_controls
    )
    qA_base = np.exp(pred_log_qA_base)
    qB_base = np.exp(pred_log_qB_base)

    rev_A_base = float(np.sum(qA_base * pA_shelf))
    rev_B_base = float(np.sum(qB_base * pB_shelf))
    tot_qty_A_base = float(np.sum(qA_base))
    tot_qty_B_base = float(np.sum(qB_base))

    # 2. Promoted Scenario (Product A discounted with display, Product B unpromoted)
    pA_promo = pA_shelf * (1.0 - discount_pct)
    log_pA_promo = np.log(np.maximum(pA_promo, 1e-4))

    promo_controls = eval_df_merged.copy()
    if baseline_mode == "zero_merch":
        promo_controls["display_share"] = 1.0 if with_display else 0.0
        promo_controls["display_share_A"] = 1.0 if with_display else 0.0
        promo_controls["display_share_B"] = 0.0
        promo_controls["mailer_share"] = 0.0
        promo_controls["mailer_share_A"] = 0.0
        promo_controls["mailer_share_B"] = 0.0
    else:
        if with_display:
            promo_controls["display_share"] = 1.0
            promo_controls["display_share_A"] = 1.0

    pred_log_qA_promo, pred_log_qB_promo = pair_model.predict_pair(
        log_pA_promo, log_pB_base, promo_controls
    )
    qA_promo = np.exp(pred_log_qA_promo)
    qB_promo = np.exp(pred_log_qB_promo)

    rev_A_promo = float(np.sum(qA_promo * pA_promo))
    rev_B_promo = float(np.sum(qB_promo * pB_shelf))
    tot_qty_A_promo = float(np.sum(qA_promo))
    tot_qty_B_promo = float(np.sum(qB_promo))

    # 3. Incremental Changes
    delta_qty_A = tot_qty_A_promo - tot_qty_A_base
    delta_qty_B = tot_qty_B_promo - tot_qty_B_base
    delta_rev_A = rev_A_promo - rev_A_base
    delta_rev_B = rev_B_promo - rev_B_base

    # 4. Cannibalization & Substitution Ratios with Explicit Denominator Guards
    # Revenue-Offset Ratio: -Delta R_B / Delta R_A (fraction of Product A's gain offset by B's loss)
    if delta_rev_A > 1e-4:
        # If delta_rev_B is negative, product B lost revenue (cannibalization occurred)
        rev_offset_ratio = max(-delta_rev_B / delta_rev_A, 0.0)
        has_valid_rev_denominator = True
    else:
        rev_offset_ratio = None
        has_valid_rev_denominator = False

    # Unit Substitution Ratio: -Delta Q_B / Delta Q_A
    if delta_qty_A > 1e-4:
        unit_sub_ratio = max(-delta_qty_B / delta_qty_A, 0.0)
        has_valid_qty_denominator = True
    else:
        unit_sub_ratio = None
        has_valid_qty_denominator = False

    net_category_rev_lift = delta_rev_A + delta_rev_B

    # Empirical Support Checks & Methodological Warning Flags
    is_extrapolative_merch = bool(baseline_mode == "zero_merch" and with_display)
    merch_warning = (
        "Extrapolative merchandising shift: zero-display baseline compared to 100% display. "
        "Display is an observational co-factor, not a proven causal lever; volume multiplier "
        "reflects joint model-implied extrapolation."
        if is_extrapolative_merch
        else None
    )
    discount_supported = bool(0.0 <= discount_pct <= 0.40)
    discount_warning = (
        f"Discount percentage {discount_pct:.1%} exceeds historical promotional support (<= 40%)."
        if not discount_supported
        else None
    )

    # Check cross-price identification strength
    cross_se_A = float(pair_model.model_A.get("cross_se", 0.0))
    p_vals_A = pair_model.model_A.get("p_val", [])
    cross_pval_A = float(p_vals_A[2]) if len(p_vals_A) > 2 else 0.0
    cross_identified = bool(cross_se_A < 1.5 and cross_pval_A < 0.20)
    cross_id_warning = (
        "Weakly identified cross-price elasticity term (large standard error or p >= 0.20). "
        "Counterfactual cannibalization is an illustrative model-implied scenario tracing, "
        "NOT empirical proof of verified substitution."
        if not cross_identified
        else None
    )

    return {
        "scenario": {
            "discount_pct": discount_pct,
            "with_display": with_display,
            "baseline_mode": baseline_mode,
            "evaluated_weeks": n_weeks,
            "empirical_support": {
                "is_extrapolative_merchandising": is_extrapolative_merch,
                "merchandising_support_warning": merch_warning,
                "discount_pct_supported": discount_supported,
                "discount_warning": discount_warning,
                "cross_price_well_identified": cross_identified,
                "cross_identification_warning": cross_id_warning,
            },
        },
        "baseline_counterfactual": {
            "qty_A": round(tot_qty_A_base, 2),
            "qty_B": round(tot_qty_B_base, 2),
            "revenue_A": round(rev_A_base, 2),
            "revenue_B": round(rev_B_base, 2),
            "category_revenue": round(rev_A_base + rev_B_base, 2),
        },
        "promoted_scenario": {
            "qty_A": round(tot_qty_A_promo, 2),
            "qty_B": round(tot_qty_B_promo, 2),
            "revenue_A": round(rev_A_promo, 2),
            "revenue_B": round(rev_B_promo, 2),
            "category_revenue": round(rev_A_promo + rev_B_promo, 2),
        },
        "counterfactual_differences": {
            "delta_qty_A": round(delta_qty_A, 2),
            "delta_qty_B": round(delta_qty_B, 2),
            "delta_revenue_A": round(delta_rev_A, 2),
            "delta_revenue_B": round(delta_rev_B, 2),
            "net_category_revenue_lift": round(net_category_rev_lift, 2),
        },
        "cannibalization_metrics": {
            "revenue_offset_ratio": round(rev_offset_ratio, 4)
            if rev_offset_ratio is not None
            else None,
            "unit_substitution_ratio": round(unit_sub_ratio, 4)
            if unit_sub_ratio is not None
            else None,
            "has_valid_revenue_denominator": has_valid_rev_denominator,
            "has_valid_quantity_denominator": has_valid_qty_denominator,
            "methodological_label": "model_implied_observational_substitution",
        },
    }
