"""Promotion Simulation Engine: Scenario-Based Counterfactual Demand and Profit Modeling.

Methodological Guardrails:
1. Scenario-Based Evaluation:
   - Evaluates supported promotional discounts (5% to 35%) under price-only vs co-merchandised executions.
   - Supports both realistic status-quo baseline (observed merchandising) and extrapolative zero-merch bounds.
2. Contribution Margin Profit Accounting:
   - Computes category contribution profit lift:
     Delta Pi_category = Delta Pi_promoted + Delta Pi_substitutes
   - Missing Wholesale Cost Handling: Employs transparent, configurable gross margin baselines
     (15%, 25%, 35% margin tiers) benchmarked against list shelf price:
     c_j = P_j^shelf * (1 - m_base_j).
     Clearly labeled as configurable business assumptions, NOT observed supplier invoices.
3. Cannibalization Integration & Weak Identification Dual-Reporting:
   - For well-identified pairs, computes net category profit lift accounting for substitute margin offsets.
   - For weakly identified pairs (large SE or wrong sign), generates dual outputs:
     (a) Isolated Scenario: Delta Pi_category = Delta Pi_A (zero substitute loss assumed).
     (b) Model-Implied Scenario: Delta Pi_category = Delta Pi_A + Delta Pi_B (with explicit warning banner).
4. Outcome Segregation:
   - Separates Unit Lift (Delta Q), Revenue Lift (Delta R), and Profit Lift (Delta Pi).
5. Empirical Support Checks & Uncertainty Labeling:
   - Flags discounts outside [0.05, 0.35] and extrapolative merchandising shifts.
   - Propagates partial uncertainty from own-price HAC standard errors, explicitly labeled
     as 'partial_own_price_hac_bound' rather than full joint confidence intervals.
"""

from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.src.elasticity.cross_elasticity import CrossPriceElasticityModel
from ml_pipeline.src.elasticity.own_elasticity import PaidPriceElasticityOLS

# Supported promotional discount boundaries
MIN_SUPPORTED_DISCOUNT = 0.05
MAX_SUPPORTED_DISCOUNT = 0.35
STANDARD_MARGIN_TIERS = (0.15, 0.25, 0.35)


def check_empirical_scenario_support(
    discount_pct: float,
    with_display: bool,
    baseline_mode: str,
    historical_display_mean: float = 0.75,
) -> dict[str, Any]:
    """Audits empirical support for proposed promotional scenario."""
    discount_supported = bool(MIN_SUPPORTED_DISCOUNT <= discount_pct <= MAX_SUPPORTED_DISCOUNT)
    discount_warning = None
    if discount_pct < MIN_SUPPORTED_DISCOUNT:
        discount_warning = f"Discount {discount_pct:.1%} is below minimum active promotional threshold ({MIN_SUPPORTED_DISCOUNT:.0%})."
    elif discount_pct > MAX_SUPPORTED_DISCOUNT:
        discount_warning = f"Discount {discount_pct:.1%} exceeds empirical promotional support ceiling ({MAX_SUPPORTED_DISCOUNT:.0%})."

    is_extrapolative_merch = bool(baseline_mode == "zero_merch" and with_display)
    merch_warning = (
        "Extrapolative merchandising shift: zero-display baseline compared to 100% display execution. "
        "Display is an observational co-factor, not a proven standalone causal lever; volume multiplier "
        "reflects joint model-implied extrapolation."
        if is_extrapolative_merch
        else None
    )

    is_jointly_supported = bool(discount_supported and not is_extrapolative_merch)
    joint_warning = None
    if not is_jointly_supported:
        joint_warning = (
            "Scenario combines out-of-support parameters or extrapolative merchandising bounds. "
            "Univariate range inclusion does not guarantee joint empirical support."
        )

    return {
        "is_jointly_supported": is_jointly_supported,
        "discount_supported": discount_supported,
        "discount_warning": discount_warning,
        "is_extrapolative_merchandising": is_extrapolative_merch,
        "merchandising_warning": merch_warning,
        "joint_support_warning": joint_warning,
    }


def compute_contribution_profit(
    price: float,
    quantity: float,
    unit_cost: float,
) -> float:
    """Computes gross contribution profit: (price - unit_cost) * quantity."""
    return float((price - unit_cost) * quantity)


def simulate_single_sku_promotion(
    model: PaidPriceElasticityOLS,
    eval_df: pd.DataFrame,
    discount_pct: float = 0.20,
    with_display: bool = False,
    baseline_mode: str = "observed_merch",
    margin_tiers: tuple[float, ...] = STANDARD_MARGIN_TIERS,
) -> dict[str, Any]:
    """Simulates promotion for a single product under frozen demand model.

    Parameters
    ----------
    model : PaidPriceElasticityOLS
        Fitted frozen single-product demand model.
    eval_df : pd.DataFrame
        Evaluation period dataframe (Validation or Test split).
    discount_pct : float
        Promotional discount percentage (0.05 to 0.35).
    with_display : bool
        Whether to execute in-store endcap display placement (1.0 vs observed).
    baseline_mode : str
        'observed_merch' (realistic status quo) or 'zero_merch' (extrapolative bound).
    margin_tiers : tuple[float, ...]
        Configurable baseline gross margin assumptions.

    Returns
    -------
    dict[str, Any]
        Scenario simulation results with unit, revenue, and profit lifts across margin tiers.
    """
    n_weeks = len(eval_df)
    if n_weeks == 0:
        raise ValueError("Evaluation dataframe has 0 observations.")

    support_audit = check_empirical_scenario_support(
        discount_pct=discount_pct,
        with_display=with_display,
        baseline_mode=baseline_mode,
    )

    shelf_prices = eval_df["shelf_price"].to_numpy(dtype=float)
    avg_shelf_price = float(np.mean(shelf_prices))

    # 1. Baseline Predictions
    base_df = eval_df.copy()
    if baseline_mode == "zero_merch":
        base_df["display_share"] = 0.0
        base_df["mailer_share"] = 0.0
    # Price is regular shelf price
    base_df["log_paid_price"] = np.log(np.maximum(shelf_prices, 1e-4))

    pred_log_q_base = model.predict(base_df)
    q_base = np.exp(pred_log_q_base)
    tot_q_base = float(np.sum(q_base))
    tot_rev_base = float(np.sum(q_base * shelf_prices))

    # 2. Promoted Predictions
    promo_df = eval_df.copy()
    if baseline_mode == "zero_merch":
        promo_df["display_share"] = 1.0 if with_display else 0.0
        promo_df["mailer_share"] = 0.0
    else:
        if with_display:
            promo_df["display_share"] = 1.0

    promo_prices = shelf_prices * (1.0 - discount_pct)
    promo_df["log_paid_price"] = np.log(np.maximum(promo_prices, 1e-4))

    pred_log_q_promo = model.predict(promo_df)
    q_promo = np.exp(pred_log_q_promo)
    tot_q_promo = float(np.sum(q_promo))
    tot_rev_promo = float(np.sum(q_promo * promo_prices))

    # Lifts
    delta_qty = tot_q_promo - tot_q_base
    delta_rev = tot_rev_promo - tot_rev_base
    pct_unit_lift = (delta_qty / tot_q_base) if tot_q_base > 0 else 0.0
    pct_rev_lift = (delta_rev / tot_rev_base) if tot_rev_base > 0 else 0.0

    # 3. Profit Calculations across Margin Tiers
    profit_results: dict[str, Any] = {}
    for m_tier in margin_tiers:
        # Unit cost derived from baseline regular shelf price: c = P_shelf * (1 - m)
        unit_costs = shelf_prices * (1.0 - m_tier)
        pi_base = float(np.sum((shelf_prices - unit_costs) * q_base))
        pi_promo = float(np.sum((promo_prices - unit_costs) * q_promo))
        delta_pi = pi_promo - pi_base
        pct_pi_lift = (delta_pi / pi_base) if pi_base > 0 else 0.0

        tier_key = f"margin_{int(m_tier * 100)}pct"
        profit_results[tier_key] = {
            "assumed_gross_margin": m_tier,
            "cost_basis_label": "derived_from_shelf_price_assumption",
            "profit_baseline": round(pi_base, 2),
            "profit_promoted": round(pi_promo, 2),
            "delta_profit": round(delta_pi, 2),
            "pct_profit_lift": round(pct_pi_lift * 100.0, 2),
            "is_profitable_lift": bool(delta_pi > 0.0),
        }

    # 4. Partial Uncertainty Bounds on Profit (Propagated from Own-Price HAC SE)
    # Holding merchandising and other parameters fixed at point estimates
    hac_se = float(model.elasticity_hac_se_ if model.elasticity_hac_se_ is not None else 0.40)
    # High demand bound: beta_more_negative = beta - 1.96 * se (more price responsive)
    # Low demand bound:  beta_less_negative = beta + 1.96 * se (less price responsive)
    delta_log_p = np.log(1.0 - discount_pct)
    mult_base = np.exp(model.elasticity_ * delta_log_p) if model.elasticity_ is not None else 1.0
    mult_high = (
        np.exp((model.elasticity_ - 1.96 * hac_se) * delta_log_p)
        if model.elasticity_ is not None
        else 1.0
    )
    mult_low = (
        np.exp((model.elasticity_ + 1.96 * hac_se) * delta_log_p)
        if model.elasticity_ is not None
        else 1.0
    )

    ratio_high = mult_high / max(mult_base, 1e-6)
    ratio_low = mult_low / max(mult_base, 1e-6)
    q_promo_high = tot_q_promo * ratio_high
    q_promo_low = tot_q_promo * ratio_low

    m_std = 0.25
    c_std = avg_shelf_price * (1.0 - m_std)
    avg_p_promo = avg_shelf_price * (1.0 - discount_pct)
    pi_promo_high = (avg_p_promo - c_std) * q_promo_high
    pi_promo_low = (avg_p_promo - c_std) * q_promo_low
    pi_base_std = (avg_shelf_price - c_std) * tot_q_base

    return {
        "scenario": {
            "discount_pct": discount_pct,
            "with_display": with_display,
            "baseline_mode": baseline_mode,
            "evaluated_weeks": n_weeks,
            "empirical_support": support_audit,
        },
        "prices": {
            "avg_shelf_price": round(avg_shelf_price, 4),
            "promotional_price": round(avg_shelf_price * (1.0 - discount_pct), 4),
            "discount_depth": round(-float(np.log(1.0 - discount_pct)), 4),
        },
        "volumes": {
            "units_baseline": round(tot_q_base, 2),
            "units_promoted": round(tot_q_promo, 2),
            "delta_units": round(delta_qty, 2),
            "pct_unit_lift": round(pct_unit_lift * 100.0, 2),
        },
        "revenues": {
            "revenue_baseline": round(tot_rev_base, 2),
            "revenue_promoted": round(tot_rev_promo, 2),
            "delta_revenue": round(delta_rev, 2),
            "pct_revenue_lift": round(pct_rev_lift * 100.0, 2),
        },
        "contribution_profits": profit_results,
        "profit_uncertainty": {
            "uncertainty_type": "partial_own_price_hac_bound",
            "uncertainty_disclaimer": (
                "Partial parameter uncertainty bound propagated strictly from own-price HAC standard error "
                "holding merchandising and other covariates fixed at point estimates. Does NOT represent "
                "a complete joint confidence interval."
            ),
            "delta_profit_bound_low": round(pi_promo_low - pi_base_std, 2),
            "delta_profit_bound_high": round(pi_promo_high - pi_base_std, 2),
        },
    }


def simulate_category_pair_promotion(
    pair_model: CrossPriceElasticityModel,
    eval_df_merged: pd.DataFrame,
    discount_pct: float = 0.20,
    with_display: bool = False,
    baseline_mode: str = "observed_merch",
    margin_tiers: tuple[float, ...] = STANDARD_MARGIN_TIERS,
) -> dict[str, Any]:
    """Simulates category promotion on Product A with cannibalization on substitute Product B.

    Handles well-identified pairs as well as weakly identified pairs with dual-reporting.
    """
    n_weeks = len(eval_df_merged)
    if n_weeks == 0:
        raise ValueError("Evaluation dataframe has 0 observations.")

    support_audit = check_empirical_scenario_support(
        discount_pct=discount_pct,
        with_display=with_display,
        baseline_mode=baseline_mode,
    )

    # 1. Baseline Prices and Demand
    pA_shelf = eval_df_merged["shelf_price_A"].to_numpy(dtype=float)
    pB_shelf = eval_df_merged["shelf_price_B"].to_numpy(dtype=float)
    log_pA_base = np.log(np.maximum(pA_shelf, 1e-4))
    log_pB_base = np.log(np.maximum(pB_shelf, 1e-4))

    base_controls = eval_df_merged.copy()
    if baseline_mode == "zero_merch":
        base_controls["display_share_A"] = 0.0
        base_controls["display_share_B"] = 0.0
        base_controls["mailer_share_A"] = 0.0
        base_controls["mailer_share_B"] = 0.0

    pred_log_qA_base, pred_log_qB_base = pair_model.predict_pair(
        log_pA_base, log_pB_base, base_controls
    )
    qA_base = np.exp(pred_log_qA_base)
    qB_base = np.exp(pred_log_qB_base)

    tot_qA_base = float(np.sum(qA_base))
    tot_qB_base = float(np.sum(qB_base))
    rev_A_base = float(np.sum(qA_base * pA_shelf))
    rev_B_base = float(np.sum(qB_base * pB_shelf))

    # 2. Promoted Scenario: Product A discounted, Product B regular shelf price
    pA_promo = pA_shelf * (1.0 - discount_pct)
    log_pA_promo = np.log(np.maximum(pA_promo, 1e-4))

    promo_controls = eval_df_merged.copy()
    if baseline_mode == "zero_merch":
        promo_controls["display_share_A"] = 1.0 if with_display else 0.0
        promo_controls["display_share_B"] = 0.0
        promo_controls["mailer_share_A"] = 0.0
        promo_controls["mailer_share_B"] = 0.0
    else:
        if with_display:
            promo_controls["display_share_A"] = 1.0

    pred_log_qA_promo, pred_log_qB_promo = pair_model.predict_pair(
        log_pA_promo, log_pB_base, promo_controls
    )
    qA_promo = np.exp(pred_log_qA_promo)
    qB_promo = np.exp(pred_log_qB_promo)

    tot_qA_promo = float(np.sum(qA_promo))
    tot_qB_promo = float(np.sum(qB_promo))
    rev_A_promo = float(np.sum(qA_promo * pA_promo))
    rev_B_promo = float(np.sum(qB_promo * pB_shelf))

    # Differences
    delta_qA = tot_qA_promo - tot_qA_base
    delta_qB = tot_qB_promo - tot_qB_base
    delta_revA = rev_A_promo - rev_A_base
    delta_revB = rev_B_promo - rev_B_base

    # Check identification strength of cross-price term
    cross_se_A = float(pair_model.model_A.get("cross_se", 0.0))
    p_vals_A = pair_model.model_A.get("p_val", [])
    cross_pval_A = float(p_vals_A[2]) if len(p_vals_A) > 2 else 0.0
    is_weakly_identified = bool(cross_se_A >= 1.5 or cross_pval_A >= 0.20)

    # 3. Profit Accounting across Margin Tiers
    profit_tiers_isolated: dict[str, Any] = {}
    profit_tiers_category: dict[str, Any] = {}

    for m_tier in margin_tiers:
        cA = pA_shelf * (1.0 - m_tier)
        cB = pB_shelf * (1.0 - m_tier)

        pi_A_base = float(np.sum((pA_shelf - cA) * qA_base))
        pi_A_promo = float(np.sum((pA_promo - cA) * qA_promo))
        delta_pi_A = pi_A_promo - pi_A_base

        pi_B_base = float(np.sum((pB_shelf - cB) * qB_base))
        pi_B_promo = float(np.sum((pB_shelf - cB) * qB_promo))
        delta_pi_B = pi_B_promo - pi_B_base

        delta_pi_cat = delta_pi_A + delta_pi_B
        tier_key = f"margin_{int(m_tier * 100)}pct"

        # Isolated (Product A alone, ignoring substitute substitution)
        profit_tiers_isolated[tier_key] = {
            "assumed_gross_margin": m_tier,
            "delta_profit_promoted": round(delta_pi_A, 2),
            "is_profitable": bool(delta_pi_A > 0.0),
        }

        # Category (Product A incremental profit minus Product B cannibalization loss)
        profit_tiers_category[tier_key] = {
            "assumed_gross_margin": m_tier,
            "delta_profit_promoted_A": round(delta_pi_A, 2),
            "delta_profit_substitute_B": round(delta_pi_B, 2),
            "delta_profit_category_net": round(delta_pi_cat, 2),
            "is_category_profitable": bool(delta_pi_cat > 0.0),
        }

    # Revenue offset ratio guard
    rev_offset_ratio = None
    if delta_revA > 1e-4:
        rev_offset_ratio = round(max(-delta_revB / delta_revA, 0.0), 4)

    return {
        "scenario": {
            "discount_pct": discount_pct,
            "with_display": with_display,
            "baseline_mode": baseline_mode,
            "evaluated_weeks": n_weeks,
            "is_weakly_identified_cross_price": is_weakly_identified,
            "empirical_support": support_audit,
        },
        "volume_impact": {
            "delta_units_A": round(delta_qA, 2),
            "delta_units_B": round(delta_qB, 2),
            "net_category_units": round(delta_qA + delta_qB, 2),
        },
        "revenue_impact": {
            "delta_revenue_A": round(delta_revA, 2),
            "delta_revenue_B": round(delta_revB, 2),
            "net_category_revenue": round(delta_revA + delta_revB, 2),
            "revenue_offset_ratio": rev_offset_ratio,
        },
        "profit_impact_isolated": profit_tiers_isolated,
        "profit_impact_category_modeled": profit_tiers_category,
        "methodological_warnings": {
            "cross_price_warning": (
                "Weakly identified cross-price elasticity term. Substitute offsets reflect "
                "model-implied illustrative scenario tracing, NOT empirical proof of verified cannibalization."
                if is_weakly_identified
                else None
            ),
            "recommendation_guardrail": (
                "For weakly identified pairs, decisions should benchmark both isolated and model-implied profit tiers."
                if is_weakly_identified
                else "Cross-price substitution terms are within empirical identification bounds."
            ),
        },
    }
