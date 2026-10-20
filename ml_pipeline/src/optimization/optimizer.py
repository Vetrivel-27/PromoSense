"""Bounded Category Profit Optimizer for Promotional Pricing.

Methodological Guardrails:
1. Bounded Search Space:
   - Evaluates discount depths strictly within empirical support: d in [0.05, 0.35].
   - Does NOT extrapolate beyond 35% or optimize regular base shelf prices.
2. Multi-Tier Margin Sensitivity:
   - Evaluates profit optimization across three standard margin tiers (15%, 25%, 35%).
   - Transparently labels wholesale costs as configurable business margin assumptions:
     c_j = P_j^shelf * (1 - m_base).
3. Cannibalization Integration:
   - Directly optimizes net category contribution profit:
     Delta Pi_category(d) = Delta Pi_A(d) + Delta Pi_B(d)
   - For weakly identified pairs, reports both isolated optimum and model-implied category optimum.
4. Profitability Safeguards:
   - When demand is inelastic and no profitable discount exists in [0.05, 0.35],
     explicitly sets recommend_discount=False and advises against price promotion.
5. Strict Temporal Quarantine:
   - Optimization runs strictly on Validation (Weeks 71-86) or Training data,
     quarantining Test data (Weeks 87-102) for post-optimization verification.
"""

from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.src.elasticity.cross_elasticity import CrossPriceElasticityModel
from ml_pipeline.src.elasticity.own_elasticity import PaidPriceElasticityOLS
from ml_pipeline.src.optimization.simulator import (
    MAX_SUPPORTED_DISCOUNT,
    MIN_SUPPORTED_DISCOUNT,
    STANDARD_MARGIN_TIERS,
    simulate_category_pair_promotion,
    simulate_single_sku_promotion,
)


class BoundedPromotionOptimizer:
    """Bounded Grid Search Optimizer for Promotion Discounting and Merchandising."""

    def __init__(
        self,
        min_discount: float = MIN_SUPPORTED_DISCOUNT,
        max_discount: float = MAX_SUPPORTED_DISCOUNT,
        discount_step: float = 0.025,
        margin_tiers: tuple[float, ...] = STANDARD_MARGIN_TIERS,
        min_profit_threshold: float = 0.0,
    ) -> None:
        if min_discount < MIN_SUPPORTED_DISCOUNT or max_discount > MAX_SUPPORTED_DISCOUNT:
            raise ValueError(
                f"Discount bounds must reside within supported range [{MIN_SUPPORTED_DISCOUNT:.2f}, {MAX_SUPPORTED_DISCOUNT:.2f}]"
            )
        if min_discount > max_discount:
            raise ValueError(
                f"min_discount ({min_discount}) cannot exceed max_discount ({max_discount})"
            )
        self.min_discount = min_discount
        self.max_discount = max_discount
        self.discount_step = discount_step
        self.margin_tiers = margin_tiers
        self.min_profit_threshold = min_profit_threshold
        self.candidate_discounts = np.round(
            np.arange(min_discount, max_discount + 1e-4, discount_step), 4
        )

    def optimize_single_sku(
        self,
        model: PaidPriceElasticityOLS,
        eval_df: pd.DataFrame,
        with_display: bool = False,
        baseline_mode: str = "observed_merch",
    ) -> dict[str, Any]:
        """Finds profit-maximizing discount depth for single product across margin tiers.

        Treats 0% discount as explicit no-promotion baseline. Evaluates permitted
        promotional candidates [min_discount, max_discount] against baseline.
        """
        if len(eval_df) == 0:
            raise ValueError("Evaluation dataframe has 0 observations.")

        grid_evaluations: list[dict[str, Any]] = []

        for d in self.candidate_discounts:
            sim = simulate_single_sku_promotion(
                model=model,
                eval_df=eval_df,
                discount_pct=float(d),
                with_display=with_display,
                baseline_mode=baseline_mode,
                margin_tiers=self.margin_tiers,
            )
            grid_evaluations.append(sim)

        optimal_by_tier: dict[str, Any] = {}

        for m_tier in self.margin_tiers:
            tier_key = f"margin_{int(m_tier * 100)}pct"
            profits = np.array(
                [sim["contribution_profits"][tier_key]["delta_profit"] for sim in grid_evaluations],
                dtype=float,
            )

            # Explicit no-promotion baseline: discount=0%, delta_profit=0.0
            no_promo_baseline = {
                "discount_pct": 0.0,
                "delta_profit": 0.0,
                "delta_revenue": 0.0,
                "delta_units": 0.0,
            }

            max_profit = float(np.max(profits))
            # Tie-breaking: if multiple discounts achieve max profit within 1e-4, pick lowest discount
            max_mask = np.isclose(profits, max_profit, atol=1e-4)
            best_idx = int(np.where(max_mask)[0][0])
            best_sim = grid_evaluations[best_idx]
            best_d = float(self.candidate_discounts[best_idx])
            best_delta_pi = float(profits[best_idx])

            # Select promotion only when modeled incremental profit exceeds baseline by min_profit_threshold
            recommend_promo = bool(best_delta_pi > self.min_profit_threshold)

            optimal_by_tier[tier_key] = {
                "assumed_gross_margin": m_tier,
                "no_promotion_baseline": no_promo_baseline,
                "optimal_discount_pct": best_d if recommend_promo else 0.0,
                "modeled_best_promotional_candidate": best_d,
                "recommend_price_promotion": recommend_promo,
                "delta_profit_at_optimum": round(best_delta_pi if recommend_promo else 0.0, 2),
                "modeled_candidate_delta_profit": round(best_delta_pi, 2),
                "pct_profit_lift_at_optimum": best_sim["contribution_profits"][tier_key][
                    "pct_profit_lift"
                ]
                if recommend_promo
                else 0.0,
                "delta_revenue_at_optimum": best_sim["revenues"]["delta_revenue"]
                if recommend_promo
                else 0.0,
                "delta_units_at_optimum": best_sim["volumes"]["delta_units"]
                if recommend_promo
                else 0.0,
                "promotional_unit_price": best_sim["prices"]["promotional_price"]
                if recommend_promo
                else best_sim["prices"]["avg_shelf_price"],
                "partial_uncertainty_bound": best_sim["profit_uncertainty"],
                "recommendation_note": (
                    f"Recommend {best_d:.1%} discount with expected incremental profit lift of ${best_delta_pi:.2f}."
                    if recommend_promo
                    else f"Promotional candidates produce non-positive incremental profit (best candidate {best_d:.1%} yields ${best_delta_pi:.2f}). Recommend maintaining regular shelf price (0% discount)."
                ),
            }

        return {
            "search_configuration": {
                "min_discount": self.min_discount,
                "max_discount": self.max_discount,
                "discount_step": self.discount_step,
                "with_display": with_display,
                "baseline_mode": baseline_mode,
                "n_grid_points": len(self.candidate_discounts),
            },
            "optimal_policy_by_margin_tier": optimal_by_tier,
            "grid_sweep_curve": [
                {
                    "discount_pct": round(float(sim["scenario"]["discount_pct"]), 4),
                    "delta_units": sim["volumes"]["delta_units"],
                    "delta_revenue": sim["revenues"]["delta_revenue"],
                    "delta_profits": {
                        t_key: sim["contribution_profits"][t_key]["delta_profit"]
                        for t_key in [f"margin_{int(m * 100)}pct" for m in self.margin_tiers]
                    },
                }
                for sim in grid_evaluations
            ],
        }

    def optimize_category_pair(
        self,
        pair_model: CrossPriceElasticityModel,
        eval_df_merged: pd.DataFrame,
        with_display: bool = False,
        baseline_mode: str = "observed_merch",
    ) -> dict[str, Any]:
        """Finds optimal discount for Product A maximizing net category profit accounting for Product B cannibalization.

        Treats 0% discount as explicit no-promotion baseline.
        """
        if len(eval_df_merged) == 0:
            raise ValueError("Evaluation merged dataframe has 0 observations.")

        grid_evaluations: list[dict[str, Any]] = []

        for d in self.candidate_discounts:
            sim = simulate_category_pair_promotion(
                pair_model=pair_model,
                eval_df_merged=eval_df_merged,
                discount_pct=float(d),
                with_display=with_display,
                baseline_mode=baseline_mode,
                margin_tiers=self.margin_tiers,
            )
            grid_evaluations.append(sim)

        optimal_by_tier: dict[str, Any] = {}
        is_weakly_identified = grid_evaluations[0]["scenario"]["is_weakly_identified_cross_price"]

        for m_tier in self.margin_tiers:
            tier_key = f"margin_{int(m_tier * 100)}pct"

            # 1. Category net profit optimum
            cat_profits = np.array(
                [
                    sim["profit_impact_category_modeled"][tier_key]["delta_profit_category_net"]
                    for sim in grid_evaluations
                ],
                dtype=float,
            )

            max_cat_profit = float(np.max(cat_profits))
            cat_max_mask = np.isclose(cat_profits, max_cat_profit, atol=1e-4)
            best_cat_idx = int(np.where(cat_max_mask)[0][0])
            best_cat_sim = grid_evaluations[best_cat_idx]
            best_cat_d = float(self.candidate_discounts[best_cat_idx])
            best_cat_delta_pi = float(cat_profits[best_cat_idx])

            recommend_cat_promo = bool(best_cat_delta_pi > self.min_profit_threshold)

            # 2. Isolated Product A optimum
            iso_profits = np.array(
                [
                    sim["profit_impact_isolated"][tier_key]["delta_profit_promoted"]
                    for sim in grid_evaluations
                ],
                dtype=float,
            )

            max_iso_profit = float(np.max(iso_profits))
            iso_max_mask = np.isclose(iso_profits, max_iso_profit, atol=1e-4)
            best_iso_idx = int(np.where(iso_max_mask)[0][0])
            best_iso_d = float(self.candidate_discounts[best_iso_idx])
            best_iso_delta_pi = float(iso_profits[best_iso_idx])

            recommend_iso_promo = bool(best_iso_delta_pi > self.min_profit_threshold)

            optimal_by_tier[tier_key] = {
                "assumed_gross_margin": m_tier,
                "no_promotion_baseline": {
                    "discount_pct": 0.0,
                    "delta_category_net_profit": 0.0,
                    "delta_promoted_A_profit": 0.0,
                    "cannibalization_loss_B": 0.0,
                },
                "category_optimum": {
                    "optimal_discount_pct": best_cat_d if recommend_cat_promo else 0.0,
                    "modeled_best_promotional_candidate": best_cat_d,
                    "delta_category_net_profit": round(
                        best_cat_delta_pi if recommend_cat_promo else 0.0, 2
                    ),
                    "modeled_candidate_category_net_profit": round(best_cat_delta_pi, 2),
                    "delta_promoted_A_profit": best_cat_sim["profit_impact_category_modeled"][
                        tier_key
                    ]["delta_profit_promoted_A"]
                    if recommend_cat_promo
                    else 0.0,
                    "cannibalization_loss_B": best_cat_sim["profit_impact_category_modeled"][
                        tier_key
                    ]["delta_profit_substitute_B"]
                    if recommend_cat_promo
                    else 0.0,
                    "revenue_offset_ratio": best_cat_sim["revenue_impact"]["revenue_offset_ratio"]
                    if recommend_cat_promo
                    else None,
                    "is_category_profitable": recommend_cat_promo,
                },
                "isolated_optimum": {
                    "optimal_discount_pct": best_iso_d if recommend_iso_promo else 0.0,
                    "modeled_best_promotional_candidate": best_iso_d,
                    "delta_promoted_profit": round(
                        best_iso_delta_pi if recommend_iso_promo else 0.0, 2
                    ),
                    "modeled_candidate_delta_profit": round(best_iso_delta_pi, 2),
                    "is_profitable": recommend_iso_promo,
                },
            }

        return {
            "search_configuration": {
                "min_discount": self.min_discount,
                "max_discount": self.max_discount,
                "discount_step": self.discount_step,
                "with_display": with_display,
                "baseline_mode": baseline_mode,
                "is_weakly_identified_cross_price": is_weakly_identified,
            },
            "optimal_policy_by_margin_tier": optimal_by_tier,
            "methodological_guidance": {
                "weak_identification_flag": is_weakly_identified,
                "guidance_note": (
                    "Substitute cross-price terms are weakly identified. Presenting both isolated and model-implied category optima. "
                    "Decisions should not rely solely on model-implied substitute offsets."
                    if is_weakly_identified
                    else "Substitute cross-price relationship is within standard empirical identification bounds."
                ),
            },
        }
