"""Promotion Optimization Service for PromoSense Backend.

Executes bounded grid search optimization over promotional discount depths,
identifying profit-maximizing depths across gross margin tiers and evaluating
net category contribution profit under substitute cannibalization.
"""

from typing import Any

from backend.app.schemas.common import (
    EmpiricalSupportMetadata,
    MethodologicalWarnings,
)
from backend.app.schemas.optimization import (
    GridSweepPoint,
    NoPromotionBaselinePoint,
    OptimalPolicyTier,
    OptimizationRequest,
    OptimizationResponse,
)
from backend.app.services.model_registry import ModelRegistry
from ml_pipeline.src.optimization.optimizer import BoundedPromotionOptimizer
from ml_pipeline.src.optimization.simulator import check_empirical_scenario_support


class OptimizationService:
    """Service orchestrating bounded promotional profit optimization."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry.get_instance()

    def optimize_promotion(self, request: OptimizationRequest) -> OptimizationResponse:
        """Finds profit-maximizing promotional discount depths across gross margin tiers."""
        if not self.registry.is_initialized:
            self.registry.load_artifacts()

        pid = request.product_id
        if pid not in self.registry.target_skus:
            raise KeyError(
                f"Product ID {pid} is not among modeled target SKUs: {self.registry.target_skus}"
            )

        # 1. Initialize Bounded Optimizer
        optimizer = BoundedPromotionOptimizer(
            min_discount=request.min_discount,
            max_discount=request.max_discount,
            discount_step=request.discount_step,
        )

        model = self.registry.get_single_model(pid)
        eval_df = self.registry.get_eval_df(pid, split="val")

        # 2. Single SKU Optimization
        single_opt = optimizer.optimize_single_sku(
            model=model,
            eval_df=eval_df,
            with_display=request.with_display,
            baseline_mode=request.baseline_mode.value,
        )

        # Format single policies
        single_policies: dict[str, OptimalPolicyTier] = {}
        for tier_key, p_tier in single_opt["optimal_policy_by_margin_tier"].items():
            single_policies[tier_key] = OptimalPolicyTier(
                assumed_gross_margin=p_tier["assumed_gross_margin"],
                optimal_discount_pct=p_tier["optimal_discount_pct"],
                modeled_best_promotional_candidate=p_tier["modeled_best_promotional_candidate"],
                recommend_price_promotion=bool(p_tier["recommend_price_promotion"]),
                delta_profit_at_optimum=p_tier["delta_profit_at_optimum"],
                modeled_candidate_delta_profit=p_tier["modeled_candidate_delta_profit"],
                pct_profit_lift_at_optimum=p_tier["pct_profit_lift_at_optimum"],
                delta_revenue_at_optimum=p_tier["delta_revenue_at_optimum"],
                delta_units_at_optimum=p_tier["delta_units_at_optimum"],
                promotional_unit_price=p_tier["promotional_unit_price"],
                partial_uncertainty_bound=p_tier.get("partial_uncertainty_bound"),
                recommendation_note=p_tier["recommendation_note"],
            )

        single_sweep: list[GridSweepPoint] = [
            GridSweepPoint(
                discount_pct=pt["discount_pct"],
                delta_units=pt["delta_units"],
                delta_revenue=pt["delta_revenue"],
                delta_profits=pt["delta_profits"],
            )
            for pt in single_opt["grid_sweep_curve"]
        ]

        # 3. Category Pair Optimization (if requested and applicable)
        pair_policies: dict[str, Any] | None = None
        pair_sweep: list[dict[str, Any]] | None = None
        cross_warning: str | None = None

        if request.include_substitute:
            pair_def = self.registry.find_pair_for_product(pid)
            if pair_def is not None:
                pair_name = pair_def["pair_name"]
                pair_model = self.registry.get_pair_model(pair_name)
                pair_eval_df = self.registry.get_pair_eval_df(pair_name, split="val")

                pair_opt = optimizer.optimize_category_pair(
                    pair_model=pair_model,
                    eval_df_merged=pair_eval_df,
                    with_display=request.with_display,
                    baseline_mode=request.baseline_mode.value,
                )

                pair_policies = pair_opt.get("optimal_policy_by_margin_tier")
                pair_sweep = pair_opt.get("grid_sweep_curve")

                if "Milk" in pair_name:
                    cross_warning = (
                        "Weakly identified cross-price elasticity term. Substitute offsets reflect "
                        "model-implied illustrative scenario tracing, NOT empirical proof of verified cannibalization."
                    )

        # 4. Support Audit
        raw_support = check_empirical_scenario_support(
            discount_pct=request.min_discount,
            with_display=request.with_display,
            baseline_mode=request.baseline_mode.value,
        )
        support_meta = EmpiricalSupportMetadata(**raw_support)

        # 5. Methodological Warnings
        methodological_warnings = MethodologicalWarnings(
            cross_price_warning=cross_warning,
        )

        return OptimizationResponse(
            product_id=pid,
            with_display=request.with_display,
            baseline_mode=request.baseline_mode.value,
            search_grid=single_opt["search_configuration"],
            no_promotion_baseline=NoPromotionBaselinePoint(),
            single_sku_policies=single_policies,
            single_sku_grid_sweep=single_sweep,
            category_pair_policies=pair_policies,
            category_pair_grid_sweep=pair_sweep,
            empirical_support=support_meta,
            methodological_warnings=methodological_warnings,
        )
