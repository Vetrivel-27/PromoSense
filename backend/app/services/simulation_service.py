"""Promotion Simulation Service for PromoSense Backend.

Executes counterfactual simulation by orchestrating frozen demand models,
margin accounting, cannibalization analysis, and support audits.
"""

from typing import Any

from backend.app.schemas.common import (
    EmpiricalSupportMetadata,
    MethodologicalWarnings,
    ProfitByMarginTier,
)
from backend.app.schemas.simulation import (
    SimulationRequest,
    SimulationResponse,
    SingleSkuSimulationDetails,
    SubstitutePairSimulationDetails,
)
from backend.app.services.model_registry import ModelRegistry
from ml_pipeline.src.optimization.simulator import (
    check_empirical_scenario_support,
    simulate_category_pair_promotion,
    simulate_single_sku_promotion,
)


class SimulationService:
    """Service orchestrating counterfactual promotion simulations."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry.get_instance()

    def simulate_promotion(self, request: SimulationRequest) -> SimulationResponse:
        """Simulates single-product and substitute category promotion effects."""
        if not self.registry.is_initialized:
            self.registry.load_artifacts()

        # 1. Product Validation
        pid = request.product_id
        if pid not in self.registry.target_skus:
            raise KeyError(
                f"Product ID {pid} is not among modeled target SKUs: {self.registry.target_skus}"
            )

        # 2. Run Single SKU Simulation
        model = self.registry.get_single_model(pid)
        eval_df = self.registry.get_eval_df(pid, split="val")

        sim_single = simulate_single_sku_promotion(
            model=model,
            eval_df=eval_df,
            discount_pct=request.discount_pct,
            with_display=request.with_display,
            baseline_mode=request.baseline_mode.value,
        )

        # 3. Format Profit Results
        profit_tiers: dict[str, ProfitByMarginTier] = {}
        for tier_key, p_data in sim_single["contribution_profits"].items():
            profit_tiers[tier_key] = ProfitByMarginTier(
                assumed_gross_margin=p_data["assumed_gross_margin"],
                cost_basis_label=p_data["cost_basis_label"],
                profit_baseline=p_data["profit_baseline"],
                profit_promoted=p_data["profit_promoted"],
                delta_profit=p_data["delta_profit"],
                pct_profit_lift=p_data["pct_profit_lift"],
                is_profitable_lift=bool(p_data["is_profitable_lift"]),
            )

        single_sku_details = SingleSkuSimulationDetails(
            product_id=pid,
            avg_shelf_price=sim_single["prices"]["avg_shelf_price"],
            promotional_price=sim_single["prices"]["promotional_price"],
            discount_depth=sim_single["prices"]["discount_depth"],
            units_baseline=sim_single["volumes"]["units_baseline"],
            units_promoted=sim_single["volumes"]["units_promoted"],
            delta_units=sim_single["volumes"]["delta_units"],
            pct_unit_lift=sim_single["volumes"]["pct_unit_lift"],
            revenue_baseline=sim_single["revenues"]["revenue_baseline"],
            revenue_promoted=sim_single["revenues"]["revenue_promoted"],
            delta_revenue=sim_single["revenues"]["delta_revenue"],
            pct_revenue_lift=sim_single["revenues"]["pct_revenue_lift"],
            contribution_profits=profit_tiers,
            profit_uncertainty_hac=sim_single["profit_uncertainty"],
        )

        # 4. Optional Substitute Pair Cannibalization Simulation
        pair_details: SubstitutePairSimulationDetails | None = None
        cross_warning: str | None = None

        if request.include_substitute:
            pair_def = self.registry.find_pair_for_product(pid)
            if pair_def is not None:
                pair_name = pair_def["pair_name"]
                pair_model = self.registry.get_pair_model(pair_name)
                pair_eval_df = self.registry.get_pair_eval_df(pair_name, split="val")

                sim_pair = simulate_category_pair_promotion(
                    pair_model=pair_model,
                    eval_df_merged=pair_eval_df,
                    discount_pct=request.discount_pct,
                    with_display=request.with_display,
                    baseline_mode=request.baseline_mode.value,
                )

                is_weak = bool(
                    "Milk" in pair_name or sim_pair.get("is_weakly_identified_cross_price", False)
                )
                if is_weak:
                    cross_warning = (
                        "Weakly identified cross-price elasticity term. Substitute offsets reflect "
                        "model-implied illustrative scenario tracing, NOT empirical proof of verified cannibalization."
                    )

                pair_details = SubstitutePairSimulationDetails(
                    pair_name=pair_name,
                    prod_A_id=pair_def["prod_A"],
                    prod_B_id=pair_def["prod_B"],
                    is_weakly_identified=is_weak,
                    delta_units_A=sim_pair["volume_impact"]["delta_units_A"],
                    delta_units_B=sim_pair["volume_impact"]["delta_units_B"],
                    net_category_units=sim_pair["volume_impact"]["net_category_units"],
                    delta_revenue_A=sim_pair["revenue_impact"]["delta_revenue_A"],
                    delta_revenue_B=sim_pair["revenue_impact"]["delta_revenue_B"],
                    net_category_revenue=sim_pair["revenue_impact"]["net_category_revenue"],
                    revenue_offset_ratio=sim_pair["revenue_impact"]["revenue_offset_ratio"],
                    profit_impact_isolated=sim_pair["profit_impact_isolated"],
                    profit_impact_category_modeled=sim_pair["profit_impact_category_modeled"],
                    weak_identification_warning=cross_warning,
                )

        # 5. Empirical Support Audit
        if request.discount_pct == 0.0:
            support_meta = EmpiricalSupportMetadata(
                is_jointly_supported=True,
                discount_supported=True,
                discount_warning=None,
                is_extrapolative_merchandising=False,
                merchandising_warning=None,
                joint_support_warning=None,
            )
        else:
            raw_support = check_empirical_scenario_support(
                discount_pct=request.discount_pct,
                with_display=request.with_display,
                baseline_mode=request.baseline_mode.value,
            )
            support_meta = EmpiricalSupportMetadata(**raw_support)

        # 6. Methodological Warnings
        methodological_warnings = MethodologicalWarnings(
            cross_price_warning=cross_warning,
        )

        scenario_meta: dict[str, Any] = {
            "product_id": pid,
            "discount_pct": request.discount_pct,
            "with_display": request.with_display,
            "baseline_mode": request.baseline_mode.value,
            "is_unpromoted_baseline_comparison": bool(request.discount_pct == 0.0),
            "evaluated_weeks": len(eval_df),
        }

        return SimulationResponse(
            scenario=scenario_meta,
            single_sku=single_sku_details,
            substitute_pair=pair_details,
            empirical_support=support_meta,
            methodological_warnings=methodological_warnings,
        )
