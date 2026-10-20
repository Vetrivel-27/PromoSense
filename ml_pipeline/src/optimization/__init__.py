"""PromoSense Optimization and Simulation Module."""

from ml_pipeline.src.optimization.optimizer import BoundedPromotionOptimizer
from ml_pipeline.src.optimization.simulator import (
    MAX_SUPPORTED_DISCOUNT,
    MIN_SUPPORTED_DISCOUNT,
    STANDARD_MARGIN_TIERS,
    check_empirical_scenario_support,
    compute_contribution_profit,
    simulate_category_pair_promotion,
    simulate_single_sku_promotion,
)

__all__ = [
    "BoundedPromotionOptimizer",
    "MAX_SUPPORTED_DISCOUNT",
    "MIN_SUPPORTED_DISCOUNT",
    "STANDARD_MARGIN_TIERS",
    "check_empirical_scenario_support",
    "compute_contribution_profit",
    "simulate_category_pair_promotion",
    "simulate_single_sku_promotion",
]
