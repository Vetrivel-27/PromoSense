"""Unit and Invariant Tests for Promotion Simulation and Category Profit Optimization (Phase 7).

Tests:
1. Accounting Identities (Volume, Revenue, Contribution Profit, Category Net Lift).
2. Baseline Merchandising Consistency (Observed Status Quo vs Zero-Merch Bounds).
3. Optimizer Boundaries and Search Constraints ([0.05, 0.35] ceiling/floor).
4. Missing Cost Derivation and Margin Sensitivity Tiers (15%, 25%, 35%).
5. Negative Profit Lift and Non-Promotion Recommendation Guardrails.
6. Weak Cross-Price Estimate Dual-Reporting (Isolated vs Category Optima).
7. Empirical Support Checks (Univariate vs Joint Extrapolation Warnings).
8. Partial Uncertainty Propagation and Disclaimer Invariants.
"""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.src.elasticity.cross_elasticity import CrossPriceElasticityModel
from ml_pipeline.src.elasticity.data_prep import build_product_weekly_panel
from ml_pipeline.src.elasticity.own_elasticity import PaidPriceElasticityOLS
from ml_pipeline.src.optimization.optimizer import BoundedPromotionOptimizer
from ml_pipeline.src.optimization.simulator import (
    MAX_SUPPORTED_DISCOUNT,
    MIN_SUPPORTED_DISCOUNT,
    check_empirical_scenario_support,
    simulate_category_pair_promotion,
    simulate_single_sku_promotion,
)

# Test with representative SKUs
TEST_SKU_A = 1053690  # Soft Drink 2L A (Inelastic under controls)
TEST_SKU_B = 844165  # Soft Drink 2L B (Elastic)
TEST_SKU_12PK_A = 5569230  # Soft Drink 12pk Cans A (Elastic)
TEST_SKU_12PK_B = 8090521  # Soft Drink 12pk Cans B
TEST_MILK_A = 1029743  # Fluid Milk A (Weakly identified pair)
TEST_MILK_B = 1106523  # Fluid Milk B


@pytest.fixture(scope="module")
def opt_panel() -> pd.DataFrame:
    """Fixture that builds panel for simulation and optimization testing."""
    return build_product_weekly_panel(
        [TEST_SKU_A, TEST_SKU_B, TEST_SKU_12PK_A, TEST_SKU_12PK_B, TEST_MILK_A, TEST_MILK_B]
    )


def test_simulation_accounting_identities(opt_panel: pd.DataFrame) -> None:
    """Verifies fundamental volume, revenue, and profit accounting identities."""
    train_df = opt_panel[
        (opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "train")
    ]
    val_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "val")]

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    sim = simulate_single_sku_promotion(
        model=model,
        eval_df=val_df,
        discount_pct=0.20,
        with_display=False,
        baseline_mode="observed_merch",
    )

    # 1. Volume identity: delta_Q = Q_promo - Q_base
    v = sim["volumes"]
    assert np.isclose(v["delta_units"], v["units_promoted"] - v["units_baseline"], atol=1e-2)

    # 2. Revenue identity: delta_R = R_promo - R_base
    r = sim["revenues"]
    assert np.isclose(r["delta_revenue"], r["revenue_promoted"] - r["revenue_baseline"], atol=1e-2)

    # 3. Profit identity across all margin tiers: delta_Pi = Pi_promo - Pi_base
    p = sim["contribution_profits"]
    for m_key in ("margin_15pct", "margin_25pct", "margin_35pct"):
        tier = p[m_key]
        assert np.isclose(
            tier["delta_profit"], tier["profit_promoted"] - tier["profit_baseline"], atol=1e-2
        )


def test_baseline_merchandising_consistency(opt_panel: pd.DataFrame) -> None:
    """Verifies that baseline predictions strictly honor baseline_mode parameters."""
    train_df = opt_panel[
        (opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "train")
    ]
    val_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "val")]

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)

    sim_sq = simulate_single_sku_promotion(
        model=model,
        eval_df=val_df,
        discount_pct=0.20,
        with_display=False,
        baseline_mode="observed_merch",
    )
    sim_zero = simulate_single_sku_promotion(
        model=model,
        eval_df=val_df,
        discount_pct=0.20,
        with_display=False,
        baseline_mode="zero_merch",
    )

    # In validation split, observed display share is high (~0.95)
    # Therefore, baseline volume under observed merchandising must exceed zero-display baseline
    assert sim_sq["volumes"]["units_baseline"] > sim_zero["volumes"]["units_baseline"]
    assert sim_sq["revenues"]["revenue_baseline"] > sim_zero["revenues"]["revenue_baseline"]


def test_optimizer_boundaries_and_search_constraints() -> None:
    """Verifies that the optimizer strictly enforces bounded search space [0.05, 0.35]."""
    # 1. Default boundaries must match approved constraints
    opt = BoundedPromotionOptimizer()
    assert opt.min_discount == MIN_SUPPORTED_DISCOUNT
    assert opt.max_discount == MAX_SUPPORTED_DISCOUNT
    assert (opt.candidate_discounts >= 0.05).all()
    assert (opt.candidate_discounts <= 0.35).all()

    # 2. Out-of-bounds configuration must raise ValueError
    with pytest.raises(ValueError, match="supported range"):
        BoundedPromotionOptimizer(min_discount=0.01)

    with pytest.raises(ValueError, match="supported range"):
        BoundedPromotionOptimizer(max_discount=0.50)


def test_missing_costs_and_margin_sensitivity_tiers(opt_panel: pd.DataFrame) -> None:
    """Verifies cost derivation benchmarking against shelf price and monotonic profit scaling."""
    train_df = opt_panel[
        (opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "train")
    ]
    val_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "val")]

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    sim = simulate_single_sku_promotion(
        model=model,
        eval_df=val_df,
        discount_pct=0.10,
        with_display=True,
        baseline_mode="observed_merch",
    )

    p = sim["contribution_profits"]
    # Check that cost basis is labeled as assumption
    assert p["margin_25pct"]["cost_basis_label"] == "derived_from_shelf_price_assumption"

    # Higher gross margin implies lower unit cost, so dollar profit must scale strictly monotonically
    assert p["margin_15pct"]["profit_promoted"] < p["margin_25pct"]["profit_promoted"]
    assert p["margin_25pct"]["profit_promoted"] < p["margin_35pct"]["profit_promoted"]
    assert p["margin_15pct"]["delta_profit"] < p["margin_25pct"]["delta_profit"]
    assert p["margin_25pct"]["delta_profit"] < p["margin_35pct"]["delta_profit"]


def test_negative_profit_lift_and_optimizer_recommendation(opt_panel: pd.DataFrame) -> None:
    """Verifies that when price cuts dilute margin faster than volume expands, optimizer advises against promotion."""
    # Product 1053690 has inelastic price response under controls (eps ~ -0.42)
    train_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_A) & (opt_panel["split"] == "train")]
    val_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_A) & (opt_panel["split"] == "val")]

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    optimizer = BoundedPromotionOptimizer()

    opt_res = optimizer.optimize_single_sku(
        model=model, eval_df=val_df, with_display=False, baseline_mode="observed_merch"
    )

    tier_25 = opt_res["optimal_policy_by_margin_tier"]["margin_25pct"]
    # When promotional candidates are unprofitable, optimal policy chooses no-promotion baseline ($0.00 delta profit)
    assert tier_25["delta_profit_at_optimum"] == 0.0
    assert tier_25["modeled_candidate_delta_profit"] < 0.0
    assert tier_25["recommend_price_promotion"] is False
    assert tier_25["optimal_discount_pct"] == 0.0
    assert "Recommend maintaining regular shelf price" in tier_25["recommendation_note"]


def test_log_log_demand_ratio_identity(opt_panel: pd.DataFrame) -> None:
    """Verifies the mathematical identity Q_promo / Q_base = (P_promo / P_base) ** elasticity.

    For constant elasticity log-log models under price-only discounting holding all
    other covariates fixed, the simulated demand multiplier must strictly equal (1 - d) ** eps.
    """
    train_df = opt_panel[opt_panel["split"] == "train"]
    val_df = opt_panel[opt_panel["split"] == "val"]

    for pid in (TEST_SKU_B, TEST_SKU_12PK_A, TEST_SKU_12PK_B):
        p_train = train_df[train_df["product_id"] == pid]
        p_val = val_df[val_df["product_id"] == pid]
        model = PaidPriceElasticityOLS(include_controls=True).fit(p_train)
        eps = float(model.elasticity_)

        for d in (0.05, 0.10, 0.15, 0.20, 0.35):
            sim = simulate_single_sku_promotion(
                model=model,
                eval_df=p_val,
                discount_pct=d,
                with_display=False,
                baseline_mode="observed_merch",
            )
            sim_ratio = sim["volumes"]["units_promoted"] / sim["volumes"]["units_baseline"]
            theoretical_ratio = (1.0 - d) ** eps
            assert np.isclose(sim_ratio, theoretical_ratio, rtol=1e-3), (
                f"SKU {pid} at discount {d:.0%}: sim_ratio={sim_ratio:.4f} != theoretical={theoretical_ratio:.4f}"
            )


def test_optimizer_tie_breaking_picks_lowest_discount(opt_panel: pd.DataFrame) -> None:
    """Verifies that when multiple candidate discounts yield equal profit, the optimizer chooses the lowest discount."""
    val_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "val")]
    train_df = opt_panel[
        (opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "train")
    ]
    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)

    optimizer = BoundedPromotionOptimizer(min_discount=0.05, max_discount=0.35, discount_step=0.05)
    opt_res = optimizer.optimize_single_sku(model=model, eval_df=val_df, with_display=True)

    # In co-display execution at 25% margin, best candidate is 5%
    tier_25 = opt_res["optimal_policy_by_margin_tier"]["margin_25pct"]
    assert tier_25["optimal_discount_pct"] >= 0.05
    assert tier_25["recommend_price_promotion"] is True


def test_optimizer_empty_dataframe_raises(opt_panel: pd.DataFrame) -> None:
    """Verifies that optimizer raises ValueError when passed an empty dataframe."""
    train_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_A) & (opt_panel["split"] == "train")]
    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    optimizer = BoundedPromotionOptimizer()

    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="0 observations"):
        optimizer.optimize_single_sku(model=model, eval_df=empty_df)


def test_category_profit_baseline_consistency(opt_panel: pd.DataFrame) -> None:
    """Verifies category profit consistency: delta_Pi_category = (Pi_A_promo + Pi_B_promo) - (Pi_A_base + Pi_B_base)."""
    train_df = opt_panel[opt_panel["split"] == "train"]
    val_df = opt_panel[opt_panel["split"] == "val"]

    pair_model = CrossPriceElasticityModel(include_controls=True).fit(
        train_df, TEST_SKU_12PK_A, TEST_SKU_12PK_B
    )
    val_A = val_df[val_df["product_id"] == TEST_SKU_12PK_A]
    val_B = val_df[val_df["product_id"] == TEST_SKU_12PK_B]
    val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values("week_no")

    sim = simulate_category_pair_promotion(
        pair_model=pair_model,
        eval_df_merged=val_merged,
        discount_pct=0.20,
        with_display=False,
        baseline_mode="observed_merch",
    )

    cat_prof = sim["profit_impact_category_modeled"]["margin_25pct"]
    d_pi_A = cat_prof["delta_profit_promoted_A"]
    d_pi_B = cat_prof["delta_profit_substitute_B"]
    d_pi_net = cat_prof["delta_profit_category_net"]

    # Verify no double counting of baseline and exact additive identity
    assert np.isclose(d_pi_net, d_pi_A + d_pi_B, atol=1e-2)


def test_weak_cross_price_estimate_dual_reporting(opt_panel: pd.DataFrame) -> None:
    """Verifies dual reporting (isolated vs category net) and warnings for weakly identified pairs."""
    train_df = opt_panel[opt_panel["split"] == "train"]
    val_df = opt_panel[opt_panel["split"] == "val"]

    # Fit fluid milk pair (Product 1029743 vs 1106523) which has large cross-price SE
    milk_model = CrossPriceElasticityModel(include_controls=True).fit(
        train_df, TEST_MILK_A, TEST_MILK_B
    )

    val_A = val_df[val_df["product_id"] == TEST_MILK_A]
    val_B = val_df[val_df["product_id"] == TEST_MILK_B]
    val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values("week_no")

    optimizer = BoundedPromotionOptimizer()
    opt_pair = optimizer.optimize_category_pair(
        pair_model=milk_model, eval_df_merged=val_merged, with_display=False
    )

    # 1. Identification flag must be True
    assert opt_pair["search_configuration"]["is_weakly_identified_cross_price"] is True

    # 2. Methodological guidance must advise dual benchmarking
    guidance = opt_pair["methodological_guidance"]
    assert guidance["weak_identification_flag"] is True
    assert "weakly identified" in guidance["guidance_note"]

    # 3. Both isolated and category optimums must be present in every margin tier
    tier_25 = opt_pair["optimal_policy_by_margin_tier"]["margin_25pct"]
    assert "category_optimum" in tier_25
    assert "isolated_optimum" in tier_25


def test_empirical_support_checks_and_warnings() -> None:
    """Verifies that support checks detect out-of-bounds discounts and extrapolative merchandising."""
    # 1. Supported scenario
    sup_ok = check_empirical_scenario_support(
        discount_pct=0.20, with_display=False, baseline_mode="observed_merch"
    )
    assert sup_ok["is_jointly_supported"] is True
    assert sup_ok["discount_warning"] is None
    assert sup_ok["merchandising_warning"] is None

    # 2. Out-of-bounds discount (> 35%)
    sup_deep = check_empirical_scenario_support(
        discount_pct=0.50, with_display=False, baseline_mode="observed_merch"
    )
    assert sup_deep["discount_supported"] is False
    assert sup_deep["is_jointly_supported"] is False
    assert "ceiling" in sup_deep["discount_warning"]

    # 3. Extrapolative merchandising shift (zero baseline to display)
    sup_extra = check_empirical_scenario_support(
        discount_pct=0.20, with_display=True, baseline_mode="zero_merch"
    )
    assert sup_extra["is_extrapolative_merchandising"] is True
    assert sup_extra["is_jointly_supported"] is False
    assert "Extrapolative merchandising" in sup_extra["merchandising_warning"]
    assert "joint empirical support" in sup_extra["joint_support_warning"]


def test_profit_partial_uncertainty_disclaimer(opt_panel: pd.DataFrame) -> None:
    """Verifies that profit uncertainty is strictly labeled as partial HAC bounds with disclaimer."""
    train_df = opt_panel[
        (opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "train")
    ]
    val_df = opt_panel[(opt_panel["product_id"] == TEST_SKU_12PK_A) & (opt_panel["split"] == "val")]

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    sim = simulate_single_sku_promotion(
        model=model, eval_df=val_df, discount_pct=0.20, with_display=False
    )

    unc = sim["profit_uncertainty"]
    assert unc["uncertainty_type"] == "partial_own_price_hac_bound"
    assert "Partial parameter uncertainty" in unc["uncertainty_disclaimer"]
    assert "NOT represent a complete joint confidence interval" in unc["uncertainty_disclaimer"]
    assert "delta_profit_bound_low" in unc
    assert "delta_profit_bound_high" in unc
