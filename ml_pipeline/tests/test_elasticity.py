"""Unit and Invariant Tests for Price Elasticity and Cannibalization Modeling.

Tests:
1. Panel Construction & Data Invariants (columns, time splits, non-negativity).
2. Collinearity Diagnostics (VIF of shelf price vs discount depth).
3. Own-Price Elasticity Models (OLS, Decomposed OLS, LightGBM).
4. Cross-Price Elasticity and Counterfactual Cannibalization Simulation.
5. Cannibalization Ratio Edge Cases (non-positive denominator, zero substitution).
"""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.src.elasticity.cross_elasticity import (
    CrossPriceElasticityModel,
    simulate_counterfactual_cannibalization,
)
from ml_pipeline.src.elasticity.data_prep import (
    audit_split_support,
    build_product_weekly_panel,
)
from ml_pipeline.src.elasticity.own_elasticity import (
    NonlinearElasticityModel,
    PaidPriceElasticityOLS,
    ShelfPriceDiscountOLS,
    compute_newey_west_cov,
)

# Test with representative Soft Drink SKUs
TEST_SKU_A = 1053690
TEST_SKU_B = 844165


@pytest.fixture(scope="module")
def sample_panel() -> pd.DataFrame:
    """Fixture that builds a panel for two test SKUs."""
    return build_product_weekly_panel([TEST_SKU_A, TEST_SKU_B])


def test_panel_construction_columns_and_splits(sample_panel: pd.DataFrame) -> None:
    """Verifies panel schema, required fields, and time split boundaries."""
    required_cols = [
        "product_id",
        "week_no",
        "quantity",
        "sales_value",
        "paid_unit_price",
        "shelf_price",
        "retail_discount_pct",
        "discount_depth",
        "display_share",
        "mailer_share",
        "log_quantity",
        "log_paid_price",
        "log_shelf_price",
        "sin_52w",
        "cos_52w",
        "split",
    ]
    for col in required_cols:
        assert col in sample_panel.columns, f"Missing required column: {col}"

    # Split boundaries
    train_df = sample_panel[sample_panel["split"] == "train"]
    val_df = sample_panel[sample_panel["split"] == "val"]
    test_df = sample_panel[sample_panel["split"] == "test"]

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    assert train_df["week_no"].max() <= 70
    assert val_df["week_no"].min() >= 71
    assert val_df["week_no"].max() <= 86
    assert test_df["week_no"].min() >= 87


def test_price_derivation_invariants(sample_panel: pd.DataFrame) -> None:
    """Verifies mathematical invariants of prices and discount metrics."""
    # Paid unit price cannot exceed shelf price (accounting for floating precision)
    assert (sample_panel["paid_unit_price"] <= sample_panel["shelf_price"] + 1e-4).all(), (
        "Paid unit price exceeded regular shelf price!"
    )

    # Discount pct is bounded in [0, 1)
    assert (sample_panel["retail_discount_pct"] >= 0.0).all()
    assert (sample_panel["retail_discount_pct"] < 1.0).all()

    # Discount depth is non-negative
    assert (sample_panel["discount_depth"] >= 0.0).all()


def test_split_support_audit(sample_panel: pd.DataFrame) -> None:
    """Verifies that split support audit correctly detects observations and price variance."""
    audit = audit_split_support(sample_panel, TEST_SKU_A)
    assert audit["product_id"] == TEST_SKU_A
    assert audit["splits"]["train"]["n_obs"] > 0
    assert audit["splits"]["val"]["n_obs"] > 0
    assert audit["splits"]["test"]["n_obs"] > 0
    # Price variation must be present
    assert audit["splits"]["train"]["has_independent_price_variation"] is True


def test_collinearity_diagnostics_in_decomposed_ols(
    sample_panel: pd.DataFrame,
) -> None:
    """Verifies that decomposing shelf price and discount depth avoids collinearity."""
    train_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "train")
    ].copy()

    model = ShelfPriceDiscountOLS(include_controls=True).fit(train_df)
    assert model.base_elasticity_ is not None
    assert model.discount_elasticity_ is not None

    # VIF between log_shelf_price and discount_depth should be modest (< 5.0)
    assert model.collinearity_vif_ is not None
    assert model.collinearity_vif_ < 5.0, f"Unexpectedly high VIF: {model.collinearity_vif_}"


def test_paid_price_elasticity_ols_fit_and_predict(
    sample_panel: pd.DataFrame,
) -> None:
    """Verifies PaidPriceElasticityOLS fits, predicts, and produces plausible negative elasticity."""
    train_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "train")
    ].copy()
    val_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "val")
    ].copy()

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    assert model.elasticity_ is not None
    # For carbonated soft drinks, demand should be price-sensitive (negative)
    assert model.elasticity_ < 0.0, f"Expected negative elasticity, got {model.elasticity_}"

    eval_res = model.evaluate(val_df)
    assert eval_res["rmse_units"] > 0.0
    assert "log_paid_price" in eval_res["coefficients"]


def test_nonlinear_lightgbm_elasticity_monotonic(
    sample_panel: pd.DataFrame,
) -> None:
    """Verifies NonlinearElasticityModel with monotonic constraints produces negative implied elasticity."""
    train_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "train")
    ].copy()
    val_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "val")
    ].copy()

    model = NonlinearElasticityModel(use_monotonic_constraint=True, random_state=42).fit(train_df)
    eval_res = model.evaluate(val_df)

    assert eval_res["use_monotonic_constraint"] is True
    # Under monotonic constraint (-1 on price), arc elasticity must be non-positive
    assert eval_res["model_implied_elasticity"] <= 0.0


def test_cross_price_elasticity_and_cannibalization(
    sample_panel: pd.DataFrame,
) -> None:
    """Verifies joint substitute pair estimation and counterfactual cannibalization simulation."""
    train_df = sample_panel[sample_panel["split"] == "train"].copy()
    val_df = sample_panel[sample_panel["split"] == "val"].copy()

    pair_model = CrossPriceElasticityModel(include_controls=True).fit(
        train_df, TEST_SKU_A, TEST_SKU_B
    )

    # Own elasticities should be negative
    assert pair_model.model_A["own_elasticity"] < 0.0
    assert pair_model.model_B["own_elasticity"] < 0.0

    # Prepare merged validation dataframe
    val_A = val_df[val_df["product_id"] == TEST_SKU_A]
    val_B = val_df[val_df["product_id"] == TEST_SKU_B]
    val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values("week_no")

    sim = simulate_counterfactual_cannibalization(
        pair_model, val_merged, discount_pct=0.20, with_display=True
    )

    # Incremental revenue on product A must be computed
    assert "delta_revenue_A" in sim["counterfactual_differences"]
    assert "delta_revenue_B" in sim["counterfactual_differences"]
    assert "revenue_offset_ratio" in sim["cannibalization_metrics"]
    assert "unit_substitution_ratio" in sim["cannibalization_metrics"]
    assert sim["cannibalization_metrics"]["has_valid_revenue_denominator"] is True


def test_cannibalization_edge_case_non_positive_denominator(
    sample_panel: pd.DataFrame,
) -> None:
    """Verifies graceful handling when promoted product delta revenue is non-positive."""
    train_df = sample_panel[sample_panel["split"] == "train"].copy()
    val_df = sample_panel[sample_panel["split"] == "val"].copy()

    pair_model = CrossPriceElasticityModel(include_controls=True).fit(
        train_df, TEST_SKU_A, TEST_SKU_B
    )

    val_A = val_df[val_df["product_id"] == TEST_SKU_A]
    val_B = val_df[val_df["product_id"] == TEST_SKU_B]
    val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values("week_no")

    # If discount is 0% and no display, delta revenue is 0.0 -> denominator is non-positive!
    sim_zero = simulate_counterfactual_cannibalization(
        pair_model, val_merged, discount_pct=0.0, with_display=False
    )

    assert sim_zero["cannibalization_metrics"]["revenue_offset_ratio"] is None
    assert sim_zero["cannibalization_metrics"]["has_valid_revenue_denominator"] is False


def test_newey_west_hac_standard_errors(sample_panel: pd.DataFrame) -> None:
    """Verifies Newey-West HAC covariance matrix computation and positive standard errors."""
    train_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "train")
    ].copy()

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    assert model.hac_se_ is not None
    assert model.elasticity_hac_se_ is not None
    assert model.elasticity_hac_se_ > 0.0

    eval_res = model.evaluate(train_df)
    assert "elasticity_hac_se" in eval_res
    assert eval_res["elasticity_hac_se"] == model.elasticity_hac_se_

    # Verify direct compute_newey_west_cov function
    X, y = model._build_design_matrix(train_df)
    resids = y - X @ model.coef_
    cov_hac = compute_newey_west_cov(X, resids, max_lags=4)
    assert cov_hac.shape == (X.shape[1], X.shape[1])
    # Diagonals must be non-negative
    assert (np.diagonal(cov_hac) >= 0.0).all()


def test_validation_vs_test_metric_distinctness(sample_panel: pd.DataFrame) -> None:
    """Verifies that validation and test metrics are computed distinctly (guards against clerical copy-paste errors)."""
    train_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "train")
    ].copy()
    val_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "val")
    ].copy()
    test_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "test")
    ].copy()

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    val_eval = model.evaluate(val_df)
    test_eval = model.evaluate(test_df)

    # Validation and test periods have different observations, prices, and errors
    assert val_eval["rmse_units"] != test_eval["rmse_units"]
    assert val_eval["r2_log"] != test_eval["r2_log"]


def test_promotion_feature_bounds_and_exogeneity(sample_panel: pd.DataFrame) -> None:
    """Verifies that display_share and mailer_share are bounded in [0, 1] without negative values or outliers."""
    assert (sample_panel["display_share"] >= 0.0).all()
    assert (sample_panel["display_share"] <= 1.0).all()
    assert (sample_panel["mailer_share"] >= 0.0).all()
    assert (sample_panel["mailer_share"] <= 1.0).all()


def test_realistic_merchandising_baseline_simulation(
    sample_panel: pd.DataFrame,
) -> None:
    """Verifies counterfactual simulation under realistic status-quo merchandising baseline."""
    train_df = sample_panel[sample_panel["split"] == "train"].copy()
    val_df = sample_panel[sample_panel["split"] == "val"].copy()

    pair_model = CrossPriceElasticityModel(include_controls=True).fit(
        train_df, TEST_SKU_A, TEST_SKU_B
    )

    val_A = val_df[val_df["product_id"] == TEST_SKU_A]
    val_B = val_df[val_df["product_id"] == TEST_SKU_B]
    val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values("week_no")

    # Realistic baseline with observed merchandising, price-only discount (-20%, with_display=False)
    sim_realistic = simulate_counterfactual_cannibalization(
        pair_model,
        val_merged,
        discount_pct=0.20,
        with_display=False,
        baseline_mode="observed_merch",
    )

    assert sim_realistic["scenario"]["baseline_mode"] == "observed_merch"
    # Because SKU A is price-inelastic (eps_AA ~ -0.49), a price-only discount reduces revenue
    delta_rev_A = sim_realistic["counterfactual_differences"]["delta_revenue_A"]
    assert delta_rev_A < 0.0, (
        f"Expected negative delta revenue under inelastic demand, got {delta_rev_A}"
    )
    # Revenue offset ratio must be None due to non-positive denominator
    assert sim_realistic["cannibalization_metrics"]["revenue_offset_ratio"] is None
    assert sim_realistic["cannibalization_metrics"]["has_valid_revenue_denominator"] is False


def test_hac_standard_errors_match_statsmodels(sample_panel: pd.DataFrame) -> None:
    """Verifies that custom Newey-West HAC covariance matches statsmodels within numerical tolerance."""
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac

    train_df = sample_panel[
        (sample_panel["product_id"] == TEST_SKU_A) & (sample_panel["split"] == "train")
    ].copy()

    model = PaidPriceElasticityOLS(include_controls=True).fit(train_df)
    X, y = model._build_design_matrix(train_df)
    resids = y - X @ model.coef_

    cov_custom = compute_newey_west_cov(X, resids, max_lags=4)
    se_custom = np.sqrt(np.maximum(np.diagonal(cov_custom), 1e-12))

    ols_sm = sm.OLS(y, X).fit()
    cov_sm = cov_hac(ols_sm, nlags=4, use_correction=True)
    se_sm = np.sqrt(np.diagonal(cov_sm))

    np.testing.assert_allclose(se_custom, se_sm, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cov_custom, cov_sm, rtol=1e-5, atol=1e-6)


def test_counterfactual_empirical_support_flags(sample_panel: pd.DataFrame) -> None:
    """Verifies that empirical support flags detect extrapolative merchandising and extreme discounts."""
    train_df = sample_panel[sample_panel["split"] == "train"].copy()
    val_df = sample_panel[sample_panel["split"] == "val"].copy()

    pair_model = CrossPriceElasticityModel(include_controls=True).fit(
        train_df, TEST_SKU_A, TEST_SKU_B
    )

    val_A = val_df[val_df["product_id"] == TEST_SKU_A]
    val_B = val_df[val_df["product_id"] == TEST_SKU_B]
    val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values("week_no")

    # 1. Extrapolative merchandising scenario (zero-merch baseline to 100% display)
    sim_extra = simulate_counterfactual_cannibalization(
        pair_model, val_merged, discount_pct=0.20, with_display=True, baseline_mode="zero_merch"
    )
    support_extra = sim_extra["scenario"]["empirical_support"]
    assert support_extra["is_extrapolative_merchandising"] is True
    assert support_extra["merchandising_support_warning"] is not None
    assert support_extra["discount_pct_supported"] is True

    # 2. Supported status-quo scenario
    sim_sq = simulate_counterfactual_cannibalization(
        pair_model,
        val_merged,
        discount_pct=0.20,
        with_display=False,
        baseline_mode="observed_merch",
    )
    support_sq = sim_sq["scenario"]["empirical_support"]
    assert support_sq["is_extrapolative_merchandising"] is False
    assert support_sq["merchandising_support_warning"] is None

    # 3. Extreme discount out-of-support scenario (e.g. 60% discount)
    sim_deep = simulate_counterfactual_cannibalization(
        pair_model,
        val_merged,
        discount_pct=0.60,
        with_display=False,
        baseline_mode="observed_merch",
    )
    support_deep = sim_deep["scenario"]["empirical_support"]
    assert support_deep["discount_pct_supported"] is False
    assert support_deep["discount_warning"] is not None
