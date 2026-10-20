"""Tests for /api/simulate/promotion endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_simulate_promotion_standard_20pct_matches_engine():
    payload = {
        "product_id": 5569230,
        "discount_pct": 0.20,
        "with_display": False,
        "baseline_mode": "observed_merch",
        "include_substitute": True,
    }
    response = client.post("/api/simulate/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check scenario echo
    assert data["scenario"]["product_id"] == 5569230
    assert data["scenario"]["discount_pct"] == 0.20
    assert data["scenario"]["is_unpromoted_baseline_comparison"] is False

    # Check Single SKU demand & revenue results against Phase 7 verified figures
    single = data["single_sku"]
    assert single["product_id"] == 5569230
    assert abs(single["avg_shelf_price"] - 4.7352) < 0.01
    assert abs(single["promotional_price"] - 3.7881) < 0.01
    assert abs(single["delta_units"] - 532.02) < 1.0  # +532.02 units
    assert abs(single["delta_revenue"] - 1354.69) < 2.0  # +$1,354.69
    assert abs(single["pct_unit_lift"] - 76.05) < 0.5  # +76.05%

    # Check margin tiers & margin erosion paradox
    profits = single["contribution_profits"]
    assert "margin_15pct" in profits
    assert "margin_25pct" in profits
    assert "margin_35pct" in profits

    # At 25% margin, 20% discount produces -$537.32 profit lift
    pi_25 = profits["margin_25pct"]
    assert abs(pi_25["delta_profit"] - (-537.32)) < 2.0
    assert pi_25["is_profitable_lift"] is False

    # Check uncertainty bounds
    assert single["profit_uncertainty_hac"]["uncertainty_type"] == "partial_own_price_hac_bound"

    # Check substitute cannibalization
    assert data["substitute_pair"] is not None
    pair = data["substitute_pair"]
    assert pair["pair_name"] == "Soft_Drinks_12pk_National"
    assert pair["prod_B_id"] == 8090521
    assert pair["delta_units_B"] < 0  # substitute loses volume
    assert pair["delta_revenue_B"] < 0  # substitute loses revenue
    assert pair["revenue_offset_ratio"] is not None
    assert abs(pair["revenue_offset_ratio"] - 0.0592) < 0.01  # 5.92% offset


def test_simulate_promotion_zero_discount_baseline_recovery():
    """Validates baseline recovery: 0.0 discount evaluates baseline against itself returning exact 0s."""
    payload = {
        "product_id": 5569230,
        "discount_pct": 0.0,
        "with_display": False,
        "baseline_mode": "observed_merch",
    }
    response = client.post("/api/simulate/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["scenario"]["is_unpromoted_baseline_comparison"] is True

    single = data["single_sku"]
    assert single["delta_units"] == 0.0
    assert single["delta_revenue"] == 0.0
    assert single["pct_unit_lift"] == 0.0
    assert single["pct_revenue_lift"] == 0.0

    for tier in single["contribution_profits"].values():
        assert tier["delta_profit"] == 0.0
        assert tier["pct_profit_lift"] == 0.0

    assert data["empirical_support"]["is_jointly_supported"] is True
    assert data["empirical_support"]["discount_supported"] is True


def test_simulate_promotion_co_display():
    payload = {
        "product_id": 5569230,
        "discount_pct": 0.20,
        "with_display": True,
        "baseline_mode": "observed_merch",
    }
    response = client.post("/api/simulate/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()

    single = data["single_sku"]
    # Co-display produces ~793.2 units lift and ~$2,354.52 revenue lift
    assert abs(single["delta_units"] - 793.22) < 2.0
    assert abs(single["delta_revenue"] - 2354.52) < 2.0
    assert abs(single["contribution_profits"]["margin_25pct"]["delta_profit"] - (-474.83)) < 2.0


def test_simulate_promotion_rejects_unsupported_discounts():
    # 2% discount is below active 5% threshold
    resp_low = client.post(
        "/api/simulate/promotion",
        json={"product_id": 5569230, "discount_pct": 0.02},
    )
    assert resp_low.status_code == 422
    assert "below minimum" in resp_low.text.lower()

    # 40% discount exceeds 35% empirical ceiling
    resp_high = client.post(
        "/api/simulate/promotion",
        json={"product_id": 5569230, "discount_pct": 0.40},
    )
    assert resp_high.status_code == 422
    assert "0.35" in resp_high.text.lower()


def test_simulate_promotion_rejects_invalid_sku():
    response = client.post(
        "/api/simulate/promotion",
        json={"product_id": 9999999, "discount_pct": 0.15},
    )
    assert response.status_code == 404
    assert "not among modeled target skus" in response.text.lower()


def test_simulate_promotion_milk_pair_warning():
    payload = {
        "product_id": 1029743,
        "discount_pct": 0.20,
        "include_substitute": True,
    }
    response = client.post("/api/simulate/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["substitute_pair"] is not None
    assert data["substitute_pair"]["is_weakly_identified"] is True
    assert data["substitute_pair"]["weak_identification_warning"] is not None
    assert "weakly identified" in data["substitute_pair"]["weak_identification_warning"].lower()


def test_simulate_promotion_exclude_substitute():
    payload = {
        "product_id": 5569230,
        "discount_pct": 0.20,
        "include_substitute": False,
    }
    response = client.post("/api/simulate/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["substitute_pair"] is None
