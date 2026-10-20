"""Tests for /api/optimize/promotion endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_optimize_promotion_price_only_5569230():
    payload = {
        "product_id": 5569230,
        "with_display": False,
        "baseline_mode": "observed_merch",
    }
    response = client.post("/api/optimize/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == 5569230
    assert data["with_display"] is False

    # Check explicit no-promotion baseline
    base_point = data["no_promotion_baseline"]
    assert base_point["discount_pct"] == 0.0
    assert base_point["delta_profit"] == 0.0
    assert base_point["delta_revenue"] == 0.0

    # For price-only, all active discounts destroy profit at standard 25% margin
    policies = data["single_sku_policies"]
    policy_25 = policies["margin_25pct"]
    assert policy_25["optimal_discount_pct"] == 0.0
    assert policy_25["recommend_price_promotion"] is False
    assert policy_25["delta_profit_at_optimum"] == 0.0
    assert policy_25["modeled_best_promotional_candidate"] == 0.05
    assert "recommend maintaining regular shelf price" in policy_25["recommendation_note"].lower()

    # Grid sweep points
    sweep = data["single_sku_grid_sweep"]
    assert len(sweep) == 13  # 0.05 to 0.35 in 0.025 steps
    discounts = [pt["discount_pct"] for pt in sweep]
    assert discounts[0] == 0.05
    assert discounts[-1] == 0.35


def test_optimize_promotion_co_display_5569230_yields_shallow_optimum():
    payload = {
        "product_id": 5569230,
        "with_display": True,
        "baseline_mode": "observed_merch",
    }
    response = client.post("/api/optimize/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()

    policies = data["single_sku_policies"]

    # At 25% margin, co-display makes a shallow 5% discount profitable (+87.94)
    p25 = policies["margin_25pct"]
    assert p25["optimal_discount_pct"] == 0.05
    assert p25["recommend_price_promotion"] is True
    assert abs(p25["delta_profit_at_optimum"] - 87.94) < 1.0
    assert abs(p25["pct_profit_lift_at_optimum"] - 10.60) < 0.5

    # At 35% margin, co-display yields +$214.85 lift at 5% discount
    p35 = policies["margin_35pct"]
    assert p35["optimal_discount_pct"] == 0.05
    assert p35["recommend_price_promotion"] is True
    assert abs(p35["delta_profit_at_optimum"] - 214.85) < 1.0


def test_optimize_promotion_inelastic_staple():
    payload = {
        "product_id": 1082185,  # Bananas
        "with_display": False,
    }
    response = client.post("/api/optimize/promotion", json=payload)
    assert response.status_code == 200
    data = response.json()

    for p_tier in data["single_sku_policies"].values():
        assert p_tier["optimal_discount_pct"] == 0.0
        assert p_tier["recommend_price_promotion"] is False
        assert p_tier["delta_profit_at_optimum"] == 0.0


def test_optimize_promotion_invalid_bounds():
    payload = {
        "product_id": 5569230,
        "min_discount": 0.30,
        "max_discount": 0.10,  # min > max
    }
    response = client.post("/api/optimize/promotion", json=payload)
    assert response.status_code == 422
    assert "cannot exceed" in response.text.lower()


def test_optimize_promotion_invalid_sku():
    payload = {
        "product_id": 9999999,
    }
    response = client.post("/api/optimize/promotion", json=payload)
    assert response.status_code == 404
    assert "not among modeled target skus" in response.text.lower()
