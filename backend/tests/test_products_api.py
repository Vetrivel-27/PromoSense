"""Tests for /api/products and /api/products/{id}/elasticity endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_list_products_endpoint():
    response = client.get("/api/products")
    assert response.status_code == 200
    data = response.json()

    assert "products" in data
    assert "substitute_pairs" in data
    assert data["total_products"] == 7

    product_ids = [p["product_id"] for p in data["products"]]
    assert 5569230 in product_ids
    assert 844165 in product_ids
    assert 1082185 in product_ids

    # Check pair coverage
    pairs = data["substitute_pairs"]
    assert len(pairs) == 3
    pair_names = [p["pair_name"] for p in pairs]
    assert "Soft_Drinks_12pk_National" in pair_names
    assert "Fluid_Milk_1GA_PrivateLabel" in pair_names

    # Check product attributes
    can_a = next(p for p in data["products"] if p["product_id"] == 5569230)
    assert can_a["commodity"] == "SOFT DRINKS"
    assert can_a["brand_type"] == "National"
    assert can_a["baseline_shelf_price"] > 3.0
    assert can_a["is_focal_substitute_a"] is True


def test_get_product_elasticity_elastic_sku():
    response = client.get("/api/products/5569230/elasticity")
    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == 5569230
    assert data["champion_elasticity"] < -2.0  # ~ -2.53
    assert data["champion_hac_se"] > 0.0
    assert len(data["champion_ci_95"]) == 2
    assert data["champion_ci_95"][0] < data["champion_elasticity"] < data["champion_ci_95"][1]
    assert "elastic" in data["interpretation"].lower()

    # Check substitute pair attachment
    assert data["substitute_pair_elasticity"] is not None
    sub = data["substitute_pair_elasticity"]
    assert sub["substitute_product_id"] == 8090521
    assert sub["cross_elasticity_AB"] > 0.0  # ~ +0.246
    assert sub["is_weakly_identified"] is False

    # Check warnings
    warnings = data["methodological_warnings"]
    assert "observational" in warnings["observational_unconfoundedness_disclaimer"].lower()


def test_get_product_elasticity_inelastic_staple():
    response = client.get("/api/products/1082185/elasticity")
    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == 1082185
    assert abs(data["champion_elasticity"]) < 0.1  # ~ +0.0018
    assert "inelastic" in data["interpretation"].lower()


def test_get_product_elasticity_weakly_identified_milk():
    response = client.get("/api/products/1029743/elasticity")
    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == 1029743
    assert data["substitute_pair_elasticity"] is not None
    sub = data["substitute_pair_elasticity"]
    assert sub["is_weakly_identified"] is True
    assert "weakly identified" in sub["identification_note"].lower()

    # Check structured warning banner
    assert data["methodological_warnings"]["cross_price_warning"] is not None
    assert "weakly identified" in data["methodological_warnings"]["cross_price_warning"].lower()


def test_get_product_elasticity_invalid_sku():
    response = client.get("/api/products/9999999/elasticity")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_get_product_detail_endpoint():
    response = client.get("/api/products/5569230")
    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == 5569230
    assert data["commodity"] == "SOFT DRINKS"
    assert data["brand_type"] == "National"
    assert data["baseline_shelf_price"] > 3.0
    assert data["min_observed_price"] <= data["max_observed_price"]
    assert data["historical_weekly_volume_mean"] > 0
    assert len(data["substitute_pair_details"]) >= 1


def test_get_product_detail_invalid_sku():
    response = client.get("/api/products/9999999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()
