"""Integration tests for Executive Dashboard endpoints (/api/dashboard/*)."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_dashboard_summary_endpoint():
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    data = response.json()

    # Fact-checking against empirical transactions panel
    assert data["total_sales_dollars"] > 8_000_000.0
    assert 0.30 < data["promotional_sales_share"] < 0.60
    assert data["total_households"] == 2500
    assert data["total_campaigns"] == 30
    assert data["modeled_target_skus"] == 7
    assert data["gross_margin_tiers"] == [0.15, 0.25, 0.35]

    # Portfolio boundary caveat
    assert "7 high-volume target SKUs" in data["portfolio_scope_disclaimer"]
    assert "92,353" in data["portfolio_scope_disclaimer"]

    # Methodological disclosure
    warnings = data["methodological_warnings"]
    assert "gross-margin tiers" in warnings["cost_basis_assumption"].lower()
    assert "observational" in warnings["observational_unconfoundedness_disclaimer"].lower()


def test_dashboard_sales_trend_endpoint():
    response = client.get("/api/dashboard/sales-trend")
    assert response.status_code == 200
    data = response.json()

    assert data["total_weeks"] == 102
    assert len(data["trend"]) == 102

    w1 = data["trend"][0]
    assert w1["week_no"] == 1
    assert w1["total_sales"] > 0
    assert round(w1["base_sales"] + w1["promotional_sales"], 2) == round(w1["total_sales"], 2)

    w102 = data["trend"][-1]
    assert w102["week_no"] == 102
    assert w102["total_sales"] > 0


def test_dashboard_promotions_endpoint():
    response = client.get("/api/dashboard/promotions")
    assert response.status_code == 200
    data = response.json()

    assert data["total_campaigns"] == 30
    assert len(data["campaigns"]) == 30
    assert data["pooled_ate_spend_lift"] > 0.0
    assert data["top_decile_lift"] >= data["pooled_ate_spend_lift"]

    c_types = data["campaign_types"]
    assert "TypeA" in c_types
    assert "TypeB" in c_types
    assert "TypeC" in c_types

    c1 = data["campaigns"][0]
    assert "campaign_id" in c1
    assert "households_targeted" in c1
    assert c1["duration_days"] > 0
