"""Integration tests for Customer Intelligence and Uplift endpoints (/api/customers/*)."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_get_customer_segments_endpoint():
    response = client.get("/api/customers/segments")
    assert response.status_code == 200
    data = response.json()

    assert data["qini_score"] > 0.0
    assert data["auuc"] > 0.0
    assert data["overall_cohort_ate"] > 0.0
    assert data["top_decile_lift"] > 0.0
    assert data["recommended_targeting_cutoff_decile"] == 3

    assert len(data["deciles"]) == 10
    d1 = data["deciles"][0]
    assert d1["decile"] == 1
    assert d1["population_fraction"] == 0.1
    assert d1["cum_gain"] > 0.0

    d10 = data["deciles"][-1]
    assert d10["decile"] == 10
    assert d10["population_fraction"] == 1.0


def test_get_customer_detail_with_demographics():
    response = client.get("/api/customers/1")
    assert response.status_code == 200
    data = response.json()

    assert data["household_key"] == 1
    demo = data["demographics"]
    assert demo["has_demographics"] is True
    assert demo["age_band"] == "65+"

    beh = data["behavior"]
    assert beh["historical_spend_dollars"] > 1000.0
    assert beh["historical_trips_count"] > 10


def test_get_customer_detail_without_demographics():
    response = client.get("/api/customers/2")
    assert response.status_code == 200
    data = response.json()

    assert data["household_key"] == 2
    demo = data["demographics"]
    assert demo["has_demographics"] is False
    assert demo["age_band"] is None

    beh = data["behavior"]
    assert beh["historical_spend_dollars"] > 0.0


def test_get_customer_detail_invalid_id():
    response = client.get("/api/customers/9999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()

    response_zero = client.get("/api/customers/0")
    assert response_zero.status_code == 404


def test_get_customer_uplift_valid():
    response = client.get("/api/customers/1/uplift")
    assert response.status_code == 200
    data = response.json()

    assert data["household_key"] == 1
    assert "uplift_score_cate" in data
    assert data["targeting_segment"] in ["Persuadables", "Sure Things", "Lost Causes"]
    assert data["recommended_action"] in ["PROMOTE", "NO PROMOTION", "AVOID DISCOUNT"]
    assert data["expected_spend_with_treatment"] >= 0.0
    assert data["expected_spend_without_treatment"] >= 0.0
    assert data["cost_threshold_dollars"] == 1.00

    # Decision-theoretic consistency
    if data["uplift_score_cate"] >= 1.00:
        assert data["is_recommended_for_treatment"] is True
        assert data["recommended_action"] == "PROMOTE"
    else:
        assert data["is_recommended_for_treatment"] is False

    warnings = data["methodological_warnings"]
    assert "cost" in warnings["cost_basis_assumption"].lower()
    assert "observational" in warnings["observational_unconfoundedness_disclaimer"].lower()


def test_get_customer_uplift_invalid_id():
    response = client.get("/api/customers/9999/uplift")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()
