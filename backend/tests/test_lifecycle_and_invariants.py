"""Tests for artifact lifecycle, error handling, and engine-to-API consistency invariants."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_registry import ModelRegistry
from ml_pipeline.src.optimization.optimizer import BoundedPromotionOptimizer
from ml_pipeline.src.optimization.simulator import simulate_single_sku_promotion

client = TestClient(app)


def test_registry_raises_on_missing_artifacts(tmp_path: Path):
    """Verifies that ModelRegistry halts with RuntimeError rather than synthesizing models."""
    empty_dir = tmp_path / "empty_artifacts"
    empty_dir.mkdir()

    registry = ModelRegistry(artifacts_dir=empty_dir)
    with pytest.raises(RuntimeError) as exc_info:
        registry.load_artifacts()

    assert "missing required promosense model artifacts" in str(exc_info.value).lower()


def test_api_simulation_matches_direct_engine():
    """Verifies that API simulation outputs exactly match direct promotion simulator calls."""
    registry = ModelRegistry.get_instance()
    if not registry.is_initialized:
        registry.load_artifacts()

    pid = 5569230
    model = registry.get_single_model(pid)
    eval_df = registry.get_eval_df(pid, split="val")

    # Direct simulator call
    direct_res = simulate_single_sku_promotion(
        model=model,
        eval_df=eval_df,
        discount_pct=0.15,
        with_display=False,
    )

    # API call
    response = client.post(
        "/api/simulate/promotion",
        json={
            "product_id": pid,
            "discount_pct": 0.15,
            "with_display": False,
            "include_substitute": False,
        },
    )
    assert response.status_code == 200
    api_res = response.json()

    api_single = api_res["single_sku"]
    assert api_single["delta_units"] == direct_res["volumes"]["delta_units"]
    assert api_single["delta_revenue"] == direct_res["revenues"]["delta_revenue"]
    assert (
        api_single["contribution_profits"]["margin_25pct"]["delta_profit"]
        == direct_res["contribution_profits"]["margin_25pct"]["delta_profit"]
    )


def test_api_optimization_matches_direct_engine():
    """Verifies that API optimization outputs match direct BoundedPromotionOptimizer calls."""
    registry = ModelRegistry.get_instance()
    if not registry.is_initialized:
        registry.load_artifacts()

    pid = 844165
    model = registry.get_single_model(pid)
    eval_df = registry.get_eval_df(pid, split="val")

    # Direct optimizer call
    optimizer = BoundedPromotionOptimizer()
    direct_opt = optimizer.optimize_single_sku(
        model=model,
        eval_df=eval_df,
        with_display=False,
    )

    # API call
    response = client.post(
        "/api/optimize/promotion",
        json={
            "product_id": pid,
            "with_display": False,
            "include_substitute": False,
        },
    )
    assert response.status_code == 200
    api_res = response.json()

    direct_p25 = direct_opt["optimal_policy_by_margin_tier"]["margin_25pct"]
    api_p25 = api_res["single_sku_policies"]["margin_25pct"]

    assert api_p25["optimal_discount_pct"] == direct_p25["optimal_discount_pct"]
    assert api_p25["recommend_price_promotion"] == bool(direct_p25["recommend_price_promotion"])
    assert api_p25["delta_profit_at_optimum"] == direct_p25["delta_profit_at_optimum"]


def test_simulation_revenue_and_profit_invariants():
    """Verifies revenue arithmetic, profit arithmetic, and 0% baseline identity for SKU 5569230."""
    response = client.post(
        "/api/simulate/promotion",
        json={
            "product_id": 5569230,
            "discount_pct": 0.20,
            "with_display": False,
            "baseline_mode": "observed_merch",
            "include_substitute": True,
        },
    )
    assert response.status_code == 200
    data = response.json()["single_sku"]

    # 1. Authoritative revenue values
    assert data["revenue_baseline"] == 3317.26
    assert data["revenue_promoted"] == 4671.95
    assert data["delta_revenue"] == 1354.69
    assert round(data["revenue_promoted"] - data["revenue_baseline"], 2) == data["delta_revenue"]

    # 2. Contribution profit arithmetic
    for tier in data["contribution_profits"].values():
        assert round(tier["profit_promoted"] - tier["profit_baseline"], 2) == tier["delta_profit"]

    # 3. Zero-discount unpromoted baseline identity
    zero_resp = client.post(
        "/api/simulate/promotion",
        json={
            "product_id": 5569230,
            "discount_pct": 0.0,
            "with_display": False,
            "baseline_mode": "observed_merch",
            "include_substitute": True,
        },
    )
    assert zero_resp.status_code == 200
    zero_data = zero_resp.json()["single_sku"]
    assert zero_data["delta_units"] == 0.0
    assert zero_data["delta_revenue"] == 0.0
    for tier in zero_data["contribution_profits"].values():
        assert tier["delta_profit"] == 0.0
