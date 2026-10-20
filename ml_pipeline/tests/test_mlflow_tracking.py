"""Unit and integration tests for MLflow Experiment Tracking & Model Registry (Phase 12).

Verifies that all PromoSense modeling experiments, hyperparameters, validation/test
metrics, and model artifacts are tracked deterministically into MLflow without network dependencies.
"""

from pathlib import Path
from typing import Any

import mlflow
import pytest

from backend.app.services.model_registry import ModelRegistry
from ml_pipeline.src.models.mlflow_tracker import MLflowTracker


@pytest.fixture
def temp_mlflow_tracker(tmp_path: Path) -> MLflowTracker:
    """Creates an MLflowTracker instance backed by a temporary local SQLite DB."""
    db_file = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_file.resolve().as_posix()}"
    return MLflowTracker(tracking_uri=tracking_uri)


def test_mlflow_tracker_initialization(temp_mlflow_tracker: MLflowTracker) -> None:
    """Verifies tracker sets and exposes the configured tracking URI."""
    assert temp_mlflow_tracker.tracking_uri.startswith("sqlite:")
    assert mlflow.get_tracking_uri() == temp_mlflow_tracker.tracking_uri


def test_get_or_create_experiment(temp_mlflow_tracker: MLflowTracker) -> None:
    """Verifies idempotent experiment creation."""
    exp_id_1 = temp_mlflow_tracker.get_or_create_experiment("Test-Experiment")
    assert exp_id_1 is not None

    # Calling again returns same experiment ID
    exp_id_2 = temp_mlflow_tracker.get_or_create_experiment("Test-Experiment")
    assert exp_id_1 == exp_id_2


def test_log_baseline_demand_experiment(temp_mlflow_tracker: MLflowTracker, tmp_path: Path) -> None:
    """Verifies baseline demand model metrics, parameters, and tags are logged."""
    fake_champ_artifact = tmp_path / "champion_baseline_demand_model.joblib"
    fake_champ_artifact.write_text("dummy_binary_payload")

    benchmark_data: dict[str, Any] = {
        "baseline_demand_validation": {
            "champion_model_name": "Ridge Demand (alpha=100)",
            "models_evaluated": {
                "Global Mean Baseline": {
                    "train_time_sec": 0.05,
                    "all_controls": {"rmse": 45.2, "mae": 32.1, "r2": 0.05, "pct_bias": 1.2},
                    "strict_clean_controls": {
                        "rmse": 44.0,
                        "mae": 31.0,
                        "r2": 0.06,
                        "pct_bias": 0.8,
                    },
                },
                "Ridge Demand (alpha=100)": {
                    "train_time_sec": 0.12,
                    "all_controls": {"rmse": 38.5, "mae": 26.4, "r2": 0.28, "pct_bias": -0.5},
                    "strict_clean_controls": {
                        "rmse": 37.1,
                        "mae": 25.0,
                        "r2": 0.31,
                        "pct_bias": -0.2,
                    },
                },
            },
        },
        "final_test_evaluation": {
            "baseline_demand_test": {
                "all_models": {
                    "Ridge Demand (alpha=100)": {
                        "all_controls": {"rmse": 39.1, "mae": 27.0, "r2": 0.27, "pct_bias": -0.4},
                        "strict_clean_controls": {
                            "rmse": 37.8,
                            "mae": 25.5,
                            "r2": 0.29,
                            "pct_bias": -0.1,
                        },
                    }
                }
            }
        },
    }

    run_ids = temp_mlflow_tracker.log_baseline_demand(
        benchmark_results=benchmark_data,
        experiment_name="Test-Baseline-Demand",
        artifacts_dir=tmp_path,
    )

    assert len(run_ids) == 2

    client = mlflow.tracking.MlflowClient()
    # Check champion run
    champ_run = client.get_run(run_ids[1])
    assert champ_run.data.tags["is_champion"] == "True"
    assert champ_run.data.tags["project"] == "PromoSense"
    assert champ_run.data.params["model_name"] == "Ridge Demand (alpha=100)"
    assert champ_run.data.metrics["val_all_rmse"] == 38.5
    assert champ_run.data.metrics["test_all_rmse"] == 39.1


def test_log_causal_uplift_experiment(temp_mlflow_tracker: MLflowTracker, tmp_path: Path) -> None:
    """Verifies causal uplift metrics (Qini, AUUC, Top-Decile lift) are logged."""
    fake_champ = tmp_path / "champion_uplift_model.joblib"
    fake_champ.write_text("dummy_uplift_model")

    uplift_data: dict[str, Any] = {
        "propensity_diagnostics": {
            "covariate_balance": {
                "total_features_checked": 24,
                "balanced_raw_count": 18,
                "balanced_ipw_count": 23,
                "mean_abs_smd_raw": 0.12,
                "mean_abs_smd_ipw": 0.04,
            }
        },
        "causal_uplift_validation": {
            "T-Learner (LightGBM)": {
                "fit_time_seconds": 1.25,
                "qini_score": 1420.5,
                "auuc": 1580.2,
                "top_decile_lift": 12.4,
                "top_decile_multiplier": 1.85,
                "cate_mean": 5.2,
                "cate_std": 3.1,
            }
        },
        "final_test_evaluation": {
            "causal_uplift_test": {
                "champion_model_name": "T-Learner (LightGBM)",
                "all_models": {
                    "T-Learner (LightGBM)": {
                        "qini_score": 1390.0,
                        "auuc": 1540.0,
                        "top_decile_lift": 11.8,
                        "top_decile_multiplier": 1.78,
                    }
                },
            }
        },
    }

    run_ids = temp_mlflow_tracker.log_causal_uplift(
        benchmark_results=uplift_data,
        experiment_name="Test-Causal-Uplift",
        artifacts_dir=tmp_path,
    )

    assert len(run_ids) == 2  # 1 propensity run + 1 uplift run
    client = mlflow.tracking.MlflowClient()

    p_run = client.get_run(run_ids[0])
    assert p_run.data.metrics["balanced_ipw_count"] == 23.0

    u_run = client.get_run(run_ids[1])
    assert u_run.data.metrics["val_qini_score"] == 1420.5
    assert u_run.data.metrics["test_qini_score"] == 1390.0
    assert u_run.data.tags["is_champion"] == "True"


def test_log_price_elasticity_experiment(temp_mlflow_tracker: MLflowTracker) -> None:
    """Verifies price elasticity and cannibalization metrics are logged."""
    elasticity_data: dict[str, Any] = {
        "own_price_elasticities": {
            "5569230": {
                "product_id": 5569230,
                "commodity": "SOFT DRINKS",
                "brand_type": "National",
                "models": {
                    "controlled_paid_ols": {
                        "elasticity": -2.35,
                        "elasticity_se": 0.22,
                        "val_eval": {"rmse": 0.18, "r2": 0.65},
                        "test_eval": {"rmse": 0.19, "r2": 0.63},
                    },
                    "decomposed_shelf_discount_ols": {
                        "base_shelf_elasticity": -2.10,
                        "discount_elasticity": 2.80,
                        "collinearity_vif": 1.12,
                    },
                },
            }
        },
        "cannibalization_analysis": {
            "Soft_Drinks_12pk_National": {
                "pair_name": "Soft_Drinks_12pk_National",
                "prod_A": 5569230,
                "prod_B": 8090521,
                "elasticities": {
                    "eps_AA_own": -2.35,
                    "eps_AB_cross": 0.45,
                    "eps_BB_own": -2.10,
                    "eps_BA_cross": 0.38,
                },
                "val_simulation": {
                    "counterfactual_differences": {
                        "delta_revenue_A": 1354.69,
                        "delta_revenue_B": -11.20,
                        "net_category_revenue_lift": 1343.49,
                    },
                    "cannibalization_metrics": {
                        "revenue_offset_ratio": 0.0083,
                    },
                },
            }
        },
    }

    run_ids = temp_mlflow_tracker.log_price_elasticity(
        elasticity_results=elasticity_data,
        experiment_name="Test-Price-Elasticity",
    )

    assert len(run_ids) == 2  # 1 SKU run + 1 Pair run
    client = mlflow.tracking.MlflowClient()

    sku_run = client.get_run(run_ids[0])
    assert sku_run.data.metrics["elasticity_controlled_paid_ols"] == -2.35
    assert sku_run.data.metrics["elasticity_hac_se"] == 0.22
    assert sku_run.data.metrics["decomposed_vif"] == 1.12

    pair_run = client.get_run(run_ids[1])
    assert pair_run.data.metrics["cross_elasticity_AB"] == 0.45
    assert pair_run.data.metrics["val_net_category_revenue_lift"] == 1343.49


def test_log_simulation_optimization_experiment(
    temp_mlflow_tracker: MLflowTracker,
) -> None:
    """Verifies promotion simulation and optimization policies are logged."""
    sim_data: dict[str, Any] = {
        "single_sku_optimization": {
            "5569230": {
                "product_id": 5569230,
                "price_only_optimization": {
                    "optimal_policy_by_margin_tier": {
                        "margin_15pct": {
                            "optimal_discount_pct": 0.15,
                            "delta_profit_at_optimum": 240.50,
                        },
                        "margin_25pct": {
                            "optimal_discount_pct": 0.20,
                            "delta_profit_at_optimum": 512.80,
                        },
                        "margin_35pct": {
                            "optimal_discount_pct": 0.25,
                            "delta_profit_at_optimum": 820.10,
                        },
                    }
                },
            }
        },
        "test_out_of_sample_realizations": {
            "5569230": {
                "applied_discount_pct": 0.20,
                "test_delta_profit_25pct": 498.20,
                "test_delta_revenue": 1310.40,
                "test_delta_units": 650.0,
            }
        },
    }

    run_ids = temp_mlflow_tracker.log_simulation_optimization(
        simulation_results=sim_data,
        experiment_name="Test-Simulation-Optimization",
    )

    assert len(run_ids) == 1
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_ids[0])

    assert run.data.metrics["margin_25pct_optimal_discount"] == 0.20
    assert run.data.metrics["margin_25pct_optimal_profit_lift"] == 512.80
    assert run.data.metrics["test_delta_profit_25pct"] == 498.20


def test_sync_all_frozen_artifacts_to_mlflow(
    temp_mlflow_tracker: MLflowTracker,
) -> None:
    """Verifies end-to-end synchronization of frozen repo artifacts to MLflow."""
    repo_artifacts_dir = Path(__file__).resolve().parents[2] / "models" / "artifacts"
    if not repo_artifacts_dir.is_dir():
        pytest.skip("models/artifacts directory not found")

    runs = temp_mlflow_tracker.sync_all_frozen_artifacts(artifacts_dir=repo_artifacts_dir)

    assert len(runs["phase_5_demand"]) >= 4
    assert len(runs["phase_5_uplift"]) >= 5
    assert len(runs["phase_6_elasticity"]) >= 7
    assert len(runs["phase_7_simulation"]) >= 7


def test_mlproject_file_validity() -> None:
    """Verifies ml_pipeline/MLProject file exists, is non-empty, and has required entry points."""
    mlproject_path = Path(__file__).resolve().parents[1] / "MLProject"
    assert mlproject_path.is_file()
    content = mlproject_path.read_text(encoding="utf-8")
    assert "name: PromoSense" in content
    assert "entry_points:" in content
    assert "track_all:" in content
    assert "run_phase5_benchmark:" in content
    assert "run_elasticity:" in content
    assert "run_simulation:" in content


def test_model_registry_mlflow_status() -> None:
    """Verifies ModelRegistry exposes get_mlflow_status cleanly without exceptions."""
    registry = ModelRegistry.get_instance()
    status = registry.get_mlflow_status()

    assert "tracking_uri" in status
    assert "is_connected" in status
    assert isinstance(status["is_connected"], bool)
