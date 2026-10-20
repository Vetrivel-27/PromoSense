"""MLflow Experiment Tracking and Model Registry Management for PromoSense.

Provides unified logging of hyperparameters, validation/test metrics, dataset versions,
and serialized model artifacts across all PromoSense modeling phases:
- Phase 5: Baseline Demand Forecasting, Propensity Diagnostics, Causal Uplift Modeling
- Phase 6: Price Elasticity (Own & Cross), Cannibalization Accounting
- Phase 7: Promotion Simulation & Category Profit Optimization

Maintains methodological integrity:
- Preserves all frozen model weights and parameter estimates.
- Does not modify established econometric splits or accounting identities.
- Supports both local file-based tracking and remote PostgreSQL-backed MLflow servers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow

from ml_pipeline.src.config.settings import ml_settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manages PromoSense experiment tracking, run logging, and artifact registration."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        """Initializes the MLflow tracker.

        Parameters
        ----------
        tracking_uri : str | None
            Tracking URI for MLflow. Defaults to MLFLOW_TRACKING_URI env var or
            a local SQLite database 'sqlite:///mlflow.db' inside the PromoSense root.
        """
        # Ensure file-store compatibility across all MLflow versions
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        if tracking_uri is None:
            env_uri = os.getenv("MLFLOW_TRACKING_URI")
            if env_uri:
                self.tracking_uri = env_uri
            else:
                db_path = (ml_settings.BASE_DIR / "mlflow.db").resolve()
                self.tracking_uri = f"sqlite:///{db_path.as_posix()}"
        else:
            self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)

    def get_or_create_experiment(self, experiment_name: str) -> str:
        """Retrieves an existing experiment ID or creates a new experiment."""
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None:
            return exp.experiment_id
        return mlflow.create_experiment(experiment_name)

    def log_baseline_demand(
        self,
        benchmark_results: dict[str, Any],
        experiment_name: str = "PromoSense-Demand-Baseline",
        artifacts_dir: Path | None = None,
    ) -> list[str]:
        """Logs Phase 5 Baseline Demand models to MLflow.

        Parameters
        ----------
        benchmark_results : dict[str, Any]
            Parsed phase_5_benchmark_results.json dictionary.
        experiment_name : str
            Target MLflow experiment name.
        artifacts_dir : Path | None
            Directory containing model artifacts.

        Returns
        -------
        list[str]
            List of MLflow run IDs created.
        """
        self.get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        run_ids: list[str] = []

        baseline_eval = benchmark_results.get("baseline_demand_validation", {})
        models_eval = baseline_eval.get("models_evaluated", {})
        champ_name = baseline_eval.get("champion_model_name", "")
        test_eval = (
            benchmark_results.get("final_test_evaluation", {})
            .get("baseline_demand_test", {})
            .get("all_models", {})
        )

        for model_name, val_perf in models_eval.items():
            run_name = f"Demand-{model_name}"
            with mlflow.start_run(run_name=run_name) as run:
                run_ids.append(run.info.run_id)

                # Tags
                mlflow.set_tags(
                    {
                        "project": "PromoSense",
                        "phase": "Phase 5: Baseline Demand Modeling",
                        "model_family": model_name,
                        "is_champion": str(model_name == champ_name),
                        "target_variable": "y_category_spend_window",
                        "cohort": "Untreated Households (T=0)",
                    }
                )

                # Params
                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "train_time_sec": val_perf.get("train_time_sec", 0.0),
                        "train_split": "Weeks 1-70",
                        "val_split": "Weeks 71-86",
                        "test_split": "Weeks 87-102",
                    }
                )

                # Validation Metrics
                val_all = val_perf.get("all_controls", {})
                for k, v in val_all.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"val_all_{k}", float(v))

                val_clean = val_perf.get("strict_clean_controls", {})
                for k, v in val_clean.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"val_clean_{k}", float(v))

                # Test Metrics (if evaluated on test)
                if model_name in test_eval:
                    t_perf = test_eval[model_name]
                    t_all = t_perf.get("all_controls", {})
                    for k, v in t_all.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"test_all_{k}", float(v))
                    t_clean = t_perf.get("strict_clean_controls", {})
                    for k, v in t_clean.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"test_clean_{k}", float(v))

                # Log Champion Model Artifact if applicable
                if model_name == champ_name and artifacts_dir:
                    champ_file = artifacts_dir / "champion_baseline_demand_model.joblib"
                    if champ_file.is_file():
                        mlflow.log_artifact(str(champ_file), artifact_path="model")

        return run_ids

    def log_causal_uplift(
        self,
        benchmark_results: dict[str, Any],
        experiment_name: str = "PromoSense-Causal-Uplift",
        artifacts_dir: Path | None = None,
    ) -> list[str]:
        """Logs Phase 5 Propensity Diagnostics and Causal Uplift Meta-Learners.

        Parameters
        ----------
        benchmark_results : dict[str, Any]
            Parsed phase_5_benchmark_results.json dictionary.
        experiment_name : str
            Target MLflow experiment name.
        artifacts_dir : Path | None
            Directory containing model artifacts.

        Returns
        -------
        list[str]
            List of MLflow run IDs created.
        """
        self.get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        run_ids: list[str] = []

        # 1. Log Propensity Diagnostics Run
        prop_data = benchmark_results.get("propensity_diagnostics", {})
        with mlflow.start_run(run_name="Propensity-Overlap-Diagnostics") as p_run:
            run_ids.append(p_run.info.run_id)
            mlflow.set_tags(
                {
                    "project": "PromoSense",
                    "phase": "Phase 5: Propensity & Overlap Diagnostics",
                    "methodology": "Inverse Propensity Weighting (IPW)",
                }
            )
            cov_bal = prop_data.get("covariate_balance", {})
            mlflow.log_metrics(
                {
                    "total_features_checked": float(cov_bal.get("total_features_checked", 0)),
                    "balanced_raw_count": float(cov_bal.get("balanced_raw_count", 0)),
                    "balanced_ipw_count": float(cov_bal.get("balanced_ipw_count", 0)),
                    "mean_abs_smd_raw": float(cov_bal.get("mean_abs_smd_raw", 0.0)),
                    "mean_abs_smd_ipw": float(cov_bal.get("mean_abs_smd_ipw", 0.0)),
                }
            )

        # 2. Log Causal Uplift Meta-Learners
        uplift_eval = benchmark_results.get("causal_uplift_validation", {})
        test_uplift = (
            benchmark_results.get("final_test_evaluation", {})
            .get("causal_uplift_test", {})
            .get("all_models", {})
        )
        champ_uplift = (
            benchmark_results.get("final_test_evaluation", {})
            .get("causal_uplift_test", {})
            .get("champion_model_name", "")
        )

        for learner_name, val_perf in uplift_eval.items():
            run_name = f"Uplift-{learner_name}"
            with mlflow.start_run(run_name=run_name) as u_run:
                run_ids.append(u_run.info.run_id)

                is_champ = learner_name == champ_uplift
                mlflow.set_tags(
                    {
                        "project": "PromoSense",
                        "phase": "Phase 5: Causal Uplift Modeling",
                        "meta_learner": learner_name,
                        "is_champion": str(is_champ),
                        "estimand": "Conditional Average Treatment Effect (CATE tau(X))",
                    }
                )

                mlflow.log_params(
                    {
                        "learner_name": learner_name,
                        "fit_time_seconds": val_perf.get("fit_time_seconds", 0.0),
                    }
                )

                # Validation Uplift Metrics
                mlflow.log_metrics(
                    {
                        "val_qini_score": float(val_perf.get("qini_score", 0.0)),
                        "val_auuc": float(val_perf.get("auuc", 0.0)),
                        "val_top_decile_lift": float(val_perf.get("top_decile_lift", 0.0)),
                        "val_top_decile_multiplier": float(
                            val_perf.get("top_decile_multiplier", 0.0)
                        ),
                        "val_cate_mean": float(val_perf.get("cate_mean", 0.0)),
                        "val_cate_std": float(val_perf.get("cate_std", 0.0)),
                    }
                )

                # Test Uplift Metrics
                if learner_name in test_uplift:
                    t_perf = test_uplift[learner_name]
                    mlflow.log_metrics(
                        {
                            "test_qini_score": float(t_perf.get("qini_score", 0.0)),
                            "test_auuc": float(t_perf.get("auuc", 0.0)),
                            "test_top_decile_lift": float(t_perf.get("top_decile_lift", 0.0)),
                            "test_top_decile_multiplier": float(
                                t_perf.get("top_decile_multiplier", 0.0)
                            ),
                        }
                    )

                if is_champ and artifacts_dir:
                    champ_file = artifacts_dir / "champion_uplift_model.joblib"
                    if champ_file.is_file():
                        mlflow.log_artifact(str(champ_file), artifact_path="model")

        return run_ids

    def log_price_elasticity(
        self,
        elasticity_results: dict[str, Any],
        experiment_name: str = "PromoSense-Price-Elasticity",
        artifacts_dir: Path | None = None,
    ) -> list[str]:
        """Logs Phase 6 Price Elasticity & Cannibalization results to MLflow.

        Parameters
        ----------
        elasticity_results : dict[str, Any]
            Parsed phase_6_elasticity_results.json dictionary.
        experiment_name : str
            Target MLflow experiment name.
        artifacts_dir : Path | None
            Directory containing model artifacts.

        Returns
        -------
        list[str]
            List of MLflow run IDs created.
        """
        self.get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        run_ids: list[str] = []

        # 1. Log Own-Price Elasticity per Target SKU
        own_res = elasticity_results.get("own_price_elasticities", {})
        for pid_str, p_info in own_res.items():
            pid = p_info.get("product_id", pid_str)
            commodity = p_info.get("commodity", "")
            brand_type = p_info.get("brand_type", "")
            models = p_info.get("models", {})

            run_name = f"Elasticity-SKU-{pid}"
            with mlflow.start_run(run_name=run_name) as run:
                run_ids.append(run.info.run_id)

                mlflow.set_tags(
                    {
                        "project": "PromoSense",
                        "phase": "Phase 6: Price Elasticity",
                        "product_id": str(pid),
                        "commodity": commodity,
                        "brand_type": brand_type,
                        "model_specification": "Log-Log OLS with Newey-West HAC Standard Errors",
                    }
                )

                # Spec A2: Controlled Paid Price OLS
                spec_a2 = models.get("controlled_paid_ols", {})
                eps_a2 = spec_a2.get("elasticity", 0.0)
                se_a2 = spec_a2.get("elasticity_se", 0.0)
                val_eval_a2 = spec_a2.get("val_eval", {})
                test_eval_a2 = spec_a2.get("test_eval", {})

                # Spec B: Decomposed Shelf + Discount OLS
                spec_b = models.get("decomposed_shelf_discount_ols", {})
                base_eps = spec_b.get("base_shelf_elasticity", 0.0)
                disc_eps = spec_b.get("discount_elasticity", 0.0)
                vif = spec_b.get("collinearity_vif", 0.0)

                mlflow.log_params(
                    {
                        "product_id": pid,
                        "commodity": commodity,
                        "brand_type": brand_type,
                    }
                )

                mlflow.log_metrics(
                    {
                        "elasticity_controlled_paid_ols": float(eps_a2)
                        if eps_a2 is not None
                        else 0.0,
                        "elasticity_hac_se": float(se_a2) if se_a2 is not None else 0.0,
                        "decomposed_base_shelf_elasticity": float(base_eps)
                        if base_eps is not None
                        else 0.0,
                        "decomposed_discount_elasticity": float(disc_eps)
                        if disc_eps is not None
                        else 0.0,
                        "decomposed_vif": float(vif) if vif is not None else 0.0,
                        "val_rmse": float(val_eval_a2.get("rmse", 0.0)),
                        "val_r2": float(val_eval_a2.get("r2", 0.0)),
                        "test_rmse": float(test_eval_a2.get("rmse", 0.0)),
                        "test_r2": float(test_eval_a2.get("r2", 0.0)),
                    }
                )

        # 2. Log Cannibalization & Cross-Price Elasticity Pairs
        pairs_res = elasticity_results.get("cannibalization_analysis", {})
        for pair_name, pair_info in pairs_res.items():
            with mlflow.start_run(run_name=f"Cannibalization-{pair_name}") as c_run:
                run_ids.append(c_run.info.run_id)

                el = pair_info.get("elasticities", {})
                val_sim = pair_info.get("val_simulation", {})
                c_diff = val_sim.get("counterfactual_differences", {})
                c_met = val_sim.get("cannibalization_metrics", {})

                mlflow.set_tags(
                    {
                        "project": "PromoSense",
                        "phase": "Phase 6: Cannibalization Modeling",
                        "pair_name": pair_name,
                        "prod_A": str(pair_info.get("prod_A", "")),
                        "prod_B": str(pair_info.get("prod_B", "")),
                    }
                )

                offset_ratio = c_met.get("revenue_offset_ratio")
                mlflow.log_metrics(
                    {
                        "own_elasticity_AA": float(el.get("eps_AA_own", 0.0)),
                        "cross_elasticity_AB": float(el.get("eps_AB_cross", 0.0)),
                        "own_elasticity_BB": float(el.get("eps_BB_own", 0.0)),
                        "cross_elasticity_BA": float(el.get("eps_BA_cross", 0.0)),
                        "val_delta_revenue_A": float(c_diff.get("delta_revenue_A", 0.0)),
                        "val_delta_revenue_B": float(c_diff.get("delta_revenue_B", 0.0)),
                        "val_net_category_revenue_lift": float(
                            c_diff.get("net_category_revenue_lift", 0.0)
                        ),
                        "val_revenue_offset_ratio": float(offset_ratio)
                        if offset_ratio is not None
                        else 0.0,
                    }
                )

        # 3. Log Artifacts
        if artifacts_dir:
            champ_elasticity = artifacts_dir / "champion_elasticity_model.joblib"
            res_json = artifacts_dir / "phase_6_elasticity_results.json"
            with mlflow.start_run(run_name="Phase6-Champion-Artifacts"):
                mlflow.set_tag("phase", "Phase 6 Summary")
                if champ_elasticity.is_file():
                    mlflow.log_artifact(str(champ_elasticity), artifact_path="model")
                if res_json.is_file():
                    mlflow.log_artifact(str(res_json), artifact_path="results")

        return run_ids

    def log_simulation_optimization(
        self,
        simulation_results: dict[str, Any],
        experiment_name: str = "PromoSense-Simulation-Optimization",
        artifacts_dir: Path | None = None,
    ) -> list[str]:
        """Logs Phase 7 Promotion Simulation and Bounded Optimization to MLflow.

        Parameters
        ----------
        simulation_results : dict[str, Any]
            Parsed phase_7_simulation_results.json dictionary.
        experiment_name : str
            Target MLflow experiment name.
        artifacts_dir : Path | None
            Directory containing model artifacts.

        Returns
        -------
        list[str]
            List of MLflow run IDs created.
        """
        self.get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        run_ids: list[str] = []

        single_opt = simulation_results.get("single_sku_optimization", {})
        test_outcomes = simulation_results.get("test_out_of_sample_realizations", {})

        for pid_str, opt_data in single_opt.items():
            pid = opt_data.get("product_id", pid_str)
            p_opt = opt_data.get("price_only_optimization", {})
            opt_by_margin = p_opt.get("optimal_policy_by_margin_tier", {})

            run_name = f"Optimization-SKU-{pid}"
            with mlflow.start_run(run_name=run_name) as run:
                run_ids.append(run.info.run_id)

                mlflow.set_tags(
                    {
                        "project": "PromoSense",
                        "phase": "Phase 7: Promotion Optimization",
                        "product_id": str(pid),
                        "search_space": "5% to 35% in 2.5% steps",
                        "objective": "Expected Category Incremental Contribution Profit",
                    }
                )

                mlflow.log_params(
                    {
                        "product_id": pid,
                        "min_discount": 0.05,
                        "max_discount": 0.35,
                        "step": 0.025,
                    }
                )

                metrics_to_log: dict[str, float] = {}
                for margin_key in ("margin_15pct", "margin_25pct", "margin_35pct"):
                    if margin_key in opt_by_margin:
                        m_data = opt_by_margin[margin_key]
                        d_star = m_data.get("optimal_discount_pct", 0.0)
                        pi_star = m_data.get("delta_profit_at_optimum", 0.0)
                        metrics_to_log[f"{margin_key}_optimal_discount"] = float(d_star)
                        metrics_to_log[f"{margin_key}_optimal_profit_lift"] = float(pi_star)

                # Test real-out-of-sample metrics
                if str(pid) in test_outcomes:
                    t_res = test_outcomes[str(pid)]
                    metrics_to_log["test_applied_discount"] = float(
                        t_res.get("applied_discount_pct", 0.0)
                    )
                    metrics_to_log["test_delta_profit_25pct"] = float(
                        t_res.get("test_delta_profit_25pct", 0.0)
                    )
                    metrics_to_log["test_delta_revenue"] = float(
                        t_res.get("test_delta_revenue", 0.0)
                    )
                    metrics_to_log["test_delta_units"] = float(t_res.get("test_delta_units", 0.0))

                mlflow.log_metrics(metrics_to_log)

        # Log Bundle Artifact
        if artifacts_dir:
            bundle_file = artifacts_dir / "frozen_simulation_models.joblib"
            res_json = artifacts_dir / "phase_7_simulation_results.json"
            with mlflow.start_run(run_name="Phase7-Simulation-Artifacts"):
                mlflow.set_tag("phase", "Phase 7 Summary")
                if bundle_file.is_file():
                    mlflow.log_artifact(str(bundle_file), artifact_path="model_bundle")
                if res_json.is_file():
                    mlflow.log_artifact(str(res_json), artifact_path="results")

        return run_ids

    def sync_all_frozen_artifacts(
        self,
        artifacts_dir: Path | None = None,
    ) -> dict[str, list[str]]:
        """Synchronizes all existing Phase 5, 6, and 7 frozen artifacts into MLflow runs.

        Guarantees that existing validated models and benchmark results are logged
        without retraining models or modifying frozen weights.

        Parameters
        ----------
        artifacts_dir : Path | None
            Directory containing frozen JSON benchmarks and joblib artifacts.
            Defaults to models/artifacts.

        Returns
        -------
        dict[str, list[str]]
            Mapping of phase names to created MLflow run IDs.
        """
        if artifacts_dir is None:
            artifacts_dir = ml_settings.BASE_DIR / "models" / "artifacts"

        created_runs: dict[str, list[str]] = {
            "phase_5_demand": [],
            "phase_5_uplift": [],
            "phase_6_elasticity": [],
            "phase_7_simulation": [],
        }

        # 1. Phase 5 Benchmark
        p5_file = artifacts_dir / "phase_5_benchmark_results.json"
        if p5_file.is_file():
            with open(p5_file, encoding="utf-8") as f:
                p5_data = json.load(f)
            created_runs["phase_5_demand"] = self.log_baseline_demand(
                p5_data, artifacts_dir=artifacts_dir
            )
            created_runs["phase_5_uplift"] = self.log_causal_uplift(
                p5_data, artifacts_dir=artifacts_dir
            )

        # 2. Phase 6 Elasticity
        p6_file = artifacts_dir / "phase_6_elasticity_results.json"
        if p6_file.is_file():
            with open(p6_file, encoding="utf-8") as f:
                p6_data = json.load(f)
            created_runs["phase_6_elasticity"] = self.log_price_elasticity(
                p6_data, artifacts_dir=artifacts_dir
            )

        # 3. Phase 7 Simulation
        p7_file = artifacts_dir / "phase_7_simulation_results.json"
        if p7_file.is_file():
            with open(p7_file, encoding="utf-8") as f:
                p7_data = json.load(f)
            created_runs["phase_7_simulation"] = self.log_simulation_optimization(
                p7_data, artifacts_dir=artifacts_dir
            )

        return created_runs


def main() -> None:
    """CLI Entry point for syncing frozen artifacts into MLflow."""
    parser = argparse.ArgumentParser(description="PromoSense MLflow Experiment Synchronization")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="models/artifacts",
        help="Path to models/artifacts directory",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="",
        help="Custom MLflow tracking URI (default uses env var or local ./mlruns)",
    )
    args = parser.parse_args()

    artifacts_path = Path(args.artifacts_dir).resolve()
    tracking_uri = args.tracking_uri if args.tracking_uri else None

    tracker = MLflowTracker(tracking_uri=tracking_uri)
    print(f"Connecting to MLflow Tracking URI: {tracker.tracking_uri}")
    print(f"Syncing artifacts from: {artifacts_path}")

    runs = tracker.sync_all_frozen_artifacts(artifacts_dir=artifacts_path)
    total_runs = sum(len(r_list) for r_list in runs.values())
    print(f"\nSuccessfully logged {total_runs} MLflow runs across 4 experiments:")
    for phase_key, r_ids in runs.items():
        print(f"  - {phase_key}: {len(r_ids)} runs")


if __name__ == "__main__":
    main()
