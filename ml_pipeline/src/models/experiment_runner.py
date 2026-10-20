"""Phase 5 Experiment Runner: Baseline Demand & Causal Uplift Modeling.

Orchestrates reproducible experiments, tracks resource consumption (runtime, memory),
evaluates multiple model families, and executes the formal model-selection gate.

Workflow:
1. Data Ingestion & Strict Segregation (Train, Val, Test).
2. Experiment A: Baseline Demand Modeling on Untreated Households (T=0).
   - Historical Prorated Heuristic
   - Global Mean Baseline
   - Ridge Regressor
   - LightGBM Regressor
3. Experiment B: Propensity Score Modeling & Overlap Diagnostics.
   - Calibrated Logistic Regression
   - LightGBM Classifier
   - Overlap bounds, common support trimming rate, Standardized Mean Differences.
4. Experiment C: Causal Uplift Modeling (CATE tau(X)).
   - S-Learner (LightGBM & Ridge)
   - T-Learner (LightGBM & Ridge)
   - X-Learner (LightGBM & Ridge)
   - Uplift curves, Qini scores, AUUC, Top-decile lift.
5. Model Selection on Validation split & Final Evaluation on Test split.
6. Exports serialized model artifacts and comprehensive run metrics.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil

from ml_pipeline.src.causal.meta_learners import SLearner, TLearner, XLearner
from ml_pipeline.src.causal.propensity import (
    PropensityScoreModel,
    compute_standardized_mean_differences,
)
from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.evaluate.uplift_metrics import evaluate_uplift_model
from ml_pipeline.src.models.baseline_demand import (
    LightGBMDemandModel,
    MeanDemandBaseline,
    ProratedHistoricalBaseline,
    RidgeDemandModel,
    compute_regression_metrics,
)
from ml_pipeline.src.models.preprocessor import FeatureSegregator


class Phase5ExperimentRunner:
    """Executes Phase 5 modeling experiments with strict split discipline and resource profiling."""

    def __init__(
        self,
        processed_dir: Path = ml_settings.PROCESSED_DATA_DIR,
        output_dir: Path | None = None,
        n_jobs: int = 2,
        random_state: int = 42,
    ) -> None:
        self.processed_dir = processed_dir
        self.output_dir = output_dir or (ml_settings.BASE_DIR / "models" / "artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.results: dict[str, Any] = {
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "n_jobs": n_jobs,
                "random_state": random_state,
                "target_outcome": "y_category_spend_window",
            },
            "resource_usage": {},
            "baseline_demand_validation": {},
            "propensity_diagnostics": {},
            "causal_uplift_validation": {},
            "final_test_evaluation": {},
        }

    def _get_process_memory_mb(self) -> float:
        """Returns current process RSS memory in MB."""
        return float(psutil.Process(os.getpid()).memory_info().rss / 1e6)

    def run_all_experiments(self) -> dict[str, Any]:
        """Executes full Phase 5 modeling benchmark."""
        t_global_start = time.time()
        mem_global_start = self._get_process_memory_mb()

        print("=" * 70)
        print("PROMOSENSE PHASE 5: BASELINE DEMAND & CAUSAL UPLIFT BENCHMARK")
        print("=" * 70)

        # 1. Load Data
        print("\n[1/5] Loading Processed Parquet Datasets...")
        train_features = pd.read_parquet(self.processed_dir / "train_features.parquet")
        train_outcomes = pd.read_parquet(self.processed_dir / "train_outcomes.parquet")
        val_features = pd.read_parquet(self.processed_dir / "val_features.parquet")
        val_outcomes = pd.read_parquet(self.processed_dir / "val_outcomes.parquet")
        test_features = pd.read_parquet(self.processed_dir / "test_features.parquet")
        test_outcomes = pd.read_parquet(self.processed_dir / "test_outcomes.parquet")

        # 2. Segregate Data
        print("\n[2/5] Segregating Governance Keys and Predictors...")
        X_tr, T_tr, y_tr, meta_tr = FeatureSegregator.extract_components(
            train_features, train_outcomes, target_col="y_category_spend_window"
        )
        X_val, T_val, y_val, meta_val = FeatureSegregator.extract_components(
            val_features, val_outcomes, target_col="y_category_spend_window"
        )
        X_te, T_te, y_te, meta_te = FeatureSegregator.extract_components(
            test_features, test_outcomes, target_col="y_category_spend_window"
        )

        print(f"  Train: {len(X_tr)} rows ({T_tr.sum()} treated, {len(T_tr) - T_tr.sum()} control)")
        print(
            f"  Val:   {len(X_val)} rows ({T_val.sum()} treated, {len(T_val) - T_val.sum()} control)"
        )
        print(f"  Test:  {len(X_te)} rows ({T_te.sum()} treated, {len(T_te) - T_te.sum()} control)")

        # 3. Baseline Demand Modeling (T=0)
        print("\n[3/5] Running Baseline Demand Benchmark on Untreated Cohort...")
        baseline_results = self._run_baseline_demand(
            X_tr, T_tr, y_tr, X_val, T_val, y_val, meta_val
        )
        self.results["baseline_demand_validation"] = baseline_results

        # 4. Propensity Score Modeling
        print("\n[4/5] Running Propensity Score & Overlap Diagnostics...")
        prop_results, prop_model = self._run_propensity_diagnostics(X_tr, T_tr, X_val, T_val)
        self.results["propensity_diagnostics"] = prop_results

        # 5. Causal Uplift Modeling
        print("\n[5/5] Running Causal Uplift Benchmark (S, T, X Learners)...")
        uplift_results, best_uplift_model, best_model_name, all_uplift_models = (
            self._run_causal_uplift(X_tr, T_tr, y_tr, X_val, T_val, y_val)
        )
        self.results["causal_uplift_validation"] = uplift_results

        # 6. Final Evaluation on Test Split (Selected Best Models)
        print("\n" + "=" * 70)
        print(f"FINAL EVALUATION ON TEST SET (Champion Model: {best_model_name})")
        print("=" * 70)
        test_eval = self._evaluate_champion_on_test(
            best_uplift_model,
            best_model_name,
            baseline_results["champion_model_obj"],
            baseline_results["champion_model_name"],
            baseline_results["fitted_model_instances"],
            all_uplift_models,
            X_te,
            T_te,
            y_te,
            meta_te,
        )
        self.results["final_test_evaluation"] = test_eval

        # Global Resource Profiling
        t_global_total = time.time() - t_global_start
        mem_global_end = self._get_process_memory_mb()
        self.results["resource_usage"] = {
            "total_wall_clock_seconds": round(t_global_total, 2),
            "start_memory_mb": round(mem_global_start, 2),
            "end_memory_mb": round(mem_global_end, 2),
            "peak_delta_mb": round(mem_global_end - mem_global_start, 2),
        }

        # Save metrics JSON (stripping un-serializable objects)
        serializable_results = self._make_serializable(self.results)
        metrics_path = self.output_dir / "phase_5_benchmark_results.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nSaved benchmark results to: {metrics_path}")

        # Save Champion Models
        joblib.dump(best_uplift_model, self.output_dir / "champion_uplift_model.joblib")
        joblib.dump(
            baseline_results["champion_model_obj"],
            self.output_dir / "champion_baseline_demand_model.joblib",
        )
        print(f"Saved champion models to: {self.output_dir}")

        return serializable_results

    def _run_baseline_demand(
        self,
        X_tr: pd.DataFrame,
        T_tr: pd.Series,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        T_val: pd.Series,
        y_val: pd.Series,
        meta_val: pd.DataFrame,
    ) -> dict[str, Any]:
        """Trains and compares baseline demand models strictly on T=0."""
        # Filter to untreated
        mask_tr_0 = T_tr == 0
        X_tr_0 = X_tr[mask_tr_0].reset_index(drop=True)
        y_tr_0 = y_tr[mask_tr_0].reset_index(drop=True)

        mask_val_0 = T_val == 0
        X_val_0 = X_val[mask_val_0].reset_index(drop=True)
        y_val_0 = y_val[mask_val_0].reset_index(drop=True)
        meta_val_0 = meta_val[mask_val_0].reset_index(drop=True)

        # Strict clean controls mask in validation
        mask_clean_val = meta_val_0["is_strict_clean_control"] == 1

        models: dict[str, Any] = {
            "Prorated Historical Baseline": ProratedHistoricalBaseline(),
            "Global Mean Baseline": MeanDemandBaseline(),
            "Ridge Demand (alpha=100)": RidgeDemandModel(alpha=100.0),
            "LightGBM Demand Regressor": LightGBMDemandModel(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=31,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
        }

        perf_records = {}
        best_name = ""
        best_rmse = float("inf")
        champion_obj = None

        for name, model in models.items():
            t0 = time.time()
            model.fit(X_tr_0, y_tr_0)
            train_sec = time.time() - t0

            # Predict on all validation controls
            preds_all = model.predict(X_val_0)
            metrics_all = compute_regression_metrics(y_val_0, preds_all)

            # Predict on strict clean validation controls
            preds_clean = preds_all[mask_clean_val]
            metrics_clean = compute_regression_metrics(y_val_0[mask_clean_val], preds_clean)

            perf_records[name] = {
                "train_time_sec": round(train_sec, 3),
                "all_controls": metrics_all,
                "strict_clean_controls": metrics_clean,
            }

            print(
                f"  {name:<28} | All RMSE: {metrics_all['rmse']:>6.2f} | "
                f"Clean RMSE: {metrics_clean['rmse']:>6.2f} | "
                f"MAE: {metrics_all['mae']:>5.2f} | "
                f"R2: {metrics_all['r2']:>5.3f} | "
                f"Bias: {metrics_all['pct_bias']:>+5.1f}% | "
                f"Time: {train_sec:.2f}s"
            )

            if metrics_all["rmse"] < best_rmse:
                best_rmse = metrics_all["rmse"]
                best_name = name
                champion_obj = model

        return {
            "models_evaluated": perf_records,
            "champion_model_name": best_name,
            "champion_model_obj": champion_obj,
            "champion_rmse": best_rmse,
            "fitted_model_instances": models,
        }

    def _run_propensity_diagnostics(
        self,
        X_tr: pd.DataFrame,
        T_tr: pd.Series,
        X_val: pd.DataFrame,
        T_val: pd.Series,
    ) -> tuple[dict[str, Any], PropensityScoreModel]:
        """Fits propensity model and computes overlap and covariate balance diagnostics."""
        prop_models = {
            "Logistic Regression (L2)": PropensityScoreModel(
                estimator_type="logistic", random_state=self.random_state
            ),
            "LightGBM Classifier": PropensityScoreModel(
                estimator_type="lightgbm",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
        }

        eval_summary = {}
        for name, p_mod in prop_models.items():
            p_mod.fit(X_tr, T_tr)
            val_disc = p_mod.evaluate_discrimination(X_val, T_val)
            overlap = p_mod.compute_overlap_diagnostics(X_val, T_val)

            eval_summary[name] = {
                "train_time_sec": round(p_mod.fit_time_seconds, 3),
                "validation_discrimination": val_disc,
                "overlap_diagnostics": overlap,
            }

            print(
                f"  {name:<26} | Val ROC-AUC: {val_disc['roc_auc']:.4f} | "
                f"Brier: {val_disc['brier_score']:.4f} | "
                f"Common Support [0.05, 0.95]: {overlap['pct_in_trim_bounds_05_95']:.1f}% | "
                f"Time: {p_mod.fit_time_seconds:.2f}s"
            )

        # Champion propensity model for downstream weighting is LightGBM
        champ_prop = prop_models["LightGBM Classifier"]
        val_e_scores = champ_prop.predict_proba(X_val)

        # Standardized Mean Differences check
        smd_df = compute_standardized_mean_differences(X_val, T_val, propensity_scores=val_e_scores)
        balanced_features_raw = int(smd_df["is_balanced_raw"].sum())
        balanced_features_ipw = int(smd_df["is_balanced_ipw"].sum())
        total_features = len(smd_df)

        print(
            f"  Covariate Balance: {balanced_features_raw}/{total_features} balanced raw (|SMD|<0.1) -> "
            f"{balanced_features_ipw}/{total_features} balanced under IPW"
        )

        return (
            {
                "propensity_models": eval_summary,
                "covariate_balance": {
                    "total_features_checked": total_features,
                    "balanced_raw_count": balanced_features_raw,
                    "balanced_ipw_count": balanced_features_ipw,
                    "mean_abs_smd_raw": float(smd_df["raw_smd"].abs().mean()),
                    "mean_abs_smd_ipw": float(smd_df["ipw_smd"].abs().mean()),
                    "top_unbalanced_features_raw": smd_df.sort_values(
                        "raw_smd", key=abs, ascending=False
                    )[["feature", "raw_smd", "ipw_smd"]]
                    .head(5)
                    .to_dict(orient="records"),
                },
            },
            champ_prop,
        )

    def _run_causal_uplift(
        self,
        X_tr: pd.DataFrame,
        T_tr: pd.Series,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        T_val: pd.Series,
        y_val: pd.Series,
    ) -> tuple[dict[str, Any], Any, str]:
        """Fits S-Learner, T-Learner, and X-Learner with LightGBM and Ridge backends."""
        uplift_candidates = {
            "S-Learner (LightGBM)": SLearner(
                base_estimator="lightgbm",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            "T-Learner (LightGBM)": TLearner(
                base_estimator="lightgbm",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            "X-Learner (LightGBM)": XLearner(
                base_estimator="lightgbm",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            "T-Learner (Ridge)": TLearner(
                base_estimator="ridge",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            "X-Learner (Ridge)": XLearner(
                base_estimator="ridge",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
        }

        perf_records = {}
        best_name = ""
        best_qini = -float("inf")
        champion_model = None

        for name, learner in uplift_candidates.items():
            t0 = time.time()
            learner.fit(X_tr, T_tr, y_tr)
            fit_time = time.time() - t0

            # Predict CATE on validation set
            cate_val = learner.predict_cate(X_val)

            # Evaluate uplift metrics
            eval_dict = evaluate_uplift_model(y_val, T_val, cate_val, model_name=name)
            eval_dict["fit_time_seconds"] = round(fit_time, 2)
            eval_dict["cate_mean"] = float(np.mean(cate_val))
            eval_dict["cate_std"] = float(np.std(cate_val))
            eval_dict["cate_min"] = float(np.min(cate_val))
            eval_dict["cate_max"] = float(np.max(cate_val))

            perf_records[name] = eval_dict

            print(
                f"  {name:<24} | Qini Score: {eval_dict['qini_score']:>9.2f} | "
                f"AUUC: {eval_dict['auuc']:>9.2f} | "
                f"Top-Decile Lift: ${eval_dict['top_decile_lift']:>6.2f} | "
                f"Lift Multiplier: {eval_dict['top_decile_multiplier']:>4.2f}x | "
                f"Time: {fit_time:.2f}s"
            )

            if eval_dict["qini_score"] > best_qini:
                best_qini = eval_dict["qini_score"]
                best_name = name
                champion_model = learner

        return perf_records, champion_model, best_name, uplift_candidates

    def _evaluate_champion_on_test(
        self,
        champion_uplift_model: Any,
        uplift_model_name: str,
        champion_baseline_model: Any,
        baseline_model_name: str,
        baseline_models_dict: dict[str, Any],
        uplift_models_dict: dict[str, Any],
        X_te: pd.DataFrame,
        T_te: pd.Series,
        y_te: pd.Series,
        meta_te: pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluates champion and candidate models on the untouched Test set."""
        # Baseline Demand on Test Controls
        mask_te_0 = T_te == 0
        X_te_0 = X_te[mask_te_0].reset_index(drop=True)
        y_te_0 = y_te[mask_te_0].reset_index(drop=True)
        meta_te_0 = meta_te[mask_te_0].reset_index(drop=True)
        mask_clean_te = meta_te_0["is_strict_clean_control"] == 1

        baseline_test_results = {}
        print("\n  [Baseline Demand Models on Test Split (T=0)]:")
        for name, b_mod in baseline_models_dict.items():
            b_preds = b_mod.predict(X_te_0)
            m_all = compute_regression_metrics(y_te_0, b_preds)
            m_clean = compute_regression_metrics(y_te_0[mask_clean_te], b_preds[mask_clean_te])
            baseline_test_results[name] = {
                "all_controls": m_all,
                "strict_clean_controls": m_clean,
            }
            print(
                f"    {name:<28} | All RMSE: {m_all['rmse']:>6.2f} | "
                f"Clean RMSE: {m_clean['rmse']:>6.2f} | "
                f"MAE: {m_all['mae']:>5.2f} | "
                f"R2: {m_all['r2']:>5.3f} | "
                f"Bias: {m_all['pct_bias']:>+5.1f}%"
            )

        # Causal Uplift Models on Test Set
        uplift_test_results = {}
        print("\n  [Causal Uplift Models on Test Split]:")
        for name, u_mod in uplift_models_dict.items():
            cate_test = u_mod.predict_cate(X_te)
            u_eval = evaluate_uplift_model(y_te, T_te, cate_test, model_name=name)
            u_eval["cate_mean"] = float(np.mean(cate_test))
            u_eval["cate_std"] = float(np.std(cate_test))
            uplift_test_results[name] = u_eval
            print(
                f"    {name:<24} | Qini Score: {u_eval['qini_score']:>9.2f} | "
                f"AUUC: {u_eval['auuc']:>9.2f} | "
                f"Top-Decile Lift: ${u_eval['top_decile_lift']:>6.2f} | "
                f"Multiplier: {u_eval['top_decile_multiplier']:>4.2f}x"
            )

        return {
            "baseline_demand_test": {
                "champion_model_name": baseline_model_name,
                "all_models": baseline_test_results,
            },
            "causal_uplift_test": {
                "champion_model_name": uplift_model_name,
                "all_models": uplift_test_results,
            },
        }

    def _make_serializable(self, obj: Any) -> Any:
        """Recursively converts non-serializable objects (DataFrames, numpy types) to dicts/floats."""
        if isinstance(obj, dict):
            clean = {}
            for k, v in obj.items():
                if k in ("champion_model_obj", "model", "preprocessor", "fitted_model_instances"):
                    continue
                if hasattr(v, "predict") or hasattr(v, "fit"):
                    continue
                clean[k] = self._make_serializable(v)
            return clean
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, "predict") or hasattr(obj, "fit"):
            return None
        return obj


if __name__ == "__main__":
    runner = Phase5ExperimentRunner(n_jobs=2, random_state=42)
    runner.run_all_experiments()
