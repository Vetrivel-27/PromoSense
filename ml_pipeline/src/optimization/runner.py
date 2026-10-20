"""Phase 7 Experiment Runner: Promotion Simulation & Optimization Pipeline.

Executes:
1. Scenario simulations across discount depths (5% to 35%) and merchandising modes.
2. Category contribution-profit calculations across 15%, 25%, 35% gross margin tiers.
3. Bounded grid search optimization for single SKUs and substitute pairs on Validation data.
4. Evaluation of frozen optimal policies on quarantined Test data.
5. Resource profiling (wall-clock runtime, peak RAM) and artifact serialization.
"""

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.elasticity.cross_elasticity import CrossPriceElasticityModel
from ml_pipeline.src.elasticity.data_prep import build_product_weekly_panel
from ml_pipeline.src.elasticity.own_elasticity import PaidPriceElasticityOLS
from ml_pipeline.src.elasticity.runner import SUBSTITUTE_PAIRS, TARGET_SKUS
from ml_pipeline.src.optimization.optimizer import BoundedPromotionOptimizer
from ml_pipeline.src.optimization.simulator import (
    simulate_category_pair_promotion,
    simulate_single_sku_promotion,
)


class Phase7ExperimentRunner:
    """Orchestrates Phase 7 simulation and optimization experiments."""

    def __init__(
        self,
        artifacts_dir: Path = ml_settings.BASE_DIR / "models" / "artifacts",
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Runs full Phase 7 pipeline."""
        tracemalloc.start()
        start_time = time.perf_counter()

        print("=" * 75)
        print("PROMOSENSE PHASE 7: PROMOTION SIMULATION & OPTIMIZATION ENGINE")
        print("=" * 75)

        # 1. Panel Construction & Model Fitting on Training Set (Weeks 1-70)
        print("\n--- 1. Loading Panel and Fitting Frozen Specifications on Train (W1-70) ---")
        panel_df = build_product_weekly_panel(TARGET_SKUS)
        train_df = panel_df[panel_df["split"] == "train"].copy()
        val_df = panel_df[panel_df["split"] == "val"].copy()
        test_df = panel_df[panel_df["split"] == "test"].copy()

        # Fit single-SKU models
        single_models: dict[int, PaidPriceElasticityOLS] = {}
        for pid in TARGET_SKUS:
            p_train = train_df[train_df["product_id"] == pid].copy()
            single_models[pid] = PaidPriceElasticityOLS(include_controls=True).fit(p_train)

        # Fit pair models
        pair_models: dict[str, CrossPriceElasticityModel] = {}
        for pair in SUBSTITUTE_PAIRS:
            p_model = CrossPriceElasticityModel(include_controls=True).fit(
                train_df, pair["prod_A"], pair["prod_B"]
            )
            pair_models[pair["pair_name"]] = p_model

        print(
            f"Fitted {len(single_models)} single-SKU models and {len(pair_models)} pair models strictly on Train W1-70."
        )

        # 2. Scenario Simulation Suite on Validation Data (Weeks 71-86)
        print("\n--- 2. Running Scenario Simulations on Validation Set (W71-86) ---")
        simulation_results: dict[str, Any] = {}

        for pid in TARGET_SKUS:
            p_val = val_df[val_df["product_id"] == pid].copy()
            m = single_models[pid]

            # Scenario A: Price-only 20% discount under observed status-quo merchandising
            sim_price_only = simulate_single_sku_promotion(
                model=m,
                eval_df=p_val,
                discount_pct=0.20,
                with_display=False,
                baseline_mode="observed_merch",
            )
            # Scenario B: Price + Display co-promotion under observed merchandising
            sim_display_promo = simulate_single_sku_promotion(
                model=m,
                eval_df=p_val,
                discount_pct=0.20,
                with_display=True,
                baseline_mode="observed_merch",
            )
            # Scenario C: Extrapolative bound (zero-display baseline to 100% display)
            sim_extrapolative = simulate_single_sku_promotion(
                model=m,
                eval_df=p_val,
                discount_pct=0.20,
                with_display=True,
                baseline_mode="zero_merch",
            )

            simulation_results[str(pid)] = {
                "product_id": pid,
                "price_only_20pct_status_quo": sim_price_only,
                "price_display_20pct_status_quo": sim_display_promo,
                "price_display_20pct_extrapolative_bound": sim_extrapolative,
            }
            print(
                f"SKU {pid}: Price-only dRev=${sim_price_only['revenues']['delta_revenue']:.2f}, "
                f"dProfit(25%)=${sim_price_only['contribution_profits']['margin_25pct']['delta_profit']:.2f} | "
                f"Co-Display dRev=${sim_display_promo['revenues']['delta_revenue']:.2f}, "
                f"dProfit(25%)=${sim_display_promo['contribution_profits']['margin_25pct']['delta_profit']:.2f}"
            )

        # 3. Category Substitute Pair Simulations
        print("\n--- 3. Running Substitute Pair Cannibalization Simulations ---")
        pair_simulation_results: dict[str, Any] = {}

        for pair in SUBSTITUTE_PAIRS:
            p_name = pair["pair_name"]
            p_model = pair_models[p_name]
            pA = pair["prod_A"]
            pB = pair["prod_B"]

            val_A = val_df[val_df["product_id"] == pA]
            val_B = val_df[val_df["product_id"] == pB]
            val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values(
                "week_no"
            )

            sim_pair_sq = simulate_category_pair_promotion(
                pair_model=p_model,
                eval_df_merged=val_merged,
                discount_pct=0.20,
                with_display=False,
                baseline_mode="observed_merch",
            )
            sim_pair_disp = simulate_category_pair_promotion(
                pair_model=p_model,
                eval_df_merged=val_merged,
                discount_pct=0.20,
                with_display=True,
                baseline_mode="observed_merch",
            )

            pair_simulation_results[p_name] = {
                "pair_name": p_name,
                "prod_A": pA,
                "prod_B": pB,
                "price_only_20pct": sim_pair_sq,
                "co_merchandised_20pct": sim_pair_disp,
            }
            print(
                f"Pair {p_name}: dRev_A=${sim_pair_sq['revenue_impact']['delta_revenue_A']:.2f}, "
                f"dRev_B=${sim_pair_sq['revenue_impact']['delta_revenue_B']:.2f}, "
                f"Net Category Profit Lift(25%)=${sim_pair_sq['profit_impact_category_modeled']['margin_25pct']['delta_profit_category_net']:.2f}"
            )

        # 4. Bounded Optimization on Validation Set
        print("\n--- 4. Running Bounded Category Profit Optimization on Validation Split ---")
        optimizer = BoundedPromotionOptimizer(
            min_discount=0.05,
            max_discount=0.35,
            discount_step=0.025,
            margin_tiers=(0.15, 0.25, 0.35),
        )

        single_sku_optima: dict[str, Any] = {}
        for pid in TARGET_SKUS:
            p_val = val_df[val_df["product_id"] == pid].copy()
            m = single_models[pid]

            opt_price_only = optimizer.optimize_single_sku(
                model=m, eval_df=p_val, with_display=False, baseline_mode="observed_merch"
            )
            opt_co_display = optimizer.optimize_single_sku(
                model=m, eval_df=p_val, with_display=True, baseline_mode="observed_merch"
            )

            single_sku_optima[str(pid)] = {
                "product_id": pid,
                "price_only_optimization": opt_price_only,
                "co_merchandised_optimization": opt_co_display,
            }
            best_d_25 = opt_price_only["optimal_policy_by_margin_tier"]["margin_25pct"][
                "optimal_discount_pct"
            ]
            best_pi_25 = opt_price_only["optimal_policy_by_margin_tier"]["margin_25pct"][
                "delta_profit_at_optimum"
            ]
            print(
                f"SKU {pid} Optimal Policy (25% Margin): d*={best_d_25:.1%}, Expected Profit Lift=${best_pi_25:.2f}"
            )

        pair_optima: dict[str, Any] = {}
        for pair in SUBSTITUTE_PAIRS:
            p_name = pair["pair_name"]
            p_model = pair_models[p_name]
            pA = pair["prod_A"]
            pB = pair["prod_B"]

            val_A = val_df[val_df["product_id"] == pA]
            val_B = val_df[val_df["product_id"] == pB]
            val_merged = val_A.merge(val_B, on="week_no", suffixes=("_A", "_B")).sort_values(
                "week_no"
            )

            opt_pair = optimizer.optimize_category_pair(
                pair_model=p_model,
                eval_df_merged=val_merged,
                with_display=False,
                baseline_mode="observed_merch",
            )
            pair_optima[p_name] = opt_pair
            cat_opt_25 = opt_pair["optimal_policy_by_margin_tier"]["margin_25pct"][
                "category_optimum"
            ]
            print(
                f"Pair {p_name} Category Optimal (25% Margin): d*={cat_opt_25['optimal_discount_pct']:.1%}, "
                f"Net Category Profit Lift=${cat_opt_25['delta_category_net_profit']:.2f}"
            )

        # 5. Out-of-Sample Verification on Frozen Test Set (Weeks 87-102)
        print("\n--- 5. Evaluating Frozen Optimal Policies on Quarantined Test Set (W87-102) ---")
        test_realization_results: dict[str, Any] = {}

        for pid in TARGET_SKUS:
            p_test = test_df[test_df["product_id"] == pid].copy()
            m = single_models[pid]
            # Use policy selected on Validation set for 25% margin tier
            selected_d = single_sku_optima[str(pid)]["price_only_optimization"][
                "optimal_policy_by_margin_tier"
            ]["margin_25pct"]["optimal_discount_pct"]

            if selected_d > 0.0:
                sim_test = simulate_single_sku_promotion(
                    model=m,
                    eval_df=p_test,
                    discount_pct=selected_d,
                    with_display=False,
                    baseline_mode="observed_merch",
                )
                test_realization_results[str(pid)] = {
                    "product_id": pid,
                    "applied_discount_pct": selected_d,
                    "test_delta_profit_25pct": sim_test["contribution_profits"]["margin_25pct"][
                        "delta_profit"
                    ],
                    "test_delta_revenue": sim_test["revenues"]["delta_revenue"],
                    "test_delta_units": sim_test["volumes"]["delta_units"],
                }
            else:
                test_realization_results[str(pid)] = {
                    "product_id": pid,
                    "applied_discount_pct": 0.0,
                    "test_delta_profit_25pct": 0.0,
                    "test_delta_revenue": 0.0,
                    "test_delta_units": 0.0,
                    "status": "policy_recommended_zero_promotion",
                }

        # Profile Resources
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time_sec = time.perf_counter() - start_time

        print("\n--- Phase 7 Resource Profiling ---")
        print(f"Total Execution Time: {total_time_sec:.2f} seconds")
        print(f"Peak Memory Allocated: {peak_mem / (1024 * 1024):.2f} MB")

        # Compile Output Results
        results: dict[str, Any] = {
            "metadata": {
                "phase": "Phase 7: Promotion Simulation & Optimization Engine",
                "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_execution_time_seconds": round(total_time_sec, 2),
                "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
                "hardware_profile": "Dell Inspiron 15 3593 (Intel Core i3-1005G1, 8 GB RAM)",
            },
            "validation_scenario_simulations": simulation_results,
            "validation_pair_simulations": pair_simulation_results,
            "single_sku_optimization": single_sku_optima,
            "category_pair_optimization": pair_optima,
            "test_out_of_sample_realizations": test_realization_results,
        }

        # Serialize Artifact
        def _make_serializable(obj: Any) -> Any:
            if isinstance(obj, np.integer | int):
                return int(obj)
            if isinstance(obj, np.floating | float):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            return obj

        with open(self.artifacts_dir / "phase_7_simulation_results.json", "w") as f_out:
            json.dump(_make_serializable(results), f_out, indent=2)

        print(f"\nArtifact saved to {self.artifacts_dir / 'phase_7_simulation_results.json'}")
        return results


if __name__ == "__main__":
    runner = Phase7ExperimentRunner()
    runner.run()
