"""Phase 6 Experiment Runner: Price Elasticity & Cannibalization Pipeline.

Executes:
1. Panel construction for representative high-velocity SKUs and substitute pairs.
2. Split support audit across Train (1-70), Val (71-86), and Test (87-102).
3. Own-price elasticity models (Unadjusted OLS, Controlled OLS, Decomposed OLS, LightGBM).
4. Segment-level pooled elasticity (National vs Private Label).
5. Cross-price elasticity and counterfactual cannibalization simulation.
6. Resource profiling (runtime and peak memory) and artifact serialization.
"""

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.elasticity.cross_elasticity import (
    CrossPriceElasticityModel,
    simulate_counterfactual_cannibalization,
)
from ml_pipeline.src.elasticity.data_prep import (
    audit_split_support,
    build_product_weekly_panel,
)
from ml_pipeline.src.elasticity.own_elasticity import (
    NonlinearElasticityModel,
    PaidPriceElasticityOLS,
    SegmentElasticityOLS,
    ShelfPriceDiscountOLS,
)

# Representative High-Velocity SKUs and Substitute Pairs
TARGET_SKUS = [
    1053690,  # Soft Drink 2L A (National)
    844165,  # Soft Drink 2L B (National)
    5569230,  # Soft Drink 12pk Cans A (National)
    8090521,  # Soft Drink 12pk Cans B (National)
    1029743,  # Fluid Milk 1GA A (Private Label)
    1106523,  # Fluid Milk 1GA B (Private Label)
    1082185,  # Tropical Fruit / Bananas (High Velocity National)
]

SUBSTITUTE_PAIRS = [
    {
        "pair_name": "Soft_Drinks_2L_National",
        "commodity": "SOFT DRINKS",
        "sub_commodity": "SFT DRNK 2 LITER BTL CARB INCL",
        "prod_A": 1053690,
        "prod_B": 844165,
    },
    {
        "pair_name": "Soft_Drinks_12pk_National",
        "commodity": "SOFT DRINKS",
        "sub_commodity": "SOFT DRINKS 12/18&15PK CAN CAR",
        "prod_A": 5569230,
        "prod_B": 8090521,
    },
    {
        "pair_name": "Fluid_Milk_1GA_PrivateLabel",
        "commodity": "FLUID MILK PRODUCTS",
        "sub_commodity": "FLUID MILK WHITE ONLY",
        "prod_A": 1029743,
        "prod_B": 1106523,
    },
]


class Phase6ExperimentRunner:
    """Orchestrates Phase 6 price elasticity and cannibalization experiments."""

    def __init__(
        self,
        artifacts_dir: Path = ml_settings.BASE_DIR / "models" / "artifacts",
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Runs full Phase 6 pipeline."""
        tracemalloc.start()
        start_time = time.perf_counter()

        print("=" * 70)
        print("PROMOSENSE PHASE 6: PRICE ELASTICITY & CANNIBALIZATION EXPERIMENTS")
        print("=" * 70)

        # 1. Panel Construction
        print("\n--- 1. Constructing Weekly Product Panels ---")
        panel_df = build_product_weekly_panel(TARGET_SKUS)
        print(f"Panel constructed: {len(panel_df)} product-week observations.")

        # 2. Split Support Audit
        print("\n--- 2. Auditing Sample Support & Price Variation by Split ---")
        split_audits = {}
        for pid in TARGET_SKUS:
            audit = audit_split_support(panel_df, pid)
            split_audits[str(pid)] = audit
            print(
                f"Product {pid}: "
                f"Train N={audit['splits']['train']['n_obs']} (std={audit['splits']['train']['paid_price_std']:.3f}), "
                f"Val N={audit['splits']['val']['n_obs']} (std={audit['splits']['val']['paid_price_std']:.3f}), "
                f"Test N={audit['splits']['test']['n_obs']} (std={audit['splits']['test']['paid_price_std']:.3f})"
            )

        # Split panels into train, val, test
        train_df = panel_df[panel_df["split"] == "train"].copy()
        val_df = panel_df[panel_df["split"] == "val"].copy()
        test_df = panel_df[panel_df["split"] == "test"].copy()

        # 3. Own-Price Elasticity Modeling
        print("\n--- 3. Estimating Own-Price Elasticity Across Specifications ---")
        own_elasticity_results: dict[str, Any] = {}

        for pid in TARGET_SKUS:
            p_train = train_df[train_df["product_id"] == pid].copy()
            p_val = val_df[val_df["product_id"] == pid].copy()
            p_test = test_df[test_df["product_id"] == pid].copy()

            # Spec A1: Unadjusted Paid Price OLS
            m_a1 = PaidPriceElasticityOLS(include_controls=False).fit(p_train)
            eval_a1_val = m_a1.evaluate(p_val)
            eval_a1_test = m_a1.evaluate(p_test)

            # Spec A2: Controlled Paid Price OLS
            m_a2 = PaidPriceElasticityOLS(include_controls=True).fit(p_train)
            eval_a2_val = m_a2.evaluate(p_val)
            eval_a2_test = m_a2.evaluate(p_test)

            # Spec B: Decomposed Shelf Price + Discount Depth OLS
            m_b = ShelfPriceDiscountOLS(include_controls=True).fit(p_train)
            eval_b_val = m_b.evaluate(p_val)
            eval_b_test = m_b.evaluate(p_test)

            # Spec C1: LightGBM Unconstrained
            m_c1 = NonlinearElasticityModel(use_monotonic_constraint=False, random_state=42).fit(
                p_train
            )
            eval_c1_val = m_c1.evaluate(p_val)
            eval_c1_test = m_c1.evaluate(p_test)

            # Spec C2: LightGBM Monotonically Constrained
            m_c2 = NonlinearElasticityModel(use_monotonic_constraint=True, random_state=42).fit(
                p_train
            )
            eval_c2_val = m_c2.evaluate(p_val)
            eval_c2_test = m_c2.evaluate(p_test)

            own_elasticity_results[str(pid)] = {
                "product_id": pid,
                "commodity": p_train["commodity"].iloc[0],
                "sub_commodity": p_train["sub_commodity"].iloc[0],
                "brand_type": p_train["brand_type"].iloc[0],
                "models": {
                    "unadjusted_paid_ols": {
                        "elasticity": m_a1.elasticity_,
                        "elasticity_se": m_a1.elasticity_se_,
                        "val_eval": eval_a1_val,
                        "test_eval": eval_a1_test,
                    },
                    "controlled_paid_ols": {
                        "elasticity": m_a2.elasticity_,
                        "elasticity_se": m_a2.elasticity_se_,
                        "val_eval": eval_a2_val,
                        "test_eval": eval_a2_test,
                    },
                    "decomposed_shelf_discount_ols": {
                        "base_shelf_elasticity": m_b.base_elasticity_,
                        "discount_elasticity": m_b.discount_elasticity_,
                        "collinearity_vif": m_b.collinearity_vif_,
                        "val_eval": eval_b_val,
                        "test_eval": eval_b_test,
                    },
                    "lightgbm_unconstrained": {
                        "val_eval": eval_c1_val,
                        "test_eval": eval_c1_test,
                    },
                    "lightgbm_monotonic": {
                        "val_eval": eval_c2_val,
                        "test_eval": eval_c2_test,
                    },
                },
            }
            print(
                f"Product {pid} ({p_train['brand_type'].iloc[0]}): "
                f"Unadjusted eps={m_a1.elasticity_:.2f}, Controlled eps={m_a2.elasticity_:.2f} (se={m_a2.elasticity_se_:.2f}), "
                f"Base Shelf eps={m_b.base_elasticity_:.2f}, Disc eps={m_b.discount_elasticity_:.2f} (VIF={m_b.collinearity_vif_:.2f})"
            )

        # 4. Segment-Level Elasticity (National Brand vs Private Label)
        print("\n--- 4. Estimating Brand Tier Heterogeneity (Pooled Model) ---")
        seg_model = SegmentElasticityOLS().fit(train_df)
        seg_eval = seg_model.evaluate(val_df)
        print(
            f"National Brand Elasticity: {seg_eval['national_brand_elasticity']:.3f} | "
            f"Private Label Elasticity: {seg_eval['private_label_elasticity']:.3f} | "
            f"Difference Beta: {seg_eval['difference_beta']:.3f} (se={seg_eval['difference_se']:.3f})"
        )

        # 5. Cannibalization & Substitution Modeling
        print("\n--- 5. Estimating Cross-Price Elasticities & Cannibalization ---")
        cannibalization_results: dict[str, Any] = {}

        for pair in SUBSTITUTE_PAIRS:
            pair_name = pair["pair_name"]
            pA = pair["prod_A"]
            pB = pair["prod_B"]

            pair_model = CrossPriceElasticityModel(include_controls=True).fit(train_df, pA, pB)

            # Prepare merged evaluation dataframe for Validation (weeks 71-86)
            val_pA = val_df[val_df["product_id"] == pA]
            val_pB = val_df[val_df["product_id"] == pB]
            val_merged = val_pA.merge(val_pB, on="week_no", suffixes=("_A", "_B")).sort_values(
                "week_no"
            )

            # Prepare merged evaluation dataframe for Test (weeks 87-102)
            test_pA = test_df[test_df["product_id"] == pA]
            test_pB = test_df[test_df["product_id"] == pB]
            test_merged = test_pA.merge(test_pB, on="week_no", suffixes=("_A", "_B")).sort_values(
                "week_no"
            )

            # Simulate counterfactual cannibalization: 20% discount on Product A with display
            sim_val = simulate_counterfactual_cannibalization(
                pair_model, val_merged, discount_pct=0.20, with_display=True
            )
            sim_test = simulate_counterfactual_cannibalization(
                pair_model, test_merged, discount_pct=0.20, with_display=True
            )

            cannibalization_results[pair_name] = {
                "pair_name": pair_name,
                "commodity": pair["commodity"],
                "sub_commodity": pair["sub_commodity"],
                "prod_A": pA,
                "prod_B": pB,
                "elasticities": {
                    "eps_AA_own": pair_model.model_A["own_elasticity"],
                    "eps_AA_se": pair_model.model_A["own_se"],
                    "eps_AB_cross": pair_model.model_A["cross_elasticity"],
                    "eps_AB_se": pair_model.model_A["cross_se"],
                    "eps_BB_own": pair_model.model_B["own_elasticity"],
                    "eps_BB_se": pair_model.model_B["own_se"],
                    "eps_BA_cross": pair_model.model_B["cross_elasticity"],
                    "eps_BA_se": pair_model.model_B["cross_se"],
                },
                "val_simulation": sim_val,
                "test_simulation": sim_test,
            }

            print(
                f"Pair '{pair_name}': "
                f"eps_AA={pair_model.model_A['own_elasticity']:.2f}, eps_AB={pair_model.model_A['cross_elasticity']:.2f} | "
                f"eps_BB={pair_model.model_B['own_elasticity']:.2f}, eps_BA={pair_model.model_B['cross_elasticity']:.2f}"
            )
            print(
                f"  Validation Cannibalization: Delta Rev A = ${sim_val['counterfactual_differences']['delta_revenue_A']:.2f}, "
                f"Delta Rev B = ${sim_val['counterfactual_differences']['delta_revenue_B']:.2f}, "
                f"Rev-Offset Ratio = {sim_val['cannibalization_metrics']['revenue_offset_ratio']}, "
                f"Net Category Lift = ${sim_val['counterfactual_differences']['net_category_revenue_lift']:.2f}"
            )

        # Champion Selection on Validation Data
        # For baseline demand elasticity: Controlled Paid Price OLS (m_a2 for representative Soft Drink A)
        champ_sku = 1053690
        p_train_champ = train_df[train_df["product_id"] == champ_sku].copy()
        champion_elasticity_model = PaidPriceElasticityOLS(include_controls=True).fit(p_train_champ)

        joblib.dump(
            champion_elasticity_model,
            self.artifacts_dir / "champion_elasticity_model.joblib",
        )

        # Profile Resources
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time_sec = time.perf_counter() - start_time

        print("\n--- Resource Profiling ---")
        print(f"Total Execution Time: {total_time_sec:.2f} seconds")
        print(f"Peak Memory Allocated: {peak_mem / (1024 * 1024):.2f} MB")

        # Compile Output Results
        results: dict[str, Any] = {
            "metadata": {
                "phase": "Phase 6: Price Elasticity & Cannibalization Modeling",
                "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_execution_time_seconds": round(total_time_sec, 2),
                "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
                "champion_model": "PaidPriceElasticityOLS (Controlled Specification)",
            },
            "split_support_audits": split_audits,
            "own_price_elasticities": own_elasticity_results,
            "segment_tier_elasticity": seg_eval,
            "cannibalization_analysis": cannibalization_results,
        }

        # Save JSON Artifact
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

        with open(self.artifacts_dir / "phase_6_elasticity_results.json", "w") as f_out:
            json.dump(_make_serializable(results), f_out, indent=2)

        print(f"\nArtifacts saved to {self.artifacts_dir / 'phase_6_elasticity_results.json'}")
        return results


if __name__ == "__main__":
    runner = Phase6ExperimentRunner()
    runner.run()
