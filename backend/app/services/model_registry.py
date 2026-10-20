"""Model Registry and Artifact Lifecycle Manager for PromoSense Backend.

Loads frozen econometric models, panels, and simulation results once at startup.
Raises explicit RuntimeError if required artifacts are missing or corrupted.
Prevents redundant per-request disk I/O and prohibits synthetic model fallbacks.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_pipeline.src.elasticity.cross_elasticity import CrossPriceElasticityModel
from ml_pipeline.src.elasticity.own_elasticity import PaidPriceElasticityOLS

# Canonical product metadata mapping
PRODUCT_CATALOG_METADATA: dict[int, dict[str, str]] = {
    1053690: {
        "commodity": "SOFT DRINKS",
        "sub_commodity": "SFT DRNK 2 LITER BTL CARB INCL",
        "brand_type": "National",
        "department": "GROCERY",
    },
    844165: {
        "commodity": "SOFT DRINKS",
        "sub_commodity": "SFT DRNK 2 LITER BTL CARB INCL",
        "brand_type": "National",
        "department": "GROCERY",
    },
    5569230: {
        "commodity": "SOFT DRINKS",
        "sub_commodity": "SOFT DRINKS 12/18&15PK CAN CAR",
        "brand_type": "National",
        "department": "GROCERY",
    },
    8090521: {
        "commodity": "SOFT DRINKS",
        "sub_commodity": "SOFT DRINKS 12/18&15PK CAN CAR",
        "brand_type": "National",
        "department": "GROCERY",
    },
    1029743: {
        "commodity": "FLUID MILK PRODUCTS",
        "sub_commodity": "FLUID MILK WHITE ONLY",
        "brand_type": "Private",
        "department": "GROCERY",
    },
    1106523: {
        "commodity": "FLUID MILK PRODUCTS",
        "sub_commodity": "FLUID MILK WHITE ONLY",
        "brand_type": "Private",
        "department": "GROCERY",
    },
    1082185: {
        "commodity": "TROPICAL FRUIT",
        "sub_commodity": "BANANAS",
        "brand_type": "National",
        "department": "PRODUCE",
    },
}


class ModelRegistry:
    """Singleton Registry managing frozen PromoSense artifacts in memory."""

    _instance: "ModelRegistry | None" = None

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        if artifacts_dir is None:
            # Default to repo root / models / artifacts
            current_file = Path(__file__).resolve()
            # current_file is backend/app/services/model_registry.py -> 4 levels up to repo root
            repo_root = current_file.parents[3]
            artifacts_dir = repo_root / "models" / "artifacts"

        self.artifacts_dir = artifacts_dir
        self.is_initialized: bool = False

        self.single_models: dict[int, PaidPriceElasticityOLS] = {}
        self.pair_models: dict[str, CrossPriceElasticityModel] = {}
        self.val_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.target_skus: list[int] = []
        self.substitute_pairs: list[dict[str, Any]] = []
        self.phase_6_results: dict[str, Any] = {}
        self.phase_7_results: dict[str, Any] = {}

    @classmethod
    def get_instance(cls, artifacts_dir: Path | None = None) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls(artifacts_dir=artifacts_dir)
        return cls._instance

    def load_artifacts(self) -> None:
        """Loads and verifies frozen model artifacts from disk.

        Raises
        ------
        RuntimeError
            If required model artifacts are missing or fail integrity validation.
        """
        if self.is_initialized:
            return

        models_joblib_path = self.artifacts_dir / "frozen_simulation_models.joblib"
        phase_6_json_path = self.artifacts_dir / "phase_6_elasticity_results.json"
        phase_7_json_path = self.artifacts_dir / "phase_7_simulation_results.json"

        # 1. Existence Check
        missing_files = []
        for p, label in [
            (models_joblib_path, "frozen_simulation_models.joblib"),
            (phase_6_json_path, "phase_6_elasticity_results.json"),
            (phase_7_json_path, "phase_7_simulation_results.json"),
        ]:
            if not p.is_file():
                missing_files.append(label)

        if missing_files:
            raise RuntimeError(
                f"Missing required PromoSense model artifacts in {self.artifacts_dir}: {', '.join(missing_files)}. "
                "Ensure Phase 6 and Phase 7 runners have executed and generated frozen artifacts."
            )

        # 2. Load JSON Artifacts
        try:
            with open(phase_6_json_path, encoding="utf-8") as f:
                self.phase_6_results = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse {phase_6_json_path}: {e}") from e

        try:
            with open(phase_7_json_path, encoding="utf-8") as f:
                self.phase_7_results = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse {phase_7_json_path}: {e}") from e

        # 3. Load Joblib Model Bundle
        try:
            bundle = joblib.load(models_joblib_path)
            self.single_models = bundle["single_models"]
            self.pair_models = bundle["pair_models"]
            self.val_df = bundle["val_df"]
            self.test_df = bundle["test_df"]
            self.target_skus = bundle["target_skus"]
            self.substitute_pairs = bundle["substitute_pairs"]
        except Exception as e:
            raise RuntimeError(f"Failed to load or deserialize {models_joblib_path}: {e}") from e

        # 4. Integrity Verification
        if not self.single_models or not self.target_skus:
            raise RuntimeError("Model registry bundle contains empty single_models or target_skus.")

        for pid in self.target_skus:
            if pid not in self.single_models:
                raise RuntimeError(f"Target SKU {pid} missing from loaded single_models.")
            model = self.single_models[pid]
            if model.elasticity_ is None:
                raise RuntimeError(f"Model for SKU {pid} has uninitialized elasticity coefficient.")

        for pair in self.substitute_pairs:
            pair_name = pair["pair_name"]
            if pair_name not in self.pair_models:
                raise RuntimeError(f"Substitute pair {pair_name} missing from loaded pair_models.")

        self.is_initialized = True

    def get_single_model(self, product_id: int) -> PaidPriceElasticityOLS:
        if not self.is_initialized:
            self.load_artifacts()
        if product_id not in self.single_models:
            raise KeyError(
                f"Product ID {product_id} is not among supported target SKUs: {self.target_skus}"
            )
        return self.single_models[product_id]

    def get_eval_df(self, product_id: int, split: str = "val") -> pd.DataFrame:
        if not self.is_initialized:
            self.load_artifacts()
        df = self.val_df if split == "val" else self.test_df
        subset = df[df["product_id"] == product_id].copy().sort_values("week_no")
        if len(subset) == 0:
            raise ValueError(f"No {split} observations found for product {product_id}")
        return subset

    def get_pair_model(self, pair_name: str) -> CrossPriceElasticityModel:
        if not self.is_initialized:
            self.load_artifacts()
        if pair_name not in self.pair_models:
            raise KeyError(
                f"Substitute pair '{pair_name}' not found. Available: {list(self.pair_models.keys())}"
            )
        return self.pair_models[pair_name]

    def get_pair_eval_df(self, pair_name: str, split: str = "val") -> pd.DataFrame:
        if not self.is_initialized:
            self.load_artifacts()
        pair_def = next((p for p in self.substitute_pairs if p["pair_name"] == pair_name), None)
        if pair_def is None:
            raise KeyError(f"Pair definition for '{pair_name}' not found.")

        df = self.val_df if split == "val" else self.test_df
        df_A = df[df["product_id"] == pair_def["prod_A"]].copy()
        df_B = df[df["product_id"] == pair_def["prod_B"]].copy()

        merged = df_A.merge(
            df_B,
            on="week_no",
            suffixes=("_A", "_B"),
            how="inner",
        ).sort_values("week_no")

        if len(merged) == 0:
            raise ValueError(f"No overlapping {split} observations for pair '{pair_name}'")
        return merged

    def find_pair_for_product(self, product_id: int) -> dict[str, Any] | None:
        """Finds substitute pair where product acts as focal Product A."""
        if not self.is_initialized:
            self.load_artifacts()
        for p in self.substitute_pairs:
            if p["prod_A"] == product_id:
                return p
        return None

    def get_product_metadata(self, product_id: int) -> dict[str, str]:
        return PRODUCT_CATALOG_METADATA.get(
            product_id,
            {
                "commodity": "UNKNOWN",
                "sub_commodity": "UNKNOWN",
                "brand_type": "UNKNOWN",
                "department": "UNKNOWN",
            },
        )

    def get_mlflow_status(self) -> dict[str, Any]:
        """Inspects connection to MLflow tracking server and experiment registry.

        Returns
        -------
        dict[str, Any]
            Tracking URI, connection status, and list of registered experiments.
        """
        import os

        try:
            import mlflow

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiments = client.search_experiments()
            return {
                "tracking_uri": tracking_uri,
                "is_connected": True,
                "experiments": [e.name for e in experiments],
            }
        except Exception as e:
            return {
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "default"),
                "is_connected": False,
                "error": str(e),
            }
