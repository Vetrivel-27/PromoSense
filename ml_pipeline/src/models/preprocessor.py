"""Data Preprocessing & Feature Segregation Module.

Enforces strict architectural boundaries:
1. Segregates identification keys (household_key, campaign_id) and governance flags
   (treatment_assigned, is_strict_clean_control) from the predictor matrix X.
2. Formats predictors for either tree-based estimators (LightGBM) or linear estimators
   (Ridge, Logistic Regression).
3. Strictly fits transformers on the training split only to prevent data leakage.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GOVERNANCE_COLS = [
    "household_key",
    "campaign_id",
    "treatment_assigned",
    "is_strict_clean_control",
]

DEMOGRAPHIC_CAT_COLS = [
    "age_band",
    "marital_status",
    "income_band",
    "homeowner_status",
    "household_comp",
    "household_size",
    "kid_category",
]

BOOLEAN_COLS = [
    "has_demographics",
]


class FeatureSegregator:
    """Segregates governance/ID columns, treatment indicators, and outcomes from predictors."""

    @staticmethod
    def extract_components(
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame | None = None,
        target_col: str = "y_category_spend_window",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, pd.DataFrame]:
        """Extracts (X, T, y, metadata_df) ensuring strict separation.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix including predictors and governance columns.
        outcomes_df : pd.DataFrame | None
            Detached outcome labels table.
        target_col : str
            The outcome column name to extract as y.

        Returns
        -------
        X : pd.DataFrame
            Predictors only (governance and treatment columns removed).
        T : pd.Series
            Binary treatment indicator (treatment_assigned).
        y : pd.Series | None
            Target outcome series if outcomes_df provided, else None.
        metadata_df : pd.DataFrame
            Table containing household_key, campaign_id, is_strict_clean_control.
        """
        # Validate that required governance columns exist
        for col in GOVERNANCE_COLS:
            if col not in features_df.columns:
                raise ValueError(f"Features dataframe missing governance column: '{col}'")

        # Extract Treatment indicator
        T = features_df["treatment_assigned"].copy().astype(int)

        # Extract Governance Metadata
        metadata_df = features_df[
            ["household_key", "campaign_id", "is_strict_clean_control"]
        ].copy()

        # Extract Predictor Matrix X (strictly excluding all governance columns)
        predictor_cols = [c for c in features_df.columns if c not in GOVERNANCE_COLS]
        X = features_df[predictor_cols].copy()

        # Extract Target Y from outcomes dataframe if supplied
        y = None
        if outcomes_df is not None:
            if target_col not in outcomes_df.columns:
                raise ValueError(f"Target column '{target_col}' not found in outcomes dataframe.")
            y = outcomes_df[target_col].copy().astype(float)

        return X, T, y, metadata_df


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer for tabular predictors supporting linear and tree-based representations."""

    def __init__(self, mode: str = "tree") -> None:
        """Initialize preprocessor.

        Parameters
        ----------
        mode : str
            'tree' (keeps DataFrame structure with category dtypes and missing indicators)
            or 'linear' (applies OneHotEncoder and StandardScaler).
        """
        if mode not in ("tree", "linear"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'tree' or 'linear'.")
        self.mode = mode
        self.cat_cols = DEMOGRAPHIC_CAT_COLS
        self.bool_cols = BOOLEAN_COLS
        self.num_cols: list[str] = []
        self._sklearn_transformer: ColumnTransformer | None = None
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TabularPreprocessor":
        """Fit preprocessor parameters on training predictors."""
        # Determine numeric columns dynamically
        self.num_cols = [
            c
            for c in X.columns
            if c not in self.cat_cols and c not in self.bool_cols and c not in GOVERNANCE_COLS
        ]

        if self.mode == "linear":
            self._sklearn_transformer = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), self.num_cols),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                        self.cat_cols,
                    ),
                    ("bool", "passthrough", self.bool_cols),
                ],
                remainder="drop",
            )
            # Impute missing categories as 'MISSING' before fitting OneHotEncoder
            X_clean = self._clean_categoricals(X)
            self._sklearn_transformer.fit(X_clean)
            self.feature_names_out_ = list(self._sklearn_transformer.get_feature_names_out())
        else:
            self.feature_names_out_ = self.num_cols + self.cat_cols + self.bool_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """Transform predictors using fitted parameters."""
        X_clean = self._clean_categoricals(X)

        if self.mode == "linear":
            if self._sklearn_transformer is None:
                raise RuntimeError("Preprocessor must be fitted before transforming.")
            return self._sklearn_transformer.transform(X_clean)

        # For tree mode: return cleaned DataFrame with categorical dtypes
        X_out = X_clean[self.num_cols + self.cat_cols + self.bool_cols].copy()
        for col in self.cat_cols:
            X_out[col] = X_out[col].astype("category")
        for col in self.bool_cols:
            X_out[col] = X_out[col].astype(int)

        return X_out

    def _clean_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces NaNs in categorical columns with 'MISSING' and casts to string."""
        X_clean = X.copy()
        for col in self.cat_cols:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna("MISSING").astype(str)
        for col in self.bool_cols:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].astype(int)
        return X_clean
