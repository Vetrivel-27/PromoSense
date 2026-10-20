"""Causal Uplift Meta-Learners Module.

Implements standard meta-learning estimators for Conditional Average Treatment Effect (CATE):
    tau(X) = E[Y(1) - Y(0) | X]

1. S-Learner (Single Model):
   mu(X, T) -> tau(X) = mu(X, 1) - mu(X, 0)
2. T-Learner (Two Models):
   mu_1(X) on T=1, mu_0(X) on T=0 -> tau(X) = mu_1(X) - mu_0(X)
3. X-Learner (Künzel et al., PNAS 2019):
   Two-stage imputation with propensity score weighting, optimized for asymmetric cohorts.
"""

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ml_pipeline.src.causal.propensity import PropensityScoreModel
from ml_pipeline.src.models.preprocessor import TabularPreprocessor


def _create_base_regressor(model_type: str, n_jobs: int = 2, random_state: int = 42) -> Any:
    """Helper to instantiate base regressor."""
    if model_type == "ridge":
        return Ridge(alpha=100.0, random_state=random_state)
    elif model_type == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")


class SLearner:
    """S-Learner (Single Model).

    Fits a single model mu(X, T) = E[Y | X, T].
    CATE: tau(X) = mu(X, 1) - mu(X, 0).
    """

    def __init__(
        self,
        base_estimator: str = "lightgbm",
        n_jobs: int = 2,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.mode = "linear" if base_estimator == "ridge" else "tree"
        self.preprocessor = TabularPreprocessor(mode=self.mode)
        self.model = _create_base_regressor(base_estimator, n_jobs, random_state)
        self.fit_time_seconds: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "SLearner":
        t0 = time.time()
        T_arr = np.asarray(T, dtype=int).reshape(-1, 1)
        y_arr = np.asarray(y, dtype=float)

        # Preprocess X
        X_prep = self.preprocessor.fit_transform(X)

        if self.mode == "linear":
            # Concatenate T as an additional feature column
            X_with_T = np.hstack([X_prep, T_arr])
        else:
            X_with_T = X_prep.copy()
            X_with_T["treatment_assigned"] = T_arr.ravel()

        self.model.fit(X_with_T, y_arr)
        self.fit_time_seconds = time.time() - t0
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimates CATE tau(X) = mu(X, 1) - mu(X, 0)."""
        X_prep = self.preprocessor.transform(X)
        n = len(X)

        if self.mode == "linear":
            X_t1 = np.hstack([X_prep, np.ones((n, 1), dtype=float)])
            X_t0 = np.hstack([X_prep, np.zeros((n, 1), dtype=float)])
        else:
            X_t1 = X_prep.copy()
            X_t1["treatment_assigned"] = 1
            X_t0 = X_prep.copy()
            X_t0["treatment_assigned"] = 0

        y1_pred = self.model.predict(X_t1)
        y0_pred = self.model.predict(X_t0)
        return np.asarray(y1_pred - y0_pred, dtype=float)


class TLearner:
    """T-Learner (Two Models).

    Fits mu_1(X) on T=1 and mu_0(X) on T=0.
    CATE: tau(X) = mu_1(X) - mu_0(X).
    """

    def __init__(
        self,
        base_estimator: str = "lightgbm",
        n_jobs: int = 2,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.mode = "linear" if base_estimator == "ridge" else "tree"
        self.preprocessor = TabularPreprocessor(mode=self.mode)
        self.model_0 = _create_base_regressor(base_estimator, n_jobs, random_state)
        self.model_1 = _create_base_regressor(base_estimator, n_jobs, random_state + 1)
        self.fit_time_seconds: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "TLearner":
        t0 = time.time()
        T_arr = np.asarray(T, dtype=int)
        y_arr = np.asarray(y, dtype=float)

        X_prep = self.preprocessor.fit_transform(X)

        mask_1 = T_arr == 1
        mask_0 = T_arr == 0

        if self.mode == "linear":
            self.model_1.fit(X_prep[mask_1], y_arr[mask_1])
            self.model_0.fit(X_prep[mask_0], y_arr[mask_0])
        else:
            self.model_1.fit(X_prep.loc[mask_1], y_arr[mask_1])
            self.model_0.fit(X_prep.loc[mask_0], y_arr[mask_0])

        self.fit_time_seconds = time.time() - t0
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimates CATE tau(X) = mu_1(X) - mu_0(X)."""
        X_prep = self.preprocessor.transform(X)
        y1_pred = self.model_1.predict(X_prep)
        y0_pred = self.model_0.predict(X_prep)
        return np.asarray(y1_pred - y0_pred, dtype=float)


class XLearner:
    """X-Learner (Künzel et al., PNAS 2019).

    Two-stage imputation with propensity score weighting, specially
    designed for asymmetric treatment/control cohorts.
    """

    def __init__(
        self,
        base_estimator: str = "lightgbm",
        n_jobs: int = 2,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.mode = "linear" if base_estimator == "ridge" else "tree"
        self.preprocessor = TabularPreprocessor(mode=self.mode)

        # Stage 1 models
        self.stage1_model_0 = _create_base_regressor(base_estimator, n_jobs, random_state)
        self.stage1_model_1 = _create_base_regressor(base_estimator, n_jobs, random_state + 1)

        # Stage 2 models (imputed residual regressors)
        self.stage2_tau_0 = _create_base_regressor(base_estimator, n_jobs, random_state + 2)
        self.stage2_tau_1 = _create_base_regressor(base_estimator, n_jobs, random_state + 3)

        # Propensity estimator for weighting
        self.propensity_model = PropensityScoreModel(
            estimator_type="lightgbm" if base_estimator == "lightgbm" else "logistic",
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.fit_time_seconds: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "XLearner":
        t0 = time.time()
        T_arr = np.asarray(T, dtype=int)
        y_arr = np.asarray(y, dtype=float)

        # Fit propensity model on raw X
        self.propensity_model.fit(X, T_arr)

        # Preprocess X for regressors
        X_prep = self.preprocessor.fit_transform(X)

        mask_1 = T_arr == 1
        mask_0 = T_arr == 0

        # Stage 1: Fit base response functions
        if self.mode == "linear":
            X_1 = X_prep[mask_1]
            X_0 = X_prep[mask_0]
        else:
            X_1 = X_prep.loc[mask_1]
            X_0 = X_prep.loc[mask_0]

        y_1 = y_arr[mask_1]
        y_0 = y_arr[mask_0]

        self.stage1_model_1.fit(X_1, y_1)
        self.stage1_model_0.fit(X_0, y_0)

        # Stage 2: Impute counterfactual effects
        # D_1 = Y_1 - mu_0(X_1)
        # D_0 = mu_1(X_0) - Y_0
        D_1 = y_1 - self.stage1_model_0.predict(X_1)
        D_0 = self.stage1_model_1.predict(X_0) - y_0

        # Stage 3: Fit effect models
        self.stage2_tau_1.fit(X_1, D_1)
        self.stage2_tau_0.fit(X_0, D_0)

        self.fit_time_seconds = time.time() - t0
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimates CATE tau(X) = e(X) * tau_0(X) + (1 - e(X)) * tau_1(X)."""
        e_scores = self.propensity_model.predict_proba(X)
        X_prep = self.preprocessor.transform(X)

        tau_0_pred = self.stage2_tau_0.predict(X_prep)
        tau_1_pred = self.stage2_tau_1.predict(X_prep)

        # Weight by propensity scores
        cate = e_scores * tau_0_pred + (1.0 - e_scores) * tau_1_pred
        return np.asarray(cate, dtype=float)
