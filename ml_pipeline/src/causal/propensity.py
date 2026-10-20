"""Propensity Score Modeling & Overlap Diagnostics Module.

Estimates the treatment assignment mechanism:
    e(X) = P(T=1 | X)

Key Responsibilities:
1. Estimate propensity scores e(X) using regularized logistic regression and LightGBM.
2. Evaluate common support and positivity overlap distributions.
3. Compute covariate balance diagnostics: Standardized Mean Differences (SMD)
   before and after Inverse Probability Weighting (IPW).
4. Provide honest methodological guardrails: Observational balance does NOT
   eliminate unmeasured confounding or guarantee strict exchangeability.
"""

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ml_pipeline.src.models.preprocessor import TabularPreprocessor


class PropensityScoreModel:
    """Estimates and evaluates propensity scores e(X) = P(T=1 | X)."""

    def __init__(
        self,
        estimator_type: str = "lightgbm",
        random_state: int = 42,
        n_jobs: int = 2,
    ) -> None:
        """Initialize propensity model.

        Parameters
        ----------
        estimator_type : str
            'lightgbm' or 'logistic'.
        random_state : int
            Reproducibility seed.
        n_jobs : int
            Number of CPU threads to use.
        """
        if estimator_type not in ("lightgbm", "logistic"):
            raise ValueError(
                f"Unknown estimator_type '{estimator_type}'. Use 'lightgbm' or 'logistic'."
            )
        self.estimator_type = estimator_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fit_time_seconds: float = 0.0

        if estimator_type == "logistic":
            self.preprocessor = TabularPreprocessor(mode="linear")
            self.model: Any = LogisticRegression(
                C=0.1,
                solver="lbfgs",
                max_iter=500,
                random_state=random_state,
            )
        else:
            self.preprocessor = TabularPreprocessor(mode="tree")
            self.model = lgb.LGBMClassifier(
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

    def fit(self, X: pd.DataFrame, T: pd.Series | np.ndarray) -> "PropensityScoreModel":
        """Fits propensity model on pre-treatment predictors X and treatment indicator T."""
        t0 = time.time()
        X_prep = self.preprocessor.fit_transform(X)
        T_arr = np.asarray(T, dtype=int)
        self.model.fit(X_prep, T_arr)
        self.fit_time_seconds = time.time() - t0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns estimated propensity score e(X) = P(T=1 | X) clipped to [1e-4, 1 - 1e-4]."""
        X_prep = self.preprocessor.transform(X)
        proba = self.model.predict_proba(X_prep)[:, 1]
        return np.clip(proba, 1e-4, 1.0 - 1e-4)

    def evaluate_discrimination(
        self, X: pd.DataFrame, T: pd.Series | np.ndarray
    ) -> dict[str, float]:
        """Computes ROC-AUC and Brier score for assignment discrimination."""
        T_arr = np.asarray(T, dtype=int)
        e_scores = self.predict_proba(X)
        auc = float(roc_auc_score(T_arr, e_scores))
        brier = float(brier_score_loss(T_arr, e_scores))
        return {
            "roc_auc": auc,
            "brier_score": brier,
        }

    def compute_overlap_diagnostics(
        self,
        X: pd.DataFrame,
        T: pd.Series | np.ndarray,
        trim_lower: float = 0.05,
        trim_upper: float = 0.95,
    ) -> dict[str, Any]:
        """Evaluates common support overlap and trimming fractions."""
        T_arr = np.asarray(T, dtype=int)
        e_scores = self.predict_proba(X)

        treated_e = e_scores[T_arr == 1]
        control_e = e_scores[T_arr == 0]

        in_support_mask = (e_scores >= trim_lower) & (e_scores <= trim_upper)
        pct_in_support = float(np.mean(in_support_mask) * 100.0)

        # Min-max empirical overlap bounds
        overlap_min = max(float(np.min(treated_e)), float(np.min(control_e)))
        overlap_max = min(float(np.max(treated_e)), float(np.max(control_e)))

        return {
            "treated_e_mean": float(np.mean(treated_e)),
            "treated_e_min": float(np.min(treated_e)),
            "treated_e_max": float(np.max(treated_e)),
            "treated_e_p10": float(np.percentile(treated_e, 10)),
            "treated_e_p90": float(np.percentile(treated_e, 90)),
            "control_e_mean": float(np.mean(control_e)),
            "control_e_min": float(np.min(control_e)),
            "control_e_max": float(np.max(control_e)),
            "control_e_p10": float(np.percentile(control_e, 10)),
            "control_e_p90": float(np.percentile(control_e, 90)),
            "empirical_overlap_min": overlap_min,
            "empirical_overlap_max": overlap_max,
            "pct_in_trim_bounds_05_95": pct_in_support,
            "n_trimmed_total": int(np.sum(~in_support_mask)),
        }


def compute_standardized_mean_differences(
    X: pd.DataFrame,
    T: pd.Series | np.ndarray,
    propensity_scores: np.ndarray | None = None,
    features_to_check: list[str] | None = None,
) -> pd.DataFrame:
    """Computes Standardized Mean Differences (SMD) before and after IPW weighting.

    Formula:
        SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)
    Threshold:
        SMD < 0.10 indicates acceptable covariate balance.
    """
    T_arr = np.asarray(T, dtype=int)
    num_cols = (
        features_to_check
        if features_to_check is not None
        else [
            c
            for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c])
            and c not in ("treatment_assigned", "household_key", "campaign_id")
        ]
    )

    records = []
    has_weights = propensity_scores is not None
    if has_weights:
        # Stabilized or standard IPW weights: w = T/e + (1-T)/(1-e)
        e = np.clip(propensity_scores, 0.01, 0.99)
        ipw_weights = np.where(T_arr == 1, 1.0 / e, 1.0 / (1.0 - e))
    else:
        ipw_weights = np.ones(len(T_arr), dtype=float)

    for col in num_cols:
        vals = X[col].to_numpy(dtype=float)
        vals_t1 = vals[T_arr == 1]
        vals_t0 = vals[T_arr == 0]

        mean_t1 = np.mean(vals_t1)
        mean_t0 = np.mean(vals_t0)
        var_t1 = np.var(vals_t1, ddof=1)
        var_t0 = np.var(vals_t0, ddof=1)
        pooled_std = np.sqrt((var_t1 + var_t0) / 2.0)

        raw_smd = float((mean_t1 - mean_t0) / (pooled_std + 1e-9))

        ipw_smd = np.nan
        if has_weights:
            w_t1 = ipw_weights[T_arr == 1]
            w_t0 = ipw_weights[T_arr == 0]

            wt_mean_t1 = np.average(vals_t1, weights=w_t1)
            wt_mean_t0 = np.average(vals_t0, weights=w_t0)

            wt_var_t1 = np.average((vals_t1 - wt_mean_t1) ** 2, weights=w_t1)
            wt_var_t0 = np.average((vals_t0 - wt_mean_t0) ** 2, weights=w_t0)
            wt_pooled_std = np.sqrt((wt_var_t1 + wt_var_t0) / 2.0)

            ipw_smd = float((wt_mean_t1 - wt_mean_t0) / (wt_pooled_std + 1e-9))

        records.append(
            {
                "feature": col,
                "mean_treated": float(mean_t1),
                "mean_control": float(mean_t0),
                "raw_smd": raw_smd,
                "ipw_smd": ipw_smd,
                "is_balanced_raw": abs(raw_smd) < 0.10,
                "is_balanced_ipw": abs(ipw_smd) < 0.10 if has_weights else False,
            }
        )

    return pd.DataFrame(records)
