"""Baseline Counterfactual Demand Modeling Module.

Estimates the counterfactual baseline demand:
    Y_0 = E[Y | X, T=0]
What would demand have looked like during the campaign window if NO promotion had occurred?

Methodological Constraints:
1. Trained strictly on untreated households (T=0) in the training split.
2. Treatment indicators, identifiers, and post-treatment outcomes are never included.
3. Evaluates multiple estimators:
   - Historical Prorated Baseline (heuristic benchmark)
   - Global Mean Baseline (simple central tendency benchmark)
   - Ridge Regressor (regularized linear baseline)
   - LightGBM Regressor (gradient boosted decision trees)
4. Assesses performance on validation untreated data across both:
   - Strict Clean Controls (is_strict_clean_control == 1)
   - Full Concurrent-Adjusted Controls (all T=0)
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_pipeline.src.models.preprocessor import TabularPreprocessor


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """Computes standard and business-level regression metrics."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    sum_actual = float(np.sum(y_true_arr))
    sum_pred = float(np.sum(y_pred_arr))
    dollar_bias = sum_pred - sum_actual
    pct_bias = float((dollar_bias / sum_actual) * 100.0) if sum_actual > 0 else 0.0
    mbe = float(np.mean(y_pred_arr - y_true_arr))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mbe": mbe,
        "sum_actual": sum_actual,
        "sum_pred": sum_pred,
        "dollar_bias": dollar_bias,
        "pct_bias": pct_bias,
    }


class ProratedHistoricalBaseline:
    """Deterministic retail heuristic: prorated 90d category spend."""

    def __init__(self) -> None:
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ProratedHistoricalBaseline":
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # spend_90d / 90 * campaign_duration_days
        spend_90d = X["hh_target_category_spend_90d"].to_numpy(dtype=float)
        duration = X["campaign_duration_days"].to_numpy(dtype=float)
        pred = (spend_90d / 90.0) * duration
        return np.clip(pred, 0.0, None)


class MeanDemandBaseline:
    """Simple baseline predicting training global mean untreated demand."""

    def __init__(self) -> None:
        self.mean_demand_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanDemandBaseline":
        self.mean_demand_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_demand_, dtype=float)


class RidgeDemandModel:
    """Regularized linear regression demand model."""

    def __init__(self, alpha: float = 100.0) -> None:
        self.alpha = alpha
        self.preprocessor = TabularPreprocessor(mode="linear")
        self.model = Ridge(alpha=alpha, random_state=42)
        self.fit_time_seconds: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeDemandModel":
        t0 = time.time()
        X_trans = self.preprocessor.fit_transform(X)
        self.model.fit(X_trans, y)
        self.fit_time_seconds = time.time() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        pred = self.model.predict(X_trans)
        return np.clip(pred, 0.0, None)


class LightGBMDemandModel:
    """LightGBM gradient boosted regression demand model."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = 2,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": -1,
        }
        self.preprocessor = TabularPreprocessor(mode="tree")
        self.model = lgb.LGBMRegressor(**self.params)
        self.fit_time_seconds: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> "LightGBMDemandModel":
        t0 = time.time()
        X_prep = self.preprocessor.fit_transform(X)

        if eval_set is not None:
            eval_data = []
            for X_ev, y_ev in eval_set:
                eval_data.append((self.preprocessor.transform(X_ev), y_ev))
            self.model.fit(
                X_prep,
                y,
                eval_set=eval_data,
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
        else:
            self.model.fit(X_prep, y)

        self.fit_time_seconds = time.time() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_prep = self.preprocessor.transform(X)
        pred = self.model.predict(X_prep)
        return np.clip(pred, 0.0, None)

    @property
    def feature_importances_(self) -> pd.Series:
        """Returns sorted feature importances."""
        return pd.Series(
            self.model.feature_importances_,
            index=self.preprocessor.feature_names_out_,
        ).sort_values(ascending=False)
