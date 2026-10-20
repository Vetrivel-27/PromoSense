"""Own-Price Elasticity Models: Interpretable Econometric Baselines and Non-Linear ML.

Specifications Implemented:
1. PaidPriceElasticityOLS (Specification A):
   ln(Q) = alpha + epsilon * ln(paid_unit_price) + controls + u
   Estimates total price elasticity with respect to actual transaction price paid.
2. ShelfPriceDiscountOLS (Specification B):
   ln(Q) = alpha + epsilon_base * ln(shelf_price) + beta_disc * discount_depth + controls + u
   Decomposes regular baseline price sensitivity from temporary promotional discount depth.
3. SegmentElasticityOLS:
   Pooled model estimating elasticity heterogeneity across brand tiers (Private vs National).
4. NonlinearElasticityModel:
   LightGBM regressor with unconstrained vs monotonic price constraints,
   computing local finite-difference elasticity curves.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_newey_west_cov(X: np.ndarray, residuals: np.ndarray, max_lags: int = 4) -> np.ndarray:
    """Computes Newey-West HAC covariance matrix for time series OLS regression."""
    n, k = X.shape
    inv_xtx = np.linalg.pinv(X.T @ X)

    # Lag 0
    Xe = X * residuals[:, np.newaxis]
    omega = Xe.T @ Xe

    # Lags 1 to max_lags with Bartlett kernel weights
    for lag in range(1, max_lags + 1):
        if lag >= n:
            break
        weight = 1.0 - (lag / (max_lags + 1.0))
        gamma_lag = Xe[lag:].T @ Xe[:-lag]
        omega += weight * (gamma_lag + gamma_lag.T)

    df_adj = n / max(n - k, 1)
    cov_hac = inv_xtx @ omega @ inv_xtx * df_adj
    return cov_hac


class PaidPriceElasticityOLS:
    """Interpretable Log-Log OLS for Actual Paid Price Sensitivity (Specification A)."""

    def __init__(self, include_controls: bool = True) -> None:
        self.include_controls = include_controls
        self.coef_: np.ndarray | None = None
        self.se_: np.ndarray | None = None
        self.hac_se_: np.ndarray | None = None
        self.t_stat_: np.ndarray | None = None
        self.p_val_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.elasticity_: float | None = None
        self.elasticity_se_: float | None = None
        self.elasticity_hac_se_: float | None = None
        self.r2_: float | None = None

    def _build_design_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feature_cols = ["log_paid_price"]
        if self.include_controls:
            feature_cols.extend(
                [
                    "display_share",
                    "mailer_share",
                    "week_trend",
                    "sin_52w",
                    "cos_52w",
                    "sin_26w",
                    "cos_26w",
                ]
            )

        self.feature_names_ = ["intercept"] + feature_cols
        X = np.column_stack(
            [np.ones(len(df))] + [df[col].to_numpy(dtype=float) for col in feature_cols]
        )
        y = df["log_quantity"].to_numpy(dtype=float)
        return X, y

    def fit(self, df: pd.DataFrame) -> "PaidPriceElasticityOLS":
        """Fits OLS on training dataframe."""
        X, y = self._build_design_matrix(df)
        n, k = X.shape

        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        self.coef_ = beta

        # Degrees of freedom and variance
        df_e = max(n - k, 1)
        resids = y - X @ beta
        sigma2 = np.sum(resids**2) / df_e

        # Covariance matrix and standard errors (conventional OLS)
        inv_xtx = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diagonal(sigma2 * inv_xtx), 1e-12))
        self.se_ = se
        self.t_stat_ = beta / se
        self.p_val_ = 2.0 * (1.0 - stats.t.cdf(np.abs(self.t_stat_), df=df_e))

        # HAC standard errors (Newey-West, max_lags=4)
        cov_hac = compute_newey_west_cov(X, resids, max_lags=4)
        hac_se = np.sqrt(np.maximum(np.diagonal(cov_hac), 1e-12))
        self.hac_se_ = hac_se
        self.elasticity_hac_se_ = float(hac_se[1])

        # Own price elasticity is coefficient on log_paid_price (index 1)
        self.elasticity_ = float(beta[1])
        self.elasticity_se_ = float(se[1])

        # R2
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(resids**2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts log quantity."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before predict.")
        feature_cols = [c for c in self.feature_names_ if c != "intercept"]
        X = np.column_stack(
            [np.ones(len(df))] + [df[col].to_numpy(dtype=float) for col in feature_cols]
        )
        return X @ self.coef_

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluates model performance on given dataset."""
        y_true_log = df["log_quantity"].to_numpy(dtype=float)
        y_pred_log = self.predict(df)

        # Level predictions: exp(log_q)
        y_true_level = df["quantity"].to_numpy(dtype=float)
        y_pred_level = np.exp(y_pred_log)

        rmse_level = float(np.sqrt(mean_squared_error(y_true_level, y_pred_level)))
        mae_level = float(mean_absolute_error(y_true_level, y_pred_level))
        r2_log = float(r2_score(y_true_log, y_pred_log))

        return {
            "model_type": "PaidPriceElasticityOLS",
            "include_controls": self.include_controls,
            "elasticity": self.elasticity_,
            "elasticity_se": self.elasticity_se_,
            "elasticity_hac_se": self.elasticity_hac_se_,
            "t_stat": float(self.t_stat_[1]) if self.t_stat_ is not None else None,
            "p_val": float(self.p_val_[1]) if self.p_val_ is not None else None,
            "rmse_units": round(rmse_level, 4),
            "mae_units": round(mae_level, 4),
            "r2_log": round(r2_log, 4),
            "coefficients": {
                name: round(float(val), 4)
                for name, val in zip(self.feature_names_, self.coef_, strict=True)
            },
        }


class ShelfPriceDiscountOLS:
    """Interpretable OLS Decomposing Base Shelf Price and Discount Depth (Specification B)."""

    def __init__(self, include_controls: bool = True) -> None:
        self.include_controls = include_controls
        self.coef_: np.ndarray | None = None
        self.se_: np.ndarray | None = None
        self.t_stat_: np.ndarray | None = None
        self.p_val_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.base_elasticity_: float | None = None
        self.discount_elasticity_: float | None = None
        self.collinearity_vif_: float | None = None

    def _build_design_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feature_cols = ["log_shelf_price", "discount_depth"]
        if self.include_controls:
            feature_cols.extend(
                [
                    "display_share",
                    "mailer_share",
                    "week_trend",
                    "sin_52w",
                    "cos_52w",
                    "sin_26w",
                    "cos_26w",
                ]
            )

        self.feature_names_ = ["intercept"] + feature_cols
        X = np.column_stack(
            [np.ones(len(df))] + [df[col].to_numpy(dtype=float) for col in feature_cols]
        )
        y = df["log_quantity"].to_numpy(dtype=float)
        return X, y

    def fit(self, df: pd.DataFrame) -> "ShelfPriceDiscountOLS":
        """Fits OLS on training dataframe."""
        X, y = self._build_design_matrix(df)
        n, k = X.shape

        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        self.coef_ = beta

        df_e = max(n - k, 1)
        resids = y - X @ beta
        sigma2 = np.sum(resids**2) / df_e

        inv_xtx = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diagonal(sigma2 * inv_xtx), 1e-12))
        self.se_ = se
        self.t_stat_ = beta / se
        self.p_val_ = 2.0 * (1.0 - stats.t.cdf(np.abs(self.t_stat_), df=df_e))

        # Base elasticity is log_shelf_price (index 1), discount elasticity is discount_depth (index 2)
        self.base_elasticity_ = float(beta[1])
        self.discount_elasticity_ = float(beta[2])

        # Collinearity diagnostic: correlation and VIF between log_shelf_price and discount_depth
        p1 = df["log_shelf_price"].to_numpy(dtype=float)
        p2 = df["discount_depth"].to_numpy(dtype=float)
        r = float(np.corrcoef(p1, p2)[0, 1]) if np.std(p1) > 0 and np.std(p2) > 0 else 0.0
        # VIF = 1 / (1 - r^2)
        self.collinearity_vif_ = float(1.0 / (1.0 - r**2)) if abs(r) < 0.999 else 999.0

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts log quantity."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before predict.")
        feature_cols = [c for c in self.feature_names_ if c != "intercept"]
        X = np.column_stack(
            [np.ones(len(df))] + [df[col].to_numpy(dtype=float) for col in feature_cols]
        )
        return X @ self.coef_

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluates model performance."""
        y_true_log = df["log_quantity"].to_numpy(dtype=float)
        y_pred_log = self.predict(df)

        y_true_level = df["quantity"].to_numpy(dtype=float)
        y_pred_level = np.exp(y_pred_log)

        rmse_level = float(np.sqrt(mean_squared_error(y_true_level, y_pred_level)))
        mae_level = float(mean_absolute_error(y_true_level, y_pred_level))
        r2_log = float(r2_score(y_true_log, y_pred_log))

        return {
            "model_type": "ShelfPriceDiscountOLS",
            "base_shelf_elasticity": self.base_elasticity_,
            "base_shelf_se": float(self.se_[1]) if self.se_ is not None else None,
            "discount_elasticity": self.discount_elasticity_,
            "discount_se": float(self.se_[2]) if self.se_ is not None else None,
            "collinearity_vif": round(float(self.collinearity_vif_ or 1.0), 3),
            "rmse_units": round(rmse_level, 4),
            "mae_units": round(mae_level, 4),
            "r2_log": round(r2_log, 4),
            "coefficients": {
                name: round(float(val), 4)
                for name, val in zip(self.feature_names_, self.coef_, strict=True)
            },
        }


class SegmentElasticityOLS:
    """Pooled Model Estimating Elasticity Heterogeneity Across Brand Tiers (National vs Private)."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.se_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.national_elasticity_: float | None = None
        self.private_label_elasticity_: float | None = None

    def fit(self, df: pd.DataFrame) -> "SegmentElasticityOLS":
        """Fits pooled interaction model."""
        log_p = df["log_paid_price"].to_numpy(dtype=float)
        is_priv = df["is_private_label"].to_numpy(dtype=float)
        interact = log_p * is_priv

        feature_cols = [
            "log_paid_price",
            "is_private_label",
            "interaction_priv_price",
            "display_share",
            "mailer_share",
            "week_trend",
            "sin_52w",
            "cos_52w",
        ]
        self.feature_names_ = ["intercept"] + feature_cols

        X = np.column_stack(
            [
                np.ones(len(df)),
                log_p,
                is_priv,
                interact,
                df["display_share"].to_numpy(dtype=float),
                df["mailer_share"].to_numpy(dtype=float),
                df["week_trend"].to_numpy(dtype=float),
                df["sin_52w"].to_numpy(dtype=float),
                df["cos_52w"].to_numpy(dtype=float),
            ]
        )
        y = df["log_quantity"].to_numpy(dtype=float)

        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        self.coef_ = beta

        n, k = X.shape
        df_e = max(n - k, 1)
        resids = y - X @ beta
        sigma2 = np.sum(resids**2) / df_e
        inv_xtx = np.linalg.pinv(X.T @ X)
        self.se_ = np.sqrt(np.maximum(np.diagonal(sigma2 * inv_xtx), 1e-12))

        # National brand elasticity = beta[1]
        # Private label elasticity = beta[1] + beta[3]
        self.national_elasticity_ = float(beta[1])
        self.private_label_elasticity_ = float(beta[1] + beta[3])

        return self

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluates pooled model."""
        return {
            "model_type": "SegmentElasticityOLS",
            "national_brand_elasticity": round(float(self.national_elasticity_ or 0.0), 4),
            "private_label_elasticity": round(float(self.private_label_elasticity_ or 0.0), 4),
            "difference_beta": round(float(self.coef_[3]), 4) if self.coef_ is not None else None,
            "difference_se": round(float(self.se_[3]), 4) if self.se_ is not None else None,
        }


class NonlinearElasticityModel:
    """Non-Linear Elasticity via LightGBM Regressor with Optional Monotonic Constraints."""

    def __init__(
        self,
        use_monotonic_constraint: bool = False,
        random_state: int = 42,
    ) -> None:
        self.use_monotonic_constraint = use_monotonic_constraint
        self.random_state = random_state
        self.feature_cols = [
            "log_paid_price",
            "display_share",
            "mailer_share",
            "week_trend",
            "sin_52w",
            "cos_52w",
            "sin_26w",
            "cos_26w",
        ]
        self.model: lgb.LGBMRegressor | None = None

    def fit(self, df: pd.DataFrame) -> "NonlinearElasticityModel":
        """Fits LightGBM on training set."""
        X = df[self.feature_cols]
        y = df["log_quantity"]

        # Monotonic constraint: log_paid_price must have non-increasing effect (-1), others unconstrained (0)
        constraints = [-1 if col == "log_paid_price" else 0 for col in self.feature_cols]

        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            monotone_constraints=constraints if self.use_monotonic_constraint else None,
            random_state=self.random_state,
            n_jobs=2,
            verbosity=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts log quantity."""
        if self.model is None:
            raise ValueError("Model must be fitted before predict.")
        return self.model.predict(df[self.feature_cols])

    def estimate_local_elasticity(
        self,
        base_df: pd.DataFrame,
        pct_delta: float = 0.05,
    ) -> float:
        """Estimates model-implied finite-difference price elasticity holding controls fixed."""
        if self.model is None:
            raise ValueError("Model must be fitted before estimate.")

        df_up = base_df.copy()
        df_down = base_df.copy()

        # Shift log price up and down by delta
        delta_log = np.log(1.0 + pct_delta)
        df_up["log_paid_price"] = base_df["log_paid_price"] + delta_log
        df_down["log_paid_price"] = base_df["log_paid_price"] - delta_log

        pred_up = self.predict(df_up)
        pred_down = self.predict(df_down)

        # Finite difference elasticity: (d ln Q) / (d ln P)
        arc_elasticity = float(np.mean((pred_up - pred_down) / (2.0 * delta_log)))
        return round(arc_elasticity, 4)

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluates model performance and implied local elasticity."""
        y_true_log = df["log_quantity"].to_numpy(dtype=float)
        y_pred_log = self.predict(df)

        y_true_level = df["quantity"].to_numpy(dtype=float)
        y_pred_level = np.exp(y_pred_log)

        rmse_level = float(np.sqrt(mean_squared_error(y_true_level, y_pred_level)))
        mae_level = float(mean_absolute_error(y_true_level, y_pred_level))
        r2_log = float(r2_score(y_true_log, y_pred_log))

        implied_elasticity = self.estimate_local_elasticity(df)

        return {
            "model_type": "NonlinearElasticityModel",
            "use_monotonic_constraint": self.use_monotonic_constraint,
            "model_implied_elasticity": implied_elasticity,
            "rmse_units": round(rmse_level, 4),
            "mae_units": round(mae_level, 4),
            "r2_log": round(r2_log, 4),
        }
