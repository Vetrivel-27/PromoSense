"""Data Preparation and Panel Construction for Price Elasticity Modeling.

Methodological Guardrails:
1. Price Definition Clarity:
   - paid_unit_price: Actual price paid per unit (sales_value / quantity).
   - shelf_price: Reconstructed regular baseline unit price before retail discounts.
   - retail_discount_pct: (shelf_price - paid_unit_price) / shelf_price.
   - discount_depth: -ln(1 - retail_discount_pct) = ln(shelf_price) - ln(paid_unit_price).
2. Panel Aggregation:
   - Aggregates valid purchase transactions to the Product x Week panel across the core chain.
   - Merges weekly store-promotion marketing flags (display_share, mailer_share).
   - Documents that transaction-derived prices reflect purchasing households and are not posted shelf prices in unobserved store-weeks.
3. Time-Aware Splits:
   - Train: Weeks 1 to 70 (70 weeks).
   - Validation: Weeks 71 to 86 (16 weeks, used for model selection and tuning).
   - Test: Weeks 87 to 102 (16 weeks, untouched final generalization evaluation).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.src.config.settings import ml_settings


def build_product_weekly_panel(
    product_ids: list[int],
    interim_dir: Path = ml_settings.INTERIM_DATA_DIR,
) -> pd.DataFrame:
    """Builds a weekly time-series panel for specified products.

    Parameters
    ----------
    product_ids : list[int]
        List of product IDs to include in the panel.
    interim_dir : Path
        Directory containing interim Parquet datasets.

    Returns
    -------
    pd.DataFrame
        Product-week panel containing demand, price variables, promotion shares,
        and temporal harmonic terms.
    """
    txns = pd.read_parquet(interim_dir / "transactions.parquet")
    prods = pd.read_parquet(interim_dir / "products.parquet")
    store_promos = pd.read_parquet(interim_dir / "store_promotions.parquet")

    # Filter to valid purchases on selected products
    valid_mask = (txns["row_classification"] == "valid_purchase") & (
        txns["product_id"].isin(product_ids)
    )
    sub_txns = txns[valid_mask].copy()

    if len(sub_txns) == 0:
        raise ValueError(f"No valid transactions found for products: {product_ids}")

    # Weekly aggregation per product
    records = []
    for pid in product_ids:
        p_txns = sub_txns[sub_txns["product_id"] == pid]
        if len(p_txns) == 0:
            continue

        p_info = prods[prods["product_id"] == pid].iloc[0]

        p_txns = p_txns.copy()
        p_txns["shelf_sales"] = p_txns["shelf_price"] * p_txns["quantity"]

        # Weekly aggregate stats
        weekly_stats = (
            p_txns.groupby("week_no")
            .agg(
                quantity=("quantity", "sum"),
                sales_value=("sales_value", "sum"),
                total_shelf_sales=("shelf_sales", "sum"),
                n_baskets=("basket_id", "nunique"),
                n_households=("household_key", "nunique"),
            )
            .reset_index()
        )

        # Merge store promotion flags (display and mailer share across active stores in week)
        p_promos = store_promos[store_promos["product_id"] == pid]
        if len(p_promos) > 0:
            promo_weekly = (
                p_promos.groupby("week_no")
                .agg(
                    display_share=("has_display_placement", "mean"),
                    mailer_share=("has_mailer_placement", "mean"),
                )
                .reset_index()
            )
            weekly_stats = weekly_stats.merge(promo_weekly, on="week_no", how="left")
        else:
            weekly_stats["display_share"] = 0.0
            weekly_stats["mailer_share"] = 0.0

        weekly_stats["display_share"] = weekly_stats["display_share"].fillna(0.0)
        weekly_stats["mailer_share"] = weekly_stats["mailer_share"].fillna(0.0)

        # Compute price variables
        weekly_stats["product_id"] = pid
        weekly_stats["commodity"] = p_info["commodity"]
        weekly_stats["sub_commodity"] = p_info["sub_commodity"]
        weekly_stats["brand_type"] = p_info["brand_type"]
        weekly_stats["is_private_label"] = int(p_info["brand_type"] == "Private")

        # Unit Prices
        weekly_stats["paid_unit_price"] = weekly_stats["sales_value"] / weekly_stats["quantity"]
        weekly_stats["shelf_price"] = weekly_stats["total_shelf_sales"] / weekly_stats["quantity"]

        # Discount features
        weekly_stats["retail_discount_pct"] = np.clip(
            (weekly_stats["shelf_price"] - weekly_stats["paid_unit_price"])
            / np.maximum(weekly_stats["shelf_price"], 1e-4),
            0.0,
            0.99,
        )
        # discount_depth = -ln(1 - discount_pct)
        weekly_stats["discount_depth"] = -np.log(1.0 - weekly_stats["retail_discount_pct"])

        records.append(weekly_stats)

    panel_df = pd.concat(records, ignore_index=True)

    # Calendar and Seasonality Harmonics
    t_norm = panel_df["week_no"] / 102.0
    panel_df["week_trend"] = t_norm
    panel_df["sin_52w"] = np.sin(2.0 * np.pi * panel_df["week_no"] / 52.0)
    panel_df["cos_52w"] = np.cos(2.0 * np.pi * panel_df["week_no"] / 52.0)
    panel_df["sin_26w"] = np.sin(4.0 * np.pi * panel_df["week_no"] / 52.0)
    panel_df["cos_26w"] = np.cos(4.0 * np.pi * panel_df["week_no"] / 52.0)

    # Log-transformed variables
    panel_df["log_quantity"] = np.log(np.maximum(panel_df["quantity"], 1e-4))
    panel_df["log_paid_price"] = np.log(np.maximum(panel_df["paid_unit_price"], 1e-4))
    panel_df["log_shelf_price"] = np.log(np.maximum(panel_df["shelf_price"], 1e-4))

    # Time-aware split assignment
    panel_df["split"] = "train"
    panel_df.loc[(panel_df["week_no"] >= 71) & (panel_df["week_no"] <= 86), "split"] = "val"
    panel_df.loc[panel_df["week_no"] >= 87, "split"] = "test"

    return panel_df.sort_values(["product_id", "week_no"]).reset_index(drop=True)


def audit_split_support(panel_df: pd.DataFrame, product_id: int) -> dict[str, Any]:
    """Audits sample size and price variation support across train, val, and test splits."""
    p_df = panel_df[panel_df["product_id"] == product_id]

    report: dict[str, Any] = {"product_id": product_id, "splits": {}}

    for split_name in ("train", "val", "test"):
        s_df = p_df[p_df["split"] == split_name]
        n_obs = len(s_df)
        if n_obs == 0:
            report["splits"][split_name] = {"n_obs": 0, "has_support": False}
            continue

        prices = s_df["paid_unit_price"].to_numpy(dtype=float)
        shelf_prices = s_df["shelf_price"].to_numpy(dtype=float)
        qtys = s_df["quantity"].to_numpy(dtype=float)

        p_std = float(np.std(prices))
        p_mean = float(np.mean(prices))
        p_min = float(np.min(prices))
        p_max = float(np.max(prices))

        report["splits"][split_name] = {
            "n_obs": n_obs,
            "quantity_total": float(np.sum(qtys)),
            "quantity_mean": float(np.mean(qtys)),
            "paid_price_mean": round(p_mean, 4),
            "paid_price_std": round(p_std, 4),
            "paid_price_min": round(p_min, 4),
            "paid_price_max": round(p_max, 4),
            "shelf_price_mean": round(float(np.mean(shelf_prices)), 4),
            "discount_depth_mean": round(float(np.mean(s_df["discount_depth"])), 4),
            "display_share_mean": round(float(np.mean(s_df["display_share"])), 4),
            "has_independent_price_variation": bool(p_std > 1e-4),
        }

    return report
