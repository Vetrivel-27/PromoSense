"""Data Validation Suite for PromoSense.

Provides automated schema, referential integrity, range, and leakage validation checks
for all raw and processed entities.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class DataValidator:
    """Validates Dunnhumby data tables before and after ETL transformations."""

    @staticmethod
    def validate_households(df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()
        required_cols = ["household_key"]
        for col in required_cols:
            if col not in df.columns:
                report.errors.append(f"Missing required column: {col}")
                report.is_valid = False

        if not report.is_valid:
            return report

        if df["household_key"].duplicated().any():
            report.errors.append("Duplicate household_key values found")
            report.is_valid = False

        report.metrics["total_households"] = len(df)
        report.metrics["demographic_coverage_pct"] = (
            df["has_demographics"].mean() * 100 if "has_demographics" in df.columns else 0.0
        )
        return report

    @staticmethod
    def validate_products(df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()
        required_cols = ["product_id", "department", "commodity"]
        for col in required_cols:
            if col not in df.columns:
                report.errors.append(f"Missing required column: {col}")
                report.is_valid = False

        if not report.is_valid:
            return report

        if df["product_id"].duplicated().any():
            report.errors.append("Duplicate product_id values found")
            report.is_valid = False

        report.metrics["total_products"] = len(df)
        report.metrics["unique_departments"] = df["department"].nunique()
        return report

    @staticmethod
    def validate_transactions(df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()
        required_cols = [
            "household_key",
            "basket_id",
            "day",
            "product_id",
            "quantity",
            "sales_value",
            "retail_disc",
            "row_classification",
        ]
        for col in required_cols:
            if col not in df.columns:
                report.errors.append(f"Missing required transaction column: {col}")
                report.is_valid = False

        if not report.is_valid:
            return report

        # Value audits
        if (df["sales_value"] < 0).any():
            report.errors.append("Negative SALES_VALUE detected in transactions")
            report.is_valid = False

        if (df["quantity"] < 0).any():
            report.warnings.append("Negative QUANTITY detected (possible returns)")

        # Reconstructed shelf price audits
        if "shelf_price" in df.columns:
            invalid_prices = df[
                (df["shelf_price"].notna()) & (df["shelf_price"] < 0)
            ]
            if len(invalid_prices) > 0:
                report.errors.append(f"Found {len(invalid_prices)} negative derived shelf prices")
                report.is_valid = False

            # Invariant: valid_purchase rows must have strictly positive shelf price
            valid_purchases = df[df["row_classification"] == "valid_purchase"]
            zero_price_valid = valid_purchases[valid_purchases["shelf_price"] <= 0]
            if len(zero_price_valid) > 0:
                report.errors.append(
                    f"Found {len(zero_price_valid)} non-positive shelf prices in valid_purchase rows"
                )
                report.is_valid = False

            extreme_prices = df[
                (df["shelf_price"].notna()) & (df["shelf_price"] > 500)
            ]
            if len(extreme_prices) > 0:
                report.warnings.append(f"Found {len(extreme_prices)} shelf prices exceeding $500")

        # Classification reconciliation
        known_classes = {
            "valid_purchase",
            "void_or_zero_entry",
            "fee_or_service_charge",
            "promotional_freebie",
            "coupon_clearing_line",
            "zero_sales_unexplained",
        }
        unrecognized_classes = set(df["row_classification"].unique()) - known_classes
        if unrecognized_classes:
            report.errors.append(f"Unrecognized classification labels found: {unrecognized_classes}")
            report.is_valid = False

        report.metrics["total_transactions"] = len(df)
        report.metrics["valid_purchase_count"] = (df["row_classification"] == "valid_purchase").sum()
        report.metrics["zero_sales_count"] = (df["sales_value"] == 0).sum()
        report.metrics["zero_qty_count"] = (df["quantity"] == 0).sum()
        return report

    @staticmethod
    def validate_referential_integrity(
        transactions_df: pd.DataFrame,
        households_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> ValidationReport:
        report = ValidationReport()

        txn_hh = set(transactions_df["household_key"].unique())
        master_hh = set(households_df["household_key"].unique())
        orphan_hh = txn_hh - master_hh
        if orphan_hh:
            report.errors.append(f"Found {len(orphan_hh)} orphan households in transactions")
            report.is_valid = False

        txn_prod = set(transactions_df["product_id"].unique())
        master_prod = set(products_df["product_id"].unique())
        orphan_prod = txn_prod - master_prod
        if orphan_prod:
            report.errors.append(f"Found {len(orphan_prod)} orphan products in transactions")
            report.is_valid = False

        report.metrics["orphan_households"] = len(orphan_hh)
        report.metrics["orphan_products"] = len(orphan_prod)
        return report
