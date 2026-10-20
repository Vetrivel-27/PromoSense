"""ETL Pipeline for PromoSense.

Extracts, validates, cleans, and transforms raw Dunnhumby CSV files into analytical
datasets (Parquet) and relational database tables.

Key Methodological Guardrails:
1. Reconstructed shelf price = (SALES_VALUE - RETAIL_DISC - COUPON_DISC - COUPON_MATCH_DISC) / QUANTITY.
   Strictly positive for valid purchases; NaN for void entries, fee adjustments, and unexplained zero-sales.
2. Anomaly classification separates void entries, service fees, promotional freebies,
   coupon clearing lines (COUP/STR & MFG), unexplained zero-sales, and valid purchases.
3. Retailer fiscal week calendar extracted directly from transactions for Days 1-711 (Observed).
   Days 712-740 are explicitly marked as Extrapolated on a 7-day cadence.
4. Campaign durations use consistent inclusive endpoints (end_day - start_day + 1).
   Campaign 24 is flagged as right-censored with full nominal duration 61d and observed duration 53d (Days 659-711).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from ml_pipeline.src.config.settings import ml_settings
from ml_pipeline.src.data.validator import DataValidator


class PromoSenseETL:
    """End-to-end ETL processor for retail transaction and campaign data."""

    def __init__(
        self,
        raw_dir: Path = ml_settings.RAW_DATA_DIR,
        output_dir: Path = ml_settings.INTERIM_DATA_DIR,
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.day_to_week_map: dict[int, int] = {}
        self.observed_day_max: int = 711

    def process_products(self) -> pd.DataFrame:
        print("Transforming products...")
        path = self.raw_dir / "product.csv"
        df = pd.read_csv(path)
        df["PRODUCT_ID"] = df["PRODUCT_ID"].astype(np.int64)
        df["MANUFACTURER"] = df["MANUFACTURER"].astype(np.int64)

        df = df.rename(
            columns={
                "PRODUCT_ID": "product_id",
                "MANUFACTURER": "manufacturer_id",
                "DEPARTMENT": "department",
                "BRAND": "brand_type",
                "COMMODITY_DESC": "commodity",
                "SUB_COMMODITY_DESC": "sub_commodity",
                "CURR_SIZE_OF_PRODUCT": "size_description",
            }
        )

        val_report = DataValidator.validate_products(df)
        if not val_report.is_valid:
            raise ValueError(f"Products validation failed: {val_report.errors}")

        out_path = self.output_dir / "products.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Products saved to {out_path} ({len(df)} rows)")
        return df

    def extract_day_to_week_mapping(self) -> dict[int, int]:
        """Extract empirical DAY -> WEEK_NO fiscal calendar from transactions.

        Days 1-711 are observed directly in transaction data.
        Days 712-740 are extrapolated using the observed 7-day weekly cadence.
        """
        print("Extracting empirical DAY -> WEEK_NO fiscal calendar from transactions...")
        txn_path = self.raw_dir / "transaction_data.csv"
        day_week_pairs = pd.read_csv(
            txn_path, usecols=["DAY", "WEEK_NO"]
        ).drop_duplicates().sort_values("DAY")

        mapping = dict(zip(day_week_pairs["DAY"], day_week_pairs["WEEK_NO"], strict=False))
        self.observed_day_max = max(mapping.keys())

        # Extrapolate for days beyond observation window (> 711) based on observed 7-day cadence
        max_day = self.observed_day_max
        max_week = mapping[max_day]
        for d in range(max_day + 1, max_day + 30):
            mapping[d] = max_week + ((d - max_day - 1) // 7) + 1

        self.day_to_week_map = mapping
        print(f"Fiscal calendar extracted: Days 1 to {self.observed_day_max} observed; Days {self.observed_day_max + 1} to {max(mapping.keys())} extrapolated.")
        return mapping

    def process_campaigns(self) -> pd.DataFrame:
        print("Transforming campaigns with fiscal calendar and censoring metadata...")
        path = self.raw_dir / "campaign_desc.csv"
        df = pd.read_csv(path)

        df["CAMPAIGN"] = df["CAMPAIGN"].astype(int)
        df["START_DAY"] = df["START_DAY"].astype(int)
        df["END_DAY"] = df["END_DAY"].astype(int)

        if not self.day_to_week_map:
            self.extract_day_to_week_mapping()

        # Map start/end weeks using empirical retailer fiscal calendar
        df["start_week"] = df["START_DAY"].map(self.day_to_week_map).astype(int)
        df["end_week"] = df["END_DAY"].map(self.day_to_week_map).astype(int)

        # Consistent inclusive duration: end_day - start_day + 1
        df["duration_days"] = df["END_DAY"] - df["START_DAY"] + 1

        # Observed endpoint and right-censoring calculation
        df["observed_end_day"] = df["END_DAY"].clip(upper=self.observed_day_max)
        df["observed_duration_days"] = df["observed_end_day"] - df["START_DAY"] + 1
        df["is_right_censored"] = df["END_DAY"] > self.observed_day_max
        df["start_week_basis"] = "observed"
        df["end_week_basis"] = np.where(df["END_DAY"] <= self.observed_day_max, "observed", "extrapolated")

        df = df.rename(
            columns={
                "CAMPAIGN": "campaign_id",
                "DESCRIPTION": "campaign_type",
                "START_DAY": "start_day",
                "END_DAY": "end_day",
            }
        )

        out_path = self.output_dir / "campaigns.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Campaigns saved to {out_path} ({len(df)} rows, right_censored={df['is_right_censored'].sum()})")
        return df

    def process_campaign_assignments(self) -> pd.DataFrame:
        print("Transforming campaign assignments...")
        path = self.raw_dir / "campaign_table.csv"
        df = pd.read_csv(path)

        df["household_key"] = df["household_key"].astype(np.int64)
        df["CAMPAIGN"] = df["CAMPAIGN"].astype(int)

        df = df.rename(
            columns={
                "CAMPAIGN": "campaign_id",
                "DESCRIPTION": "campaign_type",
            }
        )

        out_path = self.output_dir / "campaign_assignments.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Campaign assignments saved to {out_path} ({len(df)} rows)")
        return df

    def process_coupons_and_redemptions(self):
        print("Transforming coupons and redemptions...")
        c_path = self.raw_dir / "coupon.csv"
        c_df = pd.read_csv(c_path)
        c_df = c_df.rename(
            columns={
                "COUPON_UPC": "coupon_upc",
                "PRODUCT_ID": "product_id",
                "CAMPAIGN": "campaign_id",
            }
        )
        c_df["product_id"] = c_df["product_id"].astype(np.int64)
        c_df["campaign_id"] = c_df["campaign_id"].astype(int)
        # Note: raw coupon.csv contains 5,164 exact duplicate rows which are preserved here
        # for source fidelity. Downstream analytical joins MUST call drop_duplicates().
        c_df.to_parquet(self.output_dir / "coupons.parquet", index=False)

        r_path = self.raw_dir / "coupon_redempt.csv"
        r_df = pd.read_csv(r_path)
        r_df = r_df.rename(
            columns={
                "household_key": "household_key",
                "DAY": "day",
                "COUPON_UPC": "coupon_upc",
                "CAMPAIGN": "campaign_id",
            }
        )
        r_df["household_key"] = r_df["household_key"].astype(np.int64)
        r_df["day"] = r_df["day"].astype(int)

        if not self.day_to_week_map:
            self.extract_day_to_week_mapping()

        r_df["week_no"] = r_df["day"].map(self.day_to_week_map).astype(int)
        r_df["campaign_id"] = r_df["campaign_id"].astype(int)
        r_df.to_parquet(self.output_dir / "coupon_redemptions.parquet", index=False)
        print("Coupons and redemptions saved.")

    def process_households(self, all_household_keys: set) -> pd.DataFrame:
        print("Transforming households with demographics coverage...")
        demo_path = self.raw_dir / "hh_demographic.csv"
        demo_df = pd.read_csv(demo_path)
        demo_df["household_key"] = demo_df["household_key"].astype(np.int64)
        demo_df["has_demographics"] = True

        demo_df = demo_df.rename(
            columns={
                "AGE_DESC": "age_band",
                "MARITAL_STATUS_CODE": "marital_status",
                "INCOME_DESC": "income_band",
                "HOMEOWNER_DESC": "homeowner_status",
                "HH_COMP_DESC": "household_comp",
                "HOUSEHOLD_SIZE_DESC": "household_size",
                "KID_CATEGORY_DESC": "kid_category",
            }
        )

        all_hh_df = pd.DataFrame({"household_key": sorted(list(all_household_keys))})
        merged = all_hh_df.merge(demo_df, on="household_key", how="left")
        merged["has_demographics"] = merged["has_demographics"].fillna(False).astype(bool)

        val_report = DataValidator.validate_households(merged)
        if not val_report.is_valid:
            raise ValueError(f"Households validation failed: {val_report.errors}")

        out_path = self.output_dir / "households.parquet"
        merged.to_parquet(out_path, index=False)
        print(
            f"Households saved to {out_path} ({len(merged)} rows, {val_report.metrics['demographic_coverage_pct']:.1f}% demo coverage)"
        )
        return merged

    def process_transactions(self, coupon_dept_prod_ids: set) -> set:
        print("Transforming transactions in chunks with refined anomaly classification...")
        txn_path = self.raw_dir / "transaction_data.csv"

        chunk_size = 250_000
        chunks = []
        all_households = set()
        unique_stores = set()

        for chunk in pd.read_csv(txn_path, chunksize=chunk_size):
            all_households.update(chunk["household_key"].unique())
            unique_stores.update(chunk["STORE_ID"].unique())

            # Cast data types
            chunk["household_key"] = chunk["household_key"].astype(np.int64)
            chunk["PRODUCT_ID"] = chunk["PRODUCT_ID"].astype(np.int64)
            chunk["STORE_ID"] = chunk["STORE_ID"].astype(int)
            chunk["DAY"] = chunk["DAY"].astype(int)
            chunk["WEEK_NO"] = chunk["WEEK_NO"].astype(int)
            chunk["QUANTITY"] = chunk["QUANTITY"].astype(float)
            chunk["SALES_VALUE"] = chunk["SALES_VALUE"].astype(float)
            chunk["RETAIL_DISC"] = chunk["RETAIL_DISC"].astype(float)
            chunk["COUPON_DISC"] = chunk["COUPON_DISC"].astype(float)
            chunk["COUPON_MATCH_DISC"] = chunk["COUPON_MATCH_DISC"].astype(float)

            # Refined Anomaly Classification (Condition 2)
            classification = np.full(len(chunk), "valid_purchase", dtype=object)

            zero_qty = chunk["QUANTITY"] == 0
            zero_sales = chunk["SALES_VALUE"] == 0
            pos_qty = chunk["QUANTITY"] > 0
            pos_sales = chunk["SALES_VALUE"] > 0
            has_disc = (chunk["COUPON_DISC"] != 0) | (chunk["RETAIL_DISC"] != 0)
            is_coupon_line = chunk["PRODUCT_ID"].isin(coupon_dept_prod_ids)

            # 1. Void / Zero entries (qty == 0 and sales == 0)
            classification[zero_qty & zero_sales] = "void_or_zero_entry"

            # 2. Fees or service charge (qty == 0 and sales > 0)
            classification[zero_qty & pos_sales] = "fee_or_service_charge"

            # 3. Promotional freebies with recorded discount (qty > 0 and sales == 0 and has_disc)
            classification[pos_qty & zero_sales & has_disc] = "promotional_freebie"

            # 4. Coupon clearing lines (qty > 0 and sales == 0 and ~has_disc and coupon department)
            classification[pos_qty & zero_sales & (~has_disc) & is_coupon_line] = "coupon_clearing_line"

            # 5. Unexplained zero sales (qty > 0 and sales == 0 and ~has_disc and non-coupon department)
            classification[pos_qty & zero_sales & (~has_disc) & (~is_coupon_line)] = "zero_sales_unexplained"

            # 6. Valid purchase (qty > 0 and sales > 0)
            classification[pos_qty & pos_sales] = "valid_purchase"

            chunk["row_classification"] = classification

            # Shelf price reconstruction
            total_discount = chunk["RETAIL_DISC"] + chunk["COUPON_DISC"] + chunk["COUPON_MATCH_DISC"]
            gross_value = chunk["SALES_VALUE"] - total_discount

            # Mask out records where shelf price is not defined or cannot be verified
            unpriced_mask = zero_qty | (zero_sales & (~has_disc))
            safe_qty = chunk["QUANTITY"].where(~unpriced_mask, np.nan)

            chunk["shelf_price"] = np.where(unpriced_mask, np.nan, gross_value / safe_qty)
            chunk["paid_unit_price"] = np.where(unpriced_mask, np.nan, chunk["SALES_VALUE"] / safe_qty)

            # Retail discount percentage
            base_sales = chunk["SALES_VALUE"] - chunk["RETAIL_DISC"]
            raw_discount_pct = np.where(
                base_sales > 0,
                np.abs(chunk["RETAIL_DISC"]) / base_sales,
                0.0,
            )
            chunk["retail_discount_pct"] = np.clip(raw_discount_pct, 0.0, 1.0)

            chunk["in_store_promoted"] = chunk["RETAIL_DISC"] != 0.0
            chunk["coupon_used"] = chunk["COUPON_DISC"] != 0.0

            chunk = chunk.rename(
                columns={
                    "BASKET_ID": "basket_id",
                    "DAY": "day",
                    "PRODUCT_ID": "product_id",
                    "QUANTITY": "quantity",
                    "SALES_VALUE": "sales_value",
                    "STORE_ID": "store_id",
                    "RETAIL_DISC": "retail_disc",
                    "TRANS_TIME": "trans_time",
                    "WEEK_NO": "week_no",
                    "COUPON_DISC": "coupon_disc",
                    "COUPON_MATCH_DISC": "coupon_match_disc",
                }
            )

            chunks.append(chunk)

        full_txns = pd.concat(chunks, ignore_index=True)

        val_report = DataValidator.validate_transactions(full_txns)
        print(f"Transactions validation: is_valid={val_report.is_valid}")
        for w in val_report.warnings:
            print(f"  Warning: {w}")
        if not val_report.is_valid:
            raise ValueError(f"Transactions validation failed: {val_report.errors}")

        out_path = self.output_dir / "transactions.parquet"
        full_txns.to_parquet(out_path, index=False)
        print(f"Transactions saved to {out_path} ({len(full_txns)} rows)")

        stores_df = pd.DataFrame({"store_id": sorted(list(unique_stores))})
        stores_df.to_parquet(self.output_dir / "stores.parquet", index=False)
        print(f"Stores saved ({len(stores_df)} stores)")

        return all_households

    def run_all(self):
        print("Starting PromoSense Remediated ETL Pipeline...")
        prods_df = self.process_products()
        coupon_dept_prod_ids = set(
            prods_df[prods_df["department"] == "COUP/STR & MFG"]["product_id"]
        )

        self.extract_day_to_week_mapping()
        self.process_campaigns()
        self.process_campaign_assignments()
        self.process_coupons_and_redemptions()
        all_hh = self.process_transactions(coupon_dept_prod_ids)
        self.process_households(all_hh)
        print("\nRemediated ETL Pipeline completed successfully!")


if __name__ == "__main__":
    etl = PromoSenseETL()
    etl.run_all()
