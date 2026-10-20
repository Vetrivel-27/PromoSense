"""Investigation script for zero-sales, zero-quantity, and outlier rows in transaction_data.csv.

This implements Decision 3 from Phase 0.5 to rigorously classify anomalies before ETL.
"""

import csv
from collections import Counter
from pathlib import Path


def inspect_transaction_anomalies(data_path: Path):
    print(f"Reading {data_path}...")

    zero_sales_total = 0
    zero_sales_with_coupon = 0
    zero_sales_with_retail_disc = 0
    zero_sales_zero_qty = 0
    zero_sales_pos_qty = 0

    zero_qty_total = 0
    zero_qty_zero_sales = 0
    zero_qty_pos_sales = 0

    extreme_qty_rows = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            qty = float(row["QUANTITY"])
            sales = float(row["SALES_VALUE"])
            retail_disc = float(row["RETAIL_DISC"])
            coupon_disc = float(row["COUPON_DISC"])
            coupon_match = float(row["COUPON_MATCH_DISC"])

            if sales == 0.0:
                zero_sales_total += 1
                if qty == 0:
                    zero_sales_zero_qty += 1
                else:
                    zero_sales_pos_qty += 1

                if coupon_disc != 0.0 or coupon_match != 0.0:
                    zero_sales_with_coupon += 1
                if retail_disc != 0.0:
                    zero_sales_with_retail_disc += 1

            if qty == 0.0:
                zero_qty_total += 1
                if sales == 0:
                    zero_qty_zero_sales += 1
                else:
                    zero_qty_pos_sales += 1

            if qty > 500:
                if len(extreme_qty_rows) < 5:
                    extreme_qty_rows.append(
                        {
                            "row": i,
                            "product_id": row["PRODUCT_ID"],
                            "qty": qty,
                            "sales": sales,
                            "retail_disc": retail_disc,
                        }
                    )

    print("\n--- ZERO SALES_VALUE SUMMARY ---")
    print(f"Total rows with SALES_VALUE == 0: {zero_sales_total}")
    print(f"  With QUANTITY == 0: {zero_sales_zero_qty}")
    print(f"  With QUANTITY > 0:  {zero_sales_pos_qty}")
    print(f"  With COUPON_DISC != 0: {zero_sales_with_coupon}")
    print(f"  With RETAIL_DISC != 0: {zero_sales_with_retail_disc}")

    print("\n--- ZERO QUANTITY SUMMARY ---")
    print(f"Total rows with QUANTITY == 0: {zero_qty_total}")
    print(f"  With SALES_VALUE == 0: {zero_qty_zero_sales}")
    print(f"  With SALES_VALUE > 0:  {zero_qty_pos_sales}")

    print("\n--- EXTREME QUANTITY EXAMPLES (QTY > 500) ---")
    for ex in extreme_qty_rows:
        print(f"  {ex}")


if __name__ == "__main__":
    raw_path = Path("data/raw/transaction_data.csv")
    if raw_path.exists():
        inspect_transaction_anomalies(raw_path)
    else:
        print(f"File not found: {raw_path}")
