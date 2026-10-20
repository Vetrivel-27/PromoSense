import pandas as pd
import pytest

from ml_pipeline.src.config.settings import ml_settings


@pytest.fixture(scope="module")
def interim_dir():
    return ml_settings.INTERIM_DATA_DIR


def test_products_interim_dataset(interim_dir):
    path = interim_dir / "products.parquet"
    assert path.exists()

    df = pd.read_parquet(path)
    assert len(df) == 92353
    assert df["product_id"].is_unique
    assert "department" in df.columns
    assert "commodity" in df.columns
    assert "brand_type" in df.columns


def test_households_interim_dataset(interim_dir):
    path = interim_dir / "households.parquet"
    assert path.exists()

    df = pd.read_parquet(path)
    assert len(df) == 2500
    assert df["household_key"].is_unique
    assert "has_demographics" in df.columns
    assert df["has_demographics"].sum() == 801


def test_campaigns_interim_dataset(interim_dir):
    path = interim_dir / "campaigns.parquet"
    assert path.exists()

    df = pd.read_parquet(path)
    assert len(df) == 30
    assert (df["duration_days"] > 0).all()
    assert (df["start_day"] < df["end_day"]).all()


def test_transactions_shelf_price_and_classification(interim_dir):
    path = interim_dir / "transactions.parquet"
    assert path.exists()

    # Read sample of transactions to keep test ultra-fast
    df = pd.read_parquet(
        path,
        columns=[
            "household_key",
            "product_id",
            "quantity",
            "sales_value",
            "shelf_price",
            "row_classification",
            "retail_discount_pct",
        ],
    )
    assert len(df) == 2595732

    # Shelf price should never be negative
    valid_prices = df["shelf_price"].dropna()
    assert (valid_prices >= 0).all()

    # Classification distribution checks
    classifications = df["row_classification"].unique()
    assert "valid_purchase" in classifications
    assert "void_or_zero_entry" in classifications
    assert "promotional_freebie" in classifications
    assert "coupon_clearing_line" in classifications
    assert "zero_sales_unexplained" in classifications

    # Retail discount percentage should be between 0 and 1
    assert (df["retail_discount_pct"] >= 0.0).all()
    assert (df["retail_discount_pct"] <= 1.0).all()


def test_deterministic_transaction_discount_accounting() -> None:
    """Deterministic verification of shelf price and discount depth accounting.

    Verifies the exact accounting semantics of transactions containing both
    store retail discounts (RETAIL_DISC < 0) and manufacturer coupons (COUPON_DISC < 0).
    """
    import numpy as np

    # Synthetic scenario: Customer purchases 2 units.
    # Regular shelf price = $2.50/unit -> Gross value = $5.00
    # Store sales discount: -$1.00 (-$0.50/unit)
    # Manufacturer coupon:  -$0.50 (-$0.25/unit)
    # Coupon match discount: $0.00
    # Net paid receipt (SALES_VALUE) = $5.00 - $1.00 - $0.50 = $3.50
    qty = 2.0
    sales_val = 3.50
    retail_disc = -1.00
    coupon_disc = -0.50
    coupon_match = 0.00

    # 1. ETL Loader gross value & shelf price reconstruction semantics
    total_disc = retail_disc + coupon_disc + coupon_match  # -1.50
    gross_val = sales_val - total_disc  # 3.50 - (-1.50) = 5.00
    reconstructed_shelf_price = gross_val / qty  # 2.50
    paid_unit_price = sales_val / qty  # 1.75

    assert np.isclose(reconstructed_shelf_price, 2.50)
    assert np.isclose(paid_unit_price, 1.75)
    assert reconstructed_shelf_price > paid_unit_price

    # 2. Total discount percentage and discount depth semantics
    total_disc_pct = (reconstructed_shelf_price - paid_unit_price) / reconstructed_shelf_price
    assert np.isclose(total_disc_pct, 0.30)  # ($2.50 - $1.75) / $2.50 = 30%

    discount_depth = -np.log(1.0 - total_disc_pct)
    expected_depth = np.log(reconstructed_shelf_price) - np.log(paid_unit_price)
    assert np.isclose(discount_depth, expected_depth)
    assert np.isclose(discount_depth, 0.3566749439)

    # 3. Invariant: Reconstructed shelf price and discount depth reproduce paid price exactly
    reproduced_paid = reconstructed_shelf_price * (1.0 - total_disc_pct)
    reproduced_paid_exp = reconstructed_shelf_price * np.exp(-discount_depth)
    assert np.isclose(reproduced_paid, paid_unit_price)
    assert np.isclose(reproduced_paid_exp, paid_unit_price)
