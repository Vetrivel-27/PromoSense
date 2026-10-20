from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class Household(Base):
    __tablename__ = "households"

    household_key = Column(BigInteger, primary_key=True, index=True)
    age_band = Column(String(50), nullable=True)
    marital_status = Column(String(20), nullable=True)
    income_band = Column(String(50), nullable=True)
    homeowner_status = Column(String(50), nullable=True)
    household_comp = Column(String(100), nullable=True)
    household_size = Column(String(20), nullable=True)
    kid_category = Column(String(50), nullable=True)
    has_demographics = Column(Boolean, default=False, nullable=False)

    # Relationships
    transactions = relationship("Transaction", back_populates="household")
    campaign_assignments = relationship("CampaignAssignment", back_populates="household")
    coupon_redemptions = relationship("CouponRedemption", back_populates="household")


class Product(Base):
    __tablename__ = "products"

    product_id = Column(BigInteger, primary_key=True, index=True)
    manufacturer_id = Column(BigInteger, index=True, nullable=False)
    department = Column(String(100), index=True, nullable=False)
    brand_type = Column(String(50), nullable=False)
    commodity = Column(String(100), index=True, nullable=False)
    sub_commodity = Column(String(100), nullable=False)
    size_description = Column(String(100), nullable=True)

    # Relationships
    transactions = relationship("Transaction", back_populates="product")
    coupons = relationship("Coupon", back_populates="product")


class Store(Base):
    __tablename__ = "stores"

    store_id = Column(Integer, primary_key=True, index=True)

    # Relationships
    transactions = relationship("Transaction", back_populates="store")


class Campaign(Base):
    __tablename__ = "campaigns"

    campaign_id = Column(Integer, primary_key=True, index=True)
    campaign_type = Column(String(50), nullable=False)
    start_day = Column(Integer, nullable=False)
    end_day = Column(Integer, nullable=False)
    start_week = Column(Integer, nullable=False)
    end_week = Column(Integer, nullable=False)
    duration_days = Column(Integer, nullable=False)

    # Relationships
    assignments = relationship("CampaignAssignment", back_populates="campaign")
    coupons = relationship("Coupon", back_populates="campaign")
    redemptions = relationship("CouponRedemption", back_populates="campaign")


class CampaignAssignment(Base):
    __tablename__ = "campaign_assignments"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    household_key = Column(
        BigInteger, ForeignKey("households.household_key"), index=True, nullable=False
    )
    campaign_id = Column(Integer, ForeignKey("campaigns.campaign_id"), index=True, nullable=False)
    campaign_type = Column(String(50), nullable=False)

    # Relationships
    household = relationship("Household", back_populates="campaign_assignments")
    campaign = relationship("Campaign", back_populates="assignments")

    __table_args__ = (Index("idx_assignment_hh_camp", "household_key", "campaign_id", unique=True),)


class Coupon(Base):
    __tablename__ = "coupons"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    coupon_upc = Column(String(50), index=True, nullable=False)
    product_id = Column(BigInteger, ForeignKey("products.product_id"), index=True, nullable=False)
    campaign_id = Column(Integer, ForeignKey("campaigns.campaign_id"), index=True, nullable=False)

    # Relationships
    product = relationship("Product", back_populates="coupons")
    campaign = relationship("Campaign", back_populates="coupons")

    __table_args__ = (Index("idx_coupon_upc_prod_camp", "coupon_upc", "product_id", "campaign_id"),)


class CouponRedemption(Base):
    __tablename__ = "coupon_redemptions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    household_key = Column(
        BigInteger, ForeignKey("households.household_key"), index=True, nullable=False
    )
    day = Column(Integer, index=True, nullable=False)
    week_no = Column(Integer, index=True, nullable=False)
    coupon_upc = Column(String(50), index=True, nullable=False)
    campaign_id = Column(Integer, ForeignKey("campaigns.campaign_id"), index=True, nullable=False)

    # Relationships
    household = relationship("Household", back_populates="coupon_redemptions")
    campaign = relationship("Campaign", back_populates="redemptions")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    household_key = Column(
        BigInteger, ForeignKey("households.household_key"), index=True, nullable=False
    )
    basket_id = Column(String(50), index=True, nullable=False)
    day = Column(Integer, index=True, nullable=False)
    week_no = Column(Integer, index=True, nullable=False)
    product_id = Column(BigInteger, ForeignKey("products.product_id"), index=True, nullable=False)
    store_id = Column(Integer, ForeignKey("stores.store_id"), index=True, nullable=False)

    quantity = Column(Float, nullable=False)
    sales_value = Column(Float, nullable=False)
    retail_disc = Column(Float, default=0.0, nullable=False)
    coupon_disc = Column(Float, default=0.0, nullable=False)
    coupon_match_disc = Column(Float, default=0.0, nullable=False)
    trans_time = Column(Integer, nullable=True)

    # Derived and validated fields
    shelf_price = Column(Float, nullable=True)
    paid_unit_price = Column(Float, nullable=True)
    retail_discount_pct = Column(Float, default=0.0, nullable=False)
    in_store_promoted = Column(Boolean, default=False, nullable=False)
    coupon_used = Column(Boolean, default=False, nullable=False)
    row_classification = Column(String(50), default="valid_purchase", nullable=False)

    # Relationships
    household = relationship("Household", back_populates="transactions")
    product = relationship("Product", back_populates="transactions")
    store = relationship("Store", back_populates="transactions")

    __table_args__ = (
        Index("idx_txn_hh_week", "household_key", "week_no"),
        Index("idx_txn_prod_week", "product_id", "week_no"),
        Index("idx_txn_store_prod_week", "store_id", "product_id", "week_no"),
    )


class StorePromotion(Base):
    __tablename__ = "store_promotions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(BigInteger, ForeignKey("products.product_id"), index=True, nullable=False)
    store_id = Column(Integer, ForeignKey("stores.store_id"), index=True, nullable=False)
    week_no = Column(Integer, index=True, nullable=False)
    display_code = Column(String(20), nullable=False)
    mailer_code = Column(String(20), nullable=False)
    has_display_placement = Column(Boolean, default=False, nullable=False)
    has_mailer_placement = Column(Boolean, default=False, nullable=False)

    __table_args__ = (Index("idx_promo_prod_store_week", "product_id", "store_id", "week_no"),)
