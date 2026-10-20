"""Dashboard Pydantic schemas for PromoSense Executive & Portfolio Views."""

from pydantic import BaseModel, Field

from backend.app.schemas.common import MethodologicalWarnings


class DashboardSummaryResponse(BaseModel):
    total_sales_dollars: float = Field(
        ..., description="Total chain-wide sales across all 102 weeks in dollars"
    )
    promotional_sales_dollars: float = Field(
        ..., description="Total sales from transactions with retail or coupon discounts"
    )
    promotional_sales_share: float = Field(
        ..., description="Proportion of total sales generated during promotion"
    )
    total_units_sold: float = Field(
        ..., description="Total chain-wide quantity of items sold across 102 weeks"
    )
    total_transactions: int = Field(
        ..., description="Total line-item transaction count in empirical dataset"
    )
    total_baskets: int = Field(
        ..., description="Distinct checkout basket count across all stores and weeks"
    )
    total_households: int = Field(..., description="Total observed customer household panel size")
    total_campaigns: int = Field(..., description="Total Dunnhumby marketing campaigns executed")
    modeled_target_skus: int = Field(
        ..., description="Number of target high-volume products with econometric elasticity models"
    )
    modeled_product_revenue: float = Field(
        ..., description="Aggregate 102-week sales for the 7 modeled target volume SKUs"
    )
    modeled_product_revenue_share: float = Field(
        ..., description="Proportion of chain sales represented by the 7 modeled SKUs"
    )
    gross_margin_tiers: list[float] = Field(
        default=[0.15, 0.25, 0.35],
        description="Assumed gross margin tiers used across scenario profit simulations",
    )
    portfolio_scope_disclaimer: str = Field(
        ...,
        description="Explicit methodological boundary explaining unmodeled catalog extrapolation limits",
    )
    methodological_warnings: MethodologicalWarnings = Field(
        ...,
        description="Mandatory methodological disclosures regarding cost assumptions and causal validity",
    )


class WeeklySalesTrendItem(BaseModel):
    week_no: int = Field(..., description="Fiscal week number (1 to 102)")
    total_sales: float = Field(..., description="Total chain sales in week in dollars")
    promotional_sales: float = Field(..., description="Sales from discounted items in week")
    base_sales: float = Field(..., description="Base (non-discounted) sales in week")
    total_units: float = Field(..., description="Total units sold in week")
    promotional_share: float = Field(
        ..., description="Ratio of promotional sales to total sales in week"
    )


class DashboardSalesTrendResponse(BaseModel):
    start_week: int = Field(..., description="First week in trend series")
    end_week: int = Field(..., description="Last week in trend series")
    total_weeks: int = Field(..., description="Total count of weeks in panel")
    trend: list[WeeklySalesTrendItem] = Field(..., description="Weekly sales time series")


class CampaignSummaryItem(BaseModel):
    campaign_id: int = Field(..., description="Dunnhumby campaign identifier")
    campaign_type: str = Field(..., description="Campaign modality (TypeA, TypeB, TypeC)")
    start_day: int = Field(..., description="Start day offset from dataset start")
    end_day: int = Field(..., description="End day offset from dataset start")
    duration_days: int = Field(..., description="Duration in calendar days")
    households_targeted: int = Field(..., description="Number of households assigned to campaign")
    target_rate: float = Field(..., description="Proportion of 2,500 households reached")


class DashboardPromotionsResponse(BaseModel):
    total_campaigns: int = Field(..., description="Total marketing campaigns in dataset")
    campaign_types: dict[str, int] = Field(..., description="Count of campaigns by type")
    pooled_ate_spend_lift: float = Field(
        ...,
        description="Estimated cohort Average Treatment Effect (ATE) spend lift from Phase 5 champion model",
    )
    top_decile_lift: float = Field(
        ...,
        description="Top-decile incremental lift per targeted household from Qini curve",
    )
    campaigns: list[CampaignSummaryItem] = Field(
        ..., description="List of all 30 campaigns with reach and duration metrics"
    )
    methodological_note: str = Field(
        ...,
        description="Distinction between direct-mail campaign uplift and in-store price elasticity",
    )
