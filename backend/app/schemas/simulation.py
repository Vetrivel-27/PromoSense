"""Promotion simulation request and response Pydantic schemas."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.common import (
    BaselineModeEnum,
    EmpiricalSupportMetadata,
    MethodologicalWarnings,
    ProfitByMarginTier,
)


class SimulationRequest(BaseModel):
    product_id: int = Field(
        ...,
        description="Target product SKU ID (must be one of the 7 modeled core SKUs)",
        examples=[5569230],
    )
    discount_pct: float = Field(
        0.20,
        ge=0.0,
        le=0.35,
        description=(
            "Promotional discount fraction (0.0 for baseline comparison, or 0.05 to 0.35 for active promotion). "
            "Discounts between 0.0 and 0.05 or exceeding 0.35 are rejected."
        ),
        examples=[0.20],
    )
    with_display: bool = Field(
        False,
        description="Whether to simulate 100% store endcap display execution (co-merchandising)",
        examples=[False],
    )
    baseline_mode: BaselineModeEnum = Field(
        BaselineModeEnum.OBSERVED_MERCH,
        description="'observed_merch' (realistic status quo) or 'zero_merch' (extrapolative bound)",
        examples=["observed_merch"],
    )
    include_substitute: bool = Field(
        True,
        description="Whether to compute cannibalization effects on substitute product if SKU belongs to a modeled pair",
        examples=[True],
    )

    @field_validator("discount_pct")
    @classmethod
    def validate_discount_support(cls, v: float) -> float:
        # Permit exact 0.0 for direct unpromoted baseline evaluation
        if v == 0.0:
            return v
        if v < 0.05:
            raise ValueError(
                f"Discount {v:.1%} is below minimum active promotional threshold (5.0%). "
                "Use 0.0 to evaluate unpromoted baseline, or >= 0.05 for active promotions."
            )
        if v > 0.35:
            raise ValueError(
                f"Discount {v:.1%} exceeds maximum supported promotional ceiling (35.0%). "
                "Extrapolating beyond 35% is unsupported by empirical training data."
            )
        return round(v, 4)


class SingleSkuSimulationDetails(BaseModel):
    product_id: int = Field(..., description="Target product SKU ID")
    avg_shelf_price: float = Field(..., description="Average base list shelf price in dollars")
    promotional_price: float = Field(..., description="Promotional unit price in dollars")
    discount_depth: float = Field(..., description="Log discount depth -ln(1 - discount_pct)")
    units_baseline: float = Field(..., description="Baseline unit volume over evaluation period")
    units_promoted: float = Field(
        ..., description="Simulated promoted unit volume over evaluation period"
    )
    delta_units: float = Field(..., description="Incremental unit sales lift (Delta Q)")
    pct_unit_lift: float = Field(..., description="Percentage unit sales lift")
    revenue_baseline: float = Field(
        ..., description="Baseline dollar revenue over evaluation period"
    )
    revenue_promoted: float = Field(..., description="Simulated promoted dollar revenue")
    delta_revenue: float = Field(..., description="Incremental gross revenue lift (Delta R)")
    pct_revenue_lift: float = Field(..., description="Percentage revenue lift")
    contribution_profits: dict[str, ProfitByMarginTier] = Field(
        ...,
        description="Contribution profit breakdown across 15%, 25%, and 35% baseline gross margin tiers",
    )
    profit_uncertainty_hac: dict[str, Any] = Field(
        ...,
        description="Partial parameter uncertainty bounds propagated from own-price HAC standard error",
    )


class SubstitutePairSimulationDetails(BaseModel):
    pair_name: str = Field(..., description="Canonical pair name")
    prod_A_id: int = Field(..., description="Focal promoted product A SKU ID")
    prod_B_id: int = Field(..., description="Substitute product B SKU ID")
    is_weakly_identified: bool = Field(
        ...,
        description="Whether cross-price terms are weakly identified due to collinear pricing",
    )
    delta_units_A: float = Field(..., description="Incremental units for promoted product A")
    delta_units_B: float = Field(
        ..., description="Cannibalization unit volume loss for substitute product B"
    )
    net_category_units: float = Field(
        ..., description="Net category unit volume change (Delta Q_A + Delta Q_B)"
    )
    delta_revenue_A: float = Field(..., description="Incremental revenue for product A in dollars")
    delta_revenue_B: float = Field(
        ..., description="Revenue offset / loss on substitute product B in dollars"
    )
    net_category_revenue: float = Field(..., description="Net category revenue change in dollars")
    revenue_offset_ratio: float | None = Field(
        None,
        description="Fraction of product A revenue expansion offset by product B loss (|Delta R_B| / Delta R_A)",
    )
    profit_impact_isolated: dict[str, Any] = Field(
        ...,
        description="Product A profit impact in isolation across margin tiers",
    )
    profit_impact_category_modeled: dict[str, Any] = Field(
        ...,
        description="Net category profit impact including substitute B across margin tiers",
    )
    weak_identification_warning: str | None = Field(
        None,
        description="Warning banner if cross-price term is weakly identified",
    )


class SimulationResponse(BaseModel):
    scenario: dict[str, Any] = Field(
        ..., description="Evaluated scenario configuration and metadata"
    )
    single_sku: SingleSkuSimulationDetails = Field(
        ...,
        description="Single-product demand, revenue, and contribution profit projections",
    )
    substitute_pair: SubstitutePairSimulationDetails | None = Field(
        None,
        description="Substitute product cannibalization projections if product is part of a modeled pair",
    )
    empirical_support: EmpiricalSupportMetadata = Field(
        ...,
        description="Empirical support audit checking univariate and joint domain coverage",
    )
    methodological_warnings: MethodologicalWarnings = Field(
        ...,
        description="Mandatory disclosures regarding observational data, missing costs, and uncertainty",
    )
