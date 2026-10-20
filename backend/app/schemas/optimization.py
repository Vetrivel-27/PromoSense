"""Promotion optimization request and response Pydantic schemas."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from backend.app.schemas.common import (
    BaselineModeEnum,
    EmpiricalSupportMetadata,
    MethodologicalWarnings,
)


class OptimizationRequest(BaseModel):
    product_id: int = Field(
        ...,
        description="Target product SKU ID to optimize",
        examples=[5569230],
    )
    with_display: bool = Field(
        False,
        description="Whether to evaluate optimization with 100% store endcap display execution",
        examples=[False],
    )
    baseline_mode: BaselineModeEnum = Field(
        BaselineModeEnum.OBSERVED_MERCH,
        description="'observed_merch' (realistic status quo) or 'zero_merch' (extrapolative bound)",
        examples=["observed_merch"],
    )
    min_discount: float = Field(
        0.05,
        ge=0.05,
        le=0.35,
        description="Minimum candidate promotional discount (default 0.05 / 5%)",
        examples=[0.05],
    )
    max_discount: float = Field(
        0.35,
        ge=0.05,
        le=0.35,
        description="Maximum candidate promotional discount (default 0.35 / 35%)",
        examples=[0.35],
    )
    discount_step: float = Field(
        0.025,
        gt=0.0,
        le=0.10,
        description="Grid search step size (default 0.025 / 2.5%)",
        examples=[0.025],
    )
    include_substitute: bool = Field(
        True,
        description="Whether to optimize net category profit including substitute cannibalization if pair exists",
        examples=[True],
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "OptimizationRequest":
        if self.min_discount > self.max_discount:
            raise ValueError(
                f"min_discount ({self.min_discount:.1%}) cannot exceed max_discount ({self.max_discount:.1%})"
            )
        return self


class OptimalPolicyTier(BaseModel):
    assumed_gross_margin: float = Field(
        ..., description="Assumed baseline gross margin tier (e.g. 0.25)"
    )
    optimal_discount_pct: float = Field(
        ...,
        description="Profit-maximizing discount depth (0.0% if unpromoted shelf price is optimal)",
    )
    modeled_best_promotional_candidate: float = Field(
        ...,
        description="Highest-performing non-zero promotional candidate evaluated in grid",
    )
    recommend_price_promotion: bool = Field(
        ...,
        description="True if an active promotion generates strictly positive profit lift Delta Pi > 0",
    )
    delta_profit_at_optimum: float = Field(
        ..., description="Incremental contribution profit lift at recommended policy"
    )
    modeled_candidate_delta_profit: float = Field(
        ...,
        description="Incremental contribution profit lift of the best active promotional candidate",
    )
    pct_profit_lift_at_optimum: float = Field(
        ..., description="Percentage profit lift at recommended policy"
    )
    delta_revenue_at_optimum: float = Field(
        ..., description="Incremental revenue lift at recommended policy"
    )
    delta_units_at_optimum: float = Field(
        ..., description="Incremental unit volume lift at recommended policy"
    )
    promotional_unit_price: float = Field(..., description="Recommended unit price in dollars")
    partial_uncertainty_bound: dict[str, Any] | None = Field(
        None,
        description="Partial HAC parameter uncertainty bounds at recommended policy",
    )
    recommendation_note: str = Field(
        ..., description="Plain-language business rationale and policy guidance"
    )


class GridSweepPoint(BaseModel):
    discount_pct: float = Field(..., description="Candidate promotional discount fraction")
    delta_units: float = Field(..., description="Projected incremental unit volume")
    delta_revenue: float = Field(..., description="Projected incremental gross revenue")
    delta_profits: dict[str, float] = Field(
        ...,
        description="Projected incremental contribution profit across margin tiers (15%, 25%, 35%)",
    )


class NoPromotionBaselinePoint(BaseModel):
    discount_pct: float = Field(0.0, description="Baseline discount fraction")
    delta_profit: float = Field(0.0, description="Baseline incremental profit (identically 0.0)")
    delta_revenue: float = Field(0.0, description="Baseline incremental revenue (identically 0.0)")
    delta_units: float = Field(0.0, description="Baseline incremental units (identically 0.0)")


class OptimizationResponse(BaseModel):
    product_id: int = Field(..., description="Optimized product SKU ID")
    with_display: bool = Field(..., description="Merchandising mode evaluated")
    baseline_mode: str = Field(..., description="Baseline mode evaluated")
    search_grid: dict[str, Any] = Field(..., description="Search boundaries and step configuration")
    no_promotion_baseline: NoPromotionBaselinePoint = Field(
        ...,
        description="Explicit no-promotion baseline anchor against which all candidates are benchmarked",
    )
    single_sku_policies: dict[str, OptimalPolicyTier] = Field(
        ...,
        description="Optimal policies across 15%, 25%, and 35% margin tiers for single product",
    )
    single_sku_grid_sweep: list[GridSweepPoint] = Field(
        ...,
        description="Full evaluation grid sweep curve showing trade-offs across discount depths",
    )
    category_pair_policies: dict[str, Any] | None = Field(
        None,
        description="Net category optimal policies after deducting substitute margin offsets",
    )
    category_pair_grid_sweep: list[dict[str, Any]] | None = Field(
        None,
        description="Category net evaluation curve across discount depths if product belongs to pair",
    )
    empirical_support: EmpiricalSupportMetadata = Field(
        ...,
        description="Support audit for candidate search domain",
    )
    methodological_warnings: MethodologicalWarnings = Field(
        ...,
        description="Mandatory methodological disclosures and uncertainty disclaimers",
    )
