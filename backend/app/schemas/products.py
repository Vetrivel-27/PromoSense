"""Product catalog and price elasticity Pydantic schemas."""

from typing import Any

from pydantic import BaseModel, Field

from backend.app.schemas.common import MethodologicalWarnings


class ProductSummary(BaseModel):
    product_id: int = Field(..., description="Unique product SKU ID")
    commodity: str = Field(..., description="Product commodity category (e.g. SOFT DRINKS)")
    sub_commodity: str = Field(..., description="Sub-commodity category (e.g. 12pk cans)")
    brand_type: str = Field(..., description="Brand tier: National or Private")
    department: str = Field(..., description="Department name (e.g. GROCERY)")
    baseline_shelf_price: float = Field(
        ..., description="Average observed baseline shelf price in dollars"
    )
    observed_display_share: float = Field(
        ..., description="Historical average store display execution share"
    )
    observed_mailer_share: float = Field(
        ..., description="Historical average mailer feature execution share"
    )
    substitute_pair_names: list[str] = Field(
        default_factory=list,
        description="Associated substitute pair names if product is modeled in substitute pairs",
    )
    is_focal_substitute_a: bool = Field(
        False,
        description="Whether this product acts as primary promoted product A in a substitute pair",
    )
    is_substitute_b: bool = Field(
        False,
        description="Whether this product acts as competing substitute product B in a substitute pair",
    )


class SubstitutePairSummary(BaseModel):
    pair_name: str = Field(..., description="Canonical substitute pair name")
    commodity: str = Field(..., description="Commodity category")
    sub_commodity: str = Field(..., description="Sub-commodity category")
    prod_A_id: int = Field(..., description="Focal promoted product A SKU ID")
    prod_B_id: int = Field(..., description="Competing substitute product B SKU ID")
    is_weakly_identified: bool = Field(
        ...,
        description="Whether cross-price terms have large standard errors or collinear pricing",
    )


class ProductCatalogResponse(BaseModel):
    products: list[ProductSummary] = Field(..., description="List of target volume product SKUs")
    substitute_pairs: list[SubstitutePairSummary] = Field(
        ..., description="List of substitute product pairs"
    )
    total_products: int = Field(..., description="Total count of target products")


class ProductDetailResponse(ProductSummary):
    min_observed_price: float = Field(
        ..., description="Minimum observed shelf price in historical data"
    )
    max_observed_price: float = Field(
        ..., description="Maximum observed shelf price in historical data"
    )
    historical_weekly_volume_mean: float = Field(
        ..., description="Mean weekly volume in baseline panel"
    )
    historical_weekly_volume_std: float = Field(
        ..., description="Standard deviation of weekly volume"
    )
    substitute_pair_details: list[SubstitutePairSummary] = Field(
        default_factory=list, description="Details of substitute pairs where product participates"
    )


class ElasticityModelDetail(BaseModel):
    model_type: str = Field(..., description="Model class name (e.g. PaidPriceElasticityOLS)")
    elasticity: float = Field(..., description="Estimated price elasticity coefficient")
    elasticity_se: float | None = Field(None, description="Conventional OLS standard error")
    elasticity_hac_se: float | None = Field(
        None, description="Newey-West HAC robust standard error"
    )
    t_stat: float | None = Field(None, description="t-statistic")
    p_value: float | None = Field(None, description="Two-tailed p-value")
    confidence_interval_95_hac: list[float] | None = Field(
        None,
        description="95% confidence interval computed using HAC standard error [low, high]",
    )
    rmse_units: float | None = Field(None, description="Root mean squared error on unit sales")
    r2_log: float | None = Field(None, description="R-squared on log volume")
    coefficients: dict[str, float] | None = Field(None, description="Model feature coefficients")


class SubstituteElasticityDetail(BaseModel):
    pair_name: str = Field(..., description="Canonical pair name")
    substitute_product_id: int = Field(..., description="Substitute product B SKU ID")
    own_elasticity_A: float = Field(
        ..., description="Own-price elasticity of Product A in joint model"
    )
    cross_elasticity_AB: float = Field(
        ...,
        description="Cross-price elasticity: percent change in Product B sales per 1% price change in Product A",
    )
    cross_elasticity_AB_se: float = Field(
        ..., description="Standard error of cross-price elasticity term"
    )
    is_weakly_identified: bool = Field(
        ...,
        description="Whether cross-price elasticity is weakly identified (SE > 0.5 or p >= 0.20)",
    )
    identification_note: str = Field(
        ...,
        description="Methodological note regarding identification and interpretation of cross elasticity",
    )


class ProductElasticityResponse(BaseModel):
    product_id: int = Field(..., description="Product SKU ID")
    commodity: str = Field(..., description="Commodity category")
    sub_commodity: str = Field(..., description="Sub-commodity category")
    brand_type: str = Field(..., description="Brand tier: National or Private")
    champion_specification: str = Field(
        ...,
        description="Name of chosen frozen champion specification",
    )
    champion_elasticity: float = Field(
        ...,
        description="Estimated own-price elasticity from frozen champion model",
    )
    champion_hac_se: float = Field(
        ...,
        description="Newey-West HAC standard error for champion elasticity",
    )
    champion_ci_95: list[float] = Field(
        ...,
        description="95% confidence interval [low, high] under HAC standard error",
    )
    champion_p_value: float = Field(..., description="Statistical significance p-value")
    interpretation: str = Field(
        ..., description="Plain-language economic interpretation of elasticity"
    )
    competing_models: dict[str, Any] = Field(
        ...,
        description="Comparison across unadjusted, decomposed, and non-linear LightGBM specifications",
    )
    substitute_pair_elasticity: SubstituteElasticityDetail | None = Field(
        None,
        description="Cross-price elasticity metrics if product participates in substitute pair",
    )
    methodological_warnings: MethodologicalWarnings = Field(
        ...,
        description="Mandatory methodological disclosures and uncertainty disclaimers",
    )
