"""Common Pydantic schemas, enums, and structured methodological warnings for PromoSense API."""

from enum import StrEnum

from pydantic import BaseModel, Field


class BaselineModeEnum(StrEnum):
    OBSERVED_MERCH = "observed_merch"
    ZERO_MERCH = "zero_merch"


class EmpiricalSupportMetadata(BaseModel):
    is_jointly_supported: bool = Field(
        ...,
        description="Whether proposed discount and merchandising combination falls within empirical joint support",
    )
    discount_supported: bool = Field(
        ...,
        description="Whether discount depth falls within empirical supported promotion range [0.05, 0.35]",
    )
    discount_warning: str | None = Field(
        None,
        description="Warning message if discount is outside active promotional bounds",
    )
    is_extrapolative_merchandising: bool = Field(
        ...,
        description="Whether scenario uses extrapolative merchandising (e.g. 0% baseline vs 100% display)",
    )
    merchandising_warning: str | None = Field(
        None,
        description="Warning message explaining observational merchandising extrapolation",
    )
    joint_support_warning: str | None = Field(
        None,
        description="Warning when univariate range inclusion does not guarantee joint empirical support",
    )


class MethodologicalWarnings(BaseModel):
    cost_basis_assumption: str = Field(
        "Wholesale unit costs derived from baseline shelf price assumption c_j = P_shelf * (1 - m_base); not observed supplier invoices.",
        description="Disclosure that scanner data lacks wholesale costs and relies on configured margin tiers",
    )
    observational_unconfoundedness_disclaimer: str = Field(
        "Estimates reflect observational regression associations controlling for merchandising, calendar trends, and seasonality. Not randomized controlled trial (A/B test) treatment effects.",
        description="Causal inference disclaimer distinguishing observational regressions from RCTs",
    )
    partial_uncertainty_disclaimer: str = Field(
        "Uncertainty bounds reflect partial parameter uncertainty propagated strictly from own-price HAC standard errors holding other covariates fixed. Does NOT represent a complete joint confidence interval.",
        description="Disclaimer on partial parameter uncertainty bounds",
    )
    cross_price_warning: str | None = Field(
        None,
        description="Warning when cross-price elasticity term is weakly identified (e.g. fluid milk pair)",
    )
    non_causal_label: str = Field(
        "model_implied_counterfactual_projection",
        description="Explicit methodological label preventing overclaiming of causal treatment effects",
    )


class ProfitByMarginTier(BaseModel):
    assumed_gross_margin: float = Field(
        ..., description="Assumed baseline gross margin fraction (e.g. 0.25)"
    )
    cost_basis_label: str = Field(..., description="Label identifying cost derivation method")
    profit_baseline: float = Field(..., description="Baseline contribution profit in dollars")
    profit_promoted: float = Field(..., description="Promoted contribution profit in dollars")
    delta_profit: float = Field(..., description="Incremental contribution profit lift in dollars")
    pct_profit_lift: float = Field(..., description="Percentage profit lift")
    is_profitable_lift: bool = Field(
        ..., description="Whether promotion generates positive profit lift"
    )
