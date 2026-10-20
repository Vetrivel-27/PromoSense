"""Customer and Uplift Intelligence Pydantic schemas for PromoSense Backend."""

from pydantic import BaseModel, Field

from backend.app.schemas.common import MethodologicalWarnings


class CustomerDemographics(BaseModel):
    has_demographics: bool = Field(
        ..., description="Whether household has census demographic disclosures (32% coverage)"
    )
    age_band: str | None = Field(None, description="Age group band (e.g. '35-44', '65+')")
    marital_status: str | None = Field(None, description="Marital status (A=Married, B=Single)")
    income_band: str | None = Field(None, description="Household annual income tier")
    homeowner_status: str | None = Field(None, description="Homeowner status (Homeowner, Renter)")
    household_comp: str | None = Field(None, description="Household composition category")
    household_size: str | None = Field(None, description="Household member count string")
    kid_category: str | None = Field(None, description="Children count category")


class CustomerBehavior(BaseModel):
    historical_spend_dollars: float = Field(
        ..., description="Total observed transaction spend in dollars across 102 weeks"
    )
    historical_trips_count: int = Field(..., description="Total checkout visits across 102 weeks")
    campaigns_assigned_count: int = Field(
        ..., description="Number of direct-mail campaigns household was assigned to receive"
    )
    coupons_redeemed_count: int = Field(
        ..., description="Total coupons redeemed in store by household"
    )


class CustomerDetailResponse(BaseModel):
    household_key: int = Field(..., description="Unique customer household identifier (1..2500)")
    demographics: CustomerDemographics = Field(..., description="Household demographic profile")
    behavior: CustomerBehavior = Field(..., description="Historical shopping behavior summary")
    methodological_note: str = Field(
        ..., description="Note regarding household-level identification and observational coverage"
    )


class CustomerUpliftResponse(BaseModel):
    household_key: int = Field(..., description="Unique customer household identifier (1..2500)")
    evaluated_campaign_id: int = Field(
        ..., description="Marketing campaign context evaluated for uplift"
    )
    uplift_score_cate: float = Field(
        ...,
        description="Conditional Average Treatment Effect CATE: E[Y(1) - Y(0) | X] in spend dollars",
    )
    targeting_segment: str = Field(
        ...,
        description="Targeting quadrant classification: 'Persuadables', 'Sure Things', 'Lost Causes'",
    )
    recommended_action: str = Field(
        ...,
        description="Economically grounded action: 'PROMOTE', 'NO PROMOTION', 'AVOID DISCOUNT'",
    )
    expected_spend_with_treatment: float = Field(
        ..., description="Projected spend under campaign treatment Y(1)"
    )
    expected_spend_without_treatment: float = Field(
        ..., description="Projected baseline spend without treatment Y(0)"
    )
    cost_threshold_dollars: float = Field(
        ..., description="Assumed marginal mailing and fulfillment cost per household ($1.00)"
    )
    is_recommended_for_treatment: bool = Field(
        ..., description="Whether incremental CATE net of cost justifies direct mail treatment"
    )
    model_provenance: str = Field(
        ..., description="Causal estimator model description and hyperparameters"
    )
    methodological_warnings: MethodologicalWarnings = Field(
        ..., description="Causal identification warnings and unconfoundedness caveats"
    )


class QiniDecileItem(BaseModel):
    decile: int = Field(..., description="Targeting population decile (1 to 10)")
    population_fraction: float = Field(
        ..., description="Cumulative population fraction (0.1 to 1.0)"
    )
    n_targeted_cum: int = Field(..., description="Cumulative targeted customer count")
    cum_gain: float = Field(..., description="Cumulative incremental revenue gain in dollars")
    qini_val: float = Field(..., description="Qini curve value at this decile")
    mean_treated_cum: float = Field(..., description="Cumulative mean spend in treated cohort")
    mean_control_cum: float = Field(..., description="Cumulative mean spend in control cohort")


class CustomerSegmentsResponse(BaseModel):
    qini_score: float = Field(
        ..., description="Overall Qini performance metric from Phase 5 benchmark"
    )
    auuc: float = Field(..., description="Area Under Uplift Curve metric")
    overall_cohort_ate: float = Field(
        ..., description="Average Treatment Effect across entire evaluation cohort ($)"
    )
    top_decile_lift: float = Field(
        ..., description="Incremental spend lift achieved in top targeting decile ($)"
    )
    top_decile_multiplier: float = Field(
        ..., description="Ratio of top decile incremental lift to average cohort ATE"
    )
    deciles: list[QiniDecileItem] = Field(
        ..., description="Qini cumulative gain decile progression"
    )
    recommended_targeting_cutoff_decile: int = Field(
        ..., description="Optimal decile cutoff maximizing incremental net profit"
    )
    policy_guidance: str = Field(
        ..., description="Recommendation strategy distinguishing Persuadables from Sure Things"
    )
    methodological_note: str = Field(
        ..., description="Methodological caveats regarding observational causal uplift validation"
    )
