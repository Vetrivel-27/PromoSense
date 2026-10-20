/**
 * TypeScript Type Definitions for PromoSense Decision-Support Platform.
 * Mirrors backend Pydantic schemas in backend/app/schemas/
 */

export type BaselineMode = 'observed_merch' | 'zero_merch';

export interface EmpiricalSupportMetadata {
	is_jointly_supported: boolean;
	discount_supported: boolean;
	discount_warning: string | null;
	is_extrapolative_merchandising: boolean;
	merchandising_warning: string | null;
	joint_support_warning: string | null;
}

export interface MethodologicalWarnings {
	cost_basis_assumption: string;
	observational_unconfoundedness_disclaimer: string;
	partial_uncertainty_disclaimer: string;
	cross_price_warning: string | null;
	non_causal_label: string;
}

export interface ProfitByMarginTier {
	assumed_gross_margin: number;
	cost_basis_label: string;
	profit_baseline: number;
	profit_promoted: number;
	delta_profit: number;
	pct_profit_lift: number;
	is_profitable_lift: boolean;
}

export interface ProductSummary {
	product_id: number;
	commodity: string;
	sub_commodity: string;
	brand_type: string;
	department: string;
	baseline_shelf_price: number;
	observed_display_share: number;
	observed_mailer_share: number;
	substitute_pair_names: string[];
	is_focal_substitute_a: boolean;
	is_substitute_b: boolean;
}

export interface SubstitutePairSummary {
	pair_name: string;
	commodity: string;
	sub_commodity: string;
	prod_A_id: number;
	prod_B_id: number;
	is_weakly_identified: boolean;
}

export interface ProductCatalogResponse {
	products: ProductSummary[];
	substitute_pairs: SubstitutePairSummary[];
	total_products: number;
}

export interface ProductDetailResponse extends ProductSummary {
	min_observed_price: number;
	max_observed_price: number;
	historical_weekly_volume_mean: number;
	historical_weekly_volume_std: number;
	substitute_pair_details: SubstitutePairSummary[];
}

export interface WeeklySalesTrendItem {
	week_no: number;
	total_sales: number;
	promotional_sales: number;
	base_sales: number;
	total_units: number;
	promotional_share: number;
}

export interface DashboardSalesTrendResponse {
	start_week: number;
	end_week: number;
	total_weeks: number;
	trend: WeeklySalesTrendItem[];
}

export interface CampaignSummaryItem {
	campaign_id: number;
	campaign_type: string;
	start_day: number;
	end_day: number;
	duration_days: number;
	households_targeted: number;
	target_rate: number;
}

export interface DashboardPromotionsResponse {
	total_campaigns: number;
	campaign_types: Record<string, number>;
	pooled_ate_spend_lift: number;
	top_decile_lift: number;
	campaigns: CampaignSummaryItem[];
	methodological_note: string;
}

export interface DashboardSummaryResponse {
	total_sales_dollars: number;
	promotional_sales_dollars: number;
	promotional_sales_share: number;
	total_units_sold: number;
	total_transactions: number;
	total_baskets: number;
	total_households: number;
	total_campaigns: number;
	modeled_target_skus: number;
	modeled_product_revenue: number;
	modeled_product_revenue_share: number;
	gross_margin_tiers: number[];
	portfolio_scope_disclaimer: string;
	methodological_warnings: MethodologicalWarnings;
}

export interface CustomerDemographics {
	has_demographics: boolean;
	age_band: string | null;
	marital_status: string | null;
	income_band: string | null;
	homeowner_status: string | null;
	household_comp: string | null;
	household_size: string | null;
	kid_category: string | null;
}

export interface CustomerBehavior {
	historical_spend_dollars: number;
	historical_trips_count: number;
	campaigns_assigned_count: number;
	coupons_redeemed_count: number;
}

export interface CustomerDetailResponse {
	household_key: number;
	demographics: CustomerDemographics;
	behavior: CustomerBehavior;
	methodological_note: string;
}

export interface CustomerUpliftResponse {
	household_key: number;
	evaluated_campaign_id: number;
	uplift_score_cate: number;
	targeting_segment: string;
	recommended_action: string;
	expected_spend_with_treatment: number;
	expected_spend_without_treatment: number;
	cost_threshold_dollars: number;
	is_recommended_for_treatment: boolean;
	model_provenance: string;
	methodological_warnings: MethodologicalWarnings;
}

export interface QiniDecileItem {
	decile: number;
	population_fraction: number;
	n_targeted_cum: number;
	cum_gain: number;
	qini_val: number;
	mean_treated_cum: number;
	mean_control_cum: number;
}

export interface CustomerSegmentsResponse {
	qini_score: number;
	auuc: number;
	overall_cohort_ate: number;
	top_decile_lift: number;
	top_decile_multiplier: number;
	deciles: QiniDecileItem[];
	recommended_targeting_cutoff_decile: number;
	policy_guidance: string;
	methodological_note: string;
}

export interface ElasticityModelDetail {
	model_type: string;
	elasticity: number;
	elasticity_se: number | null;
	elasticity_hac_se: number | null;
	t_stat: number | null;
	p_value: number | null;
	confidence_interval_95_hac: [number, number] | null;
	rmse_units: number | null;
	r2_log: number | null;
	coefficients: Record<string, number> | null;
}

export interface SubstituteElasticityDetail {
	pair_name: string;
	substitute_product_id: number;
	own_elasticity_A: number;
	cross_elasticity_AB: number;
	cross_elasticity_AB_se: number;
	is_weakly_identified: boolean;
	identification_note: string;
}

export interface ProductElasticityResponse {
	product_id: number;
	commodity: string;
	sub_commodity: string;
	brand_type: string;
	champion_specification: string;
	champion_elasticity: number;
	champion_hac_se: number;
	champion_ci_95: [number, number];
	champion_p_value: number;
	interpretation: string;
	competing_models: Record<string, unknown>;
	substitute_pair_elasticity: SubstituteElasticityDetail | null;
	methodological_warnings: MethodologicalWarnings;
}

export interface SimulationRequest {
	product_id: number;
	discount_pct: number;
	with_display: boolean;
	baseline_mode: BaselineMode;
	include_substitute: boolean;
}

export interface SingleSkuSimulationDetails {
	product_id: number;
	avg_shelf_price: number;
	promotional_price: number;
	discount_depth: number;
	units_baseline: number;
	units_promoted: number;
	delta_units: number;
	pct_unit_lift: number;
	revenue_baseline: number;
	revenue_promoted: number;
	delta_revenue: number;
	pct_revenue_lift: number;
	contribution_profits: Record<string, ProfitByMarginTier>;
	profit_uncertainty_hac: {
		uncertainty_type: string;
		uncertainty_disclaimer?: string;
		delta_profit_bound_low: number;
		delta_profit_bound_high: number;
	};
}

export interface SubstitutePairSimulationDetails {
	pair_name: string;
	prod_A_id: number;
	prod_B_id: number;
	is_weakly_identified: boolean;
	delta_units_A: number;
	delta_units_B: number;
	net_category_units: number;
	delta_revenue_A: number;
	delta_revenue_B: number;
	net_category_revenue: number;
	revenue_offset_ratio: number | null;
	profit_impact_isolated: Record<string, unknown>;
	profit_impact_category_modeled: Record<string, unknown>;
	weak_identification_warning: string | null;
}

export interface SimulationResponse {
	scenario: {
		product_id: number;
		discount_pct: number;
		with_display: boolean;
		baseline_mode: string;
		is_unpromoted_baseline_comparison: boolean;
		evaluated_weeks: number;
	};
	single_sku: SingleSkuSimulationDetails;
	substitute_pair: SubstitutePairSimulationDetails | null;
	empirical_support: EmpiricalSupportMetadata;
	methodological_warnings: MethodologicalWarnings;
}

export interface OptimizationRequest {
	product_id: number;
	with_display: boolean;
	baseline_mode: BaselineMode;
	min_discount: number;
	max_discount: number;
	discount_step: number;
	include_substitute: boolean;
}

export interface OptimalPolicyTier {
	assumed_gross_margin: number;
	optimal_discount_pct: number;
	modeled_best_promotional_candidate: number;
	recommend_price_promotion: boolean;
	delta_profit_at_optimum: number;
	modeled_candidate_delta_profit: number;
	pct_profit_lift_at_optimum: number;
	delta_revenue_at_optimum: number;
	delta_units_at_optimum: number;
	promotional_unit_price: number;
	partial_uncertainty_bound: {
		uncertainty_type: string;
		uncertainty_disclaimer?: string;
		delta_profit_bound_low: number;
		delta_profit_bound_high: number;
	} | null;
	recommendation_note: string;
}

export interface GridSweepPoint {
	discount_pct: number;
	delta_units: number;
	delta_revenue: number;
	delta_profits: {
		margin_15pct?: number;
		margin_25pct?: number;
		margin_35pct?: number;
		[key: string]: number | undefined;
	};
}

export interface NoPromotionBaselinePoint {
	discount_pct: number;
	delta_profit: number;
	delta_revenue: number;
	delta_units: number;
}

export interface OptimizationResponse {
	product_id: number;
	with_display: boolean;
	baseline_mode: string;
	search_grid: {
		min_discount: number;
		max_discount: number;
		discount_step: number;
		with_display: number | boolean;
		baseline_mode: string;
		n_grid_points: number;
	};
	no_promotion_baseline: NoPromotionBaselinePoint;
	single_sku_policies: Record<string, OptimalPolicyTier>;
	single_sku_grid_sweep: GridSweepPoint[];
	category_pair_policies: Record<string, unknown> | null;
	category_pair_grid_sweep: Array<Record<string, unknown>> | null;
	empirical_support: EmpiricalSupportMetadata;
	methodological_warnings: MethodologicalWarnings;
}
