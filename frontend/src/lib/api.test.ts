import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
	ApiError,
	getApiBaseUrl,
	fetchHealth,
	fetchProducts,
	fetchProductElasticity,
	simulatePromotion,
	optimizePromotion
} from './api';

describe('PromoSense Typed API Client', () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it('resolves default API base URL correctly', () => {
		const url = getApiBaseUrl();
		expect(url).toBe('http://localhost:8000/api');
	});

	it('creates ApiError with status and detail', () => {
		const err = new ApiError(404, 'Product SKU not found');
		expect(err.name).toBe('ApiError');
		expect(err.status).toBe(404);
		expect(err.detail).toBe('Product SKU not found');
		expect(err.message).toContain('API Error (404): Product SKU not found');
	});

	it('fetchHealth queries /health endpoint', async () => {
		const mockResponse = {
			status: 'healthy',
			environment: 'production',
			model_registry_loaded: true,
			target_skus_count: 7,
			substitute_pairs_count: 3
		};

		vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
			ok: true,
			json: async () => mockResponse
		} as Response);

		const result = await fetchHealth();
		expect(result.status).toBe('healthy');
		expect(result.model_registry_loaded).toBe(true);
		expect(result.target_skus_count).toBe(7);
	});

	it('fetchProducts retrieves catalog response', async () => {
		const mockCatalog = {
			products: [
				{
					product_id: 5569230,
					commodity: 'SOFT DRINKS',
					sub_commodity: '12pk cans',
					brand_type: 'National',
					department: 'GROCERY',
					baseline_shelf_price: 1.59,
					observed_display_share: 0.70,
					observed_mailer_share: 0.50,
					substitute_pair_names: ['Soft_Drinks_12pk'],
					is_focal_substitute_a: true,
					is_substitute_b: false
				}
			],
			substitute_pairs: [
				{
					pair_name: 'Soft_Drinks_12pk',
					commodity: 'SOFT DRINKS',
					sub_commodity: '12pk cans',
					prod_A_id: 5569230,
					prod_B_id: 844165,
					is_weakly_identified: false
				}
			],
			total_products: 1
		};

		vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
			ok: true,
			json: async () => mockCatalog
		} as Response);

		const result = await fetchProducts();
		expect(result.total_products).toBe(1);
		expect(result.products[0].product_id).toBe(5569230);
	});

	it('fetchProductElasticity retrieves champion elasticity and CI', async () => {
		const mockElasticity = {
			product_id: 5569230,
			commodity: 'SOFT DRINKS',
			sub_commodity: '12pk cans',
			brand_type: 'National',
			champion_specification: 'PaidPriceElasticityOLS',
			champion_elasticity: -2.5346,
			champion_hac_se: 0.3436,
			champion_ci_95: [-3.2081, -1.8611],
			champion_p_value: 0.0001,
			interpretation: 'Highly price-elastic demand',
			competing_models: {},
			substitute_pair_elasticity: null,
			methodological_warnings: {
				cost_basis_assumption: 'Assumed cost basis',
				observational_unconfoundedness_disclaimer: 'Observational data',
				partial_uncertainty_disclaimer: 'Partial HAC bounds',
				cross_price_warning: null,
				non_causal_label: 'Observational Regression Model'
			}
		};

		vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
			ok: true,
			json: async () => mockElasticity
		} as Response);

		const result = await fetchProductElasticity(5569230);
		expect(result.champion_elasticity).toBe(-2.5346);
		expect(result.champion_ci_95).toEqual([-3.2081, -1.8611]);
	});

	it('simulatePromotion sends POST with expected body', async () => {
		const mockSim = {
			scenario: { product_id: 5569230, discount_pct: 0.20 },
			single_sku: {
				product_id: 5569230,
				avg_shelf_price: 1.59,
				promotional_price: 1.272,
				discount_depth: 0.2231,
				units_baseline: 1000,
				units_promoted: 1450,
				delta_units: 450,
				pct_unit_lift: 45.0,
				revenue_baseline: 1590,
				revenue_promoted: 1844.4,
				delta_revenue: 254.4,
				pct_revenue_lift: 16.0,
				contribution_profits: {},
				profit_uncertainty_hac: {
					uncertainty_type: 'hac_own_price_propagated',
					delta_profit_bound_low: -20.0,
					delta_profit_bound_high: 60.0
				}
			},
			substitute_pair: null,
			empirical_support: {
				is_jointly_supported: true,
				discount_supported: true,
				discount_warning: null,
				is_extrapolative_merchandising: false,
				merchandising_warning: null,
				joint_support_warning: null
			},
			methodological_warnings: {
				cost_basis_assumption: 'test',
				observational_unconfoundedness_disclaimer: 'test',
				partial_uncertainty_disclaimer: 'test',
				cross_price_warning: null,
				non_causal_label: 'test'
			}
		};

		let sentBody = '';
		vi.spyOn(globalThis, 'fetch').mockImplementationOnce(async (url, init) => {
			sentBody = (init?.body as string) || '';
			return {
				ok: true,
				json: async () => mockSim
			} as Response;
		});

		const result = await simulatePromotion({
			product_id: 5569230,
			discount_pct: 0.20,
			with_display: false,
			baseline_mode: 'observed_merch',
			include_substitute: false
		});

		expect(JSON.parse(sentBody)).toEqual({
			product_id: 5569230,
			discount_pct: 0.20,
			with_display: false,
			baseline_mode: 'observed_merch',
			include_substitute: false
		});
		expect(result.single_sku.delta_units).toBe(450);
	});

	it('parses structured Pydantic 422 validation errors gracefully', async () => {
		const validationErrorResponse = {
			detail: [
				{
					loc: ['body', 'discount_pct'],
					msg: 'Discount 40.0% exceeds maximum supported promotional ceiling (35.0%)'
				}
			]
		};

		vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
			ok: false,
			status: 422,
			json: async () => validationErrorResponse
		} as Response);

		await expect(
			simulatePromotion({
				product_id: 5569230,
				discount_pct: 0.40,
				with_display: false,
				baseline_mode: 'observed_merch',
				include_substitute: false
			})
		).rejects.toThrowError(
			'API Error (422): body.discount_pct: Discount 40.0% exceeds maximum supported promotional ceiling (35.0%)'
		);
	});

	it('throws connection error when backend is offline (fetch throws)', async () => {
		vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new Error('Failed to fetch'));

		await expect(fetchHealth()).rejects.toThrowError(
			'Failed to connect to PromoSense backend at http://localhost:8000/api/health. Ensure FastAPI backend is running.'
		);
	});
});
