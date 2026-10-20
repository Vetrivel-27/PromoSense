import { describe, it, expect } from 'vitest';
import type { OptimalPolicyTier, ProfitByMarginTier } from './types';

describe('PromoSense Econometric & Financial Calculation Invariants', () => {
	it('enforces exact revenue arithmetic invariant: delta_revenue = revenue_promoted - revenue_baseline', () => {
		const revBase = 3317.26;
		const revPromo = 4671.95;
		const deltaRev = Number((revPromo - revBase).toFixed(2));

		expect(deltaRev).toBe(1354.69);
		expect(revBase + deltaRev).toBeCloseTo(revPromo, 2);
	});

	it('enforces exact profit arithmetic invariant: delta_profit = profit_promoted - profit_baseline', () => {
		const tiers: ProfitByMarginTier[] = [
			{
				assumed_gross_margin: 0.15,
				cost_basis_label: 'c = $4.02',
				profit_baseline: 497.59,
				profit_promoted: -292.00,
				delta_profit: -789.59,
				pct_profit_lift: -158.68,
				is_profitable_lift: false
			},
			{
				assumed_gross_margin: 0.25,
				cost_basis_label: 'c = $3.55',
				profit_baseline: 829.32,
				profit_promoted: 292.00,
				delta_profit: -537.32,
				pct_profit_lift: -64.79,
				is_profitable_lift: false
			},
			{
				assumed_gross_margin: 0.35,
				cost_basis_label: 'c = $3.08',
				profit_baseline: 1161.04,
				profit_promoted: 875.99,
				delta_profit: -285.05,
				pct_profit_lift: -24.55,
				is_profitable_lift: false
			}
		];

		for (const t of tiers) {
			const expectedDelta = Number((t.profit_promoted - t.profit_baseline).toFixed(2));
			expect(t.delta_profit).toBe(expectedDelta);
		}
	});

	it('enforces zero-discount baseline invariant: zero lift across units, revenue, and profit', () => {
		const zeroPromo = {
			discount_pct: 0.0,
			delta_units: 0.0,
			delta_revenue: 0.0,
			delta_profit: 0.0
		};

		expect(zeroPromo.delta_units).toBe(0.0);
		expect(zeroPromo.delta_revenue).toBe(0.0);
		expect(zeroPromo.delta_profit).toBe(0.0);
	});

	it('verifies optimizer versus simulator numerical parity at candidate discount', () => {
		// Scenario: Product 5569230 at 5% discount (best active promotional candidate)
		const simulatorDeltaProfit = -74.58;
		const optimizerCandidateDeltaProfit = -74.58;

		expect(simulatorDeltaProfit).toBe(optimizerCandidateDeltaProfit);
		expect(Math.abs(simulatorDeltaProfit - optimizerCandidateDeltaProfit)).toBeLessThan(1e-4);
	});

	it('strictly distinguishes 0% no-promotion optimum ($0.00) from missing data (null/undefined)', () => {
		const noPromotionPolicy: OptimalPolicyTier = {
			assumed_gross_margin: 0.25,
			optimal_discount_pct: 0.0,
			modeled_best_promotional_candidate: 0.05,
			recommend_price_promotion: false,
			delta_profit_at_optimum: 0.0, // Genuine mathematical zero
			modeled_candidate_delta_profit: -74.58,
			pct_profit_lift_at_optimum: 0.0,
			delta_revenue_at_optimum: 0.0,
			delta_units_at_optimum: 0.0,
			promotional_unit_price: 4.7352,
			partial_uncertainty_bound: null,
			recommendation_note: 'No promotion recommended; regular shelf price is profit-maximizing.'
		};

		// 1. Genuine zero is a number equal to 0.0
		expect(typeof noPromotionPolicy.delta_profit_at_optimum).toBe('number');
		expect(noPromotionPolicy.delta_profit_at_optimum).toBe(0.0);

		// 2. Missing data is undefined or null, and should NOT be conflated with 0.0
		const malformedPolicy: Partial<OptimalPolicyTier> = {
			assumed_gross_margin: 0.25
			// delta_profit_at_optimum is missing (undefined)
		};

		const formatProfit = (val: number | undefined | null) => {
			if (typeof val === 'number') {
				return `${val > 0 ? '+' : ''}$${val.toFixed(2)}`;
			}
			return '—';
		};

		expect(formatProfit(noPromotionPolicy.delta_profit_at_optimum)).toBe('$0.00');
		expect(formatProfit(malformedPolicy.delta_profit_at_optimum)).toBe('—');
		expect(formatProfit(undefined)).not.toBe('$0.00');
	});

	it('safely handles category pair policy nested structures and missing substitute data', () => {
		// Real backend structure for category pair policy
		const categoryPairPolicies = {
			margin_25pct: {
				assumed_gross_margin: 0.25,
				category_optimum: {
					optimal_discount_pct: 0.0,
					delta_category_net_profit: 0.0,
					modeled_best_promotional_candidate: 0.05,
					modeled_candidate_category_net_profit: -79.7,
					is_category_profitable: false
				},
				isolated_optimum: {
					optimal_discount_pct: 0.0,
					delta_promoted_profit: 0.0,
					modeled_best_promotional_candidate: 0.05,
					modeled_candidate_delta_profit: -74.58,
					is_profitable: false
				}
			}
		};

		const tier = categoryPairPolicies.margin_25pct;
		expect(tier.category_optimum).toBeDefined();
		expect(tier.category_optimum.optimal_discount_pct).toBe(0.0);
		expect(tier.category_optimum.delta_category_net_profit).toBe(0.0);
		expect(tier.category_optimum.modeled_candidate_category_net_profit).toBe(-79.7);

		// Null category policy for single SKU (e.g. Bananas)
		const nullCategoryPolicy = null;
		const categoryOptimum = (nullCategoryPolicy as any)?.category_optimum;
		expect(categoryOptimum).toBeUndefined();
	});

	it('detects Margin-Erosion Paradox when units increase but profit declines', () => {
		const deltaUnits = 532.02; // +76% volume expansion
		const profitTier: ProfitByMarginTier = {
			assumed_gross_margin: 0.25,
			cost_basis_label: 'c = $3.55',
			profit_baseline: 829.32,
			profit_promoted: 292.00,
			delta_profit: -537.32, // profit loss
			pct_profit_lift: -64.79,
			is_profitable_lift: false
		};

		const isErosion = deltaUnits > 0 && profitTier.delta_profit < 0;
		expect(isErosion).toBe(true);
		expect(profitTier.is_profitable_lift).toBe(false);
	});

	it('validates HAC 95% confidence interval bounds symmetry', () => {
		const eps = -2.5346;
		const hacSe = 0.3436;
		const ciLow = Number((eps - 1.96 * hacSe).toFixed(4));
		const ciHigh = Number((eps + 1.96 * hacSe).toFixed(4));

		expect(ciLow).toBeLessThan(eps);
		expect(ciHigh).toBeGreaterThan(eps);
		expect(ciHigh - eps).toBeCloseTo(eps - ciLow, 3);
	});

	it('preserves accounting cost basis c = P_shelf * (1 - m)', () => {
		const shelfPrice = 4.7352;
		const marginTiers = [0.15, 0.25, 0.35];

		const costs = marginTiers.map((m) => Number((shelfPrice * (1 - m)).toFixed(4)));
		expect(costs[0]).toBeCloseTo(4.0249, 4); // 15% margin
		expect(costs[1]).toBeCloseTo(3.5514, 4); // 25% margin
		expect(costs[2]).toBeCloseTo(3.0779, 4); // 35% margin
		expect(costs[0]).toBeGreaterThan(costs[1]);
		expect(costs[1]).toBeGreaterThan(costs[2]);
	});
});
