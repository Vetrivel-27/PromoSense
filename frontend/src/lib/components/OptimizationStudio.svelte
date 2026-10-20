<script lang="ts">
	import type {
		BaselineMode,
		OptimizationRequest,
		OptimizationResponse,
		ProductSummary
	} from '../types';
	import MethodologicalBanners from './MethodologicalBanners.svelte';

	interface Props {
		product: ProductSummary;
		optimizationResult: OptimizationResponse | null;
		isLoading: boolean;
		error: string | null;
		onRunOptimization: (req: OptimizationRequest) => void;
	}

	let { product, optimizationResult, isLoading, error, onRunOptimization }: Props = $props();

	// Controls state (Svelte 5 runes)
	let withDisplay = $state(false);
	let baselineMode = $state<BaselineMode>('observed_merch');
	let includeSubstitute = $state(true);
	let activeMarginTier = $state<'margin_15pct' | 'margin_25pct' | 'margin_35pct'>('margin_25pct');

	function triggerOpt() {
		onRunOptimization({
			product_id: product.product_id,
			with_display: withDisplay,
			baseline_mode: baselineMode,
			min_discount: 0.05,
			max_discount: 0.35,
			discount_step: 0.025,
			include_substitute: includeSubstitute
		});
	}

	// Active optimal policy for the selected margin tier
	const currentPolicy = $derived(
		optimizationResult?.single_sku_policies?.[activeMarginTier]
	);

	// Category net policy if available
	const categoryPolicy = $derived(
		(optimizationResult?.category_pair_policies as Record<string, any> | undefined)?.[activeMarginTier]
	);

	const categoryOptimum = $derived(
		categoryPolicy?.category_optimum as Record<string, any> | undefined
	);

	const categoryNetProfit = $derived<number | null>(
		typeof categoryOptimum?.delta_category_net_profit === 'number'
			? categoryOptimum.delta_category_net_profit
			: null
	);

	const bLow = $derived<number | null>(
		typeof currentPolicy?.partial_uncertainty_bound?.delta_profit_bound_low === 'number'
			? currentPolicy.partial_uncertainty_bound.delta_profit_bound_low
			: null
	);
	const bHigh = $derived<number | null>(
		typeof currentPolicy?.partial_uncertainty_bound?.delta_profit_bound_high === 'number'
			? currentPolicy.partial_uncertainty_bound.delta_profit_bound_high
			: null
	);
</script>

<div class="optimization-section">
	<!-- Control Header & Scenario Configuration -->
	<div class="card controls-card">
		<div class="controls-header">
			<div>
				<h2 class="section-title">Promotion Optimization Studio</h2>
				<p class="section-subtitle">
					Discrete grid optimization across 5%–35% promotional candidates (2.5% increments) benchmarked against explicit 0% unpromoted baseline.
				</p>
			</div>
			<button
				type="button"
				class="btn btn-primary"
				disabled={isLoading}
				onclick={triggerOpt}
			>
				{#if isLoading}
					<span class="spinner-sm"></span> Optimizing...
				{:else}
					⚙ Solve Optimal Promotion
				{/if}
			</button>
		</div>

		<div class="controls-grid">
			<!-- Merchandising Toggle -->
			<div class="control-group">
				<span class="control-label">Merchandising Execution Mode:</span>
				<label class="toggle-checkbox">
					<input
						type="checkbox"
						checked={withDisplay}
						onchange={(e) => {
							withDisplay = e.currentTarget.checked;
							triggerOpt();
						}}
					/>
					<span class="checkbox-label">
						<strong>Co-Display Merchandising (100% Endcap)</strong>
						<span class="sub-label-text">Evaluate candidate grid with store display feature</span>
					</span>
				</label>

				<label class="toggle-checkbox mt-1">
					<input
						type="checkbox"
						checked={includeSubstitute}
						onchange={(e) => {
							includeSubstitute = e.currentTarget.checked;
							triggerOpt();
						}}
					/>
					<span class="checkbox-label">
						<strong>Solve Net Category Optimum</strong>
						<span class="sub-label-text">Deduct substitute cannibalization margin erosion</span>
					</span>
				</label>
			</div>

			<!-- Baseline Mode -->
			<div class="control-group">
				<label for="opt-baseline-mode-select" class="control-label">Baseline Reference Mode:</label>
				<select
					id="opt-baseline-mode-select"
					bind:value={baselineMode}
					onchange={triggerOpt}
					class="select-input"
				>
					<option value="observed_merch">Observed Status Quo Merchandising</option>
					<option value="zero_merch">Zero Merchandising Reference</option>
				</select>
				<span class="field-hint">Grid candidates: 5.0% to 35.0% in 2.5% step increments</span>
			</div>

			<!-- Active Margin Tier -->
			<div class="control-group">
				<span class="control-label">Active Gross Margin View:</span>
				<div class="margin-tier-pills">
					<button
						type="button"
						class="tier-pill {activeMarginTier === 'margin_15pct' ? 'active-tier' : ''}"
						onclick={() => (activeMarginTier = 'margin_15pct')}
					>
						15% Margin
					</button>
					<button
						type="button"
						class="tier-pill {activeMarginTier === 'margin_25pct' ? 'active-tier' : ''}"
						onclick={() => (activeMarginTier = 'margin_25pct')}
					>
						25% Margin (Default)
					</button>
					<button
						type="button"
						class="tier-pill {activeMarginTier === 'margin_35pct' ? 'active-tier' : ''}"
						onclick={() => (activeMarginTier = 'margin_35pct')}
					>
						35% Margin
					</button>
				</div>
			</div>
		</div>
	</div>

	<!-- Results / Error / Loading States -->
	{#if error}
		<div class="alert alert-danger" role="alert">
			<div class="alert-icon">⚠️</div>
			<div>
				<div class="alert-title">Optimization Execution Error</div>
				<p>{error}</p>
			</div>
		</div>
	{/if}

	{#if optimizationResult}
		<!-- Support & Methodological Banners -->
		<MethodologicalBanners
			support={optimizationResult.empirical_support}
			warnings={optimizationResult.methodological_warnings}
			compact={false}
		/>

		<!-- Recommended Policy Card -->
		{#if currentPolicy}
			<div class="card policy-card {currentPolicy.recommend_price_promotion ? 'policy-recommend-promo' : 'policy-no-promo'}">
				<div class="policy-header">
					<div>
						<div class="policy-badges">
							<span class="badge {currentPolicy.recommend_price_promotion ? 'badge-success' : 'badge-neutral'}">
								{currentPolicy.recommend_price_promotion ? 'Active Promotion Recommended' : 'No Promotion Recommended'}
							</span>
							<span class="badge badge-primary">
								{(currentPolicy.assumed_gross_margin * 100).toFixed(0)}% Assumed Gross Margin
							</span>
						</div>
						<h3 class="policy-title">
							{#if currentPolicy.recommend_price_promotion}
								Optimal Policy: {(currentPolicy.optimal_discount_pct * 100).toFixed(1)}% Discount Depth (d*)
							{:else}
								Optimal Policy: 0.0% Unpromoted Regular Shelf Price (d* = 0.0%)
							{/if}
						</h3>
					</div>

					<div class="policy-kpi">
						<span class="kpi-label">Projected Incremental Profit (ΔΠ*):</span>
						<span class="kpi-val {currentPolicy.delta_profit_at_optimum > 0 ? 'text-success' : ''}">
							{currentPolicy.delta_profit_at_optimum > 0 ? '+' : ''}${currentPolicy.delta_profit_at_optimum.toFixed(2)}
						</span>
					</div>
				</div>

				<div class="policy-details-grid">
					<div class="detail-box">
						<span class="detail-label">Recommended Shelf Price:</span>
						<span class="detail-val">${currentPolicy.promotional_unit_price.toFixed(2)}</span>
						<span class="detail-sub">
							{currentPolicy.optimal_discount_pct > 0 ? `${(currentPolicy.optimal_discount_pct * 100).toFixed(1)}% below list` : 'Unpromoted list price'}
						</span>
					</div>

					<div class="detail-box">
						<span class="detail-label">Projected Unit Lift (ΔQ):</span>
						<span class="detail-val {currentPolicy.delta_units_at_optimum > 0 ? 'text-success' : ''}">
							{currentPolicy.delta_units_at_optimum > 0 ? '+' : ''}{currentPolicy.delta_units_at_optimum.toFixed(1)}
						</span>
						<span class="detail-sub">Over 16-week horizon</span>
					</div>

					<div class="detail-box">
						<span class="detail-label">Projected Revenue Lift (ΔR):</span>
						<span class="detail-val {currentPolicy.delta_revenue_at_optimum > 0 ? 'text-success' : ''}">
							{currentPolicy.delta_revenue_at_optimum > 0 ? '+' : ''}${currentPolicy.delta_revenue_at_optimum.toFixed(2)}
						</span>
						<span class="detail-sub">Gross dollar sales</span>
					</div>

					<div class="detail-box">
						<span class="detail-label">Best Active Promo Candidate:</span>
						<span class="detail-val font-mono">
							{(currentPolicy.modeled_best_promotional_candidate * 100).toFixed(1)}%
						</span>
						<span class="detail-sub {currentPolicy.modeled_candidate_delta_profit < 0 ? 'text-danger' : 'text-success'}">
							ΔΠ: {currentPolicy.modeled_candidate_delta_profit > 0 ? '+' : ''}${currentPolicy.modeled_candidate_delta_profit.toFixed(2)}
						</span>
					</div>
				</div>

				<!-- Analytical Distinction Note -->
				<div class="policy-note-box">
					<div class="note-header">
						<span class="note-icon">📋</span>
						<strong>Policy Guidance & Rationale:</strong>
					</div>
					<p class="note-text">{currentPolicy.recommendation_note}</p>
					{#if !currentPolicy.recommend_price_promotion}
						<p class="note-subtext">
							* Important: All active promotional candidates (5%–35%) generate negative profit lift compared to the 0% unpromoted baseline. The highest-ranked promotional candidate evaluated was {(currentPolicy.modeled_best_promotional_candidate * 100).toFixed(1)}%, which incurs a profit penalty of ${Math.abs(currentPolicy.modeled_candidate_delta_profit).toFixed(2)}. Regular shelf price is mathematically superior.
						</p>
					{/if}
				</div>

				<!-- HAC Uncertainty Bounds if present -->
				{#if currentPolicy.partial_uncertainty_bound && bLow !== null && bHigh !== null}
					<div class="uncertainty-bar">
						<span class="unc-title">Partial HAC Profit Uncertainty Bounds at Optimum:</span>
						<span class="unc-range font-mono">
							[{bLow < 0 ? '-' : '+'}${Math.abs(bLow).toFixed(2)},
							{bHigh < 0 ? '-' : '+'}${Math.abs(bHigh).toFixed(2)}]
						</span>
						<span class="unc-sub">({currentPolicy.partial_uncertainty_bound.uncertainty_type})</span>
					</div>
				{/if}
			</div>
		{/if}

		<!-- Multi-Margin Policy Comparison Table -->
		<div class="card table-card">
			<div class="card-inner-header">
				<h3 class="subsection-title">Optimal Promotion Across Gross Margin Tiers</h3>
				<span class="header-note">Comparing optimal policy shift as baseline profit buffer changes</span>
			</div>

			<div class="table-container">
				<table class="table">
					<thead>
						<tr>
							<th>Assumed Gross Margin</th>
							<th>Optimal Policy (d*)</th>
							<th>Recommended Unit Price</th>
							<th>Incremental Profit (ΔΠ*)</th>
							<th>Profit Lift (%)</th>
							<th>Best Active Promo Candidate</th>
							<th>Active Candidate ΔΠ</th>
							<th>Action</th>
						</tr>
					</thead>
					<tbody>
						{#each Object.entries(optimizationResult.single_sku_policies) as [key, tier]}
							<tr class={activeMarginTier === key ? 'active-tier-row' : ''}>
								<td>
									<strong>{(tier.assumed_gross_margin * 100).toFixed(0)}% Margin</strong>
									{#if activeMarginTier === key}
										<span class="badge badge-primary badge-sm ml-1">Active</span>
									{/if}
								</td>
								<td>
									{#if tier.recommend_price_promotion}
										<span class="badge badge-success">Promote @ {(tier.optimal_discount_pct * 100).toFixed(1)}%</span>
									{:else}
										<span class="badge badge-neutral">No Promotion (0.0%)</span>
									{/if}
								</td>
								<td class="numeric-cell">${tier.promotional_unit_price.toFixed(2)}</td>
								<td class="numeric-cell {tier.delta_profit_at_optimum > 0 ? 'text-success font-bold' : ''}">
									{tier.delta_profit_at_optimum > 0 ? '+' : ''}${tier.delta_profit_at_optimum.toFixed(2)}
								</td>
								<td class="numeric-cell {tier.pct_profit_lift_at_optimum > 0 ? 'text-success' : ''}">
									{tier.pct_profit_lift_at_optimum > 0 ? '+' : ''}{tier.pct_profit_lift_at_optimum.toFixed(1)}%
								</td>
								<td class="numeric-cell">{(tier.modeled_best_promotional_candidate * 100).toFixed(1)}%</td>
								<td class="numeric-cell {tier.modeled_candidate_delta_profit > 0 ? 'text-success' : 'text-danger'}">
									{tier.modeled_candidate_delta_profit > 0 ? '+' : ''}${tier.modeled_candidate_delta_profit.toFixed(2)}
								</td>
								<td>
									<button
										type="button"
										class="btn btn-secondary btn-sm"
										onclick={() => (activeMarginTier = key as any)}
									>
										View
									</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Category Pair Net Optimum (if available) -->
		{#if categoryPolicy && categoryOptimum}
			<div class="card category-opt-card">
				<div class="category-opt-header">
					<div>
						<span class="badge badge-primary">Category Net Optimization</span>
						<h3 class="subsection-title">Single SKU Optimum vs Category Net Optimum</h3>
					</div>
				</div>

				<div class="comparison-grid">
					<div class="compare-col">
						<span class="col-title">Single SKU in Isolation:</span>
						<div class="compare-row">
							<span>Optimal Discount:</span>
							<strong>{currentPolicy && typeof currentPolicy.optimal_discount_pct === 'number' ? `${(currentPolicy.optimal_discount_pct * 100).toFixed(1)}%` : '—'}</strong>
						</div>
						<div class="compare-row">
							<span>Incremental Profit:</span>
							{#if currentPolicy && typeof currentPolicy.delta_profit_at_optimum === 'number'}
								<strong class={currentPolicy.delta_profit_at_optimum > 0 ? 'text-success' : ''}>
									{currentPolicy.delta_profit_at_optimum > 0 ? '+' : ''}${currentPolicy.delta_profit_at_optimum.toFixed(2)}
								</strong>
							{:else}
								<strong>—</strong>
							{/if}
						</div>
					</div>

					<div class="compare-col compare-col-highlight">
						<span class="col-title">Category Net (Post-Cannibalization):</span>
						<div class="compare-row">
							<span>Optimal Discount:</span>
							<strong>{typeof categoryOptimum.optimal_discount_pct === 'number' ? `${(categoryOptimum.optimal_discount_pct * 100).toFixed(1)}%` : '—'}</strong>
						</div>
						<div class="compare-row">
							<span>Category Net Profit:</span>
							{#if categoryNetProfit !== null}
								<strong class={categoryNetProfit > 0 ? 'text-success' : categoryNetProfit < 0 ? 'text-danger' : ''}>
									{categoryNetProfit > 0 ? '+' : ''}${categoryNetProfit.toFixed(2)}
								</strong>
							{:else}
								<strong>—</strong>
							{/if}
						</div>
					</div>
				</div>

				<p class="category-opt-note">
					{#if typeof categoryOptimum.optimal_discount_pct === 'number' && categoryOptimum.optimal_discount_pct > 0}
						Promoting Product A at {(categoryOptimum.optimal_discount_pct * 100).toFixed(1)}% remains net profitable ({categoryNetProfit !== null ? `${categoryNetProfit > 0 ? '+' : ''}$${categoryNetProfit.toFixed(2)}` : '—'}) for the category after absorbing substitute diversion.
					{:else if typeof categoryOptimum.optimal_discount_pct === 'number'}
						Cannibalization offset reinforces the recommendation to maintain unpromoted regular shelf price (0% discount).
						Active promotional candidates dilute category contribution profit{#if typeof categoryOptimum.modeled_best_promotional_candidate === 'number'} (best candidate {(categoryOptimum.modeled_best_promotional_candidate * 100).toFixed(1)}% incurs net category impact of {#if typeof categoryOptimum.modeled_candidate_category_net_profit === 'number'}${categoryOptimum.modeled_candidate_category_net_profit.toFixed(2)}{:else}—{/if}){/if}.
					{:else}
						Category substitute policy data unavailable.
					{/if}
				</p>
			</div>
		{/if}

		<!-- Candidate Grid Sweep Curve Table -->
		<div class="card table-card">
			<div class="card-inner-header">
				<div>
					<h3 class="subsection-title">Full Candidate Grid Sweep Curve (5.0%–35.0%)</h3>
					<span class="header-note">Benchmarked against explicit 0.0% unpromoted baseline anchor</span>
				</div>
			</div>

			<div class="table-container">
				<table class="table">
					<thead>
						<tr>
							<th>Candidate Discount (d)</th>
							<th>Incremental Units (ΔQ)</th>
							<th>Incremental Revenue (ΔR)</th>
							<th>ΔΠ @ 15% Margin</th>
							<th>ΔΠ @ 25% Margin</th>
							<th>ΔΠ @ 35% Margin</th>
						</tr>
					</thead>
					<tbody>
						<!-- Baseline Row (0.0%) -->
						<tr class="baseline-sweep-row">
							<td>
								<strong>0.0% (Unpromoted Baseline)</strong>
							</td>
							<td class="numeric-cell">0.0</td>
							<td class="numeric-cell">$0.00</td>
							<td class="numeric-cell">$0.00</td>
							<td class="numeric-cell">$0.00</td>
							<td class="numeric-cell">$0.00</td>
						</tr>

						{#each optimizationResult.single_sku_grid_sweep as pt}
							{@const isOptimalForActive = currentPolicy && Math.abs(currentPolicy.optimal_discount_pct - pt.discount_pct) < 0.001}
							<tr class={isOptimalForActive ? 'optimal-grid-row' : ''}>
								<td>
									<strong>{(pt.discount_pct * 100).toFixed(1)}%</strong>
									{#if isOptimalForActive}
										<span class="badge badge-success badge-sm ml-1">Optimum (d*)</span>
									{/if}
								</td>
								<td class="numeric-cell {pt.delta_units > 0 ? 'text-success' : ''}">
									{pt.delta_units > 0 ? '+' : ''}{pt.delta_units.toFixed(1)}
								</td>
								<td class="numeric-cell {pt.delta_revenue > 0 ? 'text-success' : ''}">
									{pt.delta_revenue > 0 ? '+' : ''}${pt.delta_revenue.toFixed(2)}
								</td>
								<td class="numeric-cell {typeof pt.delta_profits.margin_15pct === 'number' && pt.delta_profits.margin_15pct > 0 ? 'text-success' : 'text-danger'}">
									{typeof pt.delta_profits.margin_15pct === 'number' && pt.delta_profits.margin_15pct > 0 ? '+' : ''}${pt.delta_profits.margin_15pct?.toFixed(2) ?? '—'}
								</td>
								<td class="numeric-cell {typeof pt.delta_profits.margin_25pct === 'number' && pt.delta_profits.margin_25pct > 0 ? 'text-success' : 'text-danger'}">
									{typeof pt.delta_profits.margin_25pct === 'number' && pt.delta_profits.margin_25pct > 0 ? '+' : ''}${pt.delta_profits.margin_25pct?.toFixed(2) ?? '—'}
								</td>
								<td class="numeric-cell {typeof pt.delta_profits.margin_35pct === 'number' && pt.delta_profits.margin_35pct > 0 ? 'text-success' : 'text-danger'}">
									{typeof pt.delta_profits.margin_35pct === 'number' && pt.delta_profits.margin_35pct > 0 ? '+' : ''}${pt.delta_profits.margin_35pct?.toFixed(2) ?? '—'}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	{/if}
</div>

<style>
	.optimization-section {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.controls-card {
		background-color: var(--color-surface);
	}

	.controls-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 1.25rem;
		flex-wrap: wrap;
		gap: 0.75rem;
	}

	.section-title {
		font-size: 1.25rem;
		font-weight: 800;
		color: var(--color-text-main);
	}

	.section-subtitle {
		font-size: 0.8125rem;
		color: var(--color-text-muted);
	}

	.controls-grid {
		display: grid;
		grid-template-columns: 1.2fr 1fr 1fr;
		gap: 1.5rem;
		border-top: 1px solid #f1f5f9;
		padding-top: 1.25rem;
	}

	.control-group {
		display: flex;
		flex-direction: column;
		gap: 0.65rem;
	}

	.control-label {
		font-size: 0.75rem;
		font-weight: 700;
		color: var(--color-text-main);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.toggle-checkbox {
		display: flex;
		align-items: flex-start;
		gap: 0.6rem;
		cursor: pointer;
	}

	.checkbox-label {
		display: flex;
		flex-direction: column;
		font-size: 0.8125rem;
		color: var(--color-text-main);
		line-height: 1.3;
	}

	.sub-label-text {
		font-size: 0.7rem;
		color: var(--color-text-subtle);
	}

	.select-input {
		width: 100%;
		padding: 0.45rem 0.65rem;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		font-size: 0.8125rem;
		background-color: #ffffff;
		color: var(--color-text-main);
	}

	.field-hint {
		font-size: 0.7rem;
		color: var(--color-text-subtle);
	}

	.margin-tier-pills {
		display: flex;
		gap: 0.35rem;
		margin-top: 0.25rem;
	}

	.tier-pill {
		flex: 1;
		padding: 0.4rem 0.4rem;
		background-color: #f8fafc;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		font-size: 0.725rem;
		font-weight: 600;
		color: var(--color-text-muted);
		cursor: pointer;
		text-align: center;
		transition: all 0.15s ease;
	}

	.tier-pill:hover {
		border-color: #cbd5e1;
		color: var(--color-text-main);
	}

	.active-tier {
		background-color: var(--color-primary);
		color: #ffffff;
		border-color: var(--color-primary);
	}

	.policy-card {
		padding: 1.25rem 1.5rem;
		border-radius: var(--radius-md);
	}

	.policy-recommend-promo {
		border: 1px solid #a7f3d0;
		background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
		border-left: 4px solid var(--color-success);
	}

	.policy-no-promo {
		border: 1px solid #cbd5e1;
		background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
		border-left: 4px solid var(--color-neutral);
	}

	.policy-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 1.25rem;
		flex-wrap: wrap;
		gap: 1rem;
	}

	.policy-badges {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.35rem;
	}

	.policy-title {
		font-size: 1.2rem;
		font-weight: 800;
		color: var(--color-text-main);
	}

	.policy-kpi {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		background-color: #ffffff;
		padding: 0.5rem 0.85rem;
		border-radius: var(--radius-sm);
		border: 1px solid var(--color-border);
	}

	.kpi-label {
		font-size: 0.7rem;
		color: var(--color-text-muted);
	}

	.kpi-val {
		font-family: var(--font-mono);
		font-weight: 800;
		font-size: 1.35rem;
		color: var(--color-text-main);
	}

	.policy-details-grid {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 1rem;
		margin-bottom: 1.25rem;
	}

	.detail-box {
		background-color: #ffffff;
		border: 1px solid var(--color-border);
		padding: 0.85rem;
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}

	.detail-label {
		font-size: 0.725rem;
		color: var(--color-text-muted);
	}

	.detail-val {
		font-family: var(--font-mono);
		font-weight: 800;
		font-size: 1.15rem;
		color: var(--color-text-main);
	}

	.detail-sub {
		font-size: 0.7rem;
		color: var(--color-text-subtle);
	}

	.policy-note-box {
		background-color: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: var(--radius-sm);
		padding: 0.85rem 1rem;
		font-size: 0.8125rem;
		line-height: 1.45;
		margin-bottom: 0.75rem;
	}

	.note-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		color: var(--color-text-main);
		margin-bottom: 0.25rem;
	}

	.note-icon {
		font-size: 0.95rem;
	}

	.note-text {
		color: var(--color-text-main);
	}

	.note-subtext {
		color: var(--color-danger);
		font-size: 0.75rem;
		margin-top: 0.35rem;
		font-weight: 500;
	}

	.uncertainty-bar {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		font-size: 0.75rem;
		padding: 0.45rem 0.75rem;
		background-color: #f8fafc;
		border-radius: var(--radius-sm);
		border: 1px dashed #cbd5e1;
	}

	.unc-title {
		color: var(--color-text-muted);
	}

	.unc-range {
		font-weight: 700;
		color: var(--color-primary);
	}

	.unc-sub {
		color: var(--color-text-subtle);
		font-size: 0.7rem;
	}

	.category-opt-card {
		border-left: 4px solid var(--color-primary);
	}

	.comparison-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
		margin: 1rem 0 0.75rem;
	}

	.compare-col {
		background-color: #f8fafc;
		border: 1px solid #f1f5f9;
		padding: 0.85rem;
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		font-size: 0.8125rem;
	}

	.compare-col-highlight {
		border-color: #bfdbfe;
		background-color: #eff6ff;
	}

	.col-title {
		font-size: 0.75rem;
		font-weight: 700;
		color: var(--color-text-muted);
		text-transform: uppercase;
	}

	.compare-row {
		display: flex;
		justify-content: space-between;
	}

	.category-opt-note {
		font-size: 0.75rem;
		color: var(--color-text-muted);
		line-height: 1.4;
	}

	.card-inner-header {
		padding: 1.15rem 1.25rem 0.75rem;
		display: flex;
		justify-content: space-between;
		align-items: center;
		border-bottom: 1px solid var(--color-border);
	}

	.subsection-title {
		font-size: 0.95rem;
		font-weight: 700;
		color: var(--color-text-main);
	}

	.header-note {
		font-size: 0.75rem;
		color: var(--color-text-subtle);
	}

	.table-card {
		padding: 0;
		overflow: hidden;
	}

	.numeric-cell {
		font-family: var(--font-mono);
		font-size: 0.8125rem;
	}

	.baseline-sweep-row {
		background-color: #f8fafc !important;
		font-weight: 600;
	}

	.optimal-grid-row {
		background-color: #f0fdf4 !important;
		font-weight: 700;
	}

	.active-tier-row {
		background-color: #eff6ff !important;
	}

	.font-bold {
		font-weight: 700;
	}

	.font-mono {
		font-family: var(--font-mono);
	}

	.ml-1 {
		margin-left: 0.25rem;
	}

	.mt-1 {
		margin-top: 0.35rem;
	}

	.badge-sm {
		padding: 0.15rem 0.35rem;
		font-size: 0.65rem;
	}

	.btn-sm {
		padding: 0.35rem 0.75rem;
		font-size: 0.75rem;
	}

	.spinner-sm {
		display: inline-block;
		width: 12px;
		height: 12px;
		border: 2px solid #ffffff;
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	@media (max-width: 900px) {
		.controls-grid,
		.policy-details-grid,
		.comparison-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
