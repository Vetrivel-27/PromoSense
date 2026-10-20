<script lang="ts">
	import type {
		BaselineMode,
		ProductSummary,
		SimulationRequest,
		SimulationResponse
	} from '../types';
	import MethodologicalBanners from './MethodologicalBanners.svelte';
	import MetricCard from './MetricCard.svelte';

	interface Props {
		product: ProductSummary;
		simulationResult: SimulationResponse | null;
		isLoading: boolean;
		error: string | null;
		onRunSimulation: (req: SimulationRequest) => void;
	}

	let { product, simulationResult, isLoading, error, onRunSimulation }: Props = $props();

	// Controls state (Svelte 5 runes)
	let discountPct = $state(0.20);
	let withDisplay = $state(false);
	let baselineMode = $state<BaselineMode>('observed_merch');
	let includeSubstitute = $state(true);
	let activeMarginTier = $state<'margin_15pct' | 'margin_25pct' | 'margin_35pct'>('margin_25pct');

	// Allowed discount steps: 0.0 (baseline), or 0.05 to 0.35 in 0.025 increments
	const discountPresets = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35];

	function handlePreset(p: number) {
		discountPct = p;
		triggerSim();
	}

	function handleSliderChange(e: Event) {
		const target = e.target as HTMLInputElement;
		let val = parseFloat(target.value);
		if (val > 0.0 && val < 0.05) {
			val = val < 0.025 ? 0.0 : 0.05;
		}
		discountPct = val;
		triggerSim();
	}

	function triggerSim() {
		onRunSimulation({
			product_id: product.product_id,
			discount_pct: discountPct,
			with_display: withDisplay,
			baseline_mode: baselineMode,
			include_substitute: includeSubstitute
		});
	}

	// Derived values for active margin tier
	const currentProfitDetails = $derived(
		simulationResult?.single_sku?.contribution_profits?.[activeMarginTier]
	);

	const isMarginErosion = $derived(
		simulationResult &&
		currentProfitDetails &&
		currentProfitDetails.delta_profit < 0 &&
		simulationResult.single_sku.delta_units > 0
	);
</script>

<div class="simulation-section">
	<!-- Control Header & Scenario Configuration -->
	<div class="card controls-card">
		<div class="controls-header">
			<div>
				<h2 class="section-title">Promotion Simulation Studio</h2>
				<p class="section-subtitle">
					Simulating counterfactual price and merchandising scenarios for SKU {product.product_id} ({product.commodity}).
				</p>
			</div>
			<button
				type="button"
				class="btn btn-primary"
				disabled={isLoading}
				onclick={triggerSim}
			>
				{#if isLoading}
					<span class="spinner-sm"></span> Simulating...
				{:else}
					▶ Run Simulation
				{/if}
			</button>
		</div>

		<div class="controls-grid">
			<!-- Discount Selector -->
			<div class="control-group discount-group">
				<div class="control-label-row">
					<label for="discount-slider" class="control-label">Promotional Discount Depth:</label>
					<span class="discount-value-pill">
						{discountPct === 0 ? '0.0% (Unpromoted Baseline)' : `${(discountPct * 100).toFixed(1)}% Discount`}
					</span>
				</div>

				<input
					id="discount-slider"
					type="range"
					min="0.0"
					max="0.35"
					step="0.025"
					value={discountPct}
					oninput={handleSliderChange}
					class="range-slider"
				/>

				<div class="preset-buttons">
					{#each discountPresets as p}
						<button
							type="button"
							class="preset-btn {Math.abs(discountPct - p) < 0.001 ? 'active-preset' : ''}"
							onclick={() => handlePreset(p)}
						>
							{p === 0 ? '0% Base' : `${(p * 100).toFixed(0)}%`}
						</button>
					{/each}
				</div>
			</div>

			<!-- Toggles Column -->
			<div class="control-group">
				<span class="control-label">Merchandising & Baseline Settings:</span>
				
				<div class="checkbox-row">
					<label class="toggle-checkbox">
						<input
							type="checkbox"
							checked={withDisplay}
							onchange={(e) => {
								withDisplay = e.currentTarget.checked;
								triggerSim();
							}}
						/>
						<span class="checkbox-label">
							<strong>Co-Display Merchandising (100% Endcap)</strong>
							<span class="sub-label-text">Simulates store feature/display execution</span>
						</span>
					</label>
				</div>

				<div class="checkbox-row">
					<label class="toggle-checkbox">
						<input
							type="checkbox"
							checked={includeSubstitute}
							onchange={(e) => {
								includeSubstitute = e.currentTarget.checked;
								triggerSim();
							}}
						/>
						<span class="checkbox-label">
							<strong>Include Substitute Cross-SKU Cannibalization</strong>
							<span class="sub-label-text">Computes volume & revenue diversion</span>
						</span>
					</label>
				</div>
			</div>

			<!-- Baseline Mode & Gross Margin Selector -->
			<div class="control-group">
				<div class="sub-group">
					<label for="baseline-mode-select" class="control-label">Baseline Counterfactual Mode:</label>
					<select
						id="baseline-mode-select"
						bind:value={baselineMode}
						onchange={triggerSim}
						class="select-input"
					>
						<option value="observed_merch">Observed Status Quo Merchandising</option>
						<option value="zero_merch">Zero Merchandising Reference</option>
					</select>
				</div>

				<div class="sub-group mt-2">
					<span class="control-label">Active Gross Margin Tier Assumption:</span>
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
	</div>

	<!-- Results / Error / Loading States -->
	{#if error}
		<div class="alert alert-danger" role="alert">
			<div class="alert-icon">⚠️</div>
			<div>
				<div class="alert-title">Simulation Execution Error</div>
				<p>{error}</p>
			</div>
		</div>
	{/if}

	{#if simulationResult}
		<!-- Support & Methodological Banners -->
		<MethodologicalBanners
			support={simulationResult.empirical_support}
			warnings={simulationResult.methodological_warnings}
			compact={false}
		/>

		<!-- Margin Erosion Paradox Banner if triggered -->
		{#if isMarginErosion}
			<div class="alert alert-danger paradox-alert" role="alert">
				<div class="alert-icon">⚡</div>
				<div>
					<div class="alert-title">Margin-Erosion Paradox Detected</div>
					<p>
						Although unit volume expands by <strong>+{simulationResult.single_sku.pct_unit_lift.toFixed(1)}%</strong>
						({simulationResult.single_sku.delta_units > 0 ? '+' : ''}{simulationResult.single_sku.delta_units.toFixed(1)} units),
						contribution profit at {activeMarginTier === 'margin_15pct' ? '15%' : activeMarginTier === 'margin_25pct' ? '25%' : '35%'} gross margin declines by
						<strong>${Math.abs(currentProfitDetails?.delta_profit || 0).toFixed(2)} ({currentProfitDetails?.pct_profit_lift.toFixed(1)}%)</strong>.
						The price discount dilutes unit margin faster than volume expansion compensates, destroying net dollar profit.
					</p>
				</div>
			</div>
		{/if}

		<!-- Metric KPI Cards Grid -->
		<div class="grid grid-cols-4">
			<MetricCard
				title="Promoted Shelf Price"
				value={simulationResult.single_sku.promotional_price}
				baseline={simulationResult.single_sku.avg_shelf_price}
				delta={simulationResult.single_sku.promotional_price - simulationResult.single_sku.avg_shelf_price}
				formatType="currency"
				subtitle={`Base List: $${simulationResult.single_sku.avg_shelf_price.toFixed(2)}`}
			/>

			<MetricCard
				title="Unit Volume Sales"
				value={simulationResult.single_sku.units_promoted}
				baseline={simulationResult.single_sku.units_baseline}
				delta={simulationResult.single_sku.delta_units}
				percentLift={simulationResult.single_sku.pct_unit_lift}
				formatType="units"
				subtitle="Over 16-Week Evaluation Horizon"
			/>

			<MetricCard
				title="Gross Revenue"
				value={simulationResult.single_sku.revenue_promoted}
				baseline={simulationResult.single_sku.revenue_baseline}
				delta={simulationResult.single_sku.delta_revenue}
				percentLift={simulationResult.single_sku.pct_revenue_lift}
				formatType="currency"
				subtitle="Aggregate Dollar Volume"
			/>

			{#if currentProfitDetails}
				<MetricCard
					title={`Contribution Profit (${(currentProfitDetails.assumed_gross_margin * 100).toFixed(0)}% Margin)`}
					value={currentProfitDetails.profit_promoted}
					baseline={currentProfitDetails.profit_baseline}
					delta={currentProfitDetails.delta_profit}
					percentLift={currentProfitDetails.pct_profit_lift}
					formatType="currency"
					isProfit={true}
					subtitle={currentProfitDetails.cost_basis_label}
				/>
			{/if}
		</div>

		<!-- HAC Uncertainty Bounds Card -->
		{#if simulationResult.single_sku.profit_uncertainty_hac}
			{@const unc = simulationResult.single_sku.profit_uncertainty_hac}
			<div class="card uncertainty-card">
				<div class="uncertainty-header">
					<span class="uncertainty-title">HAC Parameter Uncertainty Bounds on Profit Lift (ΔΠ)</span>
					<span class="badge badge-neutral">{unc.uncertainty_type}</span>
				</div>
				<div class="uncertainty-body">
					<div class="bound-pill">
						<span class="bound-label">95% Lower HAC Bound:</span>
						<span class="bound-val {unc.delta_profit_bound_low < 0 ? 'text-danger' : 'text-success'}">
							{unc.delta_profit_bound_low < 0 ? '-' : '+'}${Math.abs(unc.delta_profit_bound_low).toFixed(2)}
						</span>
					</div>
					<div class="bound-pill">
						<span class="bound-label">Point Estimate (ΔΠ):</span>
						<span class="bound-val {currentProfitDetails && currentProfitDetails.delta_profit < 0 ? 'text-danger' : 'text-success'}">
							{currentProfitDetails && currentProfitDetails.delta_profit < 0 ? '-' : '+'}${Math.abs(currentProfitDetails?.delta_profit || 0).toFixed(2)}
						</span>
					</div>
					<div class="bound-pill">
						<span class="bound-label">95% Upper HAC Bound:</span>
						<span class="bound-val {unc.delta_profit_bound_high < 0 ? 'text-danger' : 'text-success'}">
							{unc.delta_profit_bound_high < 0 ? '-' : '+'}${Math.abs(unc.delta_profit_bound_high).toFixed(2)}
						</span>
					</div>
				</div>
				<p class="uncertainty-disclaimer">
					* Partial HAC bounds propagate uncertainty from own-price elasticity only. Merchandising parameters and cross-price terms are treated as fixed.
				</p>
			</div>
		{/if}

		<!-- Margin Tiers Sensitivity Table -->
		<div class="card table-card">
			<div class="card-inner-header">
				<h3 class="subsection-title">Profit Sensitivity Across Assumed Gross Margin Tiers</h3>
				<span class="header-note">Accounting cost basis derived as c = P_shelf * (1 - m_base)</span>
			</div>

			<div class="table-container">
				<table class="table">
					<thead>
						<tr>
							<th>Assumed Gross Margin</th>
							<th>Assumed Unit Cost (c)</th>
							<th>Baseline Profit (Π_0)</th>
							<th>Promoted Profit (Π)</th>
							<th>Incremental Profit (ΔΠ)</th>
							<th>Profit Lift (%)</th>
							<th>Outcome</th>
						</tr>
					</thead>
					<tbody>
						{#each Object.entries(simulationResult.single_sku.contribution_profits) as [key, tier]}
							<tr class={activeMarginTier === key ? 'active-tier-row' : ''}>
								<td>
									<strong>{(tier.assumed_gross_margin * 100).toFixed(0)}% Margin</strong>
									{#if activeMarginTier === key}
										<span class="badge badge-primary badge-sm ml-1">Active</span>
									{/if}
								</td>
								<td class="numeric-cell">{tier.cost_basis_label}</td>
								<td class="numeric-cell">${tier.profit_baseline.toFixed(2)}</td>
								<td class="numeric-cell">${tier.profit_promoted.toFixed(2)}</td>
								<td class="numeric-cell {tier.delta_profit > 0 ? 'text-success font-bold' : tier.delta_profit < 0 ? 'text-danger font-bold' : ''}">
									{tier.delta_profit > 0 ? '+' : ''}${tier.delta_profit.toFixed(2)}
								</td>
								<td class="numeric-cell {tier.pct_profit_lift > 0 ? 'text-success' : tier.pct_profit_lift < 0 ? 'text-danger' : ''}">
									{tier.pct_profit_lift > 0 ? '+' : ''}{tier.pct_profit_lift.toFixed(1)}%
								</td>
								<td>
									{#if tier.is_profitable_lift}
										<span class="badge badge-success">Profitable</span>
									{:else if Math.abs(tier.delta_profit) < 0.01}
										<span class="badge badge-neutral">Breakeven</span>
									{:else}
										<span class="badge badge-danger">Margin Erosion</span>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Substitute Pair Cannibalization Details -->
		{#if simulationResult.substitute_pair}
			{@const sub = simulationResult.substitute_pair}
			<div class="card cannibalization-card {sub.is_weakly_identified ? 'weak-cannibalization' : ''}">
				<div class="cannibalization-header">
					<div>
						<div class="sub-pair-tags">
							<span class="pair-title">{sub.pair_name.replace(/_/g, ' ')}</span>
							{#if sub.is_weakly_identified}
								<span class="badge badge-warning">Weak Identification Caution</span>
							{:else}
								<span class="badge badge-success">Estimated Cannibalization</span>
							{/if}
						</div>
						<h3 class="subsection-title">Category Net Impact & Cannibalization Offset</h3>
					</div>

					{#if sub.revenue_offset_ratio !== null}
						<div class="offset-ratio-badge">
							<span class="ratio-label">Cannibalization Revenue Offset:</span>
							<span class="ratio-val">{(sub.revenue_offset_ratio * 100).toFixed(1)}%</span>
						</div>
					{/if}
				</div>

				<div class="grid grid-cols-3 cannibalization-grid">
					<div class="cannibal-box">
						<span class="box-label">Product A Incremental Units:</span>
						<span class="box-val text-success">+{sub.delta_units_A.toFixed(1)}</span>
						<span class="box-sub">Revenue: +${sub.delta_revenue_A.toFixed(2)}</span>
					</div>

					<div class="cannibal-box">
						<span class="box-label">Substitute B Diverted Volume:</span>
						<span class="box-val text-danger">{sub.delta_units_B.toFixed(1)}</span>
						<span class="box-sub">Revenue Loss: -${Math.abs(sub.delta_revenue_B).toFixed(2)}</span>
					</div>

					<div class="cannibal-box">
						<span class="box-label">Net Category Volume Change:</span>
						<span class="box-val {sub.net_category_units > 0 ? 'text-success' : 'text-danger'}">
							{sub.net_category_units > 0 ? '+' : ''}{sub.net_category_units.toFixed(1)}
						</span>
						<span class="box-sub">
							Net Category Revenue: {sub.net_category_revenue > 0 ? '+' : ''}${sub.net_category_revenue.toFixed(2)}
						</span>
					</div>
				</div>

				{#if sub.weak_identification_warning}
					<div class="sub-warning-box">
						<span class="warning-icon">⚠️</span>
						<span>{sub.weak_identification_warning}</span>
					</div>
				{/if}
			</div>
		{/if}
	{/if}
</div>

<style>
	.simulation-section {
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
		grid-template-columns: 1.4fr 1fr 1fr;
		gap: 1.5rem;
		border-top: 1px solid #f1f5f9;
		padding-top: 1.25rem;
	}

	.control-group {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.control-label-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.control-label {
		font-size: 0.75rem;
		font-weight: 700;
		color: var(--color-text-main);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.discount-value-pill {
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: 0.8125rem;
		color: var(--color-primary);
		background-color: #eff6ff;
		padding: 0.15rem 0.5rem;
		border-radius: var(--radius-sm);
	}

	.range-slider {
		width: 100%;
		accent-color: var(--color-primary);
		cursor: pointer;
	}

	.preset-buttons {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		margin-top: 0.25rem;
	}

	.preset-btn {
		background-color: #f8fafc;
		border: 1px solid var(--color-border);
		padding: 0.25rem 0.5rem;
		border-radius: var(--radius-sm);
		font-size: 0.7rem;
		font-weight: 600;
		color: var(--color-text-muted);
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.preset-btn:hover {
		border-color: #cbd5e1;
		color: var(--color-text-main);
	}

	.active-preset {
		background-color: var(--color-primary);
		color: #ffffff;
		border-color: var(--color-primary);
	}

	.checkbox-row {
		display: flex;
		align-items: flex-start;
		margin-bottom: 0.35rem;
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

	.paradox-alert {
		border-left: 4px solid var(--color-danger);
	}

	.uncertainty-card {
		padding: 1rem 1.25rem;
		background-color: #f8fafc;
		border: 1px dashed #cbd5e1;
	}

	.uncertainty-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.75rem;
	}

	.uncertainty-title {
		font-size: 0.8125rem;
		font-weight: 700;
		color: var(--color-text-main);
	}

	.uncertainty-body {
		display: flex;
		gap: 1.5rem;
		margin-bottom: 0.5rem;
		flex-wrap: wrap;
	}

	.bound-pill {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
	}

	.bound-label {
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.bound-val {
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: 0.875rem;
	}

	.uncertainty-disclaimer {
		font-size: 0.7rem;
		color: var(--color-text-subtle);
		font-style: italic;
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

	.active-tier-row {
		background-color: #eff6ff !important;
	}

	.font-bold {
		font-weight: 700;
	}

	.ml-1 {
		margin-left: 0.25rem;
	}

	.mt-2 {
		margin-top: 0.5rem;
	}

	.badge-sm {
		padding: 0.15rem 0.35rem;
		font-size: 0.65rem;
	}

	.cannibalization-card {
		border-left: 4px solid #10b981;
	}

	.weak-cannibalization {
		border-left: 4px solid var(--color-warning);
	}

	.cannibalization-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 1rem;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.sub-pair-tags {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.25rem;
	}

	.pair-title {
		font-size: 0.8125rem;
		font-weight: 700;
		color: var(--color-text-muted);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.offset-ratio-badge {
		background-color: #f1f5f9;
		border: 1px solid #e2e8f0;
		padding: 0.4rem 0.75rem;
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		font-size: 0.75rem;
	}

	.ratio-label {
		color: var(--color-text-muted);
		font-size: 0.7rem;
	}

	.ratio-val {
		font-family: var(--font-mono);
		font-weight: 800;
		color: var(--color-danger);
		font-size: 0.875rem;
	}

	.cannibalization-grid {
		margin-bottom: 1rem;
		gap: 1rem;
	}

	.cannibal-box {
		background-color: #f8fafc;
		border: 1px solid #f1f5f9;
		padding: 0.85rem;
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}

	.box-label {
		font-size: 0.725rem;
		color: var(--color-text-muted);
	}

	.box-val {
		font-family: var(--font-mono);
		font-weight: 800;
		font-size: 1.15rem;
	}

	.box-sub {
		font-size: 0.725rem;
		color: var(--color-text-subtle);
	}

	.sub-warning-box {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.75rem;
		background-color: #fffbeb;
		border: 1px solid #fde68a;
		padding: 0.65rem 0.85rem;
		border-radius: var(--radius-sm);
		color: #92400e;
	}

	.warning-icon {
		font-size: 0.95rem;
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
		.controls-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
