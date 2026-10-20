<script lang="ts">
	import type {
		CustomerDetailResponse,
		CustomerSegmentsResponse,
		CustomerUpliftResponse
	} from '../types';
	import MetricCard from './MetricCard.svelte';
	import MethodologicalBanners from './MethodologicalBanners.svelte';

	interface Props {
		customer: CustomerDetailResponse | null;
		uplift: CustomerUpliftResponse | null;
		segments: CustomerSegmentsResponse | null;
		loadingCustomer: boolean;
		loadingUplift: boolean;
		loadingSegments: boolean;
		customerError: string | null;
		upliftError: string | null;
		segmentsError: string | null;
		onHouseholdChange: (householdKey: number) => void;
	}

	let {
		customer,
		uplift,
		segments,
		loadingCustomer,
		loadingUplift,
		loadingSegments,
		customerError,
		upliftError,
		segmentsError,
		onHouseholdChange
	}: Props = $props();

	let inputHouseholdKey = $state<number>(1);

	function handleLookupSubmit(e: Event) {
		e.preventDefault();
		if (inputHouseholdKey >= 1 && inputHouseholdKey <= 2500) {
			onHouseholdChange(inputHouseholdKey);
		}
	}

	function selectQuickHousehold(hid: number) {
		inputHouseholdKey = hid;
		onHouseholdChange(hid);
	}

	function formatCurrency(val: number): string {
		return `$${val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
	}
</script>

<div class="customer-view">
	<!-- Header -->
	<div class="customer-header">
		<div>
			<h2 class="section-title">Customer Intelligence & CATE Uplift Targeting</h2>
			<p class="section-subtitle">
				Household-level causal uplift scoring, demographic profiles, and decision-theoretic direct-mail targeting policies.
			</p>
		</div>
		<div class="badge-group">
			<span class="badge badge-info">2,500 Households</span>
			<span class="badge badge-success">S-Learner (LightGBM)</span>
			<span class="badge badge-neutral">Qini: 30,669.17</span>
		</div>
	</div>

	<!-- Household Lookup Bar -->
	<div class="card lookup-bar">
		<form class="lookup-form" onsubmit={handleLookupSubmit}>
			<label for="hh-input" class="lookup-label">Lookup Household ID:</label>
			<input
				id="hh-input"
				type="number"
				min="1"
				max="2500"
				bind:value={inputHouseholdKey}
				class="input-number"
			/>
			<button type="submit" class="btn btn-primary" disabled={loadingCustomer || loadingUplift}>
				Fetch Intelligence
			</button>
		</form>

		<div class="quick-picks">
			<span class="quick-label">Sample Households:</span>
			<button
				type="button"
				class="btn-chip {inputHouseholdKey === 1 ? 'chip-active' : ''}"
				onclick={() => selectQuickHousehold(1)}>HH #1 (Demographics + High Spend)</button
			>
			<button
				type="button"
				class="btn-chip {inputHouseholdKey === 2 ? 'chip-active' : ''}"
				onclick={() => selectQuickHousehold(2)}>HH #2 (No Demographics)</button
			>
			<button
				type="button"
				class="btn-chip {inputHouseholdKey === 10 ? 'chip-active' : ''}"
				onclick={() => selectQuickHousehold(10)}>HH #10</button
			>
			<button
				type="button"
				class="btn-chip {inputHouseholdKey === 42 ? 'chip-active' : ''}"
				onclick={() => selectQuickHousehold(42)}>HH #42</button
			>
		</div>
	</div>

	{#if customerError}
		<div class="alert alert-danger" role="alert">
			<div class="alert-icon">⚠️</div>
			<div>
				<div class="alert-title">Customer Lookup Error</div>
				<p>{customerError}</p>
			</div>
		</div>
	{/if}

	<!-- Customer Profile & Behavioral Grid -->
	{#if loadingCustomer}
		<div class="state-card loading-state">
			<div class="spinner"></div>
			<h3>Loading Household Profile...</h3>
		</div>
	{:else if customer}
		<div class="profile-grid">
			<!-- Demographic Card -->
			<div class="card demographic-card">
				<div class="card-header">
					<h3 class="card-title">Household Demographics</h3>
					{#if customer.demographics.has_demographics}
						<span class="badge badge-success">Census Disclosed</span>
					{:else}
						<span class="badge badge-neutral">Undisclosed</span>
					{/if}
				</div>

				{#if customer.demographics.has_demographics}
					<dl class="meta-dl">
						<dt>Age Band:</dt>
						<dd>{customer.demographics.age_band ?? '—'}</dd>
						<dt>Marital Status:</dt>
						<dd>
							{customer.demographics.marital_status === 'A'
								? 'Married'
								: customer.demographics.marital_status === 'B'
									? 'Single'
									: 'Unknown'}
						</dd>
						<dt>Income Tier:</dt>
						<dd>{customer.demographics.income_band ?? '—'}</dd>
						<dt>Homeowner Status:</dt>
						<dd>{customer.demographics.homeowner_status ?? '—'}</dd>
						<dt>Household Size:</dt>
						<dd>{customer.demographics.household_size ?? '—'}</dd>
						<dt>Children:</dt>
						<dd>{customer.demographics.kid_category ?? '—'}</dd>
					</dl>
				{:else}
					<div class="empty-demo-state">
						<p>
							Voluntary demographic survey not disclosed for Household #{customer.household_key}.
						</p>
						<span class="text-muted"
							>(32% of Dunnhumby households disclosed census survey attributes; remaining 68% are modeled via 100% complete transaction histories).</span
						>
					</div>
				{/if}
			</div>

			<!-- Shopping Behavior Card -->
			<div class="card behavior-card">
				<div class="card-header">
					<h3 class="card-title">102-Week Shopping History</h3>
					<span class="badge badge-info">100% Observed</span>
				</div>

				<div class="behavior-metrics">
					<div class="b-metric">
						<span class="b-label">Total Spend</span>
						<span class="b-val">{formatCurrency(customer.behavior.historical_spend_dollars)}</span>
					</div>
					<div class="b-metric">
						<span class="b-label">Store Visits</span>
						<span class="b-val">{customer.behavior.historical_trips_count} checkout baskets</span>
					</div>
					<div class="b-metric">
						<span class="b-label">Campaigns Assigned</span>
						<span class="b-val">{customer.behavior.campaigns_assigned_count} mailers</span>
					</div>
					<div class="b-metric">
						<span class="b-label">Coupons Redeemed</span>
						<span class="b-val">{customer.behavior.coupons_redeemed_count} in-store</span>
					</div>
				</div>

				<p class="card-footer-note">{customer.methodological_note}</p>
			</div>
		</div>
	{/if}

	<!-- CATE Uplift Targeting Recommendation Card -->
	{#if loadingUplift}
		<div class="state-card loading-state">
			<div class="spinner"></div>
			<h3>Evaluating Causal Uplift Model...</h3>
		</div>
	{:else if uplift}
		<div class="card uplift-card">
			<div class="card-header">
				<div>
					<h3 class="card-title">CATE Uplift Decision & Recommended Action</h3>
					<p class="card-description">
						Evaluated under marketing campaign #{uplift.evaluated_campaign_id} context.
					</p>
				</div>
				<div class="targeting-action-badge">
					<span
						class="badge action-pill {uplift.recommended_action === 'PROMOTE'
							? 'badge-success'
							: uplift.recommended_action === 'NO PROMOTION'
								? 'badge-warning'
								: 'badge-danger'}"
					>
						Action: {uplift.recommended_action}
					</span>
				</div>
			</div>

			<div class="kpi-grid">
				<MetricCard
					title="CATE Uplift Score τ(X)"
					value={uplift.uplift_score_cate}
					formatType="currency"
					subtitle="Predicted incremental spend lift E[Y(1) - Y(0) | X]"
				/>
				<MetricCard
					title="Targeting Classification"
					value={uplift.targeting_segment}
					formatType="standard"
					subtitle={`Optimal decision: ${uplift.recommended_action}`}
				/>
				<MetricCard
					title="Projected Spend with Campaign Y(1)"
					value={uplift.expected_spend_with_treatment}
					formatType="currency"
					subtitle="Expected spend under direct-mail circular"
				/>
				<MetricCard
					title="Projected Baseline Spend Y(0)"
					value={uplift.expected_spend_without_treatment}
					formatType="currency"
					subtitle="Expected baseline spend without mailer"
				/>
			</div>

			<div class="uplift-policy-summary">
				<div class="policy-item">
					<strong>Cost-Benefit Rule:</strong> Direct-mail delivery cost is assumed at ${uplift.cost_threshold_dollars.toFixed(
						2
					)}/household.
					{#if uplift.is_recommended_for_treatment}
						Incremental spend lift ({formatCurrency(uplift.uplift_score_cate)}) exceeds the marginal mailing cost, generating positive net profit lift.
					{:else}
						Incremental spend lift ({formatCurrency(uplift.uplift_score_cate)}) does not exceed the mailing cost of ${uplift.cost_threshold_dollars.toFixed(
							2
						)}, or produces negative treatment effect. Treatment is withheld.
					{/if}
				</div>
				<div class="policy-item">
					<strong>Model Provenance:</strong> {uplift.model_provenance}
				</div>
			</div>

			<!-- Uplift Methodological Disclosures -->
			<MethodologicalBanners warnings={uplift.methodological_warnings} />
		</div>
	{/if}

	<!-- Qini Deciles & Targeting Strategy Card -->
	{#if segments && segments.deciles.length > 0}
		<div class="card segments-card">
			<div class="card-header">
				<div>
					<h3 class="card-title">Population Targeting Deciles & Qini Curve</h3>
					<p class="card-description">
						Out-of-sample cumulative incremental revenue progression evaluated in Phase 5 benchmark.
					</p>
				</div>
				<div class="badge-group">
					<span class="badge badge-success">AUUC: {segments.auuc.toLocaleString()}</span>
					<span class="badge badge-info">Top Decile Lift: +${segments.top_decile_lift}</span>
					<span class="badge badge-neutral">Top Decile Multiplier: {segments.top_decile_multiplier}x</span>
				</div>
			</div>

			<div class="segments-table-container">
				<table class="data-table">
					<thead>
						<tr>
							<th>Decile</th>
							<th>Targeted Share</th>
							<th>Cumulative Targeted</th>
							<th>Cumulative Gain ($)</th>
							<th>Qini Curve Value</th>
							<th>Treated Spend ($)</th>
							<th>Control Spend ($)</th>
							<th>Policy Action</th>
						</tr>
					</thead>
					<tbody>
						{#each segments.deciles as d}
							<tr class={d.decile <= segments.recommended_targeting_cutoff_decile ? 'highlight-row' : ''}>
								<td><strong>Decile {d.decile}</strong></td>
								<td>{(d.population_fraction * 100).toFixed(0)}%</td>
								<td>{d.n_targeted_cum.toLocaleString()}</td>
								<td>{formatCurrency(d.cum_gain)}</td>
								<td>{d.qini_val.toLocaleString()}</td>
								<td>${d.mean_treated_cum.toFixed(2)}</td>
								<td>${d.mean_control_cum.toFixed(2)}</td>
								<td>
									{#if d.decile <= segments.recommended_targeting_cutoff_decile}
										<span class="badge badge-success">PROMOTE (Persuadable)</span>
									{:else if d.decile <= 7}
										<span class="badge badge-warning">NO PROMOTION (Sure Thing)</span>
									{:else}
										<span class="badge badge-danger">AVOID (Lost Cause)</span>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<div class="guidance-box">
				<h4>Targeting Policy Recommendation</h4>
				<p>{segments.policy_guidance}</p>
				<p class="text-muted"><em>{segments.methodological_note}</em></p>
			</div>
		</div>
	{/if}
</div>

<style>
	.customer-view {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.customer-header {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		gap: 1rem;
	}

	.section-title {
		font-size: 1.5rem;
		font-weight: 700;
		color: var(--color-text-main);
		margin-bottom: 0.25rem;
	}

	.section-subtitle {
		font-size: 0.9rem;
		color: var(--color-text-muted);
	}

	.badge-group {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.lookup-bar {
		display: flex;
		flex-direction: column;
		gap: 0.85rem;
	}

	.lookup-form {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
	}

	.lookup-label {
		font-size: 0.9rem;
		font-weight: 600;
		color: var(--color-text-main);
	}

	.input-number {
		width: 120px;
		padding: 0.45rem 0.65rem;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		font-size: 0.9rem;
		background: var(--color-surface);
		color: var(--color-text-main);
	}

	.quick-picks {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.quick-label {
		font-size: 0.8rem;
		color: var(--color-text-muted);
	}

	.btn-chip {
		background: var(--color-surface-soft);
		border: 1px solid var(--color-border);
		padding: 0.25rem 0.65rem;
		font-size: 0.75rem;
		border-radius: 9999px;
		cursor: pointer;
		color: var(--color-text-main);
		transition: all 0.15s ease;
	}

	.btn-chip:hover {
		border-color: var(--color-primary);
	}

	.btn-chip.chip-active {
		background: var(--color-primary);
		color: #ffffff;
		border-color: var(--color-primary);
	}

	.profile-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: 1rem;
	}

	.card-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--color-border);
	}

	.card-title {
		font-size: 1.15rem;
		font-weight: 600;
		color: var(--color-text-main);
	}

	.card-description {
		font-size: 0.85rem;
		color: var(--color-text-muted);
		margin-top: 0.15rem;
	}

	.meta-dl {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 0.5rem 1rem;
		font-size: 0.85rem;
	}

	.meta-dl dt {
		color: var(--color-text-muted);
		font-weight: 500;
	}

	.meta-dl dd {
		color: var(--color-text-main);
		font-weight: 600;
	}

	.empty-demo-state {
		padding: 1.5rem 0;
		text-align: center;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.behavior-metrics {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
		margin-bottom: 1rem;
	}

	.b-metric {
		background: var(--color-surface-soft);
		border: 1px solid var(--color-border);
		padding: 0.75rem;
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.b-label {
		font-size: 0.75rem;
		color: var(--color-text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.b-val {
		font-size: 1.05rem;
		font-weight: 700;
		color: var(--color-text-main);
	}

	.card-footer-note {
		font-size: 0.75rem;
		color: var(--color-text-muted);
		line-height: 1.4;
	}

	.uplift-card,
	.segments-card {
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
	}

	.action-pill {
		font-size: 0.85rem;
		padding: 0.35rem 0.85rem;
		font-weight: 700;
	}

	.kpi-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
		gap: 1rem;
	}

	.uplift-policy-summary {
		background: var(--color-surface-soft);
		border-left: 3px solid var(--color-primary);
		padding: 0.85rem 1rem;
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		font-size: 0.85rem;
	}

	.segments-table-container {
		max-height: 320px;
		overflow-y: auto;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
	}

	.data-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.85rem;
		text-align: left;
	}

	.data-table th,
	.data-table td {
		padding: 0.65rem 0.85rem;
		border-bottom: 1px solid var(--color-border);
	}

	.data-table th {
		background: var(--color-surface-soft);
		font-weight: 600;
		position: sticky;
		top: 0;
		z-index: 10;
	}

	.highlight-row {
		background: rgba(16, 185, 129, 0.08);
	}

	.guidance-box {
		background: var(--color-surface-soft);
		border: 1px solid var(--color-border);
		padding: 1rem;
		border-radius: var(--radius-md);
		font-size: 0.85rem;
		line-height: 1.5;
	}

	.guidance-box h4 {
		font-weight: 600;
		margin-bottom: 0.35rem;
		color: var(--color-text-main);
	}
</style>
