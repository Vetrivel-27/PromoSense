<script lang="ts">
	import type {
		DashboardPromotionsResponse,
		DashboardSalesTrendResponse,
		DashboardSummaryResponse
	} from '../types';
	import MetricCard from './MetricCard.svelte';
	import MethodologicalBanners from './MethodologicalBanners.svelte';

	interface Props {
		summary: DashboardSummaryResponse | null;
		salesTrend: DashboardSalesTrendResponse | null;
		promotions: DashboardPromotionsResponse | null;
		loading: boolean;
		error: string | null;
	}

	let { summary, salesTrend, promotions, loading, error }: Props = $props();

	let trendWindow = $state<number>(26); // Default to last 26 weeks

	const displayedTrend = $derived.by(() => {
		if (!salesTrend?.trend) return [];
		if (trendWindow === 102) return salesTrend.trend;
		return salesTrend.trend.slice(-trendWindow);
	});

	const maxWeeklySales = $derived.by(() => {
		if (!displayedTrend.length) return 100000;
		return Math.max(...displayedTrend.map((d) => d.total_sales));
	});

	function formatCurrency(val: number): string {
		return `$${val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
	}

	function formatPct(val: number): string {
		return `${(val * 100).toFixed(1)}%`;
	}
</script>

<div class="dashboard-view">
	{#if loading}
		<div class="state-card loading-state">
			<div class="spinner"></div>
			<h3>Loading Executive Dashboard...</h3>
			<p>Aggregating 102-week empirical transactions panel facts and campaign benchmarks.</p>
		</div>
	{:else if error}
		<div class="state-card error-state">
			<div class="error-icon">⚠️</div>
			<h3>Dashboard Unavailable</h3>
			<p>{error}</p>
		</div>
	{:else if summary}
		<!-- Executive Header & Caveat Banner -->
		<div class="dashboard-header">
			<div>
				<h2 class="section-title">Chain Portfolio & Executive Summary</h2>
				<p class="section-subtitle">
					Empirical 102-week transactions panel facts across 2,500 households, 92,353 catalog products, and 30 marketing campaigns.
				</p>
			</div>
			<div class="badge-group">
				<span class="badge badge-info">102 Weeks</span>
				<span class="badge badge-neutral">2,500 Households</span>
				<span class="badge badge-neutral">30 Campaigns</span>
				<span class="badge badge-success">7 Modeled Volume SKUs</span>
			</div>
		</div>

		<!-- Scope Boundary Alert Box -->
		<div class="alert alert-info scope-banner" role="alert">
			<div class="alert-icon">ℹ️</div>
			<div>
				<div class="alert-title">Portfolio Boundary & Identification Policy</div>
				<p>{summary.portfolio_scope_disclaimer}</p>
			</div>
		</div>

		<!-- KPI Metric Cards Grid -->
		<div class="kpi-grid">
			<MetricCard
				title="Total Chain Sales (102 Weeks)"
				value={summary.total_sales_dollars}
				formatType="currency"
				subtitle="2,595,732 transaction line items"
			/>
			<MetricCard
				title="Promotional Sales Volume"
				value={summary.promotional_sales_dollars}
				formatType="currency"
				subtitle={`${formatPct(summary.promotional_sales_share)} of chain sales discounted`}
			/>
			<MetricCard
				title="Customer Household Panel"
				value={summary.total_households}
				formatType="standard"
				subtitle="801 households with demographic disclosures"
			/>
			<MetricCard
				title="Checkout Baskets"
				value={summary.total_baskets}
				formatType="standard"
				subtitle="Avg basket: $29.14 across 102 weeks"
			/>
			<MetricCard
				title="Modeled Volume SKU Sales"
				value={summary.modeled_product_revenue}
				formatType="currency"
				subtitle={`7 target volume SKUs (${formatPct(summary.modeled_product_revenue_share)} of chain)`}
			/>
			<MetricCard
				title="Direct-Mail Campaigns"
				value={summary.total_campaigns}
				formatType="standard"
				subtitle="Pooled CATE Lift: +$211.80/household"
			/>
		</div>

		<!-- 102-Week Sales Trend Section -->
		{#if salesTrend && salesTrend.trend.length > 0}
			<div class="card trend-section">
				<div class="section-top-bar">
					<div>
						<h3 class="card-title">Weekly Sales Trend & Promotional Volume</h3>
						<p class="card-description">
							Historical weekly sales separated into baseline shelf volume and promotional discount lift.
						</p>
					</div>
					<div class="window-toggle">
						<span class="toggle-label">Window:</span>
						<button
							class="btn-toggle {trendWindow === 12 ? 'active' : ''}"
							onclick={() => (trendWindow = 12)}>Last 12W</button
						>
						<button
							class="btn-toggle {trendWindow === 26 ? 'active' : ''}"
							onclick={() => (trendWindow = 26)}>Last 26W</button
						>
						<button
							class="btn-toggle {trendWindow === 52 ? 'active' : ''}"
							onclick={() => (trendWindow = 52)}>Last 52W</button
						>
						<button
							class="btn-toggle {trendWindow === 102 ? 'active' : ''}"
							onclick={() => (trendWindow = 102)}>All 102W</button
						>
					</div>
				</div>

				<!-- Trend Visualizer Bar Chart -->
				<div class="trend-chart-container">
					<div class="chart-bars">
						{#each displayedTrend as point}
							{@const totalHeightPct = Math.min(100, (point.total_sales / maxWeeklySales) * 100)}
							{@const promoHeightPct = (point.promotional_sales / point.total_sales) * 100}
							{@const baseHeightPct = 100 - promoHeightPct}
							<div
								class="chart-bar-col"
								title={`Week ${point.week_no}: Total ${formatCurrency(point.total_sales)} (Promo: ${formatCurrency(point.promotional_sales)})`}
							>
								<div class="bar-stack" style={`height: ${totalHeightPct}%;`}>
									<div class="bar-promo" style={`height: ${promoHeightPct}%;`}></div>
									<div class="bar-base" style={`height: ${baseHeightPct}%;`}></div>
								</div>
								{#if displayedTrend.length <= 26 || point.week_no % 10 === 0}
									<span class="bar-label">W{point.week_no}</span>
								{/if}
							</div>
						{/each}
					</div>
					<div class="chart-legend">
						<span class="legend-item"><span class="legend-box box-base"></span> Baseline Shelf Sales</span>
						<span class="legend-item"><span class="legend-box box-promo"></span> Promotional Lift</span>
						<span class="legend-note">(Max observed: {formatCurrency(maxWeeklySales)}/week)</span>
					</div>
				</div>
			</div>
		{/if}

		<!-- Marketing Campaign Summary Section -->
		{#if promotions && promotions.campaigns.length > 0}
			<div class="card campaign-section">
				<div class="section-top-bar">
					<div>
						<h3 class="card-title">Marketing Campaign Portfolio & Causal Benchmarks</h3>
						<p class="card-description">
							Household-level direct-mail campaigns evaluated using Phase 5 Causal Meta-Learners (LightGBM).
						</p>
					</div>
					<div class="badge-group">
						<span class="badge badge-success">Pooled ATE: +${promotions.pooled_ate_spend_lift}</span>
						<span class="badge badge-info">Top Decile Lift: +${promotions.top_decile_lift}</span>
					</div>
				</div>

				<div class="campaign-table-container">
					<table class="data-table">
						<thead>
							<tr>
								<th>Campaign ID</th>
								<th>Campaign Type</th>
								<th>Duration</th>
								<th>Calendar Window</th>
								<th>Households Targeted</th>
								<th>Targeting Reach</th>
							</tr>
						</thead>
						<tbody>
							{#each promotions.campaigns as camp}
								<tr>
									<td><strong>Campaign {camp.campaign_id}</strong></td>
									<td>
										<span
											class="badge {camp.campaign_type === 'TypeA'
												? 'badge-info'
												: camp.campaign_type === 'TypeB'
													? 'badge-success'
													: 'badge-neutral'}"
										>
											{camp.campaign_type}
										</span>
									</td>
									<td>{camp.duration_days} days</td>
									<td>Day {camp.start_day} – Day {camp.end_day}</td>
									<td>{camp.households_targeted.toLocaleString()} households</td>
									<td>{formatPct(camp.target_rate)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>

				<div class="campaign-footer-note">
					<p>💡 {promotions.methodological_note}</p>
				</div>
			</div>
		{/if}

		<!-- Methodological Disclosures & Warnings -->
		<MethodologicalBanners warnings={summary.methodological_warnings} />
	{/if}
</div>

<style>
	.dashboard-view {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.dashboard-header {
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

	.scope-banner {
		margin-bottom: 0.5rem;
	}

	.kpi-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 1rem;
	}

	.trend-section,
	.campaign-section {
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
	}

	.section-top-bar {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		gap: 1rem;
	}

	.card-title {
		font-size: 1.15rem;
		font-weight: 600;
		color: var(--color-text-main);
	}

	.card-description {
		font-size: 0.85rem;
		color: var(--color-text-muted);
		margin-top: 0.2rem;
	}

	.window-toggle {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}

	.toggle-label {
		font-size: 0.8rem;
		color: var(--color-text-muted);
		margin-right: 0.25rem;
	}

	.btn-toggle {
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		padding: 0.3rem 0.65rem;
		font-size: 0.75rem;
		border-radius: var(--radius-sm);
		cursor: pointer;
		color: var(--color-text-muted);
		transition: all 0.15s ease;
	}

	.btn-toggle:hover {
		border-color: var(--color-primary);
		color: var(--color-text-main);
	}

	.btn-toggle.active {
		background: var(--color-primary);
		color: #ffffff;
		border-color: var(--color-primary);
	}

	.trend-chart-container {
		background: var(--color-surface-soft);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		padding: 1.25rem;
	}

	.chart-bars {
		display: flex;
		align-items: flex-end;
		gap: 4px;
		height: 180px;
		padding-bottom: 1.5rem;
		border-bottom: 1px solid var(--color-border);
	}

	.chart-bar-col {
		flex: 1;
		height: 100%;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: flex-end;
		position: relative;
	}

	.bar-stack {
		width: 100%;
		min-width: 4px;
		max-width: 24px;
		display: flex;
		flex-direction: column-reverse;
		border-radius: 3px 3px 0 0;
		overflow: hidden;
		transition: height 0.3s ease;
	}

	.bar-base {
		width: 100%;
		background: #3b82f6; /* Blue for baseline */
	}

	.bar-promo {
		width: 100%;
		background: #10b981; /* Green for promotional lift */
	}

	.bar-label {
		position: absolute;
		bottom: 0;
		font-size: 0.65rem;
		color: var(--color-text-muted);
		transform: translateY(1.2rem);
	}

	.chart-legend {
		display: flex;
		align-items: center;
		gap: 1.5rem;
		margin-top: 1.75rem;
		font-size: 0.8rem;
		color: var(--color-text-muted);
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}

	.legend-box {
		width: 12px;
		height: 12px;
		border-radius: 2px;
	}

	.box-base {
		background: #3b82f6;
	}

	.box-promo {
		background: #10b981;
	}

	.legend-note {
		margin-left: auto;
		font-style: italic;
		font-size: 0.75rem;
	}

	.campaign-table-container {
		max-height: 280px;
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

	.campaign-footer-note {
		font-size: 0.8rem;
		color: var(--color-text-muted);
		line-height: 1.4;
	}
</style>
