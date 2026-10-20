<script lang="ts">
	import type { ProductElasticityResponse } from '../types';
	import MethodologicalBanners from './MethodologicalBanners.svelte';

	interface Props {
		elasticityData: ProductElasticityResponse | null;
		isLoading: boolean;
		error: string | null;
		onSelectProduct?: (productId: number) => void;
	}

	let { elasticityData, isLoading, error, onSelectProduct }: Props = $props();

	function getCompetingRows(competing: Record<string, any> | undefined) {
		if (!competing) return [];
		const rows: Array<{
			id: string;
			name: string;
			type: string;
			elasticity: string;
			se: string;
			rmse: string;
			r2: string;
			isChampion: boolean;
		}> = [];

		if (competing.controlled_paid_ols) {
			const m = competing.controlled_paid_ols;
			const v = m.val_eval || {};
			rows.push({
				id: 'controlled_paid_ols',
				name: 'Controlled OLS (Champion)',
				type: 'Log-Log OLS with Merch Controls',
				elasticity: typeof m.elasticity === 'number' ? m.elasticity.toFixed(3) : '—',
				se: typeof v.elasticity_hac_se === 'number' ? `±${v.elasticity_hac_se.toFixed(3)} (HAC)` : typeof m.elasticity_se === 'number' ? `±${m.elasticity_se.toFixed(3)}` : '—',
				rmse: typeof v.rmse_units === 'number' ? v.rmse_units.toFixed(1) : '—',
				r2: typeof v.r2_log === 'number' ? `${(v.r2_log * 100).toFixed(1)}%` : '—',
				isChampion: true
			});
		}

		if (competing.unadjusted_paid_ols) {
			const m = competing.unadjusted_paid_ols;
			const v = m.val_eval || {};
			rows.push({
				id: 'unadjusted_paid_ols',
				name: 'Unadjusted OLS',
				type: 'Log-Log Paid Price (No Controls)',
				elasticity: typeof m.elasticity === 'number' ? m.elasticity.toFixed(3) : '—',
				se: typeof v.elasticity_hac_se === 'number' ? `±${v.elasticity_hac_se.toFixed(3)} (HAC)` : typeof m.elasticity_se === 'number' ? `±${m.elasticity_se.toFixed(3)}` : '—',
				rmse: typeof v.rmse_units === 'number' ? v.rmse_units.toFixed(1) : '—',
				r2: typeof v.r2_log === 'number' ? `${(v.r2_log * 100).toFixed(1)}%` : '—',
				isChampion: false
			});
		}

		if (competing.decomposed_shelf_discount_ols) {
			const m = competing.decomposed_shelf_discount_ols;
			const v = m.val_eval || {};
			const discEps = typeof m.discount_elasticity === 'number' ? `-${m.discount_elasticity.toFixed(3)}` : '—';
			rows.push({
				id: 'decomposed_shelf_discount_ols',
				name: 'Decomposed Shelf & Discount',
				type: 'Decomposed OLS (Log Shelf + Depth)',
				elasticity: discEps,
				se: typeof m.discount_se === 'number' ? `±${m.discount_se.toFixed(3)}` : '—',
				rmse: typeof v.rmse_units === 'number' ? v.rmse_units.toFixed(1) : '—',
				r2: typeof v.r2_log === 'number' ? `${(v.r2_log * 100).toFixed(1)}%` : '—',
				isChampion: false
			});
		}

		if (competing.lightgbm_unconstrained) {
			const m = competing.lightgbm_unconstrained;
			const v = m.val_eval || {};
			const lgbEps = typeof v.model_implied_elasticity === 'number' ? v.model_implied_elasticity.toFixed(3) : '—';
			rows.push({
				id: 'lightgbm_unconstrained',
				name: 'LightGBM Regressor',
				type: 'Gradient Boosted Trees (Non-linear)',
				elasticity: lgbEps,
				se: 'N/A (Tree-based)',
				rmse: typeof v.rmse_units === 'number' ? v.rmse_units.toFixed(1) : '—',
				r2: typeof v.r2_log === 'number' ? `${(v.r2_log * 100).toFixed(1)}%` : '—',
				isChampion: false
			});
		}

		return rows;
	}
</script>

<div class="elasticity-section">
	{#if isLoading}
		<div class="card loading-card">
			<div class="spinner"></div>
			<p>Retrieving econometric price elasticity models and HAC robust covariance...</p>
		</div>
	{:else if error}
		<div class="alert alert-danger" role="alert">
			<div class="alert-icon">⚠️</div>
			<div>
				<div class="alert-title">Elasticity Data Error</div>
				<p>{error}</p>
			</div>
		</div>
	{:else if elasticityData}
		<!-- Header Info -->
		<div class="section-header">
			<div>
				<div class="header-badges">
					<span class="sku-tag">SKU {elasticityData.product_id}</span>
					<span class="badge {elasticityData.brand_type === 'National' ? 'badge-primary' : 'badge-neutral'}">
						{elasticityData.brand_type} Brand
					</span>
					<span class="commodity-tag">{elasticityData.commodity} &bull; {elasticityData.sub_commodity}</span>
				</div>
				<h2 class="section-title">Price Elasticity & Econometric Diagnostics</h2>
			</div>
		</div>

		<!-- Warning Banners -->
		<MethodologicalBanners
			warnings={elasticityData.methodological_warnings}
			compact={false}
		/>

		<!-- Champion Specification Card -->
		<div class="card champion-card">
			<div class="champion-header">
				<div>
					<span class="badge badge-primary">Champion Model Specification</span>
					<h3 class="champion-title">{elasticityData.champion_specification}</h3>
				</div>
				<div class="champion-ci-badge">
					<span class="ci-label">95% HAC Confidence Interval:</span>
					<span class="ci-value">
						[{elasticityData.champion_ci_95[0].toFixed(2)}, {elasticityData.champion_ci_95[1].toFixed(2)}]
					</span>
				</div>
			</div>

			<div class="champion-grid">
				<div class="stat-box primary-stat">
					<div class="stat-label">Own-Price Elasticity (ε̂)</div>
					<div class="stat-val">{elasticityData.champion_elasticity.toFixed(3)}</div>
					<div class="stat-sub">Percentage change in sales per 1% price cut</div>
				</div>

				<div class="stat-box">
					<div class="stat-label">HAC Standard Error</div>
					<div class="stat-val">±{elasticityData.champion_hac_se.toFixed(4)}</div>
					<div class="stat-sub">Newey-West heteroskedasticity & autocorrelation robust</div>
				</div>

				<div class="stat-box">
					<div class="stat-label">Statistical Significance</div>
					<div class="stat-val">{elasticityData.champion_p_value < 0.001 ? 'p < 0.001' : `p = ${elasticityData.champion_p_value.toFixed(4)}`}</div>
					<div class="stat-sub">
						{elasticityData.champion_p_value < 0.05 ? 'Statistically significant (α = 0.05)' : 'Not statistically significant'}
					</div>
				</div>
			</div>

			<div class="interpretation-box">
				<div class="interp-heading">
					<span class="interp-icon">💡</span>
					<strong>Economic Interpretation</strong>
				</div>
				<p class="interp-text">{elasticityData.interpretation}</p>
			</div>
		</div>

		<!-- Competing Model Comparison Table -->
		<div class="card table-card">
			<div class="card-inner-header">
				<h3 class="subsection-title">Competing Econometric Specifications</h3>
				<span class="header-note">Evaluated on out-of-sample temporal validation split (Weeks 71–86)</span>
			</div>

			<div class="table-container">
				<table class="table">
					<thead>
						<tr>
							<th>Specification</th>
							<th>Family / Formulation</th>
							<th>Estimated Elasticity</th>
							<th>Standard Error</th>
							<th>RMSE (Units)</th>
							<th>R² (Log Vol)</th>
							<th>Status</th>
						</tr>
					</thead>
					<tbody>
						{#each getCompetingRows(elasticityData.competing_models) as row}
							<tr class={row.isChampion ? 'champion-row' : ''}>
								<td>
									<strong>{row.name}</strong>
								</td>
								<td class="font-secondary">{row.type}</td>
								<td class="numeric-cell">
									<strong>{row.elasticity}</strong>
								</td>
								<td class="numeric-cell">{row.se}</td>
								<td class="numeric-cell">{row.rmse}</td>
								<td class="numeric-cell">{row.r2}</td>
								<td>
									{#if row.isChampion}
										<span class="badge badge-primary">Selected Champion</span>
									{:else}
										<span class="badge badge-neutral">Benchmark</span>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Substitute Cross-Price Elasticity Card -->
		{#if elasticityData.substitute_pair_elasticity}
			{@const sub = elasticityData.substitute_pair_elasticity}
			<div class="card substitute-card {sub.is_weakly_identified ? 'weak-substitute' : ''}">
				<div class="substitute-header">
					<div>
						<div class="sub-pair-tags">
							<span class="pair-title">{sub.pair_name.replace(/_/g, ' ')}</span>
							{#if sub.is_weakly_identified}
								<span class="badge badge-warning">Weak Identification Warning</span>
							{:else}
								<span class="badge badge-success">Identified Cross-Price Elasticity</span>
							{/if}
						</div>
						<h3 class="subsection-title">Category Substitute Relationship & Cannibalization</h3>
					</div>

					{#if onSelectProduct}
						<button
							type="button"
							class="btn btn-secondary btn-sm"
							onclick={() => onSelectProduct(sub.substitute_product_id)}
						>
							Switch to Substitute (SKU {sub.substitute_product_id})
						</button>
					{/if}
				</div>

				<div class="substitute-grid">
					<div class="sub-stat">
						<span class="sub-label">Focal Product A Own-Price (ε_AA):</span>
						<span class="sub-val">{sub.own_elasticity_A.toFixed(3)}</span>
					</div>
					<div class="sub-stat">
						<span class="sub-label">Cross-Price Elasticity (ε_AB):</span>
						<span class="sub-val {sub.cross_elasticity_AB > 0 ? 'text-primary' : ''}">
							+{sub.cross_elasticity_AB.toFixed(3)}
						</span>
					</div>
					<div class="sub-stat">
						<span class="sub-label">Cross-Price HAC Standard Error:</span>
						<span class="sub-val">±{sub.cross_elasticity_AB_se.toFixed(4)}</span>
					</div>
				</div>

				<div class="sub-note-box {sub.is_weakly_identified ? 'sub-note-weak' : ''}">
					<p>{sub.identification_note}</p>
				</div>
			</div>
		{/if}
	{/if}
</div>

<style>
	.elasticity-section {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.section-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
	}

	.header-badges {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.35rem;
	}

	.sku-tag {
		font-family: var(--font-mono);
		font-weight: 800;
		font-size: 0.8125rem;
		background-color: #dbeafe;
		color: #1e40af;
		padding: 0.2rem 0.5rem;
		border-radius: var(--radius-sm);
	}

	.commodity-tag {
		font-size: 0.75rem;
		color: var(--color-text-muted);
		font-weight: 500;
	}

	.section-title {
		font-size: 1.25rem;
		font-weight: 800;
		color: var(--color-text-main);
	}

	.loading-card {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 3rem;
		gap: 1rem;
		color: var(--color-text-muted);
	}

	.spinner {
		width: 32px;
		height: 32px;
		border: 3px solid #e2e8f0;
		border-top-color: var(--color-primary);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.champion-card {
		background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
		border: 1px solid #cbd5e1;
		border-left: 4px solid var(--color-primary);
	}

	.champion-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 1.25rem;
		flex-wrap: wrap;
		gap: 0.75rem;
	}

	.champion-title {
		font-size: 1.05rem;
		font-weight: 700;
		color: var(--color-text-main);
		margin-top: 0.35rem;
	}

	.champion-ci-badge {
		background-color: #f1f5f9;
		border: 1px solid #e2e8f0;
		padding: 0.4rem 0.75rem;
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		font-size: 0.75rem;
	}

	.ci-label {
		color: var(--color-text-muted);
		font-size: 0.7rem;
	}

	.ci-value {
		font-family: var(--font-mono);
		font-weight: 700;
		color: var(--color-primary);
		font-size: 0.875rem;
	}

	.champion-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 1rem;
		margin-bottom: 1.25rem;
	}

	.stat-box {
		background-color: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		padding: 1rem;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
	}

	.primary-stat {
		border-color: #bfdbfe;
		background-color: #f0f7ff;
	}

	.stat-label {
		font-size: 0.75rem;
		font-weight: 600;
		color: var(--color-text-muted);
		margin-bottom: 0.25rem;
	}

	.stat-val {
		font-size: 1.5rem;
		font-weight: 800;
		font-family: var(--font-mono);
		color: var(--color-text-main);
		margin-bottom: 0.25rem;
	}

	.stat-sub {
		font-size: 0.7rem;
		color: var(--color-text-subtle);
		line-height: 1.3;
	}

	.interpretation-box {
		background-color: #f8fafc;
		border: 1px solid #e2e8f0;
		border-radius: var(--radius-sm);
		padding: 0.85rem 1rem;
	}

	.interp-heading {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.8125rem;
		color: var(--color-text-main);
		margin-bottom: 0.35rem;
	}

	.interp-icon {
		font-size: 0.95rem;
	}

	.interp-text {
		font-size: 0.8125rem;
		color: var(--color-text-muted);
		line-height: 1.45;
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

	.font-secondary {
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.numeric-cell {
		font-family: var(--font-mono);
		font-size: 0.8125rem;
	}

	.champion-row {
		background-color: #eff6ff !important;
		font-weight: 600;
	}

	.substitute-card {
		border-left: 4px solid #10b981;
	}

	.weak-substitute {
		border-left: 4px solid var(--color-warning);
	}

	.substitute-header {
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

	.substitute-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 1rem;
		margin-bottom: 1rem;
		background-color: #f8fafc;
		padding: 0.85rem;
		border-radius: var(--radius-sm);
		border: 1px solid #f1f5f9;
	}

	.sub-stat {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}

	.sub-label {
		font-size: 0.725rem;
		color: var(--color-text-muted);
	}

	.sub-val {
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: 1rem;
		color: var(--color-text-main);
	}

	.sub-note-box {
		font-size: 0.75rem;
		line-height: 1.4;
		color: var(--color-text-muted);
		padding: 0.65rem 0.85rem;
		background-color: #f1f5f9;
		border-radius: var(--radius-sm);
	}

	.sub-note-weak {
		background-color: #fffbeb;
		color: #92400e;
		border: 1px solid #fde68a;
	}

	.btn-sm {
		padding: 0.35rem 0.75rem;
		font-size: 0.75rem;
	}

	@media (max-width: 800px) {
		.champion-grid,
		.substitute-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
