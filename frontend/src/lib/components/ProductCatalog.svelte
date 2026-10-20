<script lang="ts">
	import type { ProductSummary, SubstitutePairSummary } from '../types';

	interface Props {
		products: ProductSummary[];
		substitutePairs: SubstitutePairSummary[];
		selectedProductId: number;
		onSelectProduct: (productId: number) => void;
	}

	let { products, substitutePairs, selectedProductId, onSelectProduct }: Props = $props();
</script>

<div class="catalog-section">
	<div class="section-header">
		<div>
			<h2 class="section-title">Modeled Product Catalog</h2>
			<p class="section-subtitle">
				7 Core volume products evaluated across 102 weeks of empirical store scanner transactions.
			</p>
		</div>
	</div>

	<!-- Product Table -->
	<div class="card table-card">
		<div class="table-container">
			<table class="table">
				<thead>
					<tr>
						<th>Product SKU</th>
						<th>Commodity & Description</th>
						<th>Brand Tier</th>
						<th>Base Shelf Price</th>
						<th>Obs. Display</th>
						<th>Obs. Mailer</th>
						<th>Category Pair Role</th>
						<th>Action</th>
					</tr>
				</thead>
				<tbody>
					{#each products as product}
						<tr class={selectedProductId === product.product_id ? 'row-selected' : ''}>
							<td>
								<span class="sku-badge">{product.product_id}</span>
							</td>
							<td>
								<div class="commodity-name">{product.commodity}</div>
								<div class="sub-commodity">{product.sub_commodity}</div>
							</td>
							<td>
								<span class="badge {product.brand_type === 'National' ? 'badge-primary' : 'badge-neutral'}">
									{product.brand_type}
								</span>
							</td>
							<td class="numeric-cell">${product.baseline_shelf_price.toFixed(2)}</td>
							<td class="numeric-cell">{(product.observed_display_share * 100).toFixed(1)}%</td>
							<td class="numeric-cell">{(product.observed_mailer_share * 100).toFixed(1)}%</td>
							<td>
								{#if product.is_focal_substitute_a}
									<span class="badge badge-primary">Focal Promoted (Prod A)</span>
								{:else if product.is_substitute_b}
									<span class="badge badge-warning">Substitute (Prod B)</span>
								{:else}
									<span class="badge badge-neutral">Single SKU (Bananas)</span>
								{/if}
							</td>
							<td>
								<button
									type="button"
									class="btn {selectedProductId === product.product_id ? 'btn-primary' : 'btn-secondary'} btn-sm"
									onclick={() => onSelectProduct(product.product_id)}
								>
									{selectedProductId === product.product_id ? 'Active' : 'Select'}
								</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>

	<!-- Category Substitute Pairs Overview -->
	<div class="pairs-overview">
		<h3 class="subsection-title">Category Substitute Pairs (Cannibalization Models)</h3>
		<div class="grid grid-cols-3">
			{#each substitutePairs as pair}
				<div class="card pair-card">
					<div class="pair-header">
						<span class="pair-name">{pair.pair_name.replace(/_/g, ' ')}</span>
						{#if pair.is_weakly_identified}
							<span class="badge badge-warning">Weak Identification</span>
						{:else}
							<span class="badge badge-success">Well-Identified</span>
						{/if}
					</div>
					<div class="pair-body">
						<div class="pair-commodity">{pair.commodity} &bull; {pair.sub_commodity}</div>
						<div class="pair-skus">
							<div class="sku-col">
								<span class="role-label">Focal (A):</span>
								<button
									type="button"
									class="sku-link"
									onclick={() => onSelectProduct(pair.prod_A_id)}
								>
									{pair.prod_A_id}
								</button>
							</div>
							<div class="sku-col">
								<span class="role-label">Substitute (B):</span>
								<button
									type="button"
									class="sku-link"
									onclick={() => onSelectProduct(pair.prod_B_id)}
								>
									{pair.prod_B_id}
								</button>
							</div>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</div>
</div>

<style>
	.catalog-section {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
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

	.table-card {
		padding: 0;
		overflow: hidden;
	}

	.sku-badge {
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: 0.8125rem;
		color: var(--color-primary);
	}

	.commodity-name {
		font-weight: 700;
		font-size: 0.8125rem;
		color: var(--color-text-main);
	}

	.sub-commodity {
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.numeric-cell {
		font-family: var(--font-mono);
		font-size: 0.8125rem;
	}

	.row-selected {
		background-color: #eff6ff !important;
	}

	.btn-sm {
		padding: 0.35rem 0.75rem;
		font-size: 0.75rem;
	}

	.pairs-overview {
		margin-top: 0.5rem;
	}

	.subsection-title {
		font-size: 1rem;
		font-weight: 700;
		color: var(--color-text-main);
		margin-bottom: 0.75rem;
	}

	.pair-card {
		padding: 1rem;
	}

	.pair-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.4rem;
	}

	.pair-name {
		font-size: 0.8125rem;
		font-weight: 700;
		color: var(--color-text-main);
	}

	.pair-commodity {
		font-size: 0.75rem;
		color: var(--color-text-muted);
		margin-bottom: 0.75rem;
	}

	.pair-skus {
		display: flex;
		gap: 1rem;
		border-top: 1px solid #f1f5f9;
		padding-top: 0.5rem;
		font-size: 0.75rem;
	}

	.sku-col {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}

	.role-label {
		color: var(--color-text-subtle);
		font-size: 0.7rem;
	}

	.sku-link {
		background: none;
		border: none;
		color: var(--color-primary);
		font-family: var(--font-mono);
		font-weight: 700;
		cursor: pointer;
		text-align: left;
		padding: 0;
	}

	.sku-link:hover {
		text-decoration: underline;
	}
</style>
