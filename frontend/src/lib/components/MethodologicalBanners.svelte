<script lang="ts">
	import type { EmpiricalSupportMetadata, MethodologicalWarnings } from '../types';

	interface Props {
		warnings?: MethodologicalWarnings | null;
		support?: EmpiricalSupportMetadata | null;
		compact?: boolean;
	}

	let { warnings, support, compact = false }: Props = $props();
</script>

<div class="methodological-container">
	<!-- 1. Weak Identification Warning Banner (Highest Priority) -->
	{#if warnings?.cross_price_warning}
		<div class="alert alert-warning" role="alert">
			<div class="alert-icon">⚠️</div>
			<div>
				<div class="alert-title">Econometric Identification Warning</div>
				<p>{warnings.cross_price_warning}</p>
			</div>
		</div>
	{/if}

	<!-- 2. Empirical Support Warnings -->
	{#if support}
		{#if !support.is_jointly_supported || support.discount_warning || support.merchandising_warning || support.joint_support_warning}
			<div class="alert alert-warning" role="alert">
				<div class="alert-icon">⚡</div>
				<div>
					<div class="alert-title">Empirical Domain Boundary Alert</div>
					{#if support.discount_warning}
						<p>{support.discount_warning}</p>
					{/if}
					{#if support.merchandising_warning}
						<p>{support.merchandising_warning}</p>
					{/if}
					{#if support.joint_support_warning && !support.merchandising_warning}
						<p>{support.joint_support_warning}</p>
					{/if}
				</div>
			</div>
		{/if}
	{/if}

	<!-- 3. Standard Methodological Disclosures -->
	{#if !compact && warnings}
		<div class="disclosure-grid">
			<div class="disclosure-card">
				<div class="disclosure-header">
					<span class="disclosure-dot dot-info"></span>
					<strong>Missing Cost Basis Disclosure</strong>
				</div>
				<p>{warnings.cost_basis_assumption}</p>
			</div>

			<div class="disclosure-card">
				<div class="disclosure-header">
					<span class="disclosure-dot dot-info"></span>
					<strong>Observational Unconfoundedness</strong>
				</div>
				<p>{warnings.observational_unconfoundedness_disclaimer}</p>
			</div>

			<div class="disclosure-card">
				<div class="disclosure-header">
					<span class="disclosure-dot dot-info"></span>
					<strong>Partial Parameter Uncertainty</strong>
				</div>
				<p>{warnings.partial_uncertainty_disclaimer}</p>
			</div>
		</div>
	{/if}
</div>

<style>
	.methodological-container {
		margin-bottom: 1.5rem;
	}

	.alert {
		display: flex;
		align-items: flex-start;
		gap: 0.85rem;
		padding: 0.9rem 1.1rem;
		border-radius: var(--radius-sm);
		font-size: 0.8125rem;
		line-height: 1.4;
		margin-bottom: 0.85rem;
	}

	.alert-icon {
		font-size: 1.1rem;
		flex-shrink: 0;
	}

	.alert-title {
		font-weight: 700;
		margin-bottom: 0.2rem;
	}

	.disclosure-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 0.85rem;
		margin-top: 0.75rem;
	}

	.disclosure-card {
		background-color: #f8fafc;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		padding: 0.75rem 0.9rem;
		font-size: 0.75rem;
		color: var(--color-text-muted);
		line-height: 1.35;
	}

	.disclosure-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		color: var(--color-text-main);
		margin-bottom: 0.25rem;
		font-size: 0.75rem;
	}

	.disclosure-dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
	}

	.dot-info {
		background-color: var(--color-info);
	}

	@media (max-width: 900px) {
		.disclosure-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
