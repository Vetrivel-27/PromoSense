<script lang="ts">
	interface Props {
		title: string;
		value: string | number;
		baseline?: string | number | null;
		delta?: string | number | null;
		percentLift?: number | null;
		formatType?: 'currency' | 'units' | 'percent' | 'standard';
		isProfit?: boolean;
		subtitle?: string | null;
	}

	let {
		title,
		value,
		baseline = null,
		delta = null,
		percentLift = null,
		formatType = 'standard',
		isProfit = false,
		subtitle = null
	}: Props = $props();

	function formatVal(v: string | number | null | undefined): string {
		if (v === null || v === undefined) return '—';
		if (typeof v === 'string') return v;
		if (formatType === 'currency') {
			const sign = v < 0 ? '-' : '';
			return `${sign}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
		}
		if (formatType === 'units') {
			return v.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
		}
		if (formatType === 'percent') {
			return `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
		}
		return v.toLocaleString('en-US');
	}

	const isPositive = $derived(
		percentLift !== null && percentLift !== undefined
			? percentLift > 0
			: typeof delta === 'number'
				? delta > 0
				: false
	);

	const isNegative = $derived(
		percentLift !== null && percentLift !== undefined
			? percentLift < 0
			: typeof delta === 'number'
				? delta < 0
				: false
	);
</script>

<div class="metric-card {isProfit ? (isPositive ? 'profit-positive' : isNegative ? 'profit-negative' : '') : ''}">
	<div class="metric-header">
		<span class="metric-title">{title}</span>
		{#if percentLift !== null && percentLift !== undefined}
			<span class="badge {percentLift > 0 ? 'badge-success' : percentLift < 0 ? 'badge-danger' : 'badge-neutral'}">
				{percentLift > 0 ? '+' : ''}{percentLift.toFixed(1)}%
			</span>
		{/if}
	</div>

	<div class="metric-body">
		<div class="metric-value">{formatVal(value)}</div>
	</div>

	{#if baseline !== null || delta !== null || subtitle}
		<div class="metric-footer">
			{#if baseline !== null}
				<div class="footer-item">
					<span class="footer-label">Baseline:</span>
					<span class="footer-val">{formatVal(baseline)}</span>
				</div>
			{/if}
			{#if delta !== null}
				<div class="footer-item">
					<span class="footer-label">Lift (Δ):</span>
					<span class="footer-val {isPositive ? 'text-success' : isNegative ? 'text-danger' : ''}">
						{typeof delta === 'number' && delta > 0 ? '+' : ''}{formatVal(delta)}
					</span>
				</div>
			{/if}
			{#if subtitle}
				<div class="metric-subtitle">{subtitle}</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.metric-card {
		background-color: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		padding: 1.15rem;
		box-shadow: var(--shadow-sm);
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		transition: all 0.15s ease;
	}

	.profit-positive {
		border-color: #a7f3d0;
		background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
	}

	.profit-negative {
		border-color: #fecaca;
		background: linear-gradient(180deg, #ffffff 0%, #fef2f2 100%);
	}

	.metric-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.4rem;
	}

	.metric-title {
		font-size: 0.8125rem;
		font-weight: 600;
		color: var(--color-text-muted);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.metric-value {
		font-size: 1.65rem;
		font-weight: 800;
		color: var(--color-text-main);
		line-height: 1.2;
		letter-spacing: -0.02em;
	}

	.metric-footer {
		margin-top: 0.85rem;
		padding-top: 0.65rem;
		border-top: 1px solid #f1f5f9;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.75rem;
	}

	.footer-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.footer-label {
		color: var(--color-text-muted);
	}

	.footer-val {
		font-weight: 600;
		color: var(--color-text-main);
	}

	.text-success {
		color: var(--color-success);
	}

	.text-danger {
		color: var(--color-danger);
	}

	.metric-subtitle {
		color: var(--color-text-subtle);
		font-size: 0.7rem;
		margin-top: 0.15rem;
	}
</style>
