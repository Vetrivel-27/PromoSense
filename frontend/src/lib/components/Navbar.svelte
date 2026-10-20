<script lang="ts">
	interface Props {
		activeTab: string;
		backendOnline: boolean | null;
		onTabChange: (tab: string) => void;
	}

	let { activeTab, backendOnline, onTabChange }: Props = $props();

	const tabs = [
		{ id: 'dashboard', label: '1. Executive Dashboard' },
		{ id: 'catalog', label: '2. Product Catalog' },
		{ id: 'elasticity', label: '3. Price Elasticity' },
		{ id: 'simulator', label: '4. Promotion Simulator' },
		{ id: 'optimizer', label: '5. Profit Optimizer' },
		{ id: 'customers', label: '6. Customer Uplift' }
	];
</script>

<header class="navbar">
	<div class="container nav-content">
		<div class="brand">
			<div class="logo-mark">PS</div>
			<div>
				<h1 class="brand-title">PromoSense</h1>
				<p class="brand-subtitle">Causal Promotion & Pricing Intelligence Platform</p>
			</div>
		</div>

		<nav class="nav-links">
			{#each tabs as tab}
				<button
					type="button"
					class="nav-tab {activeTab === tab.id ? 'active' : ''}"
					onclick={() => onTabChange(tab.id)}
				>
					{tab.label}
				</button>
			{/each}
		</nav>

		<div class="backend-status">
			{#if backendOnline === null}
				<span class="badge badge-neutral">Connecting...</span>
			{:else if backendOnline}
				<span class="badge badge-success">Backend Online</span>
			{:else}
				<span class="badge badge-danger">Backend Offline</span>
			{/if}
		</div>
	</div>
</header>

<style>
	.navbar {
		background-color: var(--color-surface);
		border-bottom: 1px solid var(--color-border);
		position: sticky;
		top: 0;
		z-index: 50;
		box-shadow: var(--shadow-sm);
	}

	.nav-content {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding-top: 0.85rem;
		padding-bottom: 0.85rem;
		gap: 1.5rem;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 0.85rem;
	}

	.logo-mark {
		width: 40px;
		height: 40px;
		background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
		color: white;
		font-weight: 800;
		font-size: 1.15rem;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: var(--radius-sm);
		box-shadow: 0 2px 4px rgb(29 78 216 / 0.3);
	}

	.brand-title {
		font-size: 1.25rem;
		font-weight: 800;
		color: var(--color-text-main);
		line-height: 1.1;
	}

	.brand-subtitle {
		font-size: 0.75rem;
		color: var(--color-text-muted);
		line-height: 1.2;
	}

	.nav-links {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		background-color: #f1f5f9;
		padding: 0.3rem;
		border-radius: var(--radius-md);
	}

	.nav-tab {
		background: transparent;
		border: none;
		padding: 0.45rem 0.95rem;
		font-size: 0.8125rem;
		font-weight: 600;
		color: var(--color-text-muted);
		border-radius: var(--radius-sm);
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.nav-tab:hover {
		color: var(--color-text-main);
	}

	.nav-tab.active {
		background-color: white;
		color: var(--color-primary);
		box-shadow: var(--shadow-sm);
	}

	@media (max-width: 900px) {
		.nav-content {
			flex-direction: column;
			align-items: stretch;
		}

		.nav-links {
			overflow-x: auto;
		}
	}
</style>
