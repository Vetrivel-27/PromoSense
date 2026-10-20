<script lang="ts">
	import { onMount } from 'svelte';
	import {
	fetchCustomerDetail,
	fetchCustomerSegments,
	fetchCustomerUplift,
	fetchDashboardPromotions,
	fetchDashboardSummary,
	fetchHealth,
	fetchProductElasticity,
	fetchProducts,
	fetchSalesTrend,
	optimizePromotion,
	simulatePromotion
} from '../lib/api';
import type {
	CustomerDetailResponse,
	CustomerSegmentsResponse,
	CustomerUpliftResponse,
	DashboardPromotionsResponse,
	DashboardSalesTrendResponse,
	DashboardSummaryResponse,
	OptimizationRequest,
	OptimizationResponse,
	ProductElasticityResponse,
	ProductSummary,
	SimulationRequest,
	SimulationResponse,
	SubstitutePairSummary
} from '../lib/types';
import CustomerIntelligence from '../lib/components/CustomerIntelligence.svelte';
import ElasticityViewer from '../lib/components/ElasticityViewer.svelte';
import ExecutiveDashboard from '../lib/components/ExecutiveDashboard.svelte';
import Navbar from '../lib/components/Navbar.svelte';
import OptimizationStudio from '../lib/components/OptimizationStudio.svelte';
import ProductCatalog from '../lib/components/ProductCatalog.svelte';
import SimulationStudio from '../lib/components/SimulationStudio.svelte';

	// Navigation & Global Connection State
	let activeTab = $state('dashboard');
	let backendOnline = $state<boolean | null>(null);
	let backendErrorMessage = $state<string | null>(null);

	// Executive Dashboard State
	let dashboardSummary = $state<DashboardSummaryResponse | null>(null);
	let dashboardSalesTrend = $state<DashboardSalesTrendResponse | null>(null);
	let dashboardPromotions = $state<DashboardPromotionsResponse | null>(null);
	let dashboardLoading = $state<boolean>(true);
	let dashboardError = $state<string | null>(null);

	// Customer Intelligence State
	let customerDetail = $state<CustomerDetailResponse | null>(null);
	let customerUplift = $state<CustomerUpliftResponse | null>(null);
	let customerSegments = $state<CustomerSegmentsResponse | null>(null);
	let customerLoading = $state<boolean>(false);
	let upliftLoading = $state<boolean>(false);
	let segmentsLoading = $state<boolean>(false);
	let customerError = $state<string | null>(null);
	let upliftError = $state<string | null>(null);
	let segmentsError = $state<string | null>(null);
	let selectedHouseholdKey = $state<number>(1);

	// Catalog State
	let products = $state<ProductSummary[]>([]);
	let substitutePairs = $state<SubstitutePairSummary[]>([]);
	let selectedProductId = $state<number>(5569230);
	let catalogLoading = $state<boolean>(true);
	let catalogError = $state<string | null>(null);

	// Elasticity State
	let elasticityData = $state<ProductElasticityResponse | null>(null);
	let elasticityLoading = $state<boolean>(false);
	let elasticityError = $state<string | null>(null);

	// Simulation State
	let simulationResult = $state<SimulationResponse | null>(null);
	let simulationLoading = $state<boolean>(false);
	let simulationError = $state<string | null>(null);

	// Optimization State
	let optimizationResult = $state<OptimizationResponse | null>(null);
	let optimizationLoading = $state<boolean>(false);
	let optimizationError = $state<string | null>(null);

	// Selected Product derived
	const selectedProduct = $derived(
		products.find((p) => p.product_id === selectedProductId) || products[0]
	);

	async function loadDashboard() {
		dashboardLoading = true;
		dashboardError = null;
		try {
			const [sum, tr, pr] = await Promise.all([
				fetchDashboardSummary(),
				fetchSalesTrend(),
				fetchDashboardPromotions()
			]);
			dashboardSummary = sum;
			dashboardSalesTrend = tr;
			dashboardPromotions = pr;
		} catch (err: any) {
			dashboardError = err?.detail || err?.message || 'Failed to load executive dashboard';
		} finally {
			dashboardLoading = false;
		}
	}

	async function loadCustomerData(hhKey: number) {
		selectedHouseholdKey = hhKey;
		customerLoading = true;
		upliftLoading = true;
		customerError = null;
		upliftError = null;

		fetchCustomerDetail(hhKey)
			.then((res) => {
				customerDetail = res;
			})
			.catch((err: any) => {
				customerError = err?.detail || err?.message || 'Failed to load customer profile';
			})
			.finally(() => {
				customerLoading = false;
			});

		fetchCustomerUplift(hhKey)
			.then((res) => {
				customerUplift = res;
			})
			.catch((err: any) => {
				upliftError = err?.detail || err?.message || 'Failed to evaluate customer uplift';
			})
			.finally(() => {
				upliftLoading = false;
			});

		if (!customerSegments && !segmentsLoading) {
			segmentsLoading = true;
			fetchCustomerSegments()
				.then((res) => {
					customerSegments = res;
				})
				.catch((err: any) => {
					segmentsError = err?.detail || err?.message || 'Failed to load segment deciles';
				})
				.finally(() => {
					segmentsLoading = false;
				});
		}
	}

	function handleSelectHousehold(hhKey: number) {
		loadCustomerData(hhKey);
	}

	async function initApp() {
		catalogLoading = true;
		backendErrorMessage = null;
		try {
			// 1. Health Check
			await fetchHealth();
			backendOnline = true;

			// 2. Load Dashboard & Customer data in background
			loadDashboard();
			loadCustomerData(1);

			// 3. Fetch Catalog
			const catResp = await fetchProducts();
			products = catResp.products;
			substitutePairs = catResp.substitute_pairs;
			catalogLoading = false;

			// 4. Default to product 5569230 if present, else first product
			const initialId = products.some((p) => p.product_id === 5569230)
				? 5569230
				: products[0]?.product_id;

			if (initialId) {
				selectedProductId = initialId;
				await loadProductDetails(initialId);
			}
		} catch (err: any) {
			backendOnline = false;
			catalogLoading = false;
			dashboardLoading = false;
			backendErrorMessage = err?.detail || err?.message || 'Failed to connect to backend';
			catalogError = backendErrorMessage;
			dashboardError = backendErrorMessage;
		}
	}

	async function loadProductDetails(productId: number) {
		loadElasticity(productId);
		loadDefaultSimulation(productId);
		loadDefaultOptimization(productId);
	}

	async function loadElasticity(productId: number) {
		elasticityLoading = true;
		elasticityError = null;
		try {
			elasticityData = await fetchProductElasticity(productId);
		} catch (err: any) {
			elasticityError = err?.detail || err?.message || 'Failed to fetch elasticity data';
		} finally {
			elasticityLoading = false;
		}
	}

	async function loadDefaultSimulation(productId: number) {
		await handleRunSimulation({
			product_id: productId,
			discount_pct: 0.20,
			with_display: false,
			baseline_mode: 'observed_merch',
			include_substitute: true
		});
	}

	async function loadDefaultOptimization(productId: number) {
		await handleRunOptimization({
			product_id: productId,
			with_display: false,
			baseline_mode: 'observed_merch',
			min_discount: 0.05,
			max_discount: 0.35,
			discount_step: 0.025,
			include_substitute: true
		});
	}

	function handleSelectProduct(productId: number) {
		selectedProductId = productId;
		loadProductDetails(productId);
	}

	async function handleRunSimulation(req: SimulationRequest) {
		simulationLoading = true;
		simulationError = null;
		try {
			simulationResult = await simulatePromotion(req);
		} catch (err: any) {
			simulationError = err?.detail || err?.message || 'Failed to run simulation';
		} finally {
			simulationLoading = false;
		}
	}

	async function handleRunOptimization(req: OptimizationRequest) {
		optimizationLoading = true;
		optimizationError = null;
		try {
			optimizationResult = await optimizePromotion(req);
		} catch (err: any) {
			optimizationError = err?.detail || err?.message || 'Failed to run optimization';
		} finally {
			optimizationLoading = false;
		}
	}

	function handleTabChange(tab: string) {
		activeTab = tab;
	}

	onMount(() => {
		initApp();
	});
</script>

<div class="app-layout">
	<!-- Top Navigation -->
	<Navbar
		{activeTab}
		{backendOnline}
		onTabChange={handleTabChange}
	/>

	<!-- Main Content Area -->
	<main class="container main-content">
		<!-- Backend Offline Warning Banner -->
		{#if backendOnline === false}
			<div class="alert alert-danger offline-banner" role="alert">
				<div class="alert-icon">🔌</div>
				<div class="offline-details">
					<div class="alert-title">PromoSense Backend Disconnected</div>
					<p>
						Could not establish connection to the FastAPI backend at <code>http://localhost:8000/api</code>.
						{backendErrorMessage || 'Please verify that the FastAPI backend service is running.'}
					</p>
					<button type="button" class="btn btn-secondary btn-sm mt-2" onclick={initApp}>
						🔄 Retry Connection
					</button>
				</div>
			</div>
		{/if}

		<!-- Active Product Bar (Visible in Tabs: Elasticity, Simulator, Optimizer) -->
		{#if (activeTab === 'elasticity' || activeTab === 'simulator' || activeTab === 'optimizer') && selectedProduct}
			<div class="active-sku-banner">
				<div class="sku-left">
					<span class="active-label">Active Analysis SKU:</span>
					<span class="sku-number">SKU {selectedProduct.product_id}</span>
					<span class="sku-name">{selectedProduct.commodity} &bull; {selectedProduct.sub_commodity}</span>
					<span class="badge {selectedProduct.brand_type === 'National' ? 'badge-primary' : 'badge-neutral'}">
						{selectedProduct.brand_type}
					</span>
					<span class="price-chip">Base List: ${selectedProduct.baseline_shelf_price.toFixed(2)}</span>
				</div>

				<div class="sku-right">
					<label for="sku-quick-select" class="quick-label">Switch SKU:</label>
					<select
						id="sku-quick-select"
						value={selectedProductId}
						onchange={(e) => handleSelectProduct(parseInt(e.currentTarget.value, 10))}
						class="sku-dropdown"
					>
						{#each products as p}
							<option value={p.product_id}>
								{p.product_id} — {p.commodity} ({p.brand_type})
							</option>
						{/each}
					</select>
				</div>
			</div>
		{/if}

		<!-- Tab 1: Executive Dashboard -->
		{#if activeTab === 'dashboard'}
			<ExecutiveDashboard
				summary={dashboardSummary}
				salesTrend={dashboardSalesTrend}
				promotions={dashboardPromotions}
				loading={dashboardLoading}
				error={dashboardError}
			/>
		{/if}

		<!-- Tab 2: Product Catalog -->
		{#if activeTab === 'catalog'}
			{#if catalogLoading}
				<div class="card loading-card">
					<div class="spinner"></div>
					<p>Loading 7 modeled target products and substitute pairs...</p>
				</div>
			{:else if catalogError && products.length === 0}
				<div class="alert alert-danger" role="alert">
					<div class="alert-icon">⚠️</div>
					<div>
						<div class="alert-title">Catalog Failed to Load</div>
						<p>{catalogError}</p>
						<button type="button" class="btn btn-primary btn-sm mt-2" onclick={initApp}>Retry</button>
					</div>
				</div>
			{:else}
				<ProductCatalog
					{products}
					{substitutePairs}
					{selectedProductId}
					onSelectProduct={(id: number) => {
						handleSelectProduct(id);
						activeTab = 'elasticity';
					}}
				/>
			{/if}
		{/if}

		<!-- Tab 3: Price Elasticity -->
		{#if activeTab === 'elasticity'}
			<ElasticityViewer
				{elasticityData}
				isLoading={elasticityLoading}
				error={elasticityError}
				onSelectProduct={handleSelectProduct}
			/>
		{/if}

		<!-- Tab 4: Promotion Simulator -->
		{#if activeTab === 'simulator'}
			{#if selectedProduct}
				<SimulationStudio
					product={selectedProduct}
					{simulationResult}
					isLoading={simulationLoading}
					error={simulationError}
					onRunSimulation={handleRunSimulation}
				/>
			{:else}
				<div class="card empty-card">
					<p>Please select a product from the catalog to run promotions simulation.</p>
				</div>
			{/if}
		{/if}

		<!-- Tab 5: Profit Optimizer -->
		{#if activeTab === 'optimizer'}
			{#if selectedProduct}
				<OptimizationStudio
					product={selectedProduct}
					{optimizationResult}
					isLoading={optimizationLoading}
					error={optimizationError}
					onRunOptimization={handleRunOptimization}
				/>
			{:else}
				<div class="card empty-card">
					<p>Please select a product from the catalog to solve optimal promotions.</p>
				</div>
			{/if}
		{/if}

		<!-- Tab 6: Customer Intelligence -->
		{#if activeTab === 'customers'}
			<CustomerIntelligence
				customer={customerDetail}
				uplift={customerUplift}
				segments={customerSegments}
				loadingCustomer={customerLoading}
				loadingUplift={upliftLoading}
				loadingSegments={segmentsLoading}
				customerError={customerError}
				upliftError={upliftError}
				segmentsError={segmentsError}
				onHouseholdChange={handleSelectHousehold}
			/>
		{/if}
	</main>

	<!-- Methodological Footer -->
	<footer class="app-footer">
		<div class="container footer-content">
			<div class="footer-meta">
				<strong>PromoSense Decision-Support Engine v1.0.0</strong> &bull;
				<span>Frozen Phase 6 Econometric Models & Phase 7 Optimization Core</span>
			</div>
			<div class="footer-disclaimer">
				<strong>Methodological Disclosures:</strong>
				(1) <em>Observational Data</em>: Models reflect conditional regressions on scanner history; estimates should not be construed as randomized controlled trial causal effects.
				(2) <em>Cost Basis Assumption</em>: Scanner data omit wholesale acquisition cost; unit cost is calculated as <code>c = P_shelf * (1 - m_base)</code> across configurable 15%, 25%, and 35% margin tiers.
				(3) <em>Identification Caution</em>: Fluid Milk cross-price terms exhibit weak empirical identification due to store-wide synchronous pricing.
			</div>
		</div>
	</footer>
</div>

<style>
	.app-layout {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
		background-color: var(--color-bg);
	}

	.main-content {
		flex: 1;
		padding-top: 1.5rem;
		padding-bottom: 3rem;
	}

	.offline-banner {
		margin-bottom: 1.5rem;
		border-left: 4px solid var(--color-danger);
	}

	.offline-details {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.active-sku-banner {
		background-color: #ffffff;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		padding: 0.75rem 1.25rem;
		margin-bottom: 1.5rem;
		display: flex;
		justify-content: space-between;
		align-items: center;
		box-shadow: var(--shadow-sm);
		flex-wrap: wrap;
		gap: 0.75rem;
	}

	.sku-left {
		display: flex;
		align-items: center;
		gap: 0.65rem;
		flex-wrap: wrap;
	}

	.active-label {
		font-size: 0.75rem;
		font-weight: 700;
		color: var(--color-text-muted);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.sku-number {
		font-family: var(--font-mono);
		font-weight: 800;
		font-size: 0.875rem;
		color: var(--color-primary);
	}

	.sku-name {
		font-size: 0.8125rem;
		font-weight: 600;
		color: var(--color-text-main);
	}

	.price-chip {
		font-family: var(--font-mono);
		font-size: 0.75rem;
		background-color: #f1f5f9;
		padding: 0.15rem 0.45rem;
		border-radius: var(--radius-sm);
		color: var(--color-text-muted);
	}

	.sku-right {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.quick-label {
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.sku-dropdown {
		padding: 0.35rem 0.65rem;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		font-size: 0.75rem;
		background-color: #ffffff;
		color: var(--color-text-main);
		font-weight: 600;
	}

	.loading-card,
	.empty-card {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 3.5rem;
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

	.mt-2 {
		margin-top: 0.5rem;
	}

	.app-footer {
		background-color: #ffffff;
		border-top: 1px solid var(--color-border);
		padding: 1.5rem 0;
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.footer-content {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
	}

	.footer-meta {
		color: var(--color-text-main);
	}

	.footer-disclaimer {
		line-height: 1.5;
		color: var(--color-text-subtle);
		background-color: #f8fafc;
		padding: 0.65rem 0.85rem;
		border-radius: var(--radius-sm);
		border: 1px solid #f1f5f9;
	}

	@media (max-width: 900px) {
		.active-sku-banner {
			flex-direction: column;
			align-items: stretch;
		}

		.sku-right {
			justify-content: space-between;
		}
	}
</style>
