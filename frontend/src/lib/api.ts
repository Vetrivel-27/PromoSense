/**
 * Dedicated Typed API Client for PromoSense Decision-Support Backend.
 * Handles environment-based configuration, error parsing, and typed request/responses.
 */

import type {
	CustomerDetailResponse,
	CustomerSegmentsResponse,
	CustomerUpliftResponse,
	DashboardPromotionsResponse,
	DashboardSalesTrendResponse,
	DashboardSummaryResponse,
	OptimizationRequest,
	OptimizationResponse,
	ProductCatalogResponse,
	ProductDetailResponse,
	ProductElasticityResponse,
	SimulationRequest,
	SimulationResponse
} from './types';

export class ApiError extends Error {
	status: number;
	detail: string;

	constructor(status: number, detail: string) {
		super(`API Error (${status}): ${detail}`);
		this.name = 'ApiError';
		this.status = status;
		this.detail = detail;
	}
}

/**
 * Resolves API Base URL from Vite environment or falls back to local FastAPI server.
 */
export function getApiBaseUrl(): string {
	// In browser/Vite environment, use VITE_API_URL if configured
	if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL) {
		return import.meta.env.VITE_API_URL;
	}
	return 'http://localhost:8000/api';
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl.replace(/\/+$/, '')}/${endpoint.replace(/^\/+/, '')}`;

	const headers: Record<string, string> = {
		Accept: 'application/json',
		...(options.headers as Record<string, string>)
	};

	if (options.body && typeof options.body === 'string') {
		headers['Content-Type'] = 'application/json';
	}

	let response: Response;
	try {
		response = await fetch(url, { ...options, headers });
	} catch (err) {
		throw new ApiError(
			0,
			`Failed to connect to PromoSense backend at ${url}. Ensure FastAPI backend is running.`
		);
	}

	if (!response.ok) {
		let errorDetail = `Request failed with status ${response.status}`;
		try {
			const errorJson = await response.json();
			if (errorJson.detail) {
				if (typeof errorJson.detail === 'string') {
					errorDetail = errorJson.detail;
				} else if (Array.isArray(errorJson.detail)) {
					// Pydantic validation errors
					errorDetail = errorJson.detail
						.map((d: { msg?: string; loc?: string[] }) => `${d.loc?.join('.') || 'param'}: ${d.msg}`)
						.join('; ');
				} else {
					errorDetail = JSON.stringify(errorJson.detail);
				}
			}
		} catch {
			// Response was not JSON
		}
		throw new ApiError(response.status, errorDetail);
	}

	return (await response.json()) as T;
}

/**
 * Health check endpoint.
 */
export async function fetchHealth(): Promise<{
	status: string;
	environment: string;
	model_registry_loaded: boolean;
	target_skus_count: number;
	substitute_pairs_count: number;
}> {
	return request('health', { method: 'GET' });
}

/**
 * Lists all target volume product SKUs and substitute pairs.
 */
export async function fetchProducts(): Promise<ProductCatalogResponse> {
	return request<ProductCatalogResponse>('products', { method: 'GET' });
}

/**
 * Retrieves frozen price elasticity estimates and confidence intervals for a SKU.
 */
export async function fetchProductElasticity(productId: number): Promise<ProductElasticityResponse> {
	return request<ProductElasticityResponse>(`products/${productId}/elasticity`, {
		method: 'GET'
	});
}

/**
 * Simulates counterfactual promotion scenario (volume, revenue, profit lift).
 */
export async function simulatePromotion(req: SimulationRequest): Promise<SimulationResponse> {
	return request<SimulationResponse>('simulate/promotion', {
		method: 'POST',
		body: JSON.stringify(req)
	});
}

/**
 * Computes profit-maximizing promotional discount depths across gross margin tiers.
 */
export async function optimizePromotion(req: OptimizationRequest): Promise<OptimizationResponse> {
	return request<OptimizationResponse>('optimize/promotion', {
		method: 'POST',
		body: JSON.stringify(req)
	});
}

/**
 * Retrieves detailed product metadata and baseline metrics for a single SKU.
 */
export async function fetchProductDetail(productId: number): Promise<ProductDetailResponse> {
	return request<ProductDetailResponse>(`products/${productId}`, { method: 'GET' });
}

/**
 * Retrieves chain-level portfolio summary facts and margin assumptions.
 */
export async function fetchDashboardSummary(): Promise<DashboardSummaryResponse> {
	return request<DashboardSummaryResponse>('dashboard/summary', { method: 'GET' });
}

/**
 * Retrieves 102-week chain-level total and promotional sales time-series.
 */
export async function fetchSalesTrend(): Promise<DashboardSalesTrendResponse> {
	return request<DashboardSalesTrendResponse>('dashboard/sales-trend', { method: 'GET' });
}

/**
 * Retrieves marketing campaign portfolio reach and causal ATE metrics.
 */
export async function fetchDashboardPromotions(): Promise<DashboardPromotionsResponse> {
	return request<DashboardPromotionsResponse>('dashboard/promotions', { method: 'GET' });
}

/**
 * Retrieves customer household demographic and shopping behavioral profile.
 */
export async function fetchCustomerDetail(householdKey: number): Promise<CustomerDetailResponse> {
	return request<CustomerDetailResponse>(`customers/${householdKey}`, { method: 'GET' });
}

/**
 * Evaluates individual customer CATE uplift score and targeting recommendation.
 */
export async function fetchCustomerUplift(
	householdKey: number,
	campaignId?: number
): Promise<CustomerUpliftResponse> {
	const query = campaignId !== undefined ? `?campaign_id=${campaignId}` : '';
	return request<CustomerUpliftResponse>(`customers/${householdKey}/uplift${query}`, {
		method: 'GET'
	});
}

/**
 * Retrieves Qini curve deciles, AUUC metrics, and recommended targeting cutoffs.
 */
export async function fetchCustomerSegments(): Promise<CustomerSegmentsResponse> {
	return request<CustomerSegmentsResponse>('customers/segments', { method: 'GET' });
}
