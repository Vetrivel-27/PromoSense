"""Dashboard and Portfolio API routes for PromoSense Executive Platform."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.dashboard import (
    DashboardPromotionsResponse,
    DashboardSalesTrendResponse,
    DashboardSummaryResponse,
)
from backend.app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/dashboard", tags=["Executive Dashboard"])


def get_dashboard_service() -> DashboardService:
    return DashboardService.get_instance()


@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
    summary="Get chain-wide executive sales summary and portfolio facts",
    description="Returns aggregate 102-week sales, promotional shares, customer counts, and margin-tier caveats.",
)
def get_dashboard_summary(
    service: DashboardService = Depends(get_dashboard_service),
) -> DashboardSummaryResponse:
    try:
        return service.get_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate dashboard summary: {e}",
        ) from e


@router.get(
    "/sales-trend",
    response_model=DashboardSalesTrendResponse,
    summary="Get weekly total and promotional sales time-series",
    description="Returns 102-week chain-level trend separating base sales from promotional lift.",
)
def get_sales_trend(
    service: DashboardService = Depends(get_dashboard_service),
) -> DashboardSalesTrendResponse:
    try:
        return service.get_sales_trend()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sales trend: {e}",
        ) from e


@router.get(
    "/promotions",
    response_model=DashboardPromotionsResponse,
    summary="Get marketing campaign summary and causal ATE benchmarks",
    description="Returns reach, durations, and causal uplift estimates across all 30 Dunnhumby marketing campaigns.",
)
def get_promotions(
    service: DashboardService = Depends(get_dashboard_service),
) -> DashboardPromotionsResponse:
    try:
        return service.get_promotions()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve promotions summary: {e}",
        ) from e
