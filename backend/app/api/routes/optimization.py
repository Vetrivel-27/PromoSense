"""Promotion optimization API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.optimization import (
    OptimizationRequest,
    OptimizationResponse,
)
from backend.app.services.optimization_service import OptimizationService

router = APIRouter(tags=["Optimization"])


def get_optimization_service() -> OptimizationService:
    return OptimizationService()


@router.post(
    "/optimize/promotion",
    response_model=OptimizationResponse,
    summary="Optimize promotional discount depth for maximum contribution profit",
    description=(
        "Executes bounded grid search over candidate discounts [0.05, 0.35] against an explicit no-promotion baseline. "
        "Returns profit-maximizing discount depth and category net optimal policies accounting for substitute cannibalization."
    ),
)
def optimize_promotion(
    request: OptimizationRequest,
    service: OptimizationService = Depends(get_optimization_service),
) -> OptimizationResponse:
    try:
        return service.optimize_promotion(request)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {e}",
        ) from e
