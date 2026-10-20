"""Promotion simulation API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.simulation import SimulationRequest, SimulationResponse
from backend.app.services.simulation_service import SimulationService

router = APIRouter(tags=["Simulation"])


def get_simulation_service() -> SimulationService:
    return SimulationService()


@router.post(
    "/simulate/promotion",
    response_model=SimulationResponse,
    summary="Simulate promotional demand, revenue, and contribution profit",
    description=(
        "Simulates counterfactual volume lift (Delta Q), revenue lift (Delta R), and contribution profit (Delta Pi) "
        "across 15%, 25%, and 35% margin tiers. Permits discount_pct=0.0 for direct unpromoted baseline comparison."
    ),
)
def simulate_promotion(
    request: SimulationRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> SimulationResponse:
    try:
        return service.simulate_promotion(request)
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
            detail=f"Simulation failed: {e}",
        ) from e
