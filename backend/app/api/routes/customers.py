"""Customer and Uplift Intelligence API routes for PromoSense Backend."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.schemas.customers import (
    CustomerDetailResponse,
    CustomerSegmentsResponse,
    CustomerUpliftResponse,
)
from backend.app.services.customer_service import CustomerService

router = APIRouter(prefix="/customers", tags=["Customer Intelligence & Uplift"])


def get_customer_service() -> CustomerService:
    return CustomerService.get_instance()


@router.get(
    "/segments",
    response_model=CustomerSegmentsResponse,
    summary="Get Qini deciles, AUUC metrics, and campaign targeting cutoffs",
    description="Returns cumulative gain progression across population deciles and recommended targeting strategy.",
)
def get_customer_segments(
    service: CustomerService = Depends(get_customer_service),
) -> CustomerSegmentsResponse:
    try:
        return service.get_customer_segments()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customer segments: {e}",
        ) from e


@router.get(
    "/{household_key}",
    response_model=CustomerDetailResponse,
    summary="Get customer household demographic profile and transaction history",
    description="Retrieves census demographic disclosures (when disclosed) and 102-week shopping activity.",
)
def get_customer_detail(
    household_key: int,
    service: CustomerService = Depends(get_customer_service),
) -> CustomerDetailResponse:
    try:
        return service.get_customer_detail(household_key)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customer detail for household {household_key}: {e}",
        ) from e


@router.get(
    "/{household_key}/uplift",
    response_model=CustomerUpliftResponse,
    summary="Get household CATE uplift score and targeting recommendation",
    description="Evaluates the frozen champion meta-learner to project incremental spend lift and optimal marketing action.",
)
def get_customer_uplift(
    household_key: int,
    campaign_id: int | None = Query(None, description="Optional marketing campaign context ID"),
    service: CustomerService = Depends(get_customer_service),
) -> CustomerUpliftResponse:
    try:
        return service.get_customer_uplift(household_key, campaign_id=campaign_id)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate customer uplift for household {household_key}: {e}",
        ) from e
