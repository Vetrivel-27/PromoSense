"""Product catalog and price elasticity API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.products import (
    ProductCatalogResponse,
    ProductDetailResponse,
    ProductElasticityResponse,
)
from backend.app.services.catalog_service import CatalogService

router = APIRouter(tags=["Products & Elasticity"])


def get_catalog_service() -> CatalogService:
    return CatalogService()


@router.get(
    "/products",
    response_model=ProductCatalogResponse,
    summary="List target products and substitute category pairs",
    description="Returns all modeled target volume product SKUs, baseline prices, merchandising shares, and substitute pairs.",
)
def list_products(
    service: CatalogService = Depends(get_catalog_service),
) -> ProductCatalogResponse:
    try:
        return service.get_catalog()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve product catalog: {e}",
        ) from e


@router.get(
    "/products/{product_id}",
    response_model=ProductDetailResponse,
    summary="Get detailed product metadata and empirical baseline stats",
    description="Retrieves comprehensive SKU metadata, price distributions, baseline sales metrics, and substitute pairs.",
)
def get_product(
    product_id: int,
    service: CatalogService = Depends(get_catalog_service),
) -> ProductDetailResponse:
    try:
        return service.get_product_detail(product_id)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product SKU {product_id} not found in modeled catalog.",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve product {product_id}: {e}",
        ) from e


@router.get(
    "/products/{product_id}/elasticity",
    response_model=ProductElasticityResponse,
    summary="Get product own- and cross-price elasticity estimates",
    description="Retrieves the frozen champion elasticity specification, HAC standard error, 95% confidence intervals, and cross-price metrics.",
)
def get_product_elasticity(
    product_id: int,
    service: CatalogService = Depends(get_catalog_service),
) -> ProductElasticityResponse:
    try:
        return service.get_product_elasticity(product_id)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product SKU {product_id} not found in modeled catalog.",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve elasticity for product {product_id}: {e}",
        ) from e
