"""FastAPI Application entrypoint for PromoSense Decision-Support Platform."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import customers, dashboard, optimization, products, simulation
from backend.app.core.config import settings
from backend.app.services.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle.

    Loads and validates frozen model artifacts once at startup.
    Raises explicit RuntimeError if required model artifacts are missing.
    """
    registry = ModelRegistry.get_instance()
    try:
        registry.load_artifacts()
    except Exception as e:
        # Re-raise to halt startup cleanly with descriptive error
        raise RuntimeError(f"PromoSense failed to initialize model registry at startup: {e}") from e
    yield


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )

    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Core status and health check endpoints
    @app.get("/")
    def root():
        return {
            "name": settings.PROJECT_NAME,
            "environment": settings.ENVIRONMENT,
            "status": "online",
        }

    @app.get(f"{settings.API_V1_PREFIX}/health")
    def health_check():
        registry = ModelRegistry.get_instance()
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "model_registry_loaded": registry.is_initialized,
            "target_skus_count": len(registry.target_skus),
            "substitute_pairs_count": len(registry.substitute_pairs),
            "mlflow_status": registry.get_mlflow_status(),
        }

    # Include decision-support API routers
    app.include_router(dashboard.router, prefix=settings.API_V1_PREFIX)
    app.include_router(products.router, prefix=settings.API_V1_PREFIX)
    app.include_router(customers.router, prefix=settings.API_V1_PREFIX)
    app.include_router(simulation.router, prefix=settings.API_V1_PREFIX)
    app.include_router(optimization.router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()
