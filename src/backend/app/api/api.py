from fastapi import APIRouter
from loguru import logger

from src.backend.app.api.endpoints import health, summarize
from src.backend.app.config import settings

# Create the main API router
api_router = APIRouter()


# Root endpoint
@api_router.get("/")
async def root():
    """Root endpoint that provides API information."""
    logger.info("[BACKEND] Root endpoint accessed")

    api_info = {
        "message": "Welcome to Synopsis API",
        "service": settings.PROJECT_NAME,
        "endpoints": {
            "health": "/health",
        },
    }

    return api_info


# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(summarize.router, prefix="/summarize", tags=["summarize"])

logger.info("[BACKEND] API router configured with endpoints")
