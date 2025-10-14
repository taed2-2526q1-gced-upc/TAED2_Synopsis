from fastapi import APIRouter
from loguru import logger

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    logger.info("[BACKEND] Health check endpoint accessed")

    return {"status": "ok", "message": "Synopsis API is healthy"}
