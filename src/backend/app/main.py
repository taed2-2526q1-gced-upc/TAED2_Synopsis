from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
import uvicorn

from app.api.api import api_router
from app.config import settings
from app.middleware import setup_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"[BACKEND] Starting Synopsis API at {settings.HOST}:{settings.PORT}")

    try:
        # Models and DB will be initialized here
        logger.info("[BACKEND] Application startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"[BACKEND] Error during startup: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("[BACKEND] Shutting down Synopsis API...")
        logger.info("[BACKEND] Application shutdown completed")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="An AI-powered news summarization tool that extracts content from article URLs.",
    lifespan=lifespan,
)

# Middleware + router
setup_middleware(app)
app.include_router(api_router, prefix=settings.API_PREFIX)

logger.info("[BACKEND] Middleware and FastAPI application configured successfully")


if __name__ == "__main__":
    logger.info("[BACKEND] Starting backend application with uvicorn...")
    uvicorn.run(
        app, host=settings.HOST, port=settings.PORT, reload=settings.DEBUG, log_level="info"
    )
