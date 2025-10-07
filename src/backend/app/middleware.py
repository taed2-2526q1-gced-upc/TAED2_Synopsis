import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.backend.app.config import settings


def add_cors_middleware(app: FastAPI) -> None:
    """Add CORS middleware to the FastAPI app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("[BACKEND] CORS middleware added")


def add_trusted_host_middleware(app: FastAPI) -> None:
    """Add trusted host middleware to the FastAPI app."""
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
    logger.info("[BACKEND] Trusted host middleware added")


def add_request_logging_middleware(app: FastAPI) -> None:
    """Add request logging middleware to the FastAPI app."""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests and responses."""
        start_time = time.time()

        # Log request
        logger.info(f"[BACKEND] Request made: {request.method} {request.url}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"[BACKEND] Received response: {response.status_code} - {process_time:.4f}s"
        )

        return response

    logger.info("[BACKEND] Request logging middleware added")


def add_global_exception_handler(app: FastAPI) -> None:
    """Add global exception handler to the FastAPI app."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(f"[BACKEND] Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": id(request),
            },
        )

    logger.info("[BACKEND] Global exception handler added")


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the FastAPI app."""
    logger.info("[BACKEND] Setting up middleware...")

    add_cors_middleware(app)
    add_trusted_host_middleware(app)
    add_request_logging_middleware(app)
    add_global_exception_handler(app)

    logger.info("[BACKEND] All middleware configured successfully")
