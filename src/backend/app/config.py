from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()


class Settings:
    """Application settings."""

    # API Settings
    API_PREFIX: str = "/api"
    PROJECT_NAME: str = "Synopsis API"
    DEBUG: bool = True

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    ALLOWED_HOSTS: list[str] = ["*"]


settings = Settings()

logger.info("[BACKEND] API configuration successfully loaded")
