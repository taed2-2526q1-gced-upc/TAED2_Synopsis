from http import HTTPStatus
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.backend.app.main import app

from src.backend.app.api.endpoints.summarize import MAX_INPUT_SIZE, MIN_INPUT_SIZE, MAX_OUTPUT_SIZE
from src.backend.app.services.news_scraper import NewsScraper

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = TESTS_DIR / "test_data"


@pytest.fixture(scope="module", autouse=True)
def scraper():
    """Fixture that returns a NewsScraper instance."""
    return NewsScraper()


@pytest.fixture
def valid_article():
    """Returns a valid article for testing."""
    return scraper.scrape_news("https://www.example.com/valid-article").get("text", "")


@pytest.fixture
def short_article():
    """Returns an article that's too short."""
    return "This is too short."


@pytest.fixture
def long_article():
    """Returns an article that exceeds the maximum length."""
    return "x" * (MAX_INPUT_SIZE + 1000)


@pytest.fixture(scope="module", autouse=True)
def client():
    """Fixture that returns a TestClient for the FastAPI app."""
    # Use the TestClient with a `with` statement to trigger the lifespan events.
    with TestClient(app) as client:
       yield client


def test_root(client):
    """Test the root endpoint."""
    response = client.get("/api/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["message"] == "Welcome to Synopsis API"
    assert json["service"] == "Synopsis API"


def test_health(client):
    """Test the health endpoint."""
    response = client.get("/api/health/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["status"] == "ok"
    assert json["message"] == "Synopsis API is healthy"


def test_summarize_success(client):
    """Test successful article summarization with a real article."""
    response = client.post("/api/summarize/", json={"url": "https://www.bbc.com/news/world-europe-67121924"})
    
    assert response.status_code == HTTPStatus.OK
    json = response.json()
    assert json["status"] == "ok"
    assert json["title"] is not None
    assert json["summary"] is not None
    assert json["full_article"] is not None
    # Verify length constraints
    assert len(json["full_article"]) >= MIN_INPUT_SIZE
    assert len(json["full_article"]) <= MAX_INPUT_SIZE
    assert len(json["summary"]) <= MAX_OUTPUT_SIZE

def test_summarize_invalid_url(client):
    """Test error handling with invalid URL."""
    response = client.post("/api/summarize/", json={"url": "not-a-url"})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

def test_summarize_nonexistent_url(client):
    """Test error handling with URL that doesn't exist."""
    response = client.post("/api/summarize/", json={"url": "https://www.bbc.com/news/nonexistent-article-12345"})
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

def test_summarize_not_news_url(client):
    """Test error handling with URL that's not a news article."""
    response = client.post("/api/summarize/", json={"url": "https://www.google.com"})
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

