from http import HTTPStatus
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.backend.app.main import app
from src.backend.app.services.news_scraper import NewsScraper


@pytest.fixture
def valid_article():
    """Returns a valid article for testing."""
    scraper = NewsScraper()
    return scraper.scrape_news(
        "https://edition.cnn.com/2025/10/16/europe/us-russia-budapest-talks-intl-hnk"
    ).get("text", "")


@pytest.fixture
def short_article():
    """Returns an article that's too short."""
    return "This is too short."


@pytest.fixture
def long_article():
    """Returns an article that exceeds the maximum length."""
    return "x" * (5000 + 1000)


@pytest.fixture(scope="module", autouse=True)
def client():
    """Fixture that returns a TestClient for the FastAPI app."""
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


def test_summarize_empty_url(client):
    """Test error handling with empty URL."""
    response = client.post("/api/summarize/", json={"url": ""})
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


def test_summarize_missing_url(client):
    """Test error handling with missing URL field."""
    response = client.post("/api/summarize/", json={})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_summarize_response_structure(client, valid_article):
    """Test that successful summarization returns the expected response structure."""
    response = client.post("/api/summarize/", json={"text": valid_article})

    if response.status_code == HTTPStatus.OK:
        json = response.json()
        assert isinstance(json, dict)
        assert all(
            key in json for key in ["status", "title", "summary", "full_article"]
        )
        assert json["status"] == "ok"
        assert isinstance(json["title"], str) and json["title"]
        assert isinstance(json["summary"], str) and json["summary"]
        assert isinstance(json["full_article"], str) and json["full_article"]

        assert len(json["summary"]) < len(json["full_article"])


def test_summarize_invalid_url_format(client):
    """Test error handling with malformed URL."""
    response = client.post("/api/summarize/", json={"url": "not-a-valid-url"})
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


def test_summarize_long_article(client, long_article):
    """Test that successful summarization returns the expected response structure from long article."""
    response = client.post("/api/summarize/", json={"text": long_article})

    if response.status_code == HTTPStatus.OK:
        json = response.json()
        assert isinstance(json, dict)
        assert all(
            key in json for key in ["status", "title", "summary", "full_article"]
        )
        assert json["status"] == "ok"
        assert isinstance(json["title"], str) and json["title"]
        assert isinstance(json["summary"], str) and json["summary"]
        assert isinstance(json["full_article"], str) and json["full_article"]

        assert len(json["summary"]) < len(json["full_article"])


def test_summarize_short_article(client, short_article):
    """Test that successful summarization returns the expected response structure from short article."""
    response = client.post("/api/summarize/", json={"text": short_article})

    if response.status_code == HTTPStatus.OK:
        json = response.json()
        assert isinstance(json, dict)
        assert all(
            key in json for key in ["status", "title", "summary", "full_article"]
        )
        assert json["status"] == "ok"
        assert isinstance(json["title"], str) and json["title"]
        assert isinstance(json["summary"], str) and json["summary"]
        assert isinstance(json["full_article"], str) and json["full_article"]

        assert len(json["summary"]) < len(json["full_article"])


def test_not_found_handling(client):
    """Test handling of non-existent endpoints."""
    response = client.get("/api/nonexistent/")
    assert response.status_code == HTTPStatus.NOT_FOUND
    json = response.json()
    assert "detail" in json


def test_method_not_allowed(client):
    """Test handling of incorrect HTTP methods."""
    response = client.get("/api/summarize/")
    assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED
    response = client.put("/api/summarize/")
    assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED


def test_analyze_success(client):
    """Test successful sentiment analysis with valid text."""
    test_text = "I am very happy today and excited about the future!"
    response = client.post("/api/analyze/", json={"text": test_text})
    assert response.status_code == HTTPStatus.OK

    json = response.json()
    assert isinstance(json, dict)
    assert "status" in json
    assert "probabilities" in json
    assert json["status"] == "ok"
    assert isinstance(json["probabilities"], dict)
    assert len(json["probabilities"]) > 0
    assert all(isinstance(v, float) for v in json["probabilities"].values())
    assert all(0 <= v <= 1 for v in json["probabilities"].values())


def test_analyze_empty_text(client):
    """Test error handling with empty text."""
    response = client.post("/api/analyze/", json={"text": ""})
    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_analyze_missing_text(client):
    """Test error handling with missing text field."""
    response = client.post("/api/analyze/", json={})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
