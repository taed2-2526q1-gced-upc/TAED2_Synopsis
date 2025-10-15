from fastapi import APIRouter
from typing import Any
from loguru import logger
from pydantic import BaseModel
from services.news_scraper import NewsScraper
import services.model as model

Scraper = NewsScraper()

router = APIRouter()

class SummaryRequest(BaseModel):
    url: str


@router.post("/")
async def summarize(request: SummaryRequest):
    """Summarize a news article."""
    logger.info("[BACKEND] Summarize endpoint accessed")

    result = Scraper.scrape_news(request.url)
    print(result)

    summary = model.summarizer(f"{result['title']} - {result['text']}", max_length=430, min_length=30, do_sample=False)

    return {"status": "ok", "message": summary[0]["summary_text"]}

    # return {"status": "ok", "message": f"{result['title']} - {result['text']}"}
