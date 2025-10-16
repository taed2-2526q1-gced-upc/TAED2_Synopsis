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

class SummaryResponse(BaseModel):
    status: str
    title: str
    message: str

@router.post("/")
async def summarize(request: SummaryRequest):
    """Summarize a news article."""
    logger.info("[BACKEND] Summarize endpoint accessed")

    result = Scraper.scrape_news(request.url)

    logger.info("[BACKEND] News article scraped successfully")
    logger.info(f"[BACKEND] Scraped article: {result}")

    logger.info("[BACKEND] Summarizing article...")
    summary = model.summarizer(result['text'], max_length=430, min_length=30, do_sample=False)
    logger.info("[BACKEND] Article summarized successfully")
    logger.info(f"[BACKEND] Summary: {summary}")

    return SummaryResponse(status="ok", title=result['title'], message=summary[0]["summary_text"])

    # return {"status": "ok", "message": f"{result['title']} - {result['text']}"}
