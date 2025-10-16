from fastapi import APIRouter, HTTPException
from typing import Any
from loguru import logger
from pydantic import BaseModel
from services.news_scraper import NewsScraper
from services.models import summarizer

Scraper = NewsScraper()

router = APIRouter()

class SummaryRequest(BaseModel):
    url: str

class SummaryResponse(BaseModel):
    status: str
    title: str
    summary: str
    full_article: str

@router.post("/", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    """Summarize a news article given its URL."""
    logger.info("[BACKEND] Summarize endpoint accessed")
    try:
        result = Scraper.scrape_news(request.url)
        if not isinstance(result, dict):
            logger.error(f"[BACKEND] Scraper did not return a dict: {result}")
            raise HTTPException(status_code=500, detail="An internal error occurred, invalid scraper result type. Please try again later.")
        if 'text' not in result or 'title' not in result:
            logger.error(f"[BACKEND] Scraped result is missing keys: {result}")
            raise HTTPException(status_code=500, detail="An internal error occurred, the scraped article is missing information. Please try again later.")
        if not isinstance(result['text'], str) or not isinstance(result['title'], str):
            logger.error(f"[BACKEND] Scraped fields have wrong types: {result}")
            raise HTTPException(status_code=500, detail="An internal error occurred, scraped fields have wrong types. Please try again later.")

        logger.info("[BACKEND] News article scraped successfully")
        logger.info(f"[BACKEND] Scraped article: {result}")
        logger.info("[BACKEND] Summarizing article...")

        summary = summarizer(result['text'], max_length=430, min_length=30, do_sample=False)
        if not isinstance(summary, list) or not summary or 'summary_text' not in summary[0]:
            logger.error(f"[BACKEND] Summarizer returned invalid output: {summary}")
            raise HTTPException(status_code=500, detail="An internal error occurred, summarizer output is invalid. Please try again later.")

        logger.info("[BACKEND] Article summarized successfully")
        logger.info(f"[BACKEND] Summary: {summary}")

        return SummaryResponse(
            status="ok",
            title=result['title'],
            summary=summary[0]["summary_text"],
            full_article=result['text']
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[BACKEND] Exception in summarize: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
