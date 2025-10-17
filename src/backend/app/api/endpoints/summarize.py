from fastapi import APIRouter, HTTPException
from typing import Any
from loguru import logger
from pydantic import BaseModel
from services.news_scraper import NewsScraper
from services.summarizer import summarizer

# Constants for model constraints
MAX_INPUT_SIZE = 4000
MAX_OUTPUT_SIZE = 2000
MIN_INPUT_SIZE = 150
MAX_TOKENS = min(MAX_OUTPUT_SIZE // 4, 430)
MIN_TOKENS = 30

# Request and response models
class SummaryRequest(BaseModel):
    url: str

class SummaryResponse(BaseModel):
    status: str
    title: str
    summary: str
    full_article: str

Scraper = NewsScraper()
router = APIRouter()

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

        article_length = len(result['text'])
     
        if article_length < MIN_INPUT_SIZE:
            logger.error(f"[BACKEND] Article too short: {article_length} characters")
            raise HTTPException(
                status_code=400,
                detail=f"Article is too short ({article_length} characters). Minimum required size is {MIN_INPUT_SIZE} characters."
            )

        logger.info("[BACKEND] News article scraped successfully")
        logger.info(f"[BACKEND] Scraped article: {result}")
        logger.info("[BACKEND] Summarizing article...")

        summary = summarizer(result['text'], max_length=MAX_TOKENS, min_length=MIN_TOKENS, do_sample=False, truncation=True)

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
