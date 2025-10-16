from fastapi import APIRouter, HTTPException
from typing import Dict
from loguru import logger
from pydantic import BaseModel
from services.analyzer import analyzer

router = APIRouter()

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    status: str
    probabilities: Dict[str, float]

@router.post("/", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze the sentiment of the article."""
    logger.info("[BACKEND] Analyze endpoint accessed")
    try:
        if not isinstance(request.text, str):
            logger.error(f"[BACKEND] Input text has wrong type: {type(request.text)}")
            raise HTTPException(status_code=400, detail="Invalid input: text must be a string.")

        logger.info("[BACKEND] Analyzing text sentiment...")

        analysis = analyzer(request.text)
        if not isinstance(analysis, list) or not analysis or not isinstance(analysis[0], list):
            logger.error(f"[BACKEND] Analyzer returned invalid output: {analysis}")
            raise HTTPException(status_code=500, detail="An internal error occurred, analyzer output is invalid. Please try again later.")

        emotions = {item['label']: item['score'] for item in analysis[0]}
        
        logger.info("[BACKEND] Text analyzed successfully")
        logger.info(f"[BACKEND] Analysis probabilities: {emotions}")

        return AnalyzeResponse(
            status="ok",
            probabilities=emotions
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[BACKEND] Exception in analyze: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
