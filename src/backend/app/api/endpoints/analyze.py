from fastapi import APIRouter, HTTPException
from typing import Dict
from loguru import logger
from pydantic import BaseModel
from services.analyzer import analyzer

# Constants for model constraints
MAX_INPUT_SIZE = 2000 

# Request and response models
class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    status: str
    probabilities: Dict[str, float]

router = APIRouter()

@router.post("/", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze the sentiment of the article."""
    logger.info("[BACKEND] Analyze endpoint accessed")
    try:
        if not isinstance(request.text, str):
            logger.error(f"[BACKEND] Input text has wrong type: {type(request.text)}")
            raise HTTPException(status_code=400, detail="Invalid input: text must be a string.")
            
        # Check input length
        text_length = len(request.text)
        if text_length > MAX_INPUT_SIZE:
            logger.error(f"[BACKEND] Input text too long: {text_length} characters")
            raise HTTPException(
                status_code=400, 
                detail=f"Text is too long ({text_length} characters). Maximum allowed size is {MAX_INPUT_SIZE} characters."
            )
            
        if text_length == 0:
            logger.error("[BACKEND] Empty input text")
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

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
