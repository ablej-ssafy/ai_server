from fastapi import APIRouter, HTTPException
from .analysis import main as analysis
from services.repo_analysis_service import text_model_response

api_router = APIRouter(
    prefix="/api"
)

@api_router.post("/text-summary")
async def test_model(content: str):
    try:
        result = text_model_response(content)
        return {"status": "success", "summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

api_router.include_router(analysis.router)
