from fastapi import APIRouter
from .analysis import main as analysis

api_router = APIRouter(
    prefix="/api"
)

api_router.include_router(analysis.router)
