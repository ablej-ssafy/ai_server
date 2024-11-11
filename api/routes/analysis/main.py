from fastapi import APIRouter
from api.routes.analysis import repo_router

router = APIRouter(
    prefix="/analysis",
    tags=["GIT 분석 API"]
)

router.include_router(repo_router.router)
