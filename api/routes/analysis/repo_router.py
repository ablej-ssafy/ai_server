from fastapi import APIRouter, HTTPException
from services.repo_analysis_service import get_repo_files, get_file_content, analyze_files, text_model_response

router = APIRouter(
    prefix="/repo",
)

@router.get("/file-path")
async def repo(
        owner: str,
        repo: str,
        branch: str,
        token: str = None
):
    try:
        files = await get_repo_files(owner, repo, branch, token)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/file-content")
async def repo(
        owner: str,
        repo: str,
        file_path: str,
        branch: str,
        token: str = None
):
    try:
        content = await get_file_content(owner, repo, file_path, branch, token)
        return {"status": "success", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_repo(
    owner: str,
    repo: str,
    branch: str,
    token: str = None
):
    try:
        result = await analyze_files(owner, repo, branch, token)
        return {"status": "success", "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
