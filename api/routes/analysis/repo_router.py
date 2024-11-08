from fastapi import APIRouter, HTTPException
from services.repo_analysis_service import get_repo_files, get_file_content, analyze_files, text_model_response
from schemas.git_repo import GitRepoRequest, GitRepoFileRequest

router = APIRouter(
    prefix="/repo",
)

@router.post("/file-path")
async def repo(
        request: GitRepoRequest
):
    try:
        files = await get_repo_files(request.owner, request.repo, request.branch, request.token)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/file-content")
async def repo(
        request: GitRepoFileRequest
):
    try:
        content = await get_file_content(request.owner, request.repo, request.file_path, request.branch, request.token)
        return {"status": "success", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_repo(
    request: GitRepoRequest
):
    try:
        result = await analyze_files(request.owner, request.repo, request.branch, request.token)
        return {"status": "success", "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
