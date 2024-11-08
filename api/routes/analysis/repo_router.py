from fastapi import APIRouter, HTTPException
from services.repo_analysis_service import get_repo_files, get_file_content, analyze_files, summation_repo_codes, project_summation
from schemas.git_repo import GitRepoRequest, GitRepoFileRequest
import json

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

async def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

@router.post("/analyze")
async def analyze_repo(
    request: GitRepoRequest
):
    try:
        analyze_result = await analyze_files(request.owner, request.repo, request.branch, request.token)
        await save_to_json(analyze_result, "analyze_result.json")

        summation_result = await summation_repo_codes(analyze_result)
        await save_to_json(summation_result, "summation_result.json")

        repo_trees = await get_repo_files(request.owner, request.repo, request.branch, request.token)
        await save_to_json(repo_trees, "repo_trees.json")

        result = await project_summation(summation_result, repo_trees)
        await save_to_json(result, "project_summation_result.json")

        return {"status": "success", "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
