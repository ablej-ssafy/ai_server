from fastapi import APIRouter, HTTPException, BackgroundTasks
from services.repo_analysis_service import get_repo_files, get_file_content
from schemas.git_repo import GitRepoRequest, GitRepoFileRequest
import json
from workers.celery import app
from workers.tasks.llm import llama_task, openai_task, openai_f_task
from core.config import settings
import redis

router = APIRouter(
    prefix="/repo",
)

redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)

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

@router.get("/status/{request_id}")
async def check_status(request_id: str):
    status_data = redis_client.get(request_id)
    if status_data:
        status = json.loads(status_data)

        if status["status"] in ["completed", "failed"]:
            redis_client.delete(request_id)
            return status

        return status
    else:
        raise HTTPException(status_code=404, detail="요청한 UUID 작업이 존재하지 않습니다.")

@router.post("/analyze")
async def analyze_repo(
        request: GitRepoRequest, background_tasks: BackgroundTasks
):
    request_id = request.request_id
    redis_client.set(request_id, json.dumps({"status": "initialized", "step": "pending", "memberId": request.memberId}))

    # llama_task.apply_async(args=[request.dict()])
    openai_f_task.apply_async(args=[request.dict()])

    return {"request_id": request_id, "status": "initialized"}

@app.task(bind=True)
def llama_callback(self, request_data, request_id):
    openai_task.apply_async(args=[request_data])