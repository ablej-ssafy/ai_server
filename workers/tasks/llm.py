from workers.celery import app
from schemas.git_repo import GitRepoRequest
from services.repo_analysis_service import summation_repo_codes, project_summation, get_repo_files, analyze_files
import asyncio
from core.config import settings
import redis
import json

redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)
gpu_semaphore = asyncio.Semaphore(1)

# LLaMA 작업 - 한 번에 하나의 작업만 가능하도록 Semaphore 사용
@app.task(bind=True)
def llama_task(self, request_data: GitRepoRequest):
    request_id = request_data["request_id"]
    async def llama_task_async():
        async with gpu_semaphore:
            await redis_client.set(request_id, json.dumps({"status": "in_progress", "step": "llama_processing"}))
            analyze_results = analyze_files(request_data["owner"], request_data["repo"], request_data["branch"], request_data["token"])
            await redis_client.set(request_id,
                                   json.dumps(
                                       {"status": "completed", "step": "llama_processing", "result": analyze_results}))

            openai_task.apply_async(args=[request_data])
    asyncio.run(llama_task_async())

# OpenAI 작업 - 여러 개의 작업 동시 실행 가능
@app.task(bind=True)
def openai_task(self, request_data: GitRepoRequest):
    request_id = request_data["request_id"]
    async def openai_task_async():
        await redis_client.set(request_id, json.dumps({"status": "in_progress", "step": "openai_processing"}))
        summation_result = await summation_repo_codes(json.loads(redis_client.get(request_id))["result"])

        await redis_client.set(request_id, json.dumps({"status": "in_progress", "step": "project_summation"}))
        repo_trees = await get_repo_files(request_data["owner"], request_data["repo"], request_data["branch"], request_data["token"])
        result = await project_summation(summation_result, repo_trees)

        await redis_client.set(request_id, json.dumps({"status": "completed", "result": result}))
    asyncio.run(openai_task_async())
