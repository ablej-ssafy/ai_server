from workers.celery import app
from schemas.git_repo import GitRepoRequest
from services.repo_analysis_service import summation_repo_codes, project_summation, get_repo_files, analyze_files
import asyncio
from core.config import settings
import redis
from redis.exceptions import LockError
import torch
import gc
import json

redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)
gpu_lock = redis_client.lock('gpu_lock', timeout=600)

# LLaMA 작업 - 한 번에 하나의 작업만 가능하도록 RedisLock 사용
@app.task(bind=True)
def llama_task(self, request_data: GitRepoRequest):
    request_id = request_data.request_id

    try:
        # RedisLock 획득 시도
        if gpu_lock.acquire(blocking=True):
            try:
                redis_client.set(request_id, json.dumps({"status": "in_progress", "step": "llama_processing"}))

                analyze_results = asyncio.run(analyze_files(
                    request_data["owner"], request_data["repo"], request_data["branch"], request_data["token"]
                ))

                redis_client.set(
                    request_id,
                    json.dumps({"status": "completed", "step": "llama_processing", "result": analyze_results})
                )

                openai_task.apply_async(args=[request_data])
            finally:
                torch.cuda.empty_cache()
                gc.collect()
                gpu_lock.release()
        else:
            redis_client.set(request_id, json.dumps({"status": "failed", "step": "llama_processing", "error": "GPU lock acquisition failed"}))
    except LockError:
        redis_client.set(request_id, json.dumps({"status": "failed", "step": "llama_processing", "error": "GPU lock acquisition error"}))
    except Exception as e:
        redis_client.set(request_id, json.dumps({"status": "failed", "step": "llama_processing", "error": str(e)}))

# OpenAI 작업 - 여러 개의 작업 동시 실행 가능
@app.task(bind=True)
def openai_task(self, request_data: dict):
    request_id = request_data["request_id"]

    async def openai_task_async():
        print(f"LOG: openai_task 호출 - {request_data}")
        try:
            # llama_task 결과
            llama_result_json = redis_client.get(request_id)
            if not llama_result_json:
                raise ValueError("Llama task result not found.")

            redis_client.set(request_id, json.dumps({"status": "in_progress", "step": "openai_processing"}))

            llama_result = json.loads(llama_result_json).get("result")
            if not llama_result:
                raise ValueError("Llama task did not complete properly.")

            summation_result = await summation_repo_codes(llama_result)

            redis_client.set(request_id, json.dumps({"status": "in_progress", "step": "project_summation"}))

            repo_trees = await get_repo_files(
                request_data["owner"], request_data["repo"], request_data["branch"], request_data["token"]
            )
            result = await project_summation(summation_result, repo_trees)

            redis_client.set(request_id, json.dumps({"status": "completed", "result": result}))
        except Exception as e:
            redis_client.set(request_id, json.dumps({"status": "failed", "step": "openai_processing", "error": str(e)}))

    asyncio.run(openai_task_async())