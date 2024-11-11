from fastapi import APIRouter, HTTPException, BackgroundTasks
from services.repo_analysis_service import get_repo_files, get_file_content, analyze_files, summation_repo_codes, \
    project_summation
from schemas.git_repo import GitRepoRequest, GitRepoFileRequest
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

router = APIRouter(
    prefix="/repo",
)

request_states = {}
gpu_semaphore = asyncio.Semaphore(1)
executor = ThreadPoolExecutor(max_workers=1)

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

# async def analyze_repo_task(request: GitRepoRequest, request_id: str):
#     try:
#         # analyze_files 시작 전 GPU 리소스 사용 대기
#         async with gpu_semaphore:
#             request_states[request_id] = {"status": "in_progress", "step": "analyze_files"}
#             analyze_result = await analyze_files(request.owner, request.repo, request.branch, request.token)
#
#         # summation_repo_codes 단계
#         request_states[request_id] = {"status": "in_progress", "step": "summation_repo_codes"}
#         summation_result = await summation_repo_codes(analyze_result)
#
#         # 프로젝트 요약 단계
#         repo_trees = await get_repo_files(request.owner, request.repo, request.branch, request.token)
#         request_states[request_id] = {"status": "in_progress", "step": "project_summation"}
#         result = await project_summation(summation_result, repo_trees)
#
#         # 작업 완료 후 결과 저장
#         await save_to_json(result, f"{request_id}_result.json")
#         request_states[request_id] = {"status": "success", "result": result}
#     except Exception as e:
#         # 오류 발생 시 상태 업데이트
#         request_states[request_id] = {"status": "failed", "error": str(e)}

def analyze_repo_sync(request: GitRepoRequest, request_id: str):
    """
    비동기 루프에서 실행하기 위해 동기적으로 GPU 작업을 처리하는 함수입니다.
    """
    try:
        loop = asyncio.get_running_loop()
        # GPU 작업 시작 전 세마포어로 잠금
        loop.run_until_complete(gpu_semaphore.acquire())
        request_states[request_id] = {"status": "in_progress", "step": "analyze_files"}

        # 동기적으로 실행되는 GPU 작업
        analyze_result = loop.run_until_complete(
            analyze_files(request.owner, request.repo, request.branch, request.token))

        # summation_repo_codes 단계
        request_states[request_id] = {"status": "in_progress", "step": "summation_repo_codes"}
        summation_result = loop.run_until_complete(summation_repo_codes(analyze_result))

        # 프로젝트 요약 단계
        repo_trees = loop.run_until_complete(get_repo_files(request.owner, request.repo, request.branch, request.token))
        request_states[request_id] = {"status": "in_progress", "step": "project_summation"}
        result = loop.run_until_complete(project_summation(summation_result, repo_trees))

        # 작업 완료 후 결과 저장
        loop.run_until_complete(save_to_json(result, f"{request_id}_result.json"))
        request_states[request_id] = {"status": "success", "result": result}
    except Exception as e:
        # 오류 발생 시 상태 업데이트
        request_states[request_id] = {"status": "failed", "error": str(e)}
    finally:
        # 세마포어 해제
        loop.run_until_complete(gpu_semaphore.release())

@router.get("/status/{request_id}")
async def check_status(request_id: str):
    if request_id in request_states:
        status = request_states[request_id]

        if status["status"] == "success" or status["status"] == "failed":
            response = status
            del request_states[request_id]
            return response

        return status
    else:
        raise HTTPException(status_code=404, detail="Request ID not found")

@router.post("/analyze")
async def analyze_repo(
        request: GitRepoRequest, background_tasks: BackgroundTasks
):
    request_id = request.request_id
    request_states[request_id] = {"status": "initialized", "step": "pending"}

    # GPU 작업을 ThreadPoolExecutor로 비동기적으로 실행
    background_tasks.add_task(
        functools.partial(executor.submit, analyze_repo_sync, request, request_id)
    )

    return {"request_id": request_id, "status": "initialized"}