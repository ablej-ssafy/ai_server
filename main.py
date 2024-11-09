import time

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import traceback
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from core.config import settings
from api.routes.main import api_router

app = FastAPI()


class TimerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"Request processing time: {process_time} seconds")
        return response


# 로깅 설정
logging.basicConfig(level=logging.ERROR, filename="error.log",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 에러 메시지 구성
    error_message = {
        "detail": "서버에서 예외가 발생했습니다.",
        "type": str(type(exc).__name__),  # 예외 타입
        "message": str(exc),  # 예외 메시지
        "traceback": traceback.format_exc()  # 자세한 트레이스백 정보
    }

    # 로그 기록
    logging.error(f"Exception: {error_message}")

    # JSON 응답 반환
    return JSONResponse(status_code=500, content=error_message)


app.add_middleware(TimerMiddleware)


@app.get("/")
async def read_root():
    return {"message": "hello"}


app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)
