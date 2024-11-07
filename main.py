import time

import uvicorn
from fastapi import FastAPI
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

app.add_middleware(TimerMiddleware)

@app.get("/")
async def read_root():
    return {"message": "hello"}

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)
