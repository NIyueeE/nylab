import os
import uuid
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from .tasks import train_model_task

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/train")
async def start_training(
    dataset: UploadFile,
    model_type: str = Form(...)
):
    # 生成唯一运行ID
    run_id = str(uuid.uuid4())
    
    # 保存上传的文件
    file_path = f"/data/uploads/{run_id}/{dataset.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await dataset.read())
    
    # 启动Celery任务
    task = train_model_task.delay(model_type, file_path, run_id)
    
    return {
        "status": "training_started",
        "run_id": run_id,
        "task_id": task.id
    }

@app.get("/api/progress/{run_id}")
async def get_progress(run_id: str):
    # 在实际应用中，这里应该从Redis或数据库中获取进度
    # 简化示例：返回模拟进度
    return {
        "run_id": run_id,
        "progress": 50,  # 实际应从任务状态获取
        "status": "running",
        "accuracy": None
    }