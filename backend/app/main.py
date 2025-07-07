import os
import tempfile
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
import logging
import uuid
from fastapi import FastAPI, UploadFile, Request, File
from fastapi.middleware.cors import CORSMiddleware
from .tasks import train_model_task
from .utils.TrainingConfig import TrainingConfig

app = FastAPI()

# 配置日志
logger = logging.getLogger(__name__)

# 计算.env文件的绝对路径
env_path = Path(__file__).resolve().parent.parent.parent / '.env'

# 加载.env文件
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"已加载环境文件: {env_path}")
else:
    logger.warning(f"未找到环境文件: {env_path}")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/train")
async def start_training_task(
    request: Request,
    files: list[UploadFile] = File(...),  # 修改为多文件上传
    script_file: Optional[UploadFile] = File(None),  # 单独脚本文件
    task_config: TrainingConfig = None  # 新增JSON配置参数
):
    # 生成唯一运行ID
    run_id = str(uuid.uuid4())
    
    # 创建数据集目录（以run_id命名）, 使用临时目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_dir = os.path.join(tmp_dir, run_id)
        os.makedirs(dataset_dir, exist_ok=True)
    
        # 保存所有上传的文件
        if task_config.use_local_dataset:
            for file in files:
                file_path = os.path.join(dataset_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    buffer.write(await file.read())
        else:

        
        # 获取所有表单参数作为训练参数
        form_data = await request.form()
        train_params = {
            key: value 
            for key, value in form_data.items() 
            if key not in ["files", "model_type", "train_name"]  # 排除基础参数
        }
        
        # 传递数据集目录路径（而不是单个文件路径）
        task = train_model_task.delay(
            dataset_path=dataset_dir,
            run_id=run_id,
            **task_config,
        )
    
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