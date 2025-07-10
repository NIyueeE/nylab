import os
from typing import Optional
from pydantic import ValidationError
from dotenv import load_dotenv
import logging
from fastapi.responses import JSONResponse
from minio import Minio
from minio.error import S3Error
import uuid
import json
from fastapi import FastAPI, UploadFile, Request, File, Form
from fastapi.middleware.cors import CORSMiddleware
from .tasks import train_model_task, minio_client
from .utils.TrainingConfig import TrainingConfig
from .utils.database import _hash_password

app = FastAPI()

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

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
    config_file: str = Form(...),
    script_file: Optional[UploadFile] = File(None)  # 单独脚本文件
):
    try:
        config_data = json.loads(config_file)
        task_config = TrainingConfig(**config_data)  # 创建Pydantic对象
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return JSONResponse(status_code=400, content={"error": "无效的JSON格式"})
    except ValidationError as e:
        logger.error(f"配置验证失败: {e}")
        return JSONResponse(status_code=400, content={"error": f"配置验证失败: {e}"})

    logger.info(f"传入配置: {task_config}")
    # 生成唯一运行ID
    run_id = str(uuid.uuid4())
    
    # 创建数据集目录（以run_id命名）, 使用临时目录
    tmp_dir = f"/data/{run_id}" # 暂时使用, 在训练结束时清理
    dataset_dir = os.path.join(tmp_dir, "datasets")
    script_dir = tmp_dir
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)

    # 保存所有本地上传的文件
    if task_config.use_local_dataset:
        for file in files:
            file_path = os.path.join(dataset_dir, file.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
    # 保存指定存储桶中的数据集
    else:
        # 尝试获取桶的元数据文件
        response = minio_client.get_object(task_config.db_dataset_bucket_name, ".bucket_meta")
        meta_content = response.data.decode()
        response.close()
        response.release_conn()
        
        # 解析元数据中的密码
        stored_pwd = None
        for line in meta_content.splitlines():
            if line.startswith("password="):
                stored_pwd = line.split("=", 1)[1].strip()
                break
        if stored_pwd and stored_pwd == _hash_password(task_config.db_dataset_bucket_pwd):
                logger.info(f"存储桶密码验证成功")

                # 检查是文件还是文件夹
                try:
                    # 尝试作为文件下载
                    minio_client.fget_object(
                        task_config.db_dataset_bucket_name,
                        task_config.db_dataset_name,
                        os.path.join(dataset_dir, os.path.basename(task_config.db_dataset_name))
                    )
                    logger.info(f"已下载数据集文件: {task_config.db_dataset_name}")
                except S3Error as e:
                    if e.code == 'NoSuchKey':
                        # 作为文件夹处理
                        prefix = task_config.db_dataset_name
                        if not prefix.endswith('/'):
                            prefix += '/'

                        objects = minio_client.list_objects(
                            task_config.db_dataset_bucket_name,
                            prefix=prefix,
                            recursive=True
                        )

                        # 下载所有对象
                        total_files = 0
                        for obj in objects:
                            total_files += 1
                            relative_path = obj.object_name[len(prefix):]
                            local_path = os.path.join(dataset_dir, relative_path)
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)

                            minio_client.fget_object(
                                task_config.db_dataset_bucket_name,
                                obj.object_name,
                                local_path
                            )
                        logger.info(f"已下载数据集文件夹: {task_config.db_dataset_name}，包含 {total_files} 个文件")
                    else:
                        logger.error(f"MinIO操作失败: {str(e)}")
                        return JSONResponse(
                            status_code=500,
                            content={"error": f"数据集访问失败: {e.message}"}
                        )
        else:
                logger.error("存储桶密码验证失败")
                return JSONResponse(
                    status_code=403,
                    content={"error": "存储桶密码错误"}
                )
    # 保存本地上传的脚本文件
    if task_config.use_local_script:
        if script_file:
            script_path = os.path.join(script_dir, script_file.filename)
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            with open(script_path, "wb") as buffer:
                buffer.write(await script_file.read())
    # 保存存储桶中的脚本文件
    else:
        script_path = os.path.join(script_dir, os.path.basename(task_config.db_script_name))
        try:
            minio_client.fget_object(
                "training-scripts",
                task_config.db_script_name,
                script_path
            )
        except S3Error as e:
            logger.error(f"下载脚本失败: {e}")
            return JSONResponse(status_code=500, content={"error": f"脚本下载失败: {e.message}"})

    try:
        # 传递数据集目录路径（而不是单个文件路径）
        task = train_model_task.delay(
            dataset_path=dataset_dir,
            script_path=script_path,
            run_id=run_id,
            task_config=task_config.dict()
        )
    except Exception as e:
        logger.error(f"任务启动失败: {e}")
        return JSONResponse(status_code=500, content={"error": f"任务启动失败: {e}"})
    
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