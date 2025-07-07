import os
from dotenv import load_dotenv
from pathlib import Path
# import time
import logging
from celery import Celery
import mlflow
from minio import Minio
from .utils.progress import update_progress
from .utils.database import (
    download_training_script, 
    load_training_module, 
    archive_dataset, 
    upload_training_script
)

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

# 环境配置集中管理
ENV_CONFIG = {
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "minio"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
}

# 设置环境变量
for key, value in ENV_CONFIG.items():
    os.environ[key] = value

# 创建Celery实例
celery_app = Celery(
    'tasks',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0'
)

# MinIO客户端配置
minio_client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    secure=False
)

@celery_app.task(bind=True)
def train_model_task(self, train_name: str, model_type: str, dataset_path: str, run_id: str, **train_params):
    """通用模型训练任务接口
    
    Args:
        train_name: 训练任务名称
        model_type: 模型类型
        dataset_path: 数据集路径
        run_id: 训练运行ID
        **train_params: 动态训练参数
    """
    # 初始化进度
    update_progress(run_id, 0, "初始化训练任务")
    logger.info(f"开始训练任务: {run_id}, 模型类型: {model_type}")
    
    try:
        # 设置MLflow跟踪
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("AutoML-Training")
        
        with mlflow.start_run(run_name=f"{train_name}-{run_id}") as run:
            # 记录基础参数
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("dataset", os.path.basename(dataset_path))
            
            # 记录所有训练参数
            for key, value in train_params.items():
                mlflow.log_param(key, value)
            
            # 是否使用自定义训练脚本
            use_custom_script = train_params.get("use_custom_script", False)
            script_path = None
            
            if use_custom_script:
                # 处理自定义脚本
                script_name = train_params.get("custom_script_name", "custom_script.py")
                script_path = f"/tmp/{run_id}_{script_name}"
                custom_script = train_params.get("custom_script", "")
                
                # 验证脚本内容
                if not custom_script.strip():
                    error_msg = "脚本内容不能为空"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # 写入脚本文件
                with open(script_path, "w") as f:
                    f.write(custom_script)
                update_progress(run_id, 20, "使用自定义脚本")
                logger.info(f"自定义脚本写入: {script_path}")

                # 保存到MinIO
                if train_params.get("save_to_db", False):
                    update_progress(run_id, 21, "保存脚本到数据库")
                    upload_training_script(
                        minio_client, 
                        script_name, 
                        script_path, 
                        update_progress, 
                        run_id
                    )
            else:
                # 下载标准脚本
                update_progress(run_id, 20, "下载训练脚本")
                script_name = train_params.get("script_name", "")
                script_path = download_training_script(
                    minio_client, 
                    model_type, 
                    run_id, 
                    script_name
                )
                logger.info(f"下载脚本完成: {script_path}")
            
            # 动态加载脚本
            update_progress(run_id, 25, "加载训练模块")
            custom_module = load_training_module(script_path)
            logger.info(f"成功加载训练模块: {script_path}")
            
            # 调用训练函数
            update_progress(run_id, 30, "开始模型训练")
            result = custom_module.train_model(
                dataset_path=dataset_path,
                run_id=run_id,
                update_progress=update_progress,
                **train_params
            )
            
            # 处理训练结果
            if 'model' in result:
                # 记录模型指标
                metrics = result.get('metrics', {})
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # 保存模型
                update_progress(run_id, 90, "保存模型")
                mlflow.sklearn.log_model(result['model'], "model")
                logger.info(f"模型保存完成: {run.info.run_id}")
            
            # 归档数据集
            archive_dataset(minio_client, dataset_path, run_id)
            logger.info(f"数据集归档完成: {dataset_path}")
            
            # 完成训练
            accuracy = result.get('accuracy')
            update_progress(run_id, 100, "训练完成", accuracy=accuracy)
            logger.info(f"训练完成: 准确率={accuracy}")
            
            return {
                "status": "success",
                "accuracy": accuracy,
                "run_id": run_id
            }
    
    except Exception as e:
        error_msg = f"训练失败: {str(e)}"
        logger.exception(error_msg)
        update_progress(run_id, 0, error_msg, status="failed")
        mlflow.log_param("error", error_msg)
        raise self.retry(exc=e, countdown=60)