import os
import io
import logging
from celery import Celery
import mlflow
from minio import Minio
from .utils.progress import update_progress
from .utils.database import (
    load_training_module, 
    archive_dataset,
    _upload_file_2_bucket
)

# 配置日志
logger = logging.getLogger(__name__)

# MinIO客户端配置
minio_client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    secure=False
)

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

@celery_app.task(bind=True)
def train_model_task(self, 
                     dataset_path: str,
                     script_path: str,
                     run_id: str,
                     **task_config
    ):
    """通用模型训练任务接口
    
    Args:
        dataset_path: 数据集在此容器下的路径
        script_path: 训练脚本在此容器下的路径
        run_id: 训练运行ID
        **params: 模型训练参数
    """
    # 初始化进度
    update_progress(run_id, 0, "初始化训练任务")
    run_name=f"{task_config.get('train_name', 'train')}-{run_id}"
    logger.info(f"开始训练任务: {run_name}")
    
    try:
        # 设置MLflow跟踪
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(task_config.train_name)
        
        with mlflow.start_run(run_name) as run:
            # 记录基础参数
            if task_config.use_local_dataset:
                mlflow.log_param(
                    "数据集来源",
                    f"本地:{task_config.local_dataset_path}"
                )
            else:
                mlflow.log_param(
                    "数据集来源", 
                    f"数据库:{task_config.db_dataset_bucket_name}/{task_config.db_dataset_name}"
                )
            if task_config.use_local_script:
                mlflow.log_param(
                    "脚本来源",
                    f"本地:{task_config.local_script_path}"
                )
            else:
                mlflow.log_param(
                    "脚本来源", 
                    f"数据库:training-scripts/{task_config.db_script_name}"
                )
            update_progress(run_id, 10, "记录基础参数")

            hyperparams = task_config.get("hyperparams", {})
            # 记录所有训练超参数
            for key, value in task_config.hyperparams.items():
                mlflow.log_param(key, value)
            update_progress(run_id, 15, "记录训练超参数")
            
            # 动态加载脚本
            update_progress(run_id, 25, "加载训练模块")
            training_model = load_training_module(script_path)
            logger.info(f"成功加载训练模块: {os.path.basename(script_path)}")
            
            # 调用训练函数
            update_progress(run_id, 30, "开始模型训练")
            result = training_model.nylab_train(
                dataset_path=dataset_path,
                run_id=run_id,
                update_progress=update_progress,
                **hyperparams
            )
            
            # 处理训练结果
            if 'model' in result:
                # 记录模型指标
                metrics = result.get('metrics', {})
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # 保存模型
                update_progress(run_id, 85, "保存模型")
                mlflow.sklearn.log_model(result['model'], "model")
                logger.info(f"模型保存完成: {run.info.run_id}")
            
            # 是否储藏数据集
            if task_config.use_local_dataset:
                store_dataset = task_config.get("store_dataset", False)
                bucket_name = task_config.get("bucket_name", "open-datasets")
                bucket_pwd = task_config.get("bucket_pwd", None)
                stored_dataset_name = task_config.get("stored_dataset_name", "收藏数据集的名称")
                stored_dataset_desc = task_config.get("stored_dataset_desc", "收藏数据集的描述")

                # 归档数据集
                archive_dataset(
                    minio_client, 
                    dataset_path, 
                    store_dataset, 
                    run_id,
                    bucket_name,
                    bucket_pwd,
                    stored_dataset_name,
                    stored_dataset_desc
                )
                update_progress(run_id, 85, "数据集归档")   
                logger.info(f"数据集归档完成: {dataset_path}")
            else:
                logger.info("非本地数据集，不进行归档")
            # 是否储藏训练脚本
            if task_config.use_local_script:
                script_name = os.path.basename(script_path)
                if task_config.store_script:
                    _upload_file_2_bucket(
                        minio_client, 
                        "training-scripts", 
                        script_name, 
                        script_path
                    )
                    meta_content = f"description={stored_dataset_desc}\nbucket=training-scripts"
                    minio_client.put_object(
                        bucket_name,
                        f"{script_name}.dataset_meta",
                        io.BytesIO(meta_content.encode()),
                        len(meta_content),
                        content_type="text/plain"
                    )
                    logger.info(f"训练脚本已存储到存储桶: {bucket_name}/{script_name}")
                else:
                    logger.info(f"训练脚本不需要存储: {script_name}")
            else:
                logger.info("非本地脚本，不进行归档")
        
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