import os
import importlib
import logging
from minio import Minio, S3Error
from minio.error import S3Error

# 配置日志
logger = logging.getLogger(__name__)

def download_training_script(minio_client: Minio, model_type: str, run_id: str, script_name: str = "") -> str:
    """从MinIO下载训练脚本
    
    Args:
        minio_client: MinIO客户端实例
        model_type: 模型类型
        run_id: 训练运行ID
        script_name: 自定义脚本名(可选)
        
    Returns:
        下载的脚本本地路径
    """
    script_bucket = "training-scripts"
    script_name = script_name or f"{model_type}.py"
    script_path = f"/tmp/{run_id}_{script_name}"
    
    try:
        # 确保存储桶存在
        if not minio_client.bucket_exists(script_bucket):
            minio_client.make_bucket(script_bucket)
            logger.info(f"创建存储桶: {script_bucket}")
        
        # 下载脚本
        minio_client.fget_object(script_bucket, script_name, script_path)
        logger.info(f"下载脚本: {script_name} -> {script_path}")
        return script_path
        
    except S3Error as e:
        logger.error(f"下载脚本失败: {e}")
        raise FileNotFoundError(f"脚本不存在: {script_name}") from e

def upload_training_script(minio_client: Minio, script_name: str, script_path: str, update_progress, run_id: str)-> None:
    """上传训练脚本到MinIO
    
    Args:
        minio_client: MinIO客户端实例
        script_name: 目标脚本名
        script_path: 本地脚本路径
        update_progress: 进度更新函数
        run_id: 训练运行ID
    """
    bucket_name = "training-scripts"
    
    try:
        # 确保存储桶存在
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(f"创建存储桶: {bucket_name}")
        
        # 检查并处理名称冲突
        original_name = script_name
        counter = 1
        while True:
            try:
                # 检查对象是否存在
                minio_client.stat_object(bucket_name, script_name)
                # 存在则生成新名称
                base, ext = os.path.splitext(original_name)
                script_name = f"{base}_{counter}{ext}"
                counter += 1
            except S3Error as e:
                if e.code == "NoSuchKey":
                    break  # 名称可用
                raise  # 其他错误重新抛出
        
        # 处理重命名
        if script_name != original_name:
            rename_msg = f"脚本名称冲突，已重命名为: {script_name}"
            update_progress(run_id, 22, rename_msg)
            logger.warning(rename_msg)
        
        # 上传脚本
        minio_client.fput_object(
            bucket_name,
            script_name,
            script_path,
            content_type="text/plain"
        )
        save_msg = f"脚本已保存到MinIO: {script_name}"
        update_progress(run_id, 23, save_msg)
        logger.info(save_msg)
        
    except Exception as e:
        logger.error(f"上传脚本失败: {e}")
        update_progress(run_id, 0, f"上传失败: {str(e)}", status="failed")
        raise

def load_training_module(script_path: str)-> object:
    """动态加载Python模块"""
    spec = importlib.util.spec_from_file_location("training_module", script_path)
    training_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_module)
    logger.info(f"加载模块: {training_module.__name__}")
    return training_module

def archive_dataset(minio_client: Minio, dataset_path: str, run_id: str)-> None:
    """归档数据集到MinIO"""
    dataset_bucket = "mlflow-datasets"
    if not minio_client.bucket_exists(dataset_bucket):
        minio_client.make_bucket(dataset_bucket)
    
    minio_client.fput_object(
        dataset_bucket,
        f"{run_id}/{os.path.basename(dataset_path)}",
        dataset_path
    )
    logger.info(f"归档数据集: {dataset_path}")