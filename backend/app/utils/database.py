import os
import io
import importlib
import logging
import hashlib
from minio import Minio
from minio.error import S3Error
from minio.datatypes import Part
from .progress import _acquire_bucket_lock

# 配置日志Part
logger = logging.getLogger(__name__)

# 哈希工具函数
def _hash_password(password: str) -> str:
    """使用 SHA-256 哈希算法处理密码"""
    if not password:
        return None
    return hashlib.sha256(password.encode()).hexdigest()

# 大文件上传
def _upload_file_2_bucket(minio_client: Minio, 
                       bucket: str, object_name: str, 
                       file_path: str
    ) -> None:
    """分块上传大文件"""
    try:
        chunk_size = os.getenv("MINIO_CHUNK_SIZE", 10 * 1024 * 1024)
        if os.path.getsize(file_path) < chunk_size:
            minio_client.fput_object(bucket, object_name, file_path)
        else:   
            upload_id = minio_client._new_multipart_upload(bucket, object_name)
            parts = []

            with open(file_path, 'rb') as file:
                part_number = 1
                while chunk := file.read(chunk_size):
                    etag = minio_client._upload_part(
                        bucket, object_name, upload_id, part_number, io.BytesIO(chunk), len(chunk)
                    )
                    parts.append(Part(part_number, etag))
                    part_number += 1

            minio_client._complete_multipart_upload(bucket, object_name, upload_id, parts)
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise

def upload_training_script(
        minio_client: Minio, 
        script_name: str, 
        script_path: str, 
        update_progress, 
        run_id: str
    )-> None:
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
        _upload_file_2_bucket(minio_client, bucket_name, script_name, script_path)

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

def archive_dataset(
    minio_client: Minio, 
    dataset_path: str, 
    store_dataset: bool, 
    run_id: str,
    bucket_name: str = None,
    bucket_pwd: str = None,
    stored_dataset_name: str = None,
    stored_dataset_desc: str = None
) -> None:
    """归档数据集到MinIO，支持收藏数据集到指定桶的功能
    
    Args:
        minio_client: MinIO客户端实例
        dataset_path: 本地数据集路径
        store_dataset: 是否作为收藏数据集存储
        run_id: 训练运行ID
        bucket_name: MinIO存储桶名称
        bucket_pwd: MinIO存储桶密码（用于验证）
        stored_dataset_name: 收藏数据集名称
        stored_dataset_desc: 收藏数据集描述
    """
    # 1. 初始化普通数据集存储桶
    dataset_bucket = "mlflow-temp-datasets"
    if not minio_client.bucket_exists(dataset_bucket):
        minio_client.make_bucket(dataset_bucket)
    
    # 2. 处理默认桶设置
    # 合并条件判断，避免覆盖用户提供的描述
    if bucket_name is None or bucket_pwd is None:
        bucket_name = "open-datasets"
        # 保留用户描述（如果提供）
        if stored_dataset_desc is None:
            stored_dataset_desc = "默认公开数据集"
        bucket_pwd = None
    
    # 3. 密码验证逻辑
    # 仅当桶存在且提供了密码时才验证
    if minio_client.bucket_exists(bucket_name) and bucket_pwd is not None:
        try:
            # 尝试获取桶的元数据文件
            response = minio_client.get_object(bucket_name, ".bucket_meta")
            meta_content = response.data.decode()
            response.close()
            response.release_conn()
            
            # 解析元数据中的密码
            stored_pwd = None
            for line in meta_content.splitlines():
                if line.startswith("password="):
                    stored_pwd = line.split("=", 1)[1].strip()
                    break
            
            # 验证密码
            if stored_pwd and stored_pwd != _hash_password(bucket_pwd):
                raise PermissionError("存储桶密码错误")
                
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.warning(f"桶 {bucket_name} 没有元数据文件，跳过密码验证")
            else:
                logger.error(f"访问桶 {bucket_name} 的元数据时出错: {str(e)}")
                raise
    
    # 4. 创建新存储桶（如果不存在）
    if not minio_client.bucket_exists(bucket_name):
        # 获取分布式锁
        lock = _acquire_bucket_lock(bucket_name)
        if lock:
            try:
                minio_client.make_bucket(bucket_name)
                logger.info(f"创建收藏数据集存储桶: {bucket_name}")

                # 为新桶创建元数据文件
                if bucket_pwd:
                    meta_content = f"password={_hash_password(bucket_pwd)}"
                    minio_client.put_object(
                        bucket_name,
                        ".bucket_meta",
                        io.BytesIO(meta_content.encode()),
                        len(meta_content),
                        content_type="text/plain"
                    )
            finally:
                lock.release()
        else:
            logger.warning(f"无法获取存储桶 {bucket_name} 的锁，跳过创建")

    # 5. 处理收藏数据集
    if store_dataset and stored_dataset_name:
        dataset_prefix = f"{stored_dataset_name}/"

        # 上传数据集
        if os.path.isdir(dataset_path):
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, dataset_path)
                    object_name = f"{dataset_prefix}{rel_path}"
                    _upload_file_2_bucket(minio_client, bucket_name, object_name, local_path)
        else:
            _upload_file_2_bucket(
                minio_client, 
                bucket_name, 
                f"{dataset_prefix}{os.path.basename(dataset_path)}", 
                dataset_path
            )
        
        meta_content = f"description={stored_dataset_desc}\nbucket={bucket_name}\nprotected={bucket_pwd is not None}"
        minio_client.put_object(
            bucket_name,
            f"{dataset_prefix}.dataset_meta",
            io.BytesIO(meta_content.encode()),
            len(meta_content),
            content_type="text/plain"
        )
        
        logger.info(f"收藏数据集已存储到存储桶: {bucket_name}/{stored_dataset_name}")
    
    # 6. 处理普通数据集
    else:
        # 获取保留数据集数量配置（修复环境变量使用）
        saved_tmp_datasets_num = os.getenv("SAVED_TMP_DATASETS_NUM", 7)
        
        # 上传数据集
        object_name = f"{run_id}/{os.path.basename(dataset_path)}"
        _upload_file_2_bucket(minio_client, dataset_bucket, object_name, dataset_path)
        
        # 清理旧数据集
        objects = minio_client.list_objects(dataset_bucket, recursive=True)
        run_dirs = {}
        
        # 收集目录及其最后修改时间
        for obj in objects:
            dir_name = obj.object_name.split('/')[0]
            if dir_name not in run_dirs or obj.last_modified > run_dirs[dir_name]:
                run_dirs[dir_name] = obj.last_modified
        
        # 按时间排序并保留指定数量
        sorted_dirs = sorted(run_dirs.items(), key=lambda x: x[1], reverse=True)
        dirs_to_keep = [d[0] for d in sorted_dirs[:saved_tmp_datasets_num]]
        
        # 删除旧数据集
        lock = _acquire_bucket_lock(dataset_bucket)
        if lock:
            try:
                for dir_name, _ in sorted_dirs[saved_tmp_datasets_num:]:
                    objects_to_remove = minio_client.list_objects(
                        dataset_bucket, prefix=dir_name, recursive=True
                    )
                    for obj in objects_to_remove:
                        minio_client.remove_object(dataset_bucket, obj.object_name)
                    logger.info(f"清理旧数据集: {dir_name}")
            finally:
                lock.release()
        else:
            logger.warning(f"无法获取存储桶 {dataset_bucket} 的锁，跳过清理")