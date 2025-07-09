from pydantic import BaseModel
from typing import Optional

class TrainingConfig(BaseModel):
    train_name: str
    
    # ========== 数据集来源相关 ==========
    use_local_dataset: bool
    
    # 本地数据集字段
    local_dataset_path: Optional[str] = None
    
    # 云存储数据集字段
    db_dataset_bucket_name: Optional[str] = None
    db_dataset_name: Optional[str] = None
    db_dataset_bucket_pwd: Optional[str] = None
    
    # ========== 训练脚本来源相关 ==========
    use_local_script: bool
    
    # 本地脚本字段
    local_script_path: Optional[str] = None
    
    # 云存储脚本字段
    db_script_name: Optional[str] = None
    
    # ========== 数据集收藏相关 ==========
    store_dataset: bool
    stored_dataset_desc: Optional[str] = None
    
    # ========== 脚本收藏相关 ==========
    store_script: bool
    stored_script_desc: Optional[str] = None
    
    # ========== 预训练模型相关 ==========
    use_pretrained_model: bool
    use_local_pretrained_model: Optional[bool] = False
    local_pretrained_model_path: Optional[str] = None
    pretrained_model_name: Optional[str] = None
    
    # 训练超参数
    hyperparams: dict = {}
    