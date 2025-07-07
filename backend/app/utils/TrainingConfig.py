from pydantic import BaseModel

class TrainingConfig(BaseModel):
    train_name: str # 训练任务名称

    # ==========数据集来源相关===========
    use_local_dataset: bool     # 是否使用本地上传数据集

    # if use_local_dataset==True:
    local_dataset_path: str     # 前端上传的数据集路径

    # if use_local_dataset==Flase:
    db_dataset_bucket_name: str # 指定桶名称
    db_dataset_name: str        # 指定桶中选择的数据集的名称
    db_dataset_bucket_pwd: str  # 输入桶密码
    # ================================

    # =======本地上传数据集处理相关=======
    # if use_local_dataset==True:
    store_dataset: bool           # 是否收藏数据集
    # 在上面已定义
    # db_dataset_bucket_name: str # 指定收藏数据集的桶名称
    # db_dataset_bucket_pwd: str  # 被指定桶的密码
    stored_dataset_desc: str      # 收藏数据集的描述
    # =================================

    # ========预训练模型来源相关==========
    use_pretrained_model: bool          # 是否使用预训练模型
    use_local_pretrained_model: bool    # 是否使用本地上传预训练模型
    # if use_local_pretrained_model==True:
    local_pretrained_model_path: str    # 前端上传的预训练模型路径
    # if use_local_pretrained_model==False:
    pretrained_model_name: str          # 预训练模型名称
    # ==================================

    hyperparams: dict  # 存储所有超参数