from pydantic import BaseModel

class TrainingConfig(BaseModel):
    train_name: str # 训练任务名称

    use_local_dataset: bool # 是否使用本地上传数据集
    local_dataset_path: str # 前端上传的数据集路径



    hyperparams: dict  # 存储所有超参数