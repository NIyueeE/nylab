import os
import time
from celery import Celery
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from minio import Minio
from .utils.progress import update_progress

# 创建Celery实例
celery_app = Celery(
    'tasks',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0'
)

# MinIO客户端配置
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    secure=False
)

@celery_app.task(bind=True)
def train_model_task(self, model_type, dataset_path, run_id):
    # 设置MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://web:5000"))
    mlflow.set_experiment("AutoML-Training")
    
    try:
        # 开始MLflow运行
        with mlflow.start_run(run_name=f"AutoML-{run_id}") as run:
            # 记录参数
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("dataset", os.path.basename(dataset_path))
            
            # 加载数据
            update_progress(run_id, 10, "Loading data")
            data = pd.read_csv(dataset_path)
            X = data.drop('target', axis=1)
            y = data['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # 训练模型
            update_progress(run_id, 30, "Training model")
            model = train_model(model_type, X_train, y_train)
            
            # 评估模型
            update_progress(run_id, 70, "Evaluating model")
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            
            # 保存模型
            update_progress(run_id, 90, "Saving model")
            mlflow.sklearn.log_model(model, "model")
            
            # 上传数据集到MinIO
            bucket_name = "mlflow-datasets"
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
            
            minio_client.fput_object(
                bucket_name,
                f"{run_id}/{os.path.basename(dataset_path)}",
                dataset_path
            )
            
            update_progress(run_id, 100, "Completed", accuracy=accuracy)
            return {"status": "success", "accuracy": accuracy}
    
    except Exception as e:
        update_progress(run_id, -1, f"Error: {str(e)}")
        raise

def train_model(model_type, X_train, y_train):
    """根据模型类型训练模型"""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
    elif model_type == "svm":
        from sklearn.svm import SVC
        model = SVC(probability=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model