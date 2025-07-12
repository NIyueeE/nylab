import os
from celery import Celery

CELERY_BROKER_URL=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
tasks_model_path = 'worker.src.tasks' # worker中的任务模块路径

def create_celery_app(include_tasks=False):
    """
    创建基础Celery应用实例
    
    :param include_tasks: 是否包含任务模块（仅Worker需要）
    :return: Celery应用实例
    """
    app = Celery(
        'tasks',
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=[tasks_model_path] if include_tasks else []
    )
    
    # 通用配置（Web和Worker共享）
    app.conf.update(
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        result_accept_content=['json'],
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        broker_connection_retry_on_startup=True
    )
    
    

    return app