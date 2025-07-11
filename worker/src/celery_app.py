from backend_common.celery_setup import create_celery_app

# 创建包含任务模块的完整Celery实例
celery_app = create_celery_app(include_tasks=True)

# Worker专用配置
celery_app.conf.update(
    worker_concurrency=4,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True
)

# 自动发现任务模块
celery_app.autodiscover_tasks(['worker.src.tasks'])