import redis
import os
import json

def get_redis_client():
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=6379,
        db=0
    )

def update_progress(run_id, progress, message, accuracy=None):
    """更新训练进度到Redis"""
    data = {
        "run_id": run_id,
        "progress": progress,
        "message": message,
        "accuracy": accuracy
    }
    
    r = get_redis_client()
    r.set(f"progress:{run_id}", json.dumps(data))
    
    # 发布进度更新
    r.publish(f"progress:{run_id}", json.dumps(data))