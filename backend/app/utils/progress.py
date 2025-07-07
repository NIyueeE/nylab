import redis
import os
import json

# 使用连接池提高Redis连接效率
REDIS_POOL = redis.ConnectionPool(
    host=os.getenv("REDIS_HOST", "redis"),
    port=6379,
    db=0,
    decode_responses=False
)

def update_progress(run_id: str, progress: int, message: str, accuracy: float = None, status: str = None):
    """更新训练进度到Redis
    
    Args:
        run_id: 训练运行ID
        progress: 进度百分比(0-100)
        message: 进度消息
        accuracy: 模型准确率(可选)
        status: 状态标记(如"failed")(可选)
    """
    # 使用连接池获取客户端
    r = redis.Redis(connection_pool=REDIS_POOL)
    
    # 构造进度数据
    data = {
        "run_id": run_id,
        "progress": progress,
        "message": message,
        "accuracy": accuracy,
        "status": status
    }
    
    # 使用管道操作提高效率
    with r.pipeline() as pipe:
        # 设置进度值
        pipe.set(f"progress:{run_id}", json.dumps(data))
        # 发布进度更新
        pipe.publish(f"progress:{run_id}", json.dumps(data))
        pipe.execute()