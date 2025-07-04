import requests
import os
import sys
import time
import json

# 添加后端路径到系统路径（仅用于导入可能需要的辅助函数）
sys.path.append(os.path.abspath("../backend"))

# 前端访问的API地址 - 使用Docker Compose映射的端口
API_URL = "http://localhost:8000/api"

def simulate_post():
    
    print("🚀 模拟前端：提交训练任务...")
    train_url = f"{API_URL}/train"
    
    # 使用实际测试数据集
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets/test.csv")
    
    # 验证数据集存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"测试数据集不存在: {dataset_path}")
    
    # 模拟前端FormData上传
    with open(dataset_path, "rb") as f:
        response = requests.post(
            train_url,
            files={"dataset": f},
            data={"model_type": "random_forest"}
        )
    
    # 检查响应状态
    if response.status_code != 200:
        print(f"❌ 提交失败: {response.status_code} - {response.text}")
        return False
    
    # 解析响应数据
    try:
        response_data = response.json()
        task_id = response_data.get("task_id")
        run_id = response_data.get("run_id")
        
        if not task_id or not run_id:
            print("❌ 响应缺少必要字段")
            return False
            
        print(f"✅ 任务提交成功! Task ID: {task_id}, Run ID: {run_id}")
    except json.JSONDecodeError:
        print(f"❌ 无效的JSON响应: {response.text}")
        return False

if __name__ == "__main__":
    simulate_post()
    