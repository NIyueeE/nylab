# backend_test.py
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
    
    # 数据集目录路径
    dataset_dir = os.path.join(os.path.dirname(__file__), "test_datasets/datasets")
    
    # 验证数据集目录存在
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"测试数据集目录不存在: {dataset_dir}")
    
    # 读取任务配置JSON文件
    config_path = os.path.join(os.path.dirname(__file__), "task_config_test.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"任务配置文件不存在: {config_path}")
    
    with open(config_path, "r") as config_file:
        task_config = json.load(config_file)
    
    # 准备多文件上传
    files_list = []
    
    # 递归遍历数据集目录
    for root, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            # 计算相对路径（保持目录结构）
            rel_path = os.path.relpath(file_path, dataset_dir)
            
            with open(file_path, "rb") as f:
                # 保留目录结构的文件上传
                files_list.append(("files", (rel_path, f, "application/octet-stream")))
    
    # 添加配置数据
    data = {"task_config": json.dumps(task_config)}
    
    # 发送请求
    response = requests.post(
        train_url,
        files=files_list,
        data=data
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
        return True
    except json.JSONDecodeError:
        print(f"❌ 无效的JSON响应: {response.text}")
        return False

if __name__ == "__main__":
    simulate_post()