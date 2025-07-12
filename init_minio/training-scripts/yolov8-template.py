import os
from pathlib import Path
from ultralytics import YOLO
import mlflow

def nylab_train(dataset_path, run_id, update_progress, **hyperparams):
    """
    YOLOv8 通用训练函数
    
    参数:
    - dataset_path: 数据集路径 (包含 dataset.yaml)
    - run_id: 训练运行唯一ID
    - update_progress: 进度回调函数 (run_id, progress, message)
    - hyperparams: 训练超参数字典
    
    支持的完整超参数列表:
    https://docs.ultralytics.com/usage/cfg/#training-arguments
    """
    # 设置默认值并覆盖传入参数
    cfg = {
        # 基础参数
        'model': './yolov8n.pt',
        'data': str(Path(dataset_path) / 'dataset.yaml'),
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'patience': 50,
        'device': '0',  # GPU ID
        'workers': 0,
        
        # 优化参数
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 损失函数参数
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        
        # 训练配置
        'save': True,
        'save_period': -1,
        'cache': False,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'project': f"runs/{run_id}",
        'name': 'train',
        'exist_ok': True,
        
        # 特殊功能
        'pretrained': True,
        'optimizer': 'auto',
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'overlap_mask': True,
        'mask_ratio': 4,
    }
    
    # 使用传入的超参数覆盖默认值
    cfg.update(hyperparams)
    
    # 记录使用的参数
    mlflow.log_params(cfg)
    
    # 初始化模型
    update_progress(run_id, 30, f"加载模型: {cfg['model']}")
    model = YOLO(cfg['model'])
    
    # 开始训练
    update_progress(run_id, 35, "开始训练")
    results = model.train(**cfg)
    
    # 创建模型保存目录
    model_dir = Path(dataset_path).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型文件
    model_path = model_dir / f"yolov8_{run_id}.pt"
    model.save(model_path)
    
    # 验证模型 (可选)
    update_progress(run_id, 85, "模型验证中")
    metrics = model.val()
    
    # 记录关键指标
    mlflow.log_metrics({
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'precision': metrics.box.p,
        'recall': metrics.box.r,
        'fitness': results.fitness
    })
    
    return {
        'model': model,
        'model_path': str(model_path),
        'metrics': {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.p,
            'recall': metrics.box.r,
            'fitness': results.fitness
        },
        'accuracy': metrics.box.map50,  # 主要指标
        'framework': 'yolo'
    }