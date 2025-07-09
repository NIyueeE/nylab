from ultralytics import YOLO

def nylab_train(dataset_path, run_id, update_progress, **kwargs):
    # 解析参数（带默认值）
    epochs = kwargs.get('epochs', 100)
    batch_size = kwargs.get('batch_size', 16)
    imgsz = kwargs.get('imgsz', 640)
    
    # 更新进度
    update_progress(run_id, 10, "Loading YOLOv8 model")
    model = YOLO("yolov8n.pt")
    
    # 训练模型
    update_progress(run_id, 30, "Starting training")
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        **kwargs
    )
    
    # 返回结果
    return {
        'model': model,
        'metrics': results.results_dict,
        'accuracy': results.results_dict.get('metrics/mAP50-95(B)', 0)
    }