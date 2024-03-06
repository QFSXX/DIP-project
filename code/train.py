from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
# 加载一个模型
    model = YOLO('yolov8s.yaml')  # 从YAML建立一个新模型
    model = YOLO('yolov8s.pt')  # 加载预训练模型（推荐用于训练）
    model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # 从YAML建立并转移权重

    # 训练模型
    results = model.train(data='cfg/train/mydaya.yaml', epochs=300, imgsz=640,batch=8,save=True,save_period=20,device=0,resume=True)