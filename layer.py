import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

#=== 1) 載入訓練好的模型 ===#
model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train3\weights\best.pt")
model.eval()

for i, layer in enumerate(model.model.model):
    print(f"{i:2d} | {layer.__class__.__name__}")
