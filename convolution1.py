import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

#=== 1) 載入訓練好的模型 ===#
model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train3\weights\best.pt")
model.eval()

#=== 2) 設定 hook 函數與儲存容器 ===#
feature_maps = {}

def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

#=== 3) 為所有 Conv2d 註冊 forward hook ===#
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        module.register_forward_hook(hook_fn(name))

#=== 4) 讀取影像並前處理（符合 YOLOv8 輸入格式）===#
img_path = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data1\test\images\35-2-155-1_jpg.rf.2259cd033dbda8df4ada610943caa858.jpg"
img0 = cv2.imread(img_path)
img = cv2.resize(img0, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
img = torch.from_numpy(img.copy()).float() / 255.0  # to float and normalize
img = img.unsqueeze(0)  # add batch dim: (1, 3, 640, 640)

#=== 5) 執行推論，觸發 forward hooks ===#
with torch.no_grad():
    _ = model.model(img)

#=== 6) 可視化其中幾層特徵圖 ===#
output_dir = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose./feature_maps_output"
os.makedirs(output_dir, exist_ok=True)

# 顯示前 N 層（太多會很慢）
N = 5
selected_layers = list(feature_maps.keys())[:N]

for layer_name in selected_layers:
    fmap = feature_maps[layer_name].squeeze(0)  # (C, H, W)
    print(f"Layer: {layer_name} - Feature shape: {fmap.shape}")
    
    # 每層最多顯示前 6 個通道
    for i in range(min(6, fmap.shape[0])):
        plt.imshow(fmap[i], cmap='viridis')
        plt.title(f"{layer_name} - channel {i}")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{layer_name.replace('.', '_')}_ch{i}.png"))
        plt.close()

print(f"完成，特徵圖已輸出至：{output_dir}")
