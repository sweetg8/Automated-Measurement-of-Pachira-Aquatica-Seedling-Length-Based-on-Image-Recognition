import os
import cv2
import numpy as np
import pandas as pd

# 你的特徵圖資料夾
feature_map_dir = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\feature_maps_output"


# 找所有 PNG 特徵圖
feature_files = sorted([f for f in os.listdir(feature_map_dir) if f.endswith(".png")])

# 計算 activation
activation_data = []

for file in feature_files:
    img_path = os.path.join(feature_map_dir, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        mean_val = np.mean(img)
        max_val = np.max(img)
        activation_data.append({
            "channel": file,
            "mean_activation": mean_val,
            "max_activation": max_val
        })

# 轉為表格
df = pd.DataFrame(activation_data)
df.sort_values(by="mean_activation", ascending=False, inplace=True)

# 顯示前幾個最活躍的通道
print(df.head(10))

# 可選：儲存 CSV
df.to_csv("channel_activation_summary.csv", index=False)
