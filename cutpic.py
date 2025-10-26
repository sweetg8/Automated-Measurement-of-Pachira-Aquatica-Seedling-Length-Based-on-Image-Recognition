import os
from PIL import Image

# 原始影像所在的資料夾路徑
input_folder = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\image\good"  # <-- 請換成你的資料夾路徑

# 輸出裁切後影像要放的資料夾路徑（建議用不同資料夾儲存）
output_folder = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\image\better"  # <-- 請換成你的輸出資料夾路徑
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 設定裁切後的高度
new_height = 280

# 批次處理資料夾內所有影像
for filename in os.listdir(input_folder):
    # 只處理常見的圖片格式
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # 取得完整路徑
        input_path = os.path.join(input_folder, filename)
        
        # 開啟影像
        with Image.open(input_path) as img:
            width, height = img.size  # 預期為 (640, 480)
            
            # 如果影像符合 640x480，才進行裁切
            # 否則可以自行決定要不要跳過或做其他處理
            if width == 640 and height == 480:
                # 【方法一】從頂部裁切，保留 640×280
                # left, top, right, bottom
                crop_box = (0, 0, 640, 280)
                cropped_img = img.crop(crop_box)
               # 儲存裁切後的影像

                output_path = os.path.join(output_folder, filename)
                cropped_img.save(output_path)
                
                print(f"已裁切並儲存：{output_path}")
            else:
                print(f"跳過 (尺寸不符合640x480)：{filename}")