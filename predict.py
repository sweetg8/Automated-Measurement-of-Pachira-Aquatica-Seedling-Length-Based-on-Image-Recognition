from ultralytics import YOLO

def main():
    # 1) 載入訓練完成的模型權重 (best.pt 為示例，可替換實際檔名)
    model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train2\weights\best.pt")

    # 2) 指定要推論的來源 (可以是單張影像、資料夾、影片路徑、攝影機ID 等)
    source = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\datatest\pic\newbb\2-115.jpg" # or "path/to/folder" / "test.mp4" / 0 (for webcam)

    # 3) 進行推論
    # conf=0.25 => 置信度閾值
    # save=True => 輸出畫有結果的圖片/影片到 runs/predict/ 資料夾
    # save_txt=True => 同時輸出標註txt檔到 runs/predict/labels/ 
    results = model.predict(source=source, save=True, save_txt=True, conf=0.5)

    # 4) 分析推論結果 (以單張圖片為例，通常 results[0] 對應第一張影像)
    print(results)  # 可顯示檢測框、分類、關鍵點等資訊
    # 例如查看框數量
    print(f"Detections on {source}: {len(results[0].boxes)} boxes")

if __name__ == "__main__":
    main()