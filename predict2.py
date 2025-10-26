from ultralytics import YOLO
import os

def predict_on_test():
    # 載入訓練好的最佳模型
    model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train2\weights\best.pt")

    # 定義 test 影像資料夾路徑
    test_img_dir = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data2\test\images"

    # 執行預測（將會儲存每張圖片的預測結果）
    results = model.predict(
        source=test_img_dir,    # test 圖片資料夾
        save_txt=True,          # 儲存每張圖片的預測為 .txt（YOLO格式）
        save_conf=True,         # 一併儲存信心值 confidence
        imgsz=640,              # 輸入尺寸
        batch=16,                # 根據你顯卡調整 batch size
        device=0                # 使用 GPU 編號（0 表示第一張）
    )

    print("預測完成！結果已儲存於預設的 runs\predict 資料夾中")

if __name__ == "__main__":
    predict_on_test()
