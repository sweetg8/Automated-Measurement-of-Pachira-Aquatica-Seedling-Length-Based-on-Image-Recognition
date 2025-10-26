from ultralytics import YOLO

def evaluate_model():
    # 載入你訓練完成後最好的權重
    model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train4\weights\best.pt")

    # 直接跑驗證，產生mAP結果
    metrics = model.val(
        data=r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data3\data.yaml",
        split="test",  # 明確指定使用 test 集合
        imgsz=640,     # 需與訓練時尺寸相同
        batch=16       # 自動設定 batch size
    )

    # 印出mAP的結果
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)

if __name__ == "__main__":
    evaluate_model()
