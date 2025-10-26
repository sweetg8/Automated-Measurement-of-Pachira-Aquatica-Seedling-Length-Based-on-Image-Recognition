from ultralytics import YOLO

def evaluate_model_at_different_iou():
    model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train7\weights\best.pt")

  
    print(f"\n===== 評估 IoU Threshold = 0.5 =====")
    metrics = model.val(
            data=r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data6\data.yaml",  # 指向你的資料集配置檔
            split="test",                    # 使用 test 資料集
            imgsz=640,
            task='pose',
            iou=0.5                    # 指定 IoU 門檻
        )
    print(f"mAP50-90:{metrics.box.map}")  # mAP50-95
    print(f"mAP50: {metrics.box.map50}")  # mAP50
    print(f"mAP75:{metrics.box.map75}")  # mAP75
    print(f"mAP50-90:{metrics.box.maps}")  # list of mAP50-95 for each category


if __name__ == "__main__":
    evaluate_model_at_different_iou()
