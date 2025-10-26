from ultralytics import YOLO

model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train33\weights\best.pt")          # 或 best.pt
results = model.val(
    data=r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data5\data.yaml",
    conf=0.001,
    iou=0.5,
    plots=True,
    workers=0,          # 👈 關閉 dataloader 多進程
    batch=1,            # 若顯示顯存不足可改 2、4…
    verbose=True)       # 讓 dataloader 印出每張圖

