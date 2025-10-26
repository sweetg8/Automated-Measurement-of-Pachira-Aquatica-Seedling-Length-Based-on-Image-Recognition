from ultralytics import YOLO

model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train2\weights\best.pt")
results = model.val(data=r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data2\data.yaml", split='test', save_json=True)
