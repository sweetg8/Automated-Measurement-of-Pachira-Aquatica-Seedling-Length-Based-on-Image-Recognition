# from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
# model = YOLO("yolo11m-pose.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data=r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\Lib\site-packages\ultralytics\cfg\datasets\coco8-pose.yaml", epochs=10, imgsz=640)

from ultralytics import YOLO

def run_training():
    model = YOLO("yolo11m-pose.yaml").load("yolo11m-pose.pt")
    results = model.train(
        data=r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\Lib\site-packages\ultralytics\cfg\datasets\coco8-pose.yaml",
        epochs=10,
        imgsz=640
    )

if __name__ == '__main__':
    run_training()