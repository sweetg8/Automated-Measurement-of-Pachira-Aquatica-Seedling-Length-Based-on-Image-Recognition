from ultralytics import YOLO

def main():
    #model = YOLO("yolo11m-pose.yaml")  # 如果有自定義架構就用它，若無此檔可省略
    model = YOLO("yolo11m-pose.yaml").load("yolo11m-pose.pt") # 載入預訓練權重
    
    results = model.train(
    data=r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data6\data.yaml",  # 你自己的資料集 yaml
    epochs=60,                # 訓練回合數
    imgsz=640,                # 輸入影像大小
    batch=-1,                  # batch size 視顯卡而定
    #workers=4,                # 資料載入的子進程數 (Windows 上需注意 if __name__ == '__main__')
    #device=0                  # 指定用哪個 GPU，0 代表第一張卡
    )














if __name__ == '__main__':
    main()