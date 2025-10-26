from ultralytics import YOLO
import cv2
import math
import os

def pixel_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def main():
    # === 一、載入模型 ===
    model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train2\weights\best.pt")

    # === 二、推論影像資料夾 ===
    source = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data2\test\images\11-3-115-2_jpg.rf.050985bf4681570f7e03cf82cbdf5929.jpg"
    results = model.predict(source=source, conf=0.5)

    # === 三、逐張處理結果 ===
    for i, result in enumerate(results):
        img_path = result.path
        img = cv2.imread(img_path)
        H, W, _ = img.shape

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            print(f"[{i}] 無偵測結果：{img_path}")
            continue

        # 分類與關鍵點列表
        keypoints_list = result.keypoints.xy
        class_list = result.boxes.cls.cpu().numpy().astype(int)  # 每個instance的類別

        # 找出類別 0（參照物）與 類別 1（苗株）各一組 keypoints
        ref_kpt = None
        plant_kpt = None
        for cls, kpt in zip(class_list, keypoints_list):
            if cls == 0 and ref_kpt is None:
                ref_kpt = kpt.cpu().numpy()
            elif cls == 1 and plant_kpt is None:
                plant_kpt = kpt.cpu().numpy()

        # 缺少任一組跳過
        if ref_kpt is None or plant_kpt is None:
            print(f"[{os.path.basename(img_path)}] 無法同時偵測到參照物與苗株，跳過。")
            continue

        # 確保關鍵點數量正確
        if len(ref_kpt) < 4 or len(plant_kpt) < 4:
            print(f"[{os.path.basename(img_path)}] 任一類別的關鍵點不足 4 個，跳過。")
            continue

        # 合併為 8 點（list of (x, y)）
        all_kpts = list(ref_kpt) + list(plant_kpt)
        pixel_points = [(int(x * W), int(y * H)) for x, y in all_kpts]

        # === 四、繪製關鍵點 ===
        for pt in pixel_points:
            cv2.circle(img, pt, 4, (0, 0, 255), -1)

        # === 五、計算參照物上下寬度並求平均 ===
        ref_top = pixel_distance(pixel_points[0], pixel_points[1])
        ref_bottom = pixel_distance(pixel_points[2], pixel_points[3])
        ref_avg_width = (ref_top + ref_bottom) / 2
        REF_REAL_WIDTH_CM = 10.0

        if ref_avg_width == 0:
            print(f"[{os.path.basename(img_path)}] 參照物像素寬度為 0，無法換算")
            continue

        px_to_cm_ratio = REF_REAL_WIDTH_CM / ref_avg_width

        # === 六、計算苗株長度（頂端→中段→根部上）===
        dist_plant_px = (
            pixel_distance(pixel_points[4], pixel_points[5]) +
            pixel_distance(pixel_points[5], pixel_points[6])
        )
        dist_plant_cm = dist_plant_px * px_to_cm_ratio

        # === 七、輸出與顯示 ===
        print(f"檔案: {os.path.basename(img_path)}")
        print(f"參照物上寬: {ref_top:.2f}px, 下寬: {ref_bottom:.2f}px, 平均: {ref_avg_width:.2f}px")
        print(f"轉換比例: 1 px = {px_to_cm_ratio:.4f} cm")
        print(f"苗株像素長度: {dist_plant_px:.2f}px → 真實長度: {dist_plant_cm:.2f} cm")

        cv2.putText(img, f"Length: {dist_plant_cm:.2f} cm", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Length Measurement", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
