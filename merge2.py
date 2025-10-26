from ultralytics import YOLO
import cv2
import math
import os
import requests

# LINE 廣播設定
CHANNEL_ACCESS_TOKEN = 'OcKhyT9AjLLkn1OfFCqDTtHx9UtCP1AuOMtJJ9XT3qp+gWjHhPT2pYmY3sHCqWmgHfrYpH2Ox1p0jjoBsmPclWs+sdGDy8+KL7RzzG4JrbuD6NcEVkN3PgXFmjbqeRqqTqWrxsyq2F82BNlyk2TKXgdB04t89/1O/w1cDnyilFU='
LINE_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"
}


def send_line_message(text):
    data = {
        "messages": [
            {
                "type": "text",
                "text": text
            }
        ]
    }
    response = requests.post(
        "https://api.line.me/v2/bot/message/broadcast",
        headers=LINE_HEADERS,
        json=data
    )
    print("LINE 狀態碼:", response.status_code)
    print("LINE 回應內容:", response.text)


def pixel_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def main():
    model = YOLO(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\runs\pose\train2\weights\best.pt")
    source = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data2\test\images\2-2-115-1_jpg.rf.0e271a31f6f0fe2885fd08a4ae79898c.jpg"
    results = model.predict(source=source, conf=0.5)

    for i, result in enumerate(results):
        img_path = result.path
        img = cv2.imread(img_path)
        H, W, _ = img.shape

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            msg = f"❌ [{os.path.basename(img_path)}] 無偵測結果。"
            print(msg)
            send_line_message(msg)
            continue

        keypoints_list = result.keypoints.xy
        class_list = result.boxes.cls.cpu().numpy().astype(int)

        ref_kpt = None
        plant_kpt = None
        for cls, kpt in zip(class_list, keypoints_list):
            if cls == 0 and ref_kpt is None:
                ref_kpt = kpt.cpu().numpy()
            elif cls == 1 and plant_kpt is None:
                plant_kpt = kpt.cpu().numpy()

        if ref_kpt is None or plant_kpt is None:
            msg = f"⚠️ [{os.path.basename(img_path)}] 無法同時偵測到參照物與苗株。"
            print(msg)
            send_line_message(msg)
            continue

        if len(ref_kpt) < 4 or len(plant_kpt) < 4:
            msg = f"⚠️ [{os.path.basename(img_path)}] 任一類別的關鍵點不足 4 個，跳過。"
            print(msg)
            send_line_message(msg)
            continue

        all_kpts = list(ref_kpt) + list(plant_kpt)
        pixel_points = [(int(x * W), int(y * H)) for x, y in all_kpts]

        for pt in pixel_points:
            cv2.circle(img, pt, 4, (0, 0, 255), -1)

        ref_top = pixel_distance(pixel_points[0], pixel_points[1])
        ref_bottom = pixel_distance(pixel_points[2], pixel_points[3])
        ref_avg_width = (ref_top + ref_bottom) / 2
        REF_REAL_WIDTH_CM = 10.0

        if ref_avg_width == 0:
            msg = f"❌ [{os.path.basename(img_path)}] 參照物像素寬度為 0，無法換算。"
            print(msg)
            send_line_message(msg)
            continue

        px_to_cm_ratio = REF_REAL_WIDTH_CM / ref_avg_width

        dist_plant_px = (
            pixel_distance(pixel_points[4], pixel_points[5]) +
            pixel_distance(pixel_points[5], pixel_points[6])
        )
        dist_plant_cm = dist_plant_px * px_to_cm_ratio

        msg = (
            
            f"📏 參照物上寬: {ref_top:.2f}px, 下寬: {ref_bottom:.2f}px\n"
            f"⚖️ 像素轉換比例: 1 px = {px_to_cm_ratio:.4f} cm\n"
            f"🌱 苗株真實長度: {dist_plant_cm:.2f} cm"
        )
        print(msg)
        send_line_message(msg)

        cv2.putText(img, f"Length: {dist_plant_cm:.2f} cm", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Length Measurement", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
