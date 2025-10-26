import cv2
import math

img = cv2.imread(r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data5\test\images\28_jpg.rf.655f0aa5dd0e2a6ed8c89da444e9ce69.jpg")
H, W, _ = img.shape

normalized_points = [
    (0.771274 ,0.588811),
    ( 0.278597, 0.616326),
    (0.495038 ,0.26033),
    (0.444426 ,0.26184)
]

# 3. 繪製點
for (x_rel, y_rel) in normalized_points:
    # 轉成像素座標
    x_px = int(x_rel * W)
    y_px = int(y_rel * H)
   
    # 以 (x_px, y_px) 當作圓心，畫一個半徑 5 的紅色實心圓點
    cv2.circle(img, (x_px, y_px), 5, (0, 0, 255), -1)


def pixel_distance(x1, y1, x2, y2):
    """
    計算平面上兩點 (x1, y1) 與 (x2, y2) 的像素距離
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
# 4. 顯示結果
cv2.imshow("Points on Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()