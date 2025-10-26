import cv2
import math

def pixel_distance(p1, p2):
    """計算兩點之間的像素距離"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def main():
    # 影像路徑（依據你的檔案位置）
    img_path = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data1\test\images\1-1-115-2_jpg.rf.8ce70c25647e5cd6cb524f87a48d4ee8.jpg"
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    
    # === 一、定義參照物和植株點（範例）===
    # 注意：這裡假設前四點是參照物 (長方形四角)
    # 接下來四點是植株: 頂端→輔助點→根部上→根部下
    normalized_points = [
        # 參照物四角
        (0.472156, 0.133756),  # 左上
        (0.473331, 0.179653),  # 左下
        (0.565375, 0.133861),  # 右上 (假設位置，請自行修正)
        (0.56738, 0.178428),  # 右下 (假設位置，請自行修正)
        # 植株四點
        (0.212837, 0.445415),  # 頂端
        (0.425392, 0.402187),  # 輔助點
        (0.645856, 0.360794),  # 根部斷層處上方
        (0.716206,  0.360442),  # 根部斷層處下方
        
    ]

    # === 二、轉為像素座標並繪製點 ===
    pixel_points = []
    for (x_rel, y_rel) in normalized_points:
        x_px = int(x_rel * W)
        y_px = int(y_rel * H)
        pixel_points.append((x_px, y_px))
        cv2.circle(img, (x_px, y_px), 4, (0, 0, 255), -1)

    # === 三、計算參照物真實長度與像素距離 ===
    # 假設參照物真實寬度為10公分（可依實際情況更改）
    REF_REAL_WIDTH_CM = 10.0

    # 計算參照物的頂邊與底邊平均作為換算依據（較準確）
    top_width = pixel_distance(pixel_points[0], pixel_points[3])
    bottom_width = pixel_distance(pixel_points[1], pixel_points[2])
    avg_ref_width_px = (top_width + bottom_width) / 2

    if avg_ref_width_px == 0:
        print("參照物寬度為0，無法換算。")
        return

    px_to_cm_ratio = REF_REAL_WIDTH_CM / avg_ref_width_px
    print(f"參照物平均像素寬度: {avg_ref_width_px:.2f} px, 實際寬度: {REF_REAL_WIDTH_CM:.2f} cm, 參照物像素寬度:{top_width:.2f} px,參照物像素寬度:{bottom_width:.2f} px")
    print(f"換算比例為: 1 px = {px_to_cm_ratio:.4f} cm")

    # === 四、計算植株前三點總距離 ===
    plant_pts = pixel_points[4:7]  # 植株前三點

    # 計算總像素距離
    total_plant_px_dist = (
        pixel_distance(plant_pts[0], plant_pts[1]) +
        pixel_distance(plant_pts[1], plant_pts[2])
    )

    # 換算真實長度
    total_plant_real_dist_cm = total_plant_px_dist * px_to_cm_ratio

    print(f"植株前三點像素總距離: {total_plant_px_dist:.2f} px")
    print(f"植株前三點真實總距離: {total_plant_real_dist_cm:.2f} cm")

    # === 五、顯示結果 ===
    cv2.imshow("Annotated Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
