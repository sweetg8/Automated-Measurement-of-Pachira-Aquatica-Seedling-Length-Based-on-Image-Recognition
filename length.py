import cv2
import math

def pixel_distance(x1, y1, x2, y2):
    """
    計算平面上兩點 (x1, y1) 與 (x2, y2) 的像素距離
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    #=== 1) 讀取影像 ===#
    img_path = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data5\test\images\28_jpg.rf.655f0aa5dd0e2a6ed8c89da444e9ce69.jpg"
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    
    #=== 2) 定義相對座標 (x_rel, y_rel) ===#
    #   這裡以你給的 normalized_points 為例
    normalized_points = [
        (0.472156  ,0.133756), # 假設作為參考物的一端 (例: P0)
        (0.494536 ,0.269336), # 參考物另一端 (例: P1)
        (0.316983 ,0.647332),  # 目標物的一端 (例: P2)
        (0.806973 ,0.569319)   # 目標物另一端 (例: P3)
    ]
    
    #=== 3) 轉成像素座標並畫點 ===#
    pixel_points = []
    for (x_rel, y_rel) in normalized_points:
        # 轉成像素座標
        x_px = int(x_rel * W)
        y_px = int(y_rel * H)
        pixel_points.append((x_px, y_px))
        
        # 以 (x_px, y_px) 當作圓心，畫一個半徑 5 的紅色實心圓點
        cv2.circle(img, (x_px, y_px), 4, (0, 0, 255), -1)

    #=== 4) 假設前兩個點(P0, P1)是參考物，後兩個點(P2, P3)是目標物 ===#
    #  取得參考物的像素距離
    ref_px_dist = pixel_distance(pixel_points[0][0], pixel_points[0][1],
                                 pixel_points[1][0], pixel_points[1][1])
    
    #=== 5) 已知參考物真實長度 (可自行改單位，ex: cm, mm) ===#
    REF_REAL_LENGTH = 5.0  # 例如 10 cm
    
    if ref_px_dist == 0:
        print("參考物兩點重疊，像素距離 = 0，無法換算。")
        return
    
    #=== 6) 計算「1 像素 對應 幾公分」 ===#
    px_to_cm = REF_REAL_LENGTH / ref_px_dist
    print(f"參考物像素距離 = {ref_px_dist:.2f} px, 真實長度 = {REF_REAL_LENGTH} cm")
    print(f"換算比例: 1 px ~ {px_to_cm:.4f} cm")

    #=== 7) 取得目標物的像素距離 (P2, P3) ===#
    obj_px_dist = pixel_distance(pixel_points[2][0], pixel_points[2][1],
                                 pixel_points[3][0], pixel_points[3][1])
    
    #=== 8) 換算目標物真實長度 ===#
    obj_real_length = obj_px_dist * px_to_cm
    print(f"目標物像素距離 = {obj_px_dist:.2f} px, 推算真實長度 = {obj_real_length:.2f} cm")

    #=== 9) 顯示影像結果 ===#
    cv2.imshow("Points on Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()