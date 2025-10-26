import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# 設定影像和標註的路徑
image_dir = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data2\train\images"
annotation_file = r"C:\Users\Tim12\anaconda3\envs\YOLOv11_3\project\data2\train\labels\1_jpg.rf.41f28b33fe849f8d4f1d3bfe13fc1239.txt"

# 初始化 COCO API
coco = COCO(annotation_file)

# 獲取所有影像的 ID
image_ids = coco.getImgIds()

# 隨機選擇一個影像 ID
image_id = image_ids[0]

# 加載影像資訊
image_info = coco.loadImgs(image_id)[0]
image_path = os.path.join(image_dir, image_info['1_jpg.rf.41f28b33fe849f8d4f1d3bfe13fc1239'])

# 讀取影像
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 獲取該影像的所有標註 ID
annotation_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(annotation_ids)

# 繪製影像和標註
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis('off')

# 繪製標註的邊界框
for annotation in annotations:
    # 獲取邊界框座標
    bbox = annotation['bbox']
    x, y, width, height = bbox
    rect = plt.Rectangle((x, y), width, height, fill=False, color='red', linewidth=2)
    plt.gca().add_patch(rect)

    # 獲取類別名稱
    category_id = annotation['category_id']
    category_name = coco.loadCats(category_id)[0]['name']
    plt.text(x, y, category_name, color='red', fontsize=12, backgroundcolor='white')

plt.show()