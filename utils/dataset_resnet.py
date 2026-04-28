import cv2
import os
import random
from pathlib import Path

# ===== CONFIG =====
# IMAGE_DIR = r"datasets/train/images" 
# LABEL_DIR = r"datasets/train/labels"
IMAGE_DIR = r"datasets_resnet/train/images" 
LABEL_DIR = r"datasets_resnet/train/labels"
OUTPUT_ROOT = r"resnet_data"
CLASSES = ["adult", "child"]

# Tỉ lệ chia
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

def split_crop_and_rename():
    # 1. Tạo cấu trúc folder
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, cls), exist_ok=True)

    # 2. Thu thập và xáo trộn
    image_files = list(Path(IMAGE_DIR).glob("*.jpg")) + list(Path(IMAGE_DIR).glob("*.png"))
    random.seed(42) # Cố định seed để nếu chạy lại kết quả vẫn giống nhau
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    # Biến đếm để đổi tên (reset cho mỗi split/class)
    # Cấu trúc: counters['train']['adult'] = 0
    counters = {split: {cls: 1 for cls in CLASSES} for split in ['train', 'val', 'test']}

    print(f"📦 Tổng cộng: {total} ảnh gốc. Đang xử lý...")

    for i, img_path in enumerate(image_files):
        # Xác định tập split
        if i < train_end: current_split = 'train'
        elif i < val_end: current_split = 'val'
        else: current_split = 'test'

        img = cv2.imread(str(img_path))
        label_path = os.path.join(LABEL_DIR, img_path.stem + ".txt")
        if img is None or not os.path.exists(label_path): continue
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) < 5: continue
                
                cls_id = int(data[0])
                class_name = CLASSES[cls_id]
                x_c, y_c, wb, hb = map(float, data[1:])

                # Tính tọa độ pixel
                x1 = int((x_c - wb/2) * w)
                y1 = int((y_c - hb/2) * h)
                x2 = int((x_c + wb/2) * w)
                y2 = int((y_c + hb/2) * h)

                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                
                if crop.size > 0:
                    # ĐỔI TÊN Ở ĐÂY: format nhãn_stt.jpg (vd: adult_001.jpg)
                    num = counters[current_split][class_name]
                    new_name = f"{class_name}_{num:05d}.jpg" 
                    
                    save_path = os.path.join(OUTPUT_ROOT, current_split, class_name, new_name)
                    cv2.imwrite(save_path, crop)
                    
                    counters[current_split][class_name] += 1

    print("✅ Đã hoàn thành đổi tên và chia tập dữ liệu!")
    for split in ['train', 'val', 'test']:
        print(f"--- {split.upper()} ---")
        for cls in CLASSES:
            print(f" + {cls}: {counters[split][cls]-1} ảnh")

if __name__ == "__main__":
    split_crop_and_rename()