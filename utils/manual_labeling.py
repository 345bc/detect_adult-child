from ultralytics import YOLO
import cv2
import os
import random
import shutil

# ===== 1. CẤU HÌNH =====
root_input = "dataroot"
# root_output = "datasets_resnet"
root_output = "datasets"
model_path = "yolo26s.pt"

if not os.path.exists(root_input):
    print(f"❌ Lỗi: Không tìm thấy thư mục {root_input}!")
    exit()

model = YOLO(model_path) 

def show_keep_ratio(window_name, image, target_size=600):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cv2.imshow(window_name, resized)

# ===== 2. CHUẨN BỊ DANH SÁCH =====
images = [f for f in os.listdir(root_input) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

total = len(images)
train_end = int(0.7 * total)
val_end   = int(0.9 * total)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

for split in splits:
    os.makedirs(f"{root_output}/{split}/images", exist_ok=True)
    os.makedirs(f"{root_output}/{split}/labels", exist_ok=True)

# ===== 3. QUY TRÌNH XỬ LÝ =====
for split, img_list in splits.items():
    print(f"\n📂 ĐANG XỬ LÝ TẬP: {split.upper()}")
    # Lấy counter dựa trên số lượng file đã có sẵn trong folder để không bị ghi đè khi chạy lại
    counter = len(os.listdir(f"{root_output}/{split}/images")) + 1

    for img_name in img_list:
        img_path = os.path.join(root_input, img_name)
        
        # --- CẬP NHẬT QUAN TRỌNG: Kiểm tra file tồn tại ---
        if not os.path.exists(img_path):
            continue # Nếu file đã bị move trước đó thì bỏ qua

        img = cv2.imread(img_path)
        if img is None: continue

        h, w, _ = img.shape
        results = model(img, device=0, classes=[0], conf=0.4)

        labels_to_write = []
        has_label = False

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if crop.size == 0: continue

                temp = img.copy()
                cv2.rectangle(temp, (x1,y1), (x2,y2), (0,255,0), 2)
                
                show_keep_ratio("FULL IMAGE - GREEN BOX", temp, 800)
                show_keep_ratio("CROP - 0:Adult | 1:Child", crop, 400)

                print(f"--- [{split}] Image {counter}: {img_name} ---")
                print("Press: [0]=Adult  [1]=Child  [S]=Skip Box  [Q]=Quit")

                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
                elif key == ord('0'): cls = 0
                elif key == ord('1'): cls = 1
                else: 
                    print("⏭️ Skip.")
                    continue

                x_c = ((x1 + x2) / 2) / w
                y_c = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                labels_to_write.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
                has_label = True

        # ===== 4. LƯU VÀ DI CHUYỂN =====
        if has_label:
            ext = img_name.rsplit('.', 1)[-1]
            new_name = f"{split}_{counter:05d}.{ext}"
            new_txt_name = f"{split}_{counter:05d}.txt"

            # Di chuyển file thay vì copy để dọn dẹp dataroot
            try:
                shutil.move(img_path, os.path.join(root_output, split, "images", new_name))
                with open(os.path.join(root_output, split, "labels", new_txt_name), "w") as f:
                    f.write("\n".join(labels_to_write))
                counter += 1
            except Exception as e:
                print(f"❌ Lỗi khi di chuyển file: {e}")

cv2.destroyAllWindows()
print("\n✅ HOÀN THÀNH! Dataroot của bạn sẽ sạch bóng những ảnh đã làm.")