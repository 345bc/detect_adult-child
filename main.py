# import cv2
# from ultralytics import YOLO

# # ===== CONFIG =====
# # Dùng r"..." để tránh lỗi ký tự đặc biệt trong đường dẫn Windows
# model_path = r"runs/detect/runs/train/adult_child_v1/weights/best.pt"
# # source = r"C:\Users\Tuan\Desktop\detect_adult-child\datasets\test\images\test_00006.jpg"  # 0: Webcam | "path/to/video.mp4"
# source = 0
# def run_inference():
#     # 1. Load model an toàn
#     model = YOLO(model_path)
    
#     # 2. Gọi fuse riêng để tối ưu (nếu lỗi ở đây thì có thể xóa dòng này)
#     try:
#         model.fuse()
#     except Exception:
#         print("⚠️ Không thể fuse model, vẫn tiếp tục chạy...")

#     # 3. Chạy nhận diện
#     # device=0 để ép chạy trên card RTX 2050
#     results = model.predict(
#         source=source, 
#         conf=0.5, 
#         show=False, 
#         stream=True, 
#         device=0
#     )

#     print("🚀 Đang khởi chạy... Nhấn 'Q' tại cửa sổ ảnh để thoát.")

#     for r in results:
#         img = r.orig_img.copy()

#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])

#             # Phân loại màu sắc: Adult (Xanh lá), Child (Đỏ)
#             # model.names sẽ lấy đúng tên từ file data.yaml của bạn
#             label = f"{model.names[cls]} {conf:.2f}"
#             color = (0, 255, 0) if cls == 0 else (0, 0, 255)

#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label, (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         cv2.imshow("Adult - Child Detection", img)

#         # Thoát khi nhấn 'q'
#         if isinstance(source, int): 
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             print("📸 Đã xử lý xong ảnh. Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
#             cv2.waitKey(0) 
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     run_inference()
import cv2
from ultralytics import YOLO
import os

# ===== CONFIG =====
model_path = r"runs/detect/runs/train/adult_child_v1/weights/best.pt"
# source = 0                          # Chế độ 1: Webcam
# source = r"test_case/test_video_00001.mp4" # Chế độ 2: Video
source = r"datasets/test/images/test_00020.jpg" # Chế độ 3: Ảnh


print(os.path.exists(source))
def run_inference():
    # 1. Load model
    model = YOLO(model_path)
    
    # Kiểm tra xem source là ảnh hay video dựa vào đuôi file
    is_image = False
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp','.avif']:
            is_image = True

    # 2. Chạy nhận diện
    results = model.predict(
        source=source, 
        conf=0.5, 
        stream=True, 
        device=0,
        verbose=False
    )

    print(f"🚀 Đang xử lý: {source}")
    print("🎬 Nhấn 'Q' để thoát.")

    for r in results:
        img = r.orig_img.copy()

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{model.names[cls]} {conf:.2f}"
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        screen_res = (400, 400) 
        img_resized = cv2.resize(img, screen_res)

        cv2.imshow("HUIT Detection", img)

        # LOGIC DỪNG MÀN HÌNH:
        if is_image:
            # Nếu là ẢNH: Dừng lại cho đến khi nhấn phím bất kỳ
            cv2.waitKey(0)
            break 
        else:
            # Nếu là VIDEO/WEBCAM: Chờ 1ms rồi chạy frame tiếp theo
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("✅ Đã hoàn thành.")

if __name__ == "__main__":
    run_inference()