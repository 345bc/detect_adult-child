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

# Lựa chọn chế độ chạy:
# 1. "predict": Nhận diện đối tượng (Webcam, Video, Ảnh đơn, hoặc Thư mục chứa ảnh)
# 2. "test_eval": Đánh giá các chỉ số Precision, Recall, mAP50, mAP50-95 trên tập Test (datasets/test)
mode = "predict"

# Cấu hình nguồn nhận diện (chỉ dùng cho chế độ "predict"):
# source = 0                          # Chế độ 1: Webcam
# source = r"test_case/test_video_00002.mp4"  # Chế độ 2: Video
# source = r"test_case/image_00040.jpg"  # Chế độ 3: Ảnh đơn
source = r"datasets/test/images"  # Chế độ 4: Thư mục ảnh kiểm thử (Test split)


if mode == "predict" and isinstance(source, str):
    print(f"Kiểm tra nguồn '{source}': Tồn tại = {os.path.exists(source)}")


def run_inference():
    model = YOLO(model_path)

    # Phân loại loại nguồn để xử lý sự kiện bấm phím chuyển ảnh/thoát
    is_single_image = False
    is_folder = False
    is_video_or_webcam = True

    if isinstance(source, str):
        if os.path.isdir(source):
            is_folder = True
            is_video_or_webcam = False
        elif os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".avif"]:
                is_single_image = True
                is_video_or_webcam = False

    import torch

    device = 0 if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=source, conf=0.5, stream=True, device=device, verbose=False
    )

    print(f"🚀 Đang xử lý nguồn: {source} (trên {device})")
    if is_folder:
        print(
            "🎬 Nhấn phím bất kỳ (trên cửa sổ ảnh) để sang ảnh tiếp theo. Nhấn 'Q' để thoát."
        )
    else:
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
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("detection", img)

        if is_single_image:
            cv2.waitKey(0)
            break
        elif is_folder:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    cv2.destroyAllWindows()
    print("✅ Đã hoàn thành nhận diện.")


def run_evaluation():
    print("🚀 Đang khởi chạy đánh giá mô hình trên tập kiểm thử (Test Split)...")
    model = YOLO(model_path)

    # Chạy validation trên split='test'
    metrics = model.val(data="data.yaml", split="test")

    print("\n📊 ========================================")
    print("📊       KẾT QUẢ ĐÁNH GIÁ TẬP TEST        ")
    print("📊 ========================================")
    print(f"🔹 Precision (P): {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"🔹 Recall (R):    {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"🔹 mAP50:         {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"🔹 mAP50-95:      {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
    print("===========================================")


if __name__ == "__main__":
    if mode == "test_eval":
        run_evaluation()
    else:
        run_inference()
