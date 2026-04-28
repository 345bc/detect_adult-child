# 🕵️ Adult & Child Detection System (YOLO)

Dự án nhận diện Người lớn (Adult) và Trẻ em (Child) sử dụng framework YOLO26 (Ultralytics). Hệ thống được tối ưu hóa để chạy trên các dòng máy có GPU tầm trung (như RTX 2050).

---

## 🏗️ Cấu trúc dự án

```text
detect_adult-child/
├── datasets/           # Dữ liệu huấn luyện (images, labels)
├── runs/               # Kết quả sau khi huấn luyện (weights, logs)
├── src/                # Các công cụ bổ trợ (labeling, rename, utils)
├── test_case/          # Video/Ảnh dùng để test thực tế
├── data.yaml           # Cấu hình đường dẫn dataset cho YOLO
├── main.py             # Script chạy nhận diện (Inference)
├── train.py            # Script huấn luyện mô hình
├── requirements.txt    # Danh sách thư viện cần thiết
└── yolo26s.pt          # Trọng số mô hình gốc (Pre-trained)
```

---

## 🚀 Cài đặt

1. **Yêu cầu hệ thống:** Python 3.9+ và CUDA (nếu muốn chạy GPU).
2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (`train.py`)

Đảm bảo bạn đã chuẩn bị dataset trong thư mục `datasets/` và cấu hình đúng file `data.yaml`.

```bash
python train.py
```

_Mặc định script sẽ sử dụng GPU 0 (RTX 2050) với batch size là 8._

### 2. Chạy nhận diện (`main.py`)

Script hỗ trợ 3 chế độ truyền dữ liệu đầu vào:

- **Webcam:** Đặt `source = 0`
- **Video:** Đặt `source = "đường/dẫn/video.mp4"`
- **Ảnh:** Đặt `source = "đường/dẫn/ảnh.jpg"`

```bash
python main.py
```

_Nhấn phím **'Q'** để thoát khi đang chạy video/webcam._

### 3. Công cụ hỗ trợ (`src/`)

- `manual_labeling.py`: Công cụ hỗ trợ gán nhãn thủ công (nếu cần).
- `rename.py`: Đổi tên file hàng loạt để chuẩn hóa dataset.

---

## 📊 Cấu hình Dataset (`data.yaml`)

```yaml
path: C:/Users/Tuan/Desktop/detect_adult-child/datasets
train: train/images
val: valid/images
test: test/images

names:
  0: Adult
  1: Child
```

---

## 📝 Nhật ký cập nhật

- **V1.0:** Khởi tạo dự án, thiết lập môi trường YOLO11.
- **V1.1:** Fix lỗi Unicode escape path trên Windows.
- **V1.2:** Tối ưu hóa logic xử lý ảnh/video trong `main.py`.

---

_Phát triển bởi **Tuấn** - HUIT._
