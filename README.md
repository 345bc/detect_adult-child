# Adult & Child Detection

Nhận diện người lớn và trẻ em theo thời gian thực bằng YOLO trên ảnh, video và webcam.

![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Framework](https://img.shields.io/badge/ultralytics-≥8.4.37-orange?style=flat-square)


## Tính năng

- Nhận diện 2 lớp: **Adult** (xanh lá) và **Child** (đỏ)
- Hỗ trợ 3 nguồn đầu vào: webcam, video, ảnh tĩnh
- Chạy GPU (RTX 2050 / 4 GB VRAM) với độ trễ thấp
- Fine-tune từ `yolo26s.pt` — 100 epoch, batch 8, imgsz 640
- Bounding box + confidence score hiển thị trực tiếp qua OpenCV


## Tech stack

Python 3.9+ · Ultralytics YOLO · OpenCV · NumPy · scikit-learn · pandas


## Bắt đầu

### Yêu cầu

- Python 3.9+
- CUDA toolkit (để chạy GPU)
- GPU có ≥ 4 GB VRAM (khuyến nghị RTX 2050 trở lên)

### Cài đặt

```bash
git clone https://github.com/<your-username>/detect_adult-child.git
cd detect_adult-child
pip install -r requirements.txt
```

### Chạy

```bash
python main.py
```


## Sử dụng

Chỉnh `source` trong `main.py` trước khi chạy:

```python
# Webcam
source = 0

# Video
source = r"test_case/test_video_00001.mp4"

# Ảnh
source = r"datasets/test/images/test_00020.jpg"
```

Nhấn **Q** để thoát khi đang chạy video hoặc webcam.

### Huấn luyện lại

```bash
python train.py
```

Kết quả lưu tại `runs/train/adult_child_v1/weights/best.pt`.


## Cấu hình

| Biến / Tham số | Mặc định | Mô tả |
|---|---|---|
| `model_path` | `runs/detect/runs/train/adult_child_v1/weights/best.pt` | Đường dẫn tới file trọng số |
| `source` | `datasets/test/images/test_00020.jpg` | Nguồn đầu vào (int = webcam) |
| `conf` | `0.5` | Ngưỡng confidence tối thiểu |
| `device` | `0` | GPU index (`0`) hoặc `"cpu"` |
| `epochs` | `100` | Số epoch huấn luyện |
| `batch` | `8` | Batch size khi train |


## Cấu trúc dự án

```text
detect_adult-child/
├── datasets/           # Ảnh + nhãn (train / valid / test)
├── runs/               # Kết quả train và inference
├── utils/              # Công cụ hỗ trợ (labeling, rename)
├── test_case/          # Video / ảnh test thực tế
├── data.yaml           # Cấu hình dataset cho YOLO
├── main.py             # Script inference
├── train.py            # Script huấn luyện
├── requirements.txt    # Thư viện phụ thuộc
└── yolo26s.pt          # Trọng số pre-trained
```


## Đóng góp

1. Fork repo và tạo branch từ `main`.
2. Commit theo format: `feat:`, `fix:`, `docs:`.
3. Mở Pull Request — mô tả rõ thay đổi và lý do.

<!-- TODO: thêm CONTRIBUTING.md -->


## License

MIT
