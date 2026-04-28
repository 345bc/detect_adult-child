# Adult & Child Detection

Nhận diện người lớn và trẻ em theo thời gian thực bằng YOLO trên ảnh, video và webcam.

![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)
![Ultralytics](https://img.shields.io/badge/ultralytics-%3E%3D8.4.37-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)


## Tính năng

- Nhận diện 2 lớp: **Adult** (xanh lá) và **Child** (đỏ)
- Hỗ trợ 3 nguồn đầu vào: webcam, video, ảnh tĩnh
- Chạy GPU với độ trễ thấp, tối ưu cho RTX 2050 (4 GB VRAM)
- Fine-tune từ `yolo26s.pt` — 100 epoch, batch 8, imgsz 640
- Bounding box + confidence score hiển thị trực tiếp qua OpenCV


## Tech stack

Python 3.9+ · ultralytics ≥8.4.37 · opencv-python · numpy · pandas · scikit-learn · joblib


## Bắt đầu

### Yêu cầu

- Python 3.9+
- CUDA toolkit tương thích với PyTorch
- GPU ≥ 4 GB VRAM (khuyến nghị)

### Cài đặt

```bash
git clone https://github.com/<!-- TODO: verify -->/detect_adult-child.git
cd detect_adult-child
pip install -r requirements.txt
```

### Chạy inference

```bash
python main.py
```


## Sử dụng

Chỉnh biến `source` trong `main.py` trước khi chạy:

```python
source = 0                                          # Webcam
source = r"test_case/test_video_00001.mp4"          # Video
source = r"datasets/test/images/test_00020.jpg"     # Ảnh
```

Nhấn **Q** để thoát khi đang chạy video hoặc webcam.

### Huấn luyện lại

```bash
python train.py
```

Trọng số tốt nhất lưu tại `runs/train/adult_child_v1/weights/best.pt`.


## Cấu hình

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `model_path` | `runs/detect/runs/train/adult_child_v1/weights/best.pt` | Đường dẫn file trọng số |
| `source` | `datasets/test/images/test_00020.jpg` | Nguồn đầu vào (`int` = webcam) |
| `conf` | `0.5` | Ngưỡng confidence tối thiểu |
| `device` | `0` | GPU index hoặc `"cpu"` |
| `epochs` | `100` | Số epoch khi train |
| `batch` | `8` | Batch size khi train |
| `imgsz` | `640` | Kích thước ảnh đầu vào |


## Cấu trúc dự án

```text
detect_adult-child/
├── datasets/           # Ảnh + nhãn (train / valid / test)
├── runs/               # Kết quả train và inference
├── utils/              # Công cụ hỗ trợ (labeling, rename)
├── test_case/          # Video / ảnh dùng để test
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
