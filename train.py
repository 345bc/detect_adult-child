from ultralytics import YOLO
import torch

def train_model():
    # 1. Khởi tạo mô hình YOLO26s (bản small phù hợp với RTX 2050)
    model = YOLO("yolo26s.pt")

    # 2. Cấu hình huấn luyện
    model.train(
        data="data.yaml",    # File cấu hình dataset Tuấn vừa tạo
        epochs=100,          # Số vòng lặp huấn luyện (có thể tăng lên nếu chưa hội tụ)
        imgsz=640,           # Kích thước ảnh chuẩn của YOLO
        batch=8,            # Số lượng ảnh xử lý cùng lúc (RTX 2050 4GB để 16 là vừa đẹp)
        device=0,            # Ép buộc chạy trên GPU (RTX 2050)
        workers=4,           # Số luồng xử lý dữ liệu (tùy CPU, thường để 4 hoặc 8)
        project="runs/train",# Thư mục lưu kết quả
        name="adult_child_v1",# Tên đợt train này
        exist_ok=True,       # Ghi đè lên folder cũ nếu cùng tên
        pretrained=True,     # Sử dụng trọng số đã huấn luyện sẵn để học nhanh hơn
        optimizer='auto',    # Tự động chọn thuật toán tối ưu (AdamW hoặc SGD)
        verbose=True
    )

if __name__ == '__main__':
    # Kiểm tra xem có nhận GPU không
    if torch.cuda.is_available():
        print(f"✅ Đang chạy trên GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Không tìm thấy GPU, code sẽ chạy trên CPU (rất chậm)!")
        
    train_model()