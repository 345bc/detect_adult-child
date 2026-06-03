from ultralytics import YOLO
import torch


def train_model():
    model = YOLO("yolo26s.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        project="runs/train",
        name="adult_child_v1",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"✅ Đang chạy trên GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Không tìm thấy GPU, code sẽ chạy trên CPU (rất chậm)!")

    train_model()
