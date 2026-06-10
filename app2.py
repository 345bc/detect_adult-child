import cv2
import numpy as np
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime
import torch

# 1. KIỂM TRA VÀ PHÂN PHỐI PHẦN CỨNG CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Trạng thái phần cứng thực thi: {device.upper()}")

model = YOLO("best.pt")
model.to(device)
class_mapping = model.names

# Khởi tạo bộ đếm và trạng thái bộ nhớ đệm
counters = {"adult_in": 0, "child_in": 0, "adult_out": 0, "child_out": 0}
counted_ids = set()
track_history = {}
log_list = []

# Bộ tham số hình học cố định hiệu chuẩn chiều cao
W1, W2, BIAS = 0.35, 0.12, 45.0

WINDOW_NAME = "HE THONG SOAT VE AI - OPENCV PRO"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

def nothing(x):
    pass

# Tạo bộ thanh trượt điều khiển không dấu chuẩn OpenCV
cv2.createTrackbar("Nguong tin cay (%)", WINDOW_NAME, 40, 100, nothing)
cv2.createTrackbar("Vi tri vung bien (%)", WINDOW_NAME, 40, 100, nothing)
cv2.createTrackbar("Do day vung bien", WINDOW_NAME, 60, 200, nothing)
cv2.createTrackbar("Huong chuan VAO (0-3)", WINDOW_NAME, 0, 3, nothing)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Không thể kết nối thiết bị Webcam!")
    exit()

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    h, w, _ = frame.shape

    # Đọc thông số cấu hình từ thanh điều hướng trượt chuột
    conf_val = cv2.getTrackbarPos("Nguong tin cay (%)", WINDOW_NAME) / 100.0
    zone_start_val = cv2.getTrackbarPos("Vi tri vung bien (%)", WINDOW_NAME)
    zone_thick_val = cv2.getTrackbarPos("Do day vung bien", WINDOW_NAME)
    dir_idx = cv2.getTrackbarPos("Huong chuan VAO (0-3)", WINDOW_NAME)

    # Chạy mô hình bám vết đối tượng trên luồng ma trận sạch
    results = model.track(frame, persist=True, conf=conf_val, tracker="bytetrack.yaml", verbose=False)

    # Thiết lập tọa độ vùng kiểm soát biên ảo
    is_vertical_direction = dir_idx in (0, 1)
    if is_vertical_direction:
        bound_start = int(h * (zone_start_val / 100))
        bound_end = bound_start + zone_thick_val
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bound_start), (w, bound_end), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.line(frame, (0, bound_start), (w, bound_start), (0, 200, 200), 1)
        cv2.line(frame, (0, bound_end), (w, bound_end), (0, 200, 200), 1)
    else:
        bound_start = int(w * (zone_start_val / 100))
        bound_end = bound_start + zone_thick_val
        overlay = frame.copy()
        cv2.rectangle(overlay, (bound_start, 0), (bound_end, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.line(frame, (bound_start, 0), (bound_start, h), (0, 200, 200), 1)
        cv2.line(frame, (bound_end, 0), (bound_end, h), (0, 200, 200), 1)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, id_obj, cls in zip(boxes, ids, clss):
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cx = int((bx1 + bx2) / 2)
            cy = int((by1 + by2) / 2)

            estimated_height = (W1 * (by2 - by1)) + (W2 * by2) + BIAS
            
            raw_cls_name = class_mapping[cls].lower()
            is_adult = "adult" in raw_cls_name or "lon" in raw_cls_name or "nguoi" in raw_cls_name
            is_child = "child" in raw_cls_name or "em" in raw_cls_name or "tre" in raw_cls_name

            if id_obj not in track_history:
                track_history[id_obj] = []
            track_history[id_obj].append((cx, cy))

            if len(track_history[id_obj]) > 15:
                track_history[id_obj].pop(0)

            obj_label = "Nguoi lon" if is_adult else "Tre em"
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {id_obj} | {obj_label}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            hist = track_history[id_obj]
            if len(hist) >= 5 and id_obj not in counted_ids:
                in_zone = False
                if is_vertical_direction:
                    if bound_start <= cy <= bound_end: in_zone = True
                else:
                    if bound_start <= cx <= bound_end: in_zone = True

                if in_zone:
                    v_x = hist[-1][0] - hist[-5][0]
                    v_y = hist[-1][1] - hist[-5][1]

                    is_in = False
                    is_out = False

                    if dir_idx == 0:
                        if v_y > 3: is_in = True
                        elif v_y < -3: is_out = True
                    elif dir_idx == 1:
                        if v_y < -3: is_in = True
                        elif v_y > 3: is_out = True
                    elif dir_idx == 2:
                        if v_x > 3: is_in = True
                        elif v_x < -3: is_out = True
                    elif dir_idx == 3:
                        if v_x < -3: is_in = True
                        elif v_x > 3: is_out = True

                    now_str = datetime.now().strftime("%H:%M:%S")
                    obj_type = "Người lớn" if is_adult else "Trẻ em"

                    if is_in:
                        if is_adult: counters["adult_in"] += 1
                        elif is_child: counters["child_in"] += 1
                        counted_ids.add(id_obj)
                        log_list.append({"Thoi_Gian": now_str, "Doi_Tuong": obj_type, "Chieu_Di": "VÀO", "Chieu_Cao_CM": round(estimated_height, 1)})
                    elif is_out:
                        if is_adult: counters["adult_out"] += 1
                        elif is_child: counters["child_out"] += 1
                        counted_ids.add(id_obj)
                        log_list.append({"Thoi_Gian": now_str, "Doi_Tuong": obj_type, "Chieu_Di": "RA", "Chieu_Cao_CM": round(estimated_height, 1)})

    # 5. THIẾT KẾ MÀN HÌNH ĐIỀU KHIỂN ĐỒ HỌA TÁCH BIỆT (SIDEBAR DASHBOARD)
    # Khởi tạo một ma trận ảnh đen tĩnh có độ cao bằng video (720px) và rộng 400px
    sidebar = np.zeros((h, 400, 3), dtype=np.uint8)
    
    # Tính toán doanh thu thực tế
    revenue = (counters["adult_in"] * 100000) + (counters["child_in"] * 50000)
    dir_labels = ["Tren -> Duoi", "Duoi -> Tren", "Trai -> Phai", "Phai -> Trai"]

    # Đổ màu nền xám tối cho các tiêu đề danh mục (Category Header Backgrounds)
    cv2.rectangle(sidebar, (15, 60), (385, 95), (45, 45, 45), -1)
    cv2.rectangle(sidebar, (15, 250), (385, 285), (45, 45, 45), -1)
    cv2.rectangle(sidebar, (15, 440), (385, 520), (30, 60, 30), -1) # Hộp doanh thu màu xanh lá tối

    # Vẽ chữ tiêu đề hệ thống
    cv2.putText(sidebar, "BANG DIEU KHIEN SOAT VE", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Khối hiển thị thông số LƯỢT VÀO (Màu Xanh Lá)
    cv2.putText(sidebar, "THONG KE LUOT VAO", (25, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(sidebar, f"Nguoi lon vao:  {counters['adult_in']}", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(sidebar, f"Tre em vao:    {counters['child_in']}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(sidebar, f"TONG LUOT VAO:  {counters['adult_in'] + counters['child_in']}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Khối hiển thị thông số LƯỢT RA (Màu Đỏ)
    cv2.putText(sidebar, "THONG KE LUOT RA", (25, 273), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(sidebar, f"Nguoi lon ra:  {counters['adult_out']}", (30, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(sidebar, f"Tre em ra:    {counters['child_out']}", (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(sidebar, f"TONG LUOT RA:   {counters['adult_out'] + counters['child_out']}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Khối hiển thị DOANH THU & HIỆU NĂNG
    cv2.putText(sidebar, f"TONG DOANH THU:", (30, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(sidebar, f"{revenue:,} VND", (30, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(sidebar, f"Toc do xu ly: {fps:.1f} FPS", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(sidebar, f"Huong cua vao: {dir_labels[dir_idx]}", (20, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(sidebar, "Phim 'R': Reset | Phim 'Q': Thoat", (20, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # NỐI MA TRẬN ẢNH THEO CHIỀU NGANG: Gắn Sidebar đen vào bên phải khung hình Video gốc
    combined_display = np.hstack((frame, sidebar))

    # Đẩy ảnh tổng hợp sau phân tách ra cửa sổ giao diện chính
    cv2.imshow(WINDOW_NAME, combined_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('r') or key == ord('R'):
        counters = {"adult_in": 0, "child_in": 0, "adult_out": 0, "child_out": 0}
        counted_ids.clear()
        track_history.clear()
        log_list.clear()
        print("[INFO] Đã đặt lại toàn bộ thông số đếm về 0!")

cap.release()
cv2.destroyAllWindows()

if log_list:
    df_report = pd.DataFrame(log_list)
    filename = f"Bao_Cao_Soat_Ve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df_report.to_excel(filename, index=False)
    print(f"[SUCCESS] Đã lưu tệp báo cáo vào file: {filename}")