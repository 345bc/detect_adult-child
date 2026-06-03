import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
import pandas as pd
from datetime import datetime
import torch

st.set_page_config(page_title="AI Ticket Counter Enterprise", layout="wide")
model_path = r"runs/detect/runs/train/adult_child_v1/weights/best.pt"


@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    # Tu dong cau hinh phan cung toi uu hoa GPU hoac CPU de tang toc
    device = "cuda" if torch.torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


model = load_model()

# Khoi tao cac bien luu tru trang thai va lich su thong ke hieu nang cao
if "adult_in" not in st.session_state:
    st.session_state.adult_in = 0
if "child_in" not in st.session_state:
    st.session_state.child_in = 0
if "adult_out" not in st.session_state:
    st.session_state.adult_out = 0
if "child_out" not in st.session_state:
    st.session_state.child_out = 0
if "counted_in_ids" not in st.session_state:
    st.session_state.counted_in_ids = set()
if "counted_out_ids" not in st.session_state:
    st.session_state.counted_out_ids = set()
if "track_history" not in st.session_state:
    st.session_state.track_history = {}
# Dung danh sach thuan de tranh thoi gian cap phat bo nho cua Pandas DataFrame
if "log_list" not in st.session_state:
    st.session_state.log_list = []


def reset_counters():
    st.session_state.adult_in = 0
    st.session_state.child_in = 0
    st.session_state.adult_out = 0
    st.session_state.child_out = 0
    st.session_state.counted_in_ids = set()
    st.session_state.counted_out_ids = set()
    st.session_state.track_history = {}
    st.session_state.log_list = []


st.title("Hệ Thống Kiểm Soát Vé Và Phân Tích Nhân Khẩu Học Real-time")
st.markdown("---")

st.sidebar.title("Cấu Hình Hệ Thống")
source_type = st.sidebar.selectbox(
    "Chọn nguồn đầu vào:",
    ("Hình ảnh", "Video (File)", "Webcam Máy Tính", "Điện Thoại (IP Camera)"),
)

ip_url = ""
if source_type == "Điện Thoại (IP Camera)":
    ip_url = st.sidebar.text_input(
        "Nhập URL IP Webcam:",
        "http://192.168.1.5:8080/video",
        help="Nhập địa chỉ truyền luồng dữ liệu cấp phát từ ứng dụng IP Webcam trên điện thoại.",
    )

conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy (Confidence):",
    0.1,
    1.0,
    0.4,
    0.05,
    help="Độ chính xác tối thiểu để AI giữ lại khung nhận diện đối tượng.",
)

line_position = st.sidebar.slider(
    "Vị trí vạch kiểm soát (Tỷ lệ %):",
    10,
    90,
    50,
    5,
    help="Điều chỉnh cao độ hoặc tọa độ ngang của vạch kiểm soát ảo tính theo phần trăm.",
)

direction = st.sidebar.selectbox(
    "Chiều di chuyển xác định VÀO:",
    (
        "Từ Trên xuống Dưới",
        "Từ Dưới lên Trên",
        "Từ Trái sang Phải",
        "Từ Phải sang Trái",
    ),
    help="Chọn hướng di chuyển hợp lệ cắt qua vạch để tính toán lượt vào cổng.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tham số Hồi quy Chiều cao")

w1 = st.sidebar.number_input("Trọng số khung hình (w1):", value=0.35)
w2 = st.sidebar.number_input("Trọng số vị trí chân (w2):", value=0.12)
bias = st.sidebar.number_input("Sai số hệ thống (bias):", value=45.0)

if st.sidebar.button("Reset hệ thống", on_click=reset_counters):
    st.toast("Đã xóa toàn bộ cơ sở dữ liệu đếm!")

slot_fps = st.sidebar.empty()

# Chia bo cuc hien thi thanh hai cot song song ty le 7:3 de tranh cuon trang
main_col1, main_col2 = st.columns([7, 3])

with main_col1:
    st_frame = st.empty()

with main_col2:
    st.markdown("### Thống kê lượt VÀO")
    slot_adult_in = st.empty()
    slot_child_in = st.empty()
    slot_total_in = st.empty()
    st.markdown("---")
    st.markdown("### Thống kê lượt RA")
    slot_adult_out = st.empty()
    slot_child_out = st.empty()
    slot_total_out = st.empty()
    st.markdown("---")
    slot_revenue = st.empty()
    chart_slot = st.empty()


def update_ui_metrics():
    total_in = st.session_state.adult_in + st.session_state.child_in
    total_out = st.session_state.adult_out + st.session_state.child_out
    revenue = (st.session_state.adult_in * 100000) + (st.session_state.child_in * 50000)

    slot_adult_in.metric(label="Người lớn VÀO (Lượt)", value=st.session_state.adult_in)
    slot_child_in.metric(label="Trẻ em VÀO (Lượt)", value=st.session_state.child_in)
    slot_total_in.metric(label="TỔNG LƯỢT VÀO", value=total_in)

    slot_adult_out.metric(label="Người lớn RA (Lượt)", value=st.session_state.adult_out)
    slot_child_out.metric(label="Trẻ em RA (Lượt)", value=st.session_state.child_out)
    slot_total_out.metric(label="TỔNG LƯỢT RA", value=total_out)

    slot_revenue.metric(label="Ước Tính Doanh Thu (VNĐ)", value=f"{revenue:,}")


update_ui_metrics()

if source_type == "Hình ảnh":
    uploaded_file = main_col1.file_uploader(
        "Tải ảnh lên...", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

else:
    run_tracking = False
    video_path = None

    if source_type == "Video (File)":
        uploaded_video = main_col1.file_uploader(
            "Tải video lên...", type=["mp4", "avi", "mov"]
        )
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            run_tracking = main_col1.checkbox("Bắt đầu xử lý Video")
    elif source_type == "Webcam Máy Tính":
        video_path = 0
        run_tracking = main_col1.checkbox("Mở Webcam Máy Tính")
    elif source_type == "Điện Thoại (IP Camera)":
        video_path = ip_url
        run_tracking = main_col1.checkbox("Kết nối IP Camera")

    if run_tracking and video_path is not None:
        cap = cv2.VideoCapture(video_path)

        # Ha do phan giai cua camera phan cung de tiet kiem bang thong truyen tai
        if source_type in ("Webcam Máy Tính", "Điện Thoại (IP Camera)"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            slot_fps.markdown(f"**Tốc độ xử lý:** {fps:.1f} FPS")

            h, w, _ = frame.shape

            # TOI UU HIEU NANG: Vo hieu hoa verbose de triet tieu I/O overhead tren Terminal
            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if direction in ("Từ Trên xuống Dưới", "Từ Dưới lên Trên"):
                cy_line = int(h * (line_position / 100))
                cv2.line(frame, (0, cy_line), (w, cy_line), (255, 0, 0), 3)
                cv2.putText(
                    frame,
                    f"Line Y: {cy_line}",
                    (10, cy_line - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
            else:
                cx_line = int(w * (line_position / 100))
                cv2.line(frame, (cx_line, 0), (cx_line, h), (255, 0, 0), 3)
                cv2.putText(
                    frame,
                    f"Line X: {cx_line}",
                    (cx_line + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                # Co kiem soat trang thai thay doi thong ke de toi uu luong ve UI
                ui_needs_update = False

                for box, id_obj, cls in zip(boxes, ids, clss):
                    bx1, by1, bx2, by2 = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                    )
                    cx = int((bx1 + bx2) / 2)
                    cy = int((by1 + by2) / 2)
                    box_h = by2 - by1

                    estimated_height = (w1 * box_h) + (w2 * by2) + bias

                    if id_obj not in st.session_state.track_history:
                        st.session_state.track_history[id_obj] = []
                    st.session_state.track_history[id_obj].append((cx, cy))

                    # TOI UU BO NHO: Cat tiat do dai mang lich su tracking ngan lag phinh bo nho
                    if len(st.session_state.track_history[id_obj]) > 30:
                        st.session_state.track_history[id_obj].pop(0)

                    hist_len = len(st.session_state.track_history[id_obj])
                    if hist_len > 1:
                        for i in range(1, min(15, hist_len)):
                            pt1 = st.session_state.track_history[id_obj][-i]
                            pt2 = st.session_state.track_history[id_obj][-i - 1]
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

                    coord_text = (
                        f"Y:{cy}"
                        if direction in ("Từ Trên xuống Dưới", "Từ Dưới lên Trên")
                        else f"X:{cx}"
                    )
                    label = f"ID: {id_obj} | {'Adult' if cls == 0 else 'Child'} | {coord_text}"

                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    if hist_len > 1:
                        prev_cx, prev_cy = st.session_state.track_history[id_obj][-2]
                        current_cx, current_cy = st.session_state.track_history[id_obj][
                            -1
                        ]

                        is_in = False
                        is_out = False

                        if direction == "Từ Trên xuống Dưới":
                            if prev_cy < cy_line <= current_cy:
                                is_in = True
                            elif prev_cy > cy_line >= current_cy:
                                is_out = True
                        elif direction == "Từ Dưới lên Trên":
                            if prev_cy > cy_line >= current_cy:
                                is_in = True
                            elif prev_cy < cy_line <= current_cy:
                                is_out = True
                        elif direction == "Từ Trái sang Phải":
                            if prev_cx < cx_line <= current_cx:
                                is_in = True
                            elif prev_cx > cx_line >= current_cx:
                                is_out = True
                        elif direction == "Từ Phải sang Trái":
                            if prev_cx > cx_line >= current_cx:
                                is_in = True
                            elif prev_cx < cx_line <= current_cx:
                                is_out = True

                        now_str = datetime.now().strftime("%H:%M:%S")
                        obj_type = "Người lớn" if cls == 0 else "Trẻ em"

                        if is_in and id_obj not in st.session_state.counted_in_ids:
                            if cls == 0:
                                st.session_state.adult_in += 1
                            elif cls == 1:
                                st.session_state.child_in += 1
                            st.session_state.counted_in_ids.add(id_obj)
                            st.session_state.log_list.append(
                                {
                                    "Thoi_Gian": now_str,
                                    "Doi_Tuong": obj_type,
                                    "Chieu_Di": "VÀO",
                                    "Chieu_Cao_CM": round(estimated_height, 1),
                                }
                            )
                            ui_needs_update = True

                        if is_out and id_obj not in st.session_state.counted_out_ids:
                            if cls == 0:
                                st.session_state.adult_out += 1
                            elif cls == 1:
                                st.session_state.child_out += 1
                            st.session_state.counted_out_ids.add(id_obj)
                            st.session_state.log_list.append(
                                {
                                    "Thoi_Gian": now_str,
                                    "Doi_Tuong": obj_type,
                                    "Chieu_Di": "RA",
                                    "Chieu_Cao_CM": round(estimated_height, 1),
                                }
                            )
                            ui_needs_update = True

                # TOI UU HIEU NANG: Chi cap nhat UI metric khi co bien dong so luong thuc te
                if ui_needs_update:
                    update_ui_metrics()

            st_frame.image(frame, channels="BGR", use_container_width=True)

            # TOI UU DO THI: Chi build DataFrame tu list khi can ve bieu do chuoi thoi gian
            if st.session_state.log_list:
                df_chart = pd.DataFrame(st.session_state.log_list)
                chart_data = (
                    df_chart.groupby(["Thoi_Gian", "Chieu_Di"])
                    .size()
                    .unstack(fill_value=0)
                )
                chart_slot.line_chart(chart_data)

        cap.release()
        if source_type == "Video (File)" and video_path:
            try:
                os.remove(video_path)
            except:
                pass
