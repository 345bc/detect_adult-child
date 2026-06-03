import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="AI Ticket Counter Enterprise", layout="wide")
model_path = r"runs/detect/runs/train/adult_child_v1/weights/best.pt"


@st.cache_resource
def load_model():
    return YOLO(model_path)


model = load_model()

# Khoi tao cac bien luu tru trang thai va lich su thong ke
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
if "log_data" not in st.session_state:
    st.session_state.log_data = pd.DataFrame(
        columns=["Thoi_Gian", "Doi_Tuong", "Chieu_Di", "Chieu_Cao_CM"]
    )


def reset_counters():
    st.session_state.adult_in = 0
    st.session_state.child_in = 0
    st.session_state.adult_out = 0
    st.session_state.child_out = 0
    st.session_state.counted_in_ids = set()
    st.session_state.counted_out_ids = set()
    st.session_state.track_history = {}
    st.session_state.log_data = pd.DataFrame(
        columns=["Thoi_Gian", "Doi_Tuong", "Chieu_Di", "Chieu_Cao_CM"]
    )


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
        help="Nhập địa chỉ truyền luồng dữ liệu dạng RTSP hoặc HTTP được cấp phát từ ứng dụng IP Webcam trên điện thoại.",
    )

conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy (Confidence):",
    0.1,
    1.0,
    0.4,
    0.05,
    help="Độ chính xác tối thiểu để AI giữ lại khung nhận diện. Đặt quá thấp sẽ dễ nhận diện nhầm vật thể, đặt quá cao sẽ bỏ sót người.",
)

line_position = st.sidebar.slider(
    "Vị trí vạch kiểm soát (Tỷ lệ %):",
    10,
    90,
    50,
    5,
    help="Điều chỉnh cao độ hoặc tọa độ ngang của vạch kiểm soát ảo tính theo phần trăm kích thước khung hình video.",
)

direction = st.sidebar.selectbox(
    "Chiều di chuyển xác định VÀO:",
    (
        "Từ Trên xuống Dưới",
        "Từ Dưới lên Trên",
        "Từ Trái sang Phải",
        "Từ Phải sang Trái",
    ),
    help="Chọn hướng di chuyển hợp lệ cắt qua vạch để tính toán lượt vào cổng và cộng tiền vé. Hướng đối diện sẽ tự động tính là lượt ra.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tham số Hồi quy Chiều cao")

w1 = st.sidebar.number_input(
    "Trọng số khung hình (w1):",
    value=0.35,
    help="Hệ số tỷ lệ quy đổi kích thước chiều cao của hộp bao (pixel) sang đơn vị vật lý (centimet).",
)

w2 = st.sidebar.number_input(
    "Trọng số vị trí chân (w2):",
    value=0.12,
    help="Hệ số hiệu chỉnh độ sâu phối cảnh. Thường mang giá trị âm khi camera chĩa từ trên cao xuống nhằm bù trừ sai số gần to xa nhỏ.",
)

bias = st.sidebar.number_input(
    "Sai số hệ thống (bias):",
    value=45.0,
    help="Hằng số tự do bù trừ cho cao độ nền, phụ thuộc vào khoảng cách lắp đặt từ trần nhà tới mặt sàn và góc nghiêng của camera.",
)

if st.sidebar.button("Reset hệ thống", on_click=reset_counters):
    st.toast("Đã xóa toàn bộ cơ sở dữ liệu đếm!")

st.markdown("### Thống kê lưu lượng và Doanh thu")
col1, col2, col3 = st.columns(3)
slot_adult_in = col1.empty()
slot_child_in = col2.empty()
slot_total_in = col3.empty()

col4, col5, col6 = st.columns(3)
slot_adult_out = col4.empty()
slot_child_out = col5.empty()
slot_total_out = col6.empty()

slot_revenue = st.empty()
slot_fps = st.sidebar.empty()


def update_ui_metrics():
    total_in = st.session_state.adult_in + st.session_state.child_in
    total_out = st.session_state.adult_out + st.session_state.child_out
    revenue = (st.session_state.adult_in * 100000) + (st.session_state.child_in * 50000)

    slot_adult_in.metric(label="Người lớn VÀO", value=st.session_state.adult_in)
    slot_child_in.metric(label="Trẻ em VÀO", value=st.session_state.child_in)
    slot_total_in.metric(label="TỔNG LƯỢT VÀO", value=total_in)

    slot_adult_out.metric(label="Người lớn RA", value=st.session_state.adult_out)
    slot_child_out.metric(label="Trẻ em RA", value=st.session_state.child_out)
    slot_total_out.metric(label="TỔNG LƯỢT RA", value=total_out)

    slot_revenue.metric(
        label="Ước Tính Doanh Thu Vé Vào Cổng (VNĐ)", value=f"{revenue:,}"
    )


update_ui_metrics()
st.markdown("---")

if source_type == "Hình ảnh":
    uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, channels="BGR", width="stretch")

else:
    run_tracking = False
    video_path = None

    if source_type == "Video (File)":
        uploaded_video = st.file_uploader(
            "Tải video lên...", type=["mp4", "avi", "mov"]
        )
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            run_tracking = st.checkbox("Bắt đầu xử lý Video")
    elif source_type == "Webcam Máy Tính":
        video_path = 0
        run_tracking = st.checkbox("Mở Webcam Máy Tính")
    elif source_type == "Điện Thoại (IP Camera)":
        video_path = ip_url
        run_tracking = st.checkbox("Kết nối IP Camera")

    if run_tracking and video_path is not None:
        cap = cv2.VideoCapture(video_path)
        st_frame = st.empty()

        st.markdown("### Biểu đồ phân tích tần suất ra vào")
        chart_slot = st.empty()

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

            if direction in ("Từ Trên xuống Dưới", "Từ Dưới lên Trên"):
                cy_line = int(h * (line_position / 100))
                cv2.line(frame, (0, cy_line), (w, cy_line), (255, 0, 0), 3)
            else:
                cx_line = int(w * (line_position / 100))
                cv2.line(frame, (cx_line, 0), (cx_line, h), (255, 0, 0), 3)

            results = model.track(
                frame, persist=True, conf=conf_threshold, tracker="bytetrack.yaml"
            )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

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

                    if len(st.session_state.track_history[id_obj]) > 1:
                        for i in range(
                            1, min(20, len(st.session_state.track_history[id_obj]))
                        ):
                            pt1 = st.session_state.track_history[id_obj][-i]
                            pt2 = st.session_state.track_history[id_obj][-i - 1]
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

                    label = f"ID: {id_obj} | {'Adult' if cls == 0 else 'Child'} | {estimated_height:.1f}cm"
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

                    if len(st.session_state.track_history[id_obj]) > 1:
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

                            new_row = pd.DataFrame(
                                [
                                    {
                                        "Thoi_Gian": now_str,
                                        "Doi_Tuong": obj_type,
                                        "Chieu_Di": "VÀO",
                                        "Chieu_Cao_CM": round(estimated_height, 1),
                                    }
                                ]
                            )
                            st.session_state.log_data = pd.concat(
                                [st.session_state.log_data, new_row],
                                ignore_index=True,
                            )
                            update_ui_metrics()

                        if (
                            is_out
                            and id_obj not in st.session_state.counted_out_ids
                        ):
                            if cls == 0:
                                st.session_state.adult_out += 1
                            elif cls == 1:
                                st.session_state.child_out += 1
                            st.session_state.counted_out_ids.add(id_obj)

                            new_row = pd.DataFrame(
                                [
                                    {
                                        "Thoi_Gian": now_str,
                                        "Doi_Tuong": obj_type,
                                        "Chieu_Di": "RA",
                                        "Chieu_Cao_CM": round(estimated_height, 1),
                                    }
                                ]
                            )
                            st.session_state.log_data = pd.concat(
                                [st.session_state.log_data, new_row],
                                ignore_index=True,
                            )
                            update_ui_metrics()

            st_frame.image(frame, channels="BGR", width="stretch")

            if not st.session_state.log_data.empty:
                chart_data = (
                    st.session_state.log_data.groupby(["Thoi_Gian", "Chieu_Di"])
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

