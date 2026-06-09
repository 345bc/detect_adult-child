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

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AI Ticket Counter Pro",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== CUSTOM CSS =====================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.96);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
        .main-title {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(37, 99, 235, .28), rgba(14, 165, 233, .13));
            border: 1px solid rgba(96, 165, 250, .25);
            box-shadow: 0 18px 45px rgba(0,0,0,.25);
            margin-bottom: 1.2rem;
        }
        .main-title h1 {
            margin: 0;
            font-size: 2.1rem;
            color: #f8fafc;
            letter-spacing: -0.03em;
        }
        .main-title p {
            margin: .45rem 0 0 0;
            color: #cbd5e1;
            font-size: 1rem;
        }
        .glass-card {
            padding: 1.05rem;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 18px 35px rgba(0,0,0,.22);
            margin-bottom: 1rem;
        }
        .status-pill {
            display: inline-block;
            padding: .38rem .72rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, .16);
            border: 1px solid rgba(34, 197, 94, .35);
            color: #86efac;
            font-weight: 700;
            font-size: .85rem;
        }
        .small-muted { color: #94a3b8; font-size: .9rem; }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, .18);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 12px 28px rgba(0,0,0,.18);
        }
        div[data-testid="stMetricLabel"] p { color: #cbd5e1 !important; }
        div[data-testid="stMetricValue"] { color: #f8fafc !important; }
        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(96, 165, 250, .35);
            background: linear-gradient(135deg, #2563eb, #0891b2);
            color: white;
            font-weight: 700;
            padding: .75rem 1rem;
        }
        .stButton > button:hover {
            border-color: #93c5fd;
            filter: brightness(1.08);
        }
        .section-title {
            color: #f8fafc;
            font-weight: 800;
            margin: .2rem 0 .7rem 0;
            font-size: 1.1rem;
        }
        hr { border-color: rgba(148, 163, 184, .2); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== MODEL =====================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

model, device = load_model()

# ===================== SESSION STATE =====================
def init_state():
    defaults = {
        "adult_in": 0,
        "child_in": 0,
        "adult_out": 0,
        "child_out": 0,
        "counted_in_ids": set(),
        "counted_out_ids": set(),
        "track_history": {},
        "log_list": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_counters():
    st.session_state.adult_in = 0
    st.session_state.child_in = 0
    st.session_state.adult_out = 0
    st.session_state.child_out = 0
    st.session_state.counted_in_ids = set()
    st.session_state.counted_out_ids = set()
    st.session_state.track_history = {}
    st.session_state.log_list = []

init_state()

# ===================== SIDEBAR =====================
st.sidebar.markdown("## 🎛️ Bảng điều khiển")
st.sidebar.markdown(f"<span class='status-pill'>Model: {device.upper()}</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

source_type = st.sidebar.selectbox(
    "Nguồn đầu vào",
    ("Hình ảnh", "Video (File)", "Webcam Máy Tính", "Điện Thoại (IP Camera)"),
)

ip_url = ""
if source_type == "Điện Thoại (IP Camera)":
    ip_url = st.sidebar.text_input("URL IP Camera", "http://192.168.1.5:8080/video")

conf_threshold = st.sidebar.slider("Ngưỡng tin cậy", 0.1, 1.0, 0.4, 0.05)
line_position = st.sidebar.slider("Vị trí vạch kiểm soát (%)", 10, 90, 50, 5)
direction = st.sidebar.selectbox(
    "Chiều tính là VÀO",
    ("Từ Trên xuống Dưới", "Từ Dưới lên Trên", "Từ Trái sang Phải", "Từ Phải sang Trái"),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 Hồi quy chiều cao")
w1 = st.sidebar.number_input("w1 - trọng số chiều cao box", value=0.35)
w2 = st.sidebar.number_input("w2 - trọng số vị trí chân", value=0.12)
bias = st.sidebar.number_input("bias - sai số hệ thống", value=45.0)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset hệ thống", on_click=reset_counters):
    st.toast("Đã reset toàn bộ bộ đếm")
slot_fps = st.sidebar.empty()

# ===================== HEADER =====================
st.markdown(
    """
    <div class="main-title">
        <h1>🎟️ AI Ticket Counter Pro</h1>
        <p>Nhận diện người lớn / trẻ em, đếm lượt vào - ra, ước tính doanh thu và theo dõi thời gian thực bằng YOLO.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== METRICS =====================
def calculate_metrics():
    total_in = st.session_state.adult_in + st.session_state.child_in
    total_out = st.session_state.adult_out + st.session_state.child_out
    current_inside = max(total_in - total_out, 0)
    revenue = (st.session_state.adult_in * 100000) + (st.session_state.child_in * 50000)
    return total_in, total_out, current_inside, revenue


def render_metrics():
    total_in, total_out, current_inside, revenue = calculate_metrics()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng lượt VÀO", total_in)
    m2.metric("Tổng lượt RA", total_out)
    m3.metric("Đang bên trong", current_inside)
    m4.metric("Doanh thu ước tính", f"{revenue:,} VNĐ")

    a1, a2, c1, c2 = st.columns(4)
    a1.metric("Người lớn vào", st.session_state.adult_in)
    a2.metric("Người lớn ra", st.session_state.adult_out)
    c1.metric("Trẻ em vào", st.session_state.child_in)
    c2.metric("Trẻ em ra", st.session_state.child_out)

metrics_slot = st.empty()
with metrics_slot.container():
    render_metrics()

# ===================== LAYOUT =====================
left_col, right_col = st.columns([7, 3], gap="large")

with left_col:
    st.markdown('<div class="glass-card"><div class="section-title">📹 Màn hình nhận diện</div>', unsafe_allow_html=True)
    st_frame = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card"><div class="section-title">📊 Biểu đồ lưu lượng</div>', unsafe_allow_html=True)
    chart_slot = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-title">🧾 Nhật ký sự kiện</div>', unsafe_allow_html=True)
    log_slot = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


def refresh_dashboard():
    with metrics_slot.container():
        render_metrics()

    if st.session_state.log_list:
        df_log = pd.DataFrame(st.session_state.log_list)
        log_slot.dataframe(df_log.tail(12), use_container_width=True, hide_index=True)
        chart_data = df_log.groupby(["Thoi_Gian", "Chieu_Di"]).size().unstack(fill_value=0)
        chart_slot.line_chart(chart_data)
    else:
        log_slot.info("Chưa có lượt nào được ghi nhận.")
        chart_slot.info("Biểu đồ sẽ hiển thị khi có dữ liệu.")

refresh_dashboard()

# ===================== IMAGE MODE =====================
if source_type == "Hình ảnh":
    uploaded_file = left_col.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

# ===================== VIDEO / CAMERA MODE =====================
else:
    run_tracking = False
    video_path = None

    if source_type == "Video (File)":
        uploaded_video = left_col.file_uploader("Tải video lên", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            run_tracking = left_col.checkbox("▶️ Bắt đầu xử lý video")
    elif source_type == "Webcam Máy Tính":
        video_path = 0
        run_tracking = left_col.checkbox("📷 Mở webcam")
    elif source_type == "Điện Thoại (IP Camera)":
        video_path = ip_url
        run_tracking = left_col.checkbox("📱 Kết nối IP Camera")

    if run_tracking and video_path is not None:
        cap = cv2.VideoCapture(video_path)
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
            slot_fps.markdown(f"**⚡ FPS:** {fps:.1f}")

            h, w, _ = frame.shape
            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if direction in ("Từ Trên xuống Dưới", "Từ Dưới lên Trên"):
                cy_line = int(h * (line_position / 100))
                cv2.line(frame, (0, cy_line), (w, cy_line), (255, 190, 0), 3)
                cv2.putText(frame, "CONTROL LINE", (15, cy_line - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 190, 0), 2)
            else:
                cx_line = int(w * (line_position / 100))
                cv2.line(frame, (cx_line, 0), (cx_line, h), (255, 190, 0), 3)
                cv2.putText(frame, "CONTROL LINE", (cx_line + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 190, 0), 2)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                ui_needs_update = False

                for box, id_obj, cls in zip(boxes, ids, clss):
                    bx1, by1, bx2, by2 = map(int, box[:4])
                    cx = int((bx1 + bx2) / 2)
                    cy = int((by1 + by2) / 2)
                    box_h = by2 - by1
                    estimated_height = (w1 * box_h) + (w2 * by2) + bias

                    st.session_state.track_history.setdefault(id_obj, []).append((cx, cy))
                    if len(st.session_state.track_history[id_obj]) > 30:
                        st.session_state.track_history[id_obj].pop(0)

                    hist = st.session_state.track_history[id_obj]
                    for i in range(1, min(15, len(hist))):
                        cv2.line(frame, hist[-i], hist[-i - 1], (0, 255, 255), 2)

                    obj_type = "Người lớn" if cls == 0 else "Trẻ em"
                    color = (34, 197, 94) if cls == 0 else (239, 68, 68)
                    label = f"ID {id_obj} | {obj_type} | {estimated_height:.0f}cm"
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cv2.rectangle(frame, (bx1, by1 - 28), (bx1 + 260, by1), color, -1)
                    cv2.putText(frame, label, (bx1 + 6, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

                    if len(hist) > 1:
                        prev_cx, prev_cy = hist[-2]
                        current_cx, current_cy = hist[-1]
                        is_in = is_out = False

                        if direction == "Từ Trên xuống Dưới":
                            is_in = prev_cy < cy_line <= current_cy
                            is_out = prev_cy > cy_line >= current_cy
                        elif direction == "Từ Dưới lên Trên":
                            is_in = prev_cy > cy_line >= current_cy
                            is_out = prev_cy < cy_line <= current_cy
                        elif direction == "Từ Trái sang Phải":
                            is_in = prev_cx < cx_line <= current_cx
                            is_out = prev_cx > cx_line >= current_cx
                        elif direction == "Từ Phải sang Trái":
                            is_in = prev_cx > cx_line >= current_cx
                            is_out = prev_cx < cx_line <= current_cx

                        now_str = datetime.now().strftime("%H:%M:%S")
                        if is_in and id_obj not in st.session_state.counted_in_ids:
                            if cls == 0:
                                st.session_state.adult_in += 1
                            else:
                                st.session_state.child_in += 1
                            st.session_state.counted_in_ids.add(id_obj)
                            st.session_state.log_list.append({"Thoi_Gian": now_str, "Doi_Tuong": obj_type, "Chieu_Di": "VÀO", "Chieu_Cao_CM": round(estimated_height, 1)})
                            ui_needs_update = True

                        if is_out and id_obj not in st.session_state.counted_out_ids:
                            if cls == 0:
                                st.session_state.adult_out += 1
                            else:
                                st.session_state.child_out += 1
                            st.session_state.counted_out_ids.add(id_obj)
                            st.session_state.log_list.append({"Thoi_Gian": now_str, "Doi_Tuong": obj_type, "Chieu_Di": "RA", "Chieu_Cao_CM": round(estimated_height, 1)})
                            ui_needs_update = True

                if ui_needs_update:
                    refresh_dashboard()

            st_frame.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        if source_type == "Video (File)" and video_path:
            try:
                os.remove(video_path)
            except OSError:
                pass
