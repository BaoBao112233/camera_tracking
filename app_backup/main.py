import os
import cv2
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Đọc biến môi trường
# =========================
RTSP_URL = os.getenv("RTSP_URL")
V4L2_DEVICE = os.getenv("V4L2_DEVICE", "/dev/video0")
CAMERA_BUFFER_SIZE = int(os.getenv("CAMERA_BUFFER_SIZE", "1"))
CAMERA_TIMEOUT = int(os.getenv("CAMERA_TIMEOUT", "30"))
RECONNECT_ATTEMPTS = int(os.getenv("RECONNECT_ATTEMPTS", "5"))
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "2"))

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|loglevel;quiet"

# =========================
# Khởi tạo FastAPI
# =========================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# =========================
# Load YOLOv8n TFLite (INT8 / FLOAT32 đều OK)
# =========================
tflite_path = os.getenv("MODEL_PATH", "detector/yolov8n_saved_model/yolov8n_float32.tflite")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

logger.info(f"Model outputs: {len(output_details)}")
for i, d in enumerate(output_details):
    logger.info(f"Output {i}: name={d['name']}, shape={d['shape']}, dtype={d['dtype']}")

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']

# Biến lưu số người
people_count = 0
error_count = 0
MAX_ERRORS = 50

# =========================
# Hàm mở camera
# =========================
def open_camera():
    cap = None
    if RTSP_URL:
        logger.info(f"🔌 Thử kết nối RTSP: {RTSP_URL}")
        cap = cv2.VideoCapture(f"{RTSP_URL}?overrun_nonfatal=1&fifo_size=5000000", cv2.CAP_FFMPEG)
        if cap.isOpened():
            logger.info("✅ Kết nối RTSP thành công")
            return cap
        else:
            logger.error("❌ Không kết nối được RTSP")

    if V4L2_DEVICE and os.path.exists(V4L2_DEVICE):
        logger.info(f"🔌 Thử kết nối webcam: {V4L2_DEVICE}")
        cap = cv2.VideoCapture(V4L2_DEVICE)
        if cap.isOpened():
            logger.info("✅ Kết nối webcam thành công")
            return cap
        else:
            logger.error("❌ Không mở được webcam")

    logger.error("⚠️ Không có camera nào khả dụng")
    return None

# =========================
# Hàm detect với TFLite
# =========================
def detect_objects(frame):
    global interpreter, input_details, output_details, input_dtype
    img_resized = cv2.resize(frame, (input_width, input_height))

    input_data = np.expand_dims(img_resized, axis=0).astype(input_dtype)
    if input_dtype == np.float32:
        input_data = input_data / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # YOLOv8 raw output [1, 84, 8400] -> (8400, 84)
    preds = interpreter.get_tensor(output_details[0]['index'])[0].T

    boxes = preds[:, :4]   # (x, y, w, h)
    scores = preds[:, 4:]  # class scores

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    rects = []
    h, w, _ = frame.shape
    for i in range(len(boxes)):
        if confidences[i] > 0.5 and class_ids[i] == 0:  # class 0 = person
            x, y, bw, bh = boxes[i]
            # YOLOv8 boxes là center_x, center_y, width, height (normalized)
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            rects.append((x1, y1, x2, y2))

    return rects


# =========================
# Stream video + detect
# =========================
def generate_frames():
    global people_count, error_count
    attempts = 0
    cap = open_camera()

    while attempts < RECONNECT_ATTEMPTS:
        if not cap or not cap.isOpened():
            logger.info(f"🔁 Thử reconnect ({attempts+1}/{RECONNECT_ATTEMPTS})...")
            time.sleep(RECONNECT_DELAY)
            cap = open_camera()
            attempts += 1
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            error_count += 1
            if error_count > MAX_ERRORS:
                logger.error("🚨 Quá nhiều frame lỗi, reconnect camera...")
                cap.release()
                cap = open_camera()
                error_count = 0
            continue
        else:
            error_count = 0
            attempts = 0

        try:
            rects = detect_objects(frame)
            people_count = len(rects)

            for (x1, y1, x2, y2) in rects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"People: {people_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        except Exception as e:
            logger.warning(f"⚠️ Lỗi xử lý frame: {e}")
            continue

        time.sleep(0.05)

# =========================
# API endpoints
# =========================
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
async def stats():
    global people_count
    return {"people_count": people_count}

@app.post("/reset_count")
async def reset_count():
    global people_count
    people_count = 0
    return {"message": "Bộ đếm đã được reset"}

@app.get("/ip_camera")
async def ip_camera():
    return {"ip_camera": RTSP_URL}

# =========================
# Run service
# =========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
