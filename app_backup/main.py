import os
import cv2
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
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

# Dùng TCP thay UDP cho RTSP và ẩn bớt log FFmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|loglevel;quiet"

# =========================
# Khởi tạo FastAPI
# =========================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# =========================
# Load YOLO model
# =========================
yolo_model = YOLO("detector/yolov8n.pt", task="detect")

# Biến lưu số người
people_count = 0

error_count = 0
MAX_ERRORS = 50

# =========================
# Hàm mở camera
# =========================
def open_camera():
    """Mở camera theo ưu tiên: RTSP -> webcam -> None"""
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
# Stream video + detect
# =========================
def generate_frames():
    global people_count

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


        # Reset attempts nếu đã đọc được frame
        attempts = 0

        try:
            # Detect YOLO
            rects = []
            results = yolo_model(frame, conf=0.5, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > 0.5:  # person
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        rects.append((int(x1), int(y1), int(x2), int(y2)))

            # Update count
            people_count = len(rects)

            # Vẽ bbox
            for (x1, y1, x2, y2) in rects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"People: {people_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode frame
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
