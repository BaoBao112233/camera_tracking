import cv2
import numpy as np
import time
import socket
from datetime import datetime
import json
import os
import threading
from typing import Generator
from collections import OrderedDict
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import torch
import asyncio
from pathlib import Path

# Thêm safe_globals khi load model
yolo_model = YOLO('detector/yolov8n.pt', task='detect')

app = FastAPI()
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Load cấu hình từ file JSON
def load_config():
    return {
        "rtsp_url": os.getenv("RTSP_URL"),
        "confidence_threshold": 0.5,
        "tracker_config": {
            "max_disappeared": 100,
            "max_distance": 100
        },
        "server_config": {
            "host": "0.0.0.0",
            "port": 8000
        },
        "model_paths": {
            "yolo_model": "detector/yolov8n.pt"
        },
        "video_config": {
            "fps_limit": 15,
            "frame_skip": 1,
            "buffer_size": 1,
            "reconnect_delay": 1
        },
        "detection_config": {
            "person_class_id": 0,
            "input_size": [640, 640],
            "model_type": "yolov8"
        },
        "ui_config": {
            "line_color": [0, 255, 255],
            "line_thickness": 2,
            "bbox_color": [0, 255, 0],
            "bbox_thickness": 2,
            "text_color": [0, 255, 0],
            "text_scale": 0.5,
            "text_thickness": 2
        },
    }

# Load cấu hình
config = load_config()

# Cấu hình từ file JSON
RTSP_URL = config["rtsp_url"]
YOLO_MODEL_PATH = config["model_paths"]["yolo_model"]
CONFIDENCE_THRESHOLD = config["confidence_threshold"]

# Khởi tạo detector
yolo_model = None

# Biến đếm người (we only count people per-frame now)
people_count_in = 0
people_count_out = 0
total_people = 0

# Thông tin camera hiện tại (dùng để đưa vào JSON)
CURRENT_CAMERA_SRC = None
last_frame_stats = {}

# Lưu trữ lịch sử vị trí của objects để theo dõi hướng di chuyển (tối đa 30 frame)
previous_positions = {}

# Biến cho fall detection
fall_detection_active = False
total_falls = 0
people_detected_fall = 0
last_fall_time = None
last_fall_tracker_id = None  # Lưu tracker_id của sự kiện ngã gần đây nhất
fall_sensitivity = "medium"  # low, medium, high
fall_confidence_threshold = 0.1
yolo_pose_model = None

# Tracking fall events để tránh đếm trùng lặp
fall_tracker = {}  # {tracker_id: {'is_fallen': bool, 'fall_start_time': float, 'fall_counted': bool}}
fall_cooldown_time = 3.0  # Thời gian cooldown giữa các lần đếm ngã (giây)
max_fall_duration = 10.0  # Thời gian tối đa một lần ngã (giây)

# Threaded Camera class để giảm delay
class ThreadedCamera:
    def __init__(self, src=0):
        """
        Khởi tạo threaded camera để giảm delay
        """
        print(f"🔄 Đang khởi tạo camera với URL: {src}")
        self.capture = cv2.VideoCapture(src)
        
        # Kiểm tra camera có mở được không
        if not self.capture.isOpened():
            print(f"❌ Không thể mở camera với URL: {src}")
            raise Exception(f"Không thể kết nối camera: {src}")
        
        # Cấu hình camera
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        
        # Khởi tạo biến
        self.frame = None
        self.status = False
        self.running = True
        self.frame_count = 0
        
        # Bắt đầu thread
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        
        print(f"✅ Camera đã khởi tạo thành công")
    
    def update(self):
        """
        Cập nhật frame liên tục trong thread riêng
        """
        try:
            while hasattr(self, 'running') and self.running:
                if self.capture.isOpened():
                    (self.status, self.frame) = self.capture.read()
                    if self.status:
                        self.frame_count += 1
                        if self.frame_count % 100 == 0:  # Log mỗi 100 frames
                            print(f"📹 Đã đọc {self.frame_count} frames")
                    else:
                        print(f"⚠️ Không đọc được frame, status: {self.status}")
                else:
                    print("❌ Camera không mở được")
                    break
                time.sleep(0.01)
        except Exception as e:
            print(f"❌ Lỗi trong ThreadedCamera.update: {e}")
    
    def read(self):
        """
        Đọc frame hiện tại
        """
        if self.frame is None:
            print("⚠️ Frame là None")
        return self.status, self.frame
    
    def stop(self):
        """
        Dừng camera thread
        """
        try:
            self.running = False
            if hasattr(self, 'thread') and self.thread.is_alive():
                self.thread.join(timeout=2)
            if hasattr(self, 'capture') and self.capture.isOpened():
                self.capture.release()
        except Exception as e:
            print(f"Lỗi khi dừng ThreadedCamera: {e}")

# Khởi tạo threaded camera
threaded_camera = None

# Centroid tracker cải tiến với Kalman Filter
class CentroidTracker:
    def __init__(self, max_disappeared=100, max_distance=100):
        """
        Khởi tạo centroid tracker cải tiến với Kalman Filter
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # Vị trí hiện tại
        self.disappeared = OrderedDict()
        self.velocities = OrderedDict()  # Vận tốc của objects
        self.bboxes = OrderedDict()  # Lưu bounding boxes
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox=None):
        """
        Đăng ký object mới với centroid và bbox
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.velocities[self.next_object_id] = np.array([0.0, 0.0])  # Vận tốc ban đầu
        self.bboxes[self.next_object_id] = bbox if bbox is not None else (0, 0, 0, 0)
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Hủy đăng ký object và xóa tất cả thông tin liên quan
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.velocities[object_id]
        del self.bboxes[object_id]

    def calculate_iou(self, box1, box2):
        """
        Tính Intersection over Union (IoU) giữa hai bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Tính diện tích giao nhau
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Tính diện tích hợp nhất
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def predict_position(self, object_id):
        """
        Dự đoán vị trí tiếp theo dựa trên vận tốc
        """
        if object_id in self.objects and object_id in self.velocities:
            current_pos = np.array(self.objects[object_id])
            velocity = self.velocities[object_id]
            predicted_pos = current_pos + velocity
            return predicted_pos.astype(int)
        return None

    def update(self, rects):
        """
        Cập nhật tracker với danh sách bounding boxes - cải tiến với IoU và dự đoán
        """
        if len(rects) == 0:
            # Dự đoán vị trí cho các objects bị mất
            for object_id in list(self.disappeared.keys()):
                predicted_pos = self.predict_position(object_id)
                if predicted_pos is not None:
                    self.objects[object_id] = predicted_pos
                
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Tính centroids và lưu bounding boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_bboxes = []
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            cx = int((start_x + end_x) / 2.0)
            cy = int((start_y + end_y) / 2.0)
            input_centroids[i] = (cx, cy)
            input_bboxes.append((start_x, start_y, end_x, end_y))

        if len(self.objects) == 0:
            # Đăng ký tất cả objects mới
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            # Tính ma trận khoảng cách và IoU
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Ma trận khoảng cách centroid
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Ma trận IoU
            IoU_matrix = np.zeros((len(object_ids), len(input_bboxes)))
            for i, obj_id in enumerate(object_ids):
                for j, input_bbox in enumerate(input_bboxes):
                    if obj_id in self.bboxes:
                        iou = self.calculate_iou(self.bboxes[obj_id], input_bbox)
                        IoU_matrix[i, j] = iou
            
            # Kết hợp distance và IoU để tạo cost matrix
            # Normalize distance matrix
            D_norm = D / (self.max_distance + 1e-6)
            # IoU cost (1 - IoU để có cost thấp hơn cho IoU cao hơn)
            IoU_cost = 1 - IoU_matrix
            
            # Combined cost: 70% distance + 30% IoU
            cost_matrix = 0.7 * D_norm + 0.3 * IoU_cost
            
            # Hungarian algorithm simulation với greedy approach
            rows = cost_matrix.min(axis=1).argsort()
            cols = cost_matrix.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                # Kiểm tra cả distance và IoU thresholds
                if D[row, col] > self.max_distance and IoU_matrix[row, col] < 0.3:
                    continue

                object_id = object_ids[row]
                old_centroid = self.objects[object_id]
                new_centroid = input_centroids[col]
                
                # Cập nhật vận tốc
                velocity = np.array(new_centroid) - np.array(old_centroid)
                self.velocities[object_id] = 0.7 * self.velocities[object_id] + 0.3 * velocity
                
                # Cập nhật vị trí và bbox
                self.objects[object_id] = new_centroid
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, len(object_ids))).difference(used_row_indices)
            unused_col_indices = set(range(0, len(input_centroids))).difference(used_col_indices)

            # Xử lý objects không được match
            for row in unused_row_indices:
                object_id = object_ids[row]
                # Dự đoán vị trí mới
                predicted_pos = self.predict_position(object_id)
                if predicted_pos is not None:
                    self.objects[object_id] = predicted_pos
                
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Đăng ký detections mới
            for col in unused_col_indices:
                self.register(input_centroids[col], input_bboxes[col])

        return self.objects

# Khởi tạo tracker với cấu hình
tracker_config = config.get("tracker_config", {"max_disappeared": 50, "max_distance": 50})
tracker = CentroidTracker(
    max_disappeared=tracker_config["max_disappeared"],
    max_distance=tracker_config["max_distance"]
)

# Pose tracker để tracking người với pose detection
pose_tracker = CentroidTracker(max_disappeared=30, max_distance=80)

def initialize_detector():
    """
    Khởi tạo YOLOv8n detector
    """
    global yolo_model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLOv8n detector đã được khởi tạo thành công từ {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"Lỗi khởi tạo YOLOv8n detector: {e}")
        yolo_model = None

def initialize_pose_detector():
    """
    Khởi tạo YOLO pose detector cho fall detection
    """
    global yolo_pose_model
    try:
        pose_model_path = "detector/yolov8s-pose.pt"
        print(f"🔄 Đang tải YOLO pose model từ: {pose_model_path}")
        yolo_pose_model = YOLO(pose_model_path)
        print(f"✅ YOLO pose model đã được tải thành công")
    except Exception as e:
        print(f"❌ Lỗi khi tải YOLO pose model: {e}")
        yolo_pose_model = None


def get_device_ip() -> str:
    """Return the primary IP address of the device (not loopback)"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # doesn't need to be reachable
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def get_device_name() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "raspberrypi"


def build_frame_json(ip_camera, n_person: int) -> dict:
    return {
        "ip_rpi": get_device_ip(),
        "name_rpi": get_device_name(),
        "ip_camera": str(ip_camera) if ip_camera is not None else None,
        "n_person": int(n_person),
        "date": datetime.utcnow().isoformat() + "Z"
    }

def detect_people(frame):
    """
    Phát hiện người trong frame sử dụng YOLOv8n
    Returns:
        List of bounding boxes cho người được phát hiện
    """
    if yolo_model is None:
        print("⚠️ YOLOv8n detector chưa được khởi tạo (yolo_model is None)")
        return []
    
    try:
        # Chạy inference với YOLOv8
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        rects = []
        detection_config = config.get("detection_config", {})
        person_class_id = detection_config.get("person_class_id", 0)  # Class 0 là person trong YOLO
        
        # Xử lý kết quả detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy class id và confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Chỉ lấy detection của người (class 0)
                    if cls == person_class_id and conf > CONFIDENCE_THRESHOLD:
                        # Lấy tọa độ bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        rects.append((int(x1), int(y1), int(x2), int(y2)))
        
        return rects
        
    except Exception as e:
        print(f"❌ Lỗi trong detect_people với YOLOv8: {e}")
        return []

def detect_poses_and_falls(frame):
    """
    Phát hiện pose và người ngã sử dụng YOLO pose detection với centroid tracking để tránh đếm trùng lặp
    """
    global yolo_pose_model, total_falls, people_detected_fall, last_fall_time, fall_tracker, pose_tracker
    
    if yolo_pose_model is None:
        return [], False, None
    
    try:
        # Chạy pose detection
        results = yolo_pose_model(frame, conf=fall_confidence_threshold)
        
        poses_data = []
        fall_detected = False
        fall_person_bbox = None
        current_time = time.time()
        
        # Cleanup old fall tracker entries
        cleanup_fall_tracker(current_time)
        
        # Thu thập tất cả bounding boxes và pose results để tracking
        detected_boxes = []
        pose_results = []
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and keypoints is not None:
                for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                    # Lấy bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence >= fall_confidence_threshold:
                        # Lấy keypoints (17 điểm cho COCO format)
                        kpts_xy = kpts.xy[0].cpu().numpy()  # Shape: (17, 2)
                        kpts_conf = kpts.conf[0].cpu().numpy()  # Shape: (17,)
                        
                        # Phân tích tư thế để phát hiện ngã
                        is_fallen = analyze_pose_for_fall(kpts_xy, kpts_conf, (x1, y1, x2, y2))
                        
                        detected_boxes.append((x1, y1, x2, y2))
                        pose_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'keypoints': kpts_xy,
                            'keypoints_conf': kpts_conf,
                            'is_fallen': is_fallen,
                            'confidence': float(confidence)
                        })
        
        # Cập nhật tracker với các bounding boxes
        tracked_objects = pose_tracker.update(detected_boxes)
        
        # Kết hợp tracking results với pose results
        for tracker_id, centroid in tracked_objects.items():
            # Tìm pose result tương ứng với tracker này
            best_match = None
            min_distance = float('inf')
            
            for pose_result in pose_results:
                bbox = pose_result['bbox']
                bbox_centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                distance = ((centroid[0] - bbox_centroid[0])**2 + (centroid[1] - bbox_centroid[1])**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = pose_result
            
            if best_match and min_distance < 50:  # Ngưỡng khoảng cách để match
                # Xử lý fall tracking với tracker_id
                fall_event_detected = process_fall_tracking(tracker_id, best_match['is_fallen'], current_time)
                
                pose_info = {
                    'bbox': best_match['bbox'],
                    'keypoints': [{'x': float(kpt[0]), 'y': float(kpt[1]), 'confidence': float(conf)} 
                                for kpt, conf in zip(best_match['keypoints'], best_match['keypoints_conf'])],
                    'is_fallen': best_match['is_fallen'],
                    'confidence': best_match['confidence'],
                    'tracker_id': tracker_id,
                    'centroid': centroid
                }
                poses_data.append(pose_info)
                
                if fall_event_detected:
                    fall_detected = True
                    fall_person_bbox = best_match['bbox']
                    last_fall_time = current_time
                    print(f"🚨 PHÁT HIỆN NGƯỜI NGÃ MỚI! Tracker ID: {tracker_id}, Tổng số lần ngã: {total_falls}")
        
        people_detected_fall = len(poses_data)
        return poses_data, fall_detected, fall_person_bbox
        
    except Exception as e:
        print(f"❌ Lỗi trong detect_poses_and_falls: {e}")
        return [], False, None

def cleanup_fall_tracker(current_time):
    """
    Dọn dẹp các entry cũ trong fall_tracker
    """
    global fall_tracker
    
    to_remove = []
    for tracker_id, data in fall_tracker.items():
        # Xóa entry nếu quá lâu không cập nhật
        if current_time - data.get('last_update', 0) > max_fall_duration:
            to_remove.append(tracker_id)
    
    for tracker_id in to_remove:
        del fall_tracker[tracker_id]

def process_fall_tracking(tracker_id, is_fallen, current_time):
    """
    Xử lý tracking fall events để tránh đếm trùng lặp
    Trả về True nếu đây là một fall event mới cần đếm
    """
    global fall_tracker, total_falls, last_fall_tracker_id
    
    # Khởi tạo tracking cho tracker mới
    if tracker_id not in fall_tracker:
        fall_tracker[tracker_id] = {
            'is_fallen': False,
            'fall_start_time': None,
            'fall_counted': False,
            'last_update': current_time
        }
    
    tracker_data = fall_tracker[tracker_id]
    tracker_data['last_update'] = current_time
    
    # Nếu người này đang ngã
    if is_fallen:
        # Nếu chưa được đánh dấu là ngã trước đó
        if not tracker_data['is_fallen']:
            # Bắt đầu fall event mới
            tracker_data['is_fallen'] = True
            tracker_data['fall_start_time'] = current_time
            tracker_data['fall_counted'] = False
        
        # Kiểm tra xem có nên đếm fall này không
        if not tracker_data['fall_counted']:
            # Đếm fall này và đánh dấu đã đếm
            tracker_data['fall_counted'] = True
            total_falls += 1
            last_fall_tracker_id = tracker_id  # Lưu tracker_id của fall event mới
            return True  # Đây là fall event mới
    
    else:
        # Người không còn ngã
        if tracker_data['is_fallen']:
            # Kết thúc fall event
            tracker_data['is_fallen'] = False
            tracker_data['fall_start_time'] = None
            # Reset fall_counted sau cooldown time
            if tracker_data['fall_counted'] and current_time - tracker_data.get('fall_start_time', 0) > fall_cooldown_time:
                tracker_data['fall_counted'] = False
    
    return False  # Không phải fall event mới

def analyze_pose_for_fall(keypoints, confidences, bbox):
    """
    Phân tích keypoints để xác định người có bị ngã không
    """
    try:
        # COCO keypoints indices
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        
        # Lấy các điểm quan trọng
        nose = keypoints[0] if confidences[0] > 0.2 else None
        left_shoulder = keypoints[5] if confidences[5] > 0.2 else None
        right_shoulder = keypoints[6] if confidences[6] > 0.2 else None
        left_hip = keypoints[11] if confidences[11] > 0.2 else None
        right_hip = keypoints[12] if confidences[12] > 0.2 else None
        left_ankle = keypoints[15] if confidences[15] > 0.2 else None
        right_ankle = keypoints[16] if confidences[16] > 0.2 else None
        
        # Tính toán các chỉ số để phát hiện ngã
        fall_indicators = 0
        total_checks = 0
        
        # 1. Kiểm tra tỷ lệ chiều cao/chiều rộng của bounding box
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        
        # Nếu tỷ lệ rộng/cao > 1.2, có thể là người nằm
        if aspect_ratio > 1.2:
            fall_indicators += 1
        total_checks += 1
        
        # 2. Kiểm tra vị trí đầu so với hông
        if nose is not None and (left_hip is not None or right_hip is not None):
            hip_y = left_hip[1] if left_hip is not None else right_hip[1]
            if right_hip is not None and left_hip is not None:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            
            # Nếu đầu không cao hơn hông nhiều (< 20% chiều cao bbox)
            head_hip_diff = hip_y - nose[1]
            if head_hip_diff < bbox_height * 0.2:
                fall_indicators += 1
            total_checks += 1
        
        # 3. Kiểm tra vị trí vai so với hông
        if (left_shoulder is not None or right_shoulder is not None) and (left_hip is not None or right_hip is not None):
            shoulder_y = left_shoulder[1] if left_shoulder is not None else right_shoulder[1]
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            hip_y = left_hip[1] if left_hip is not None else right_hip[1]
            if left_hip is not None and right_hip is not None:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            
            # Nếu vai và hông gần ngang nhau
            shoulder_hip_diff = abs(shoulder_y - hip_y)
            if shoulder_hip_diff < bbox_height * 0.15:
                fall_indicators += 1
            total_checks += 1
        
        # 4. Kiểm tra vị trí chân so với thân
        if (left_ankle is not None or right_ankle is not None) and (left_hip is not None or right_hip is not None):
            ankle_y = left_ankle[1] if left_ankle is not None else right_ankle[1]
            if left_ankle is not None and right_ankle is not None:
                ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            
            hip_y = left_hip[1] if left_hip is not None else right_hip[1]
            if left_hip is not None and right_hip is not None:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            
            # Nếu chân không thấp hơn hông nhiều
            ankle_hip_diff = ankle_y - hip_y
            if ankle_hip_diff < bbox_height * 0.3:
                fall_indicators += 1
            total_checks += 1
        
        # Điều chỉnh ngưỡng theo độ nhạy - cải thiện độ chính xác
        sensitivity_thresholds = {
            "low": {"aspect_ratio": 2.2, "head_hip_ratio": 0.25, "shoulder_hip_ratio": 0.35, "knee_body_ratio": 0.5},
            "medium": {"aspect_ratio": 1.8, "head_hip_ratio": 0.35, "shoulder_hip_ratio": 0.45, "knee_body_ratio": 0.6},
            "high": {"aspect_ratio": 1.4, "head_hip_ratio": 0.45, "shoulder_hip_ratio": 0.55, "knee_body_ratio": 0.7}
        }
        
        thresholds = sensitivity_thresholds.get(fall_sensitivity, sensitivity_thresholds["medium"])
        
        fall_indicators = 0
        
        # 1. Kiểm tra tỷ lệ khung hình (người nằm ngang) - chỉ số quan trọng nhất
        if aspect_ratio > thresholds["aspect_ratio"]:
            fall_indicators += 2  # Tăng trọng số cho chỉ số này
        
        # 2. Kiểm tra vị trí đầu so với hông (đầu thấp hơn hông)
        if nose is not None and (left_hip is not None or right_hip is not None):
            hip_y = left_hip[1] if left_hip is not None else right_hip[1]
            if right_hip is not None and left_hip is not None:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            
            if nose[1] > hip_y:  # Y tăng xuống dưới
                head_hip_diff = abs(nose[1] - hip_y) / bbox_height
                if head_hip_diff > thresholds["head_hip_ratio"]:
                    fall_indicators += 1
        
        # 3. Kiểm tra vai so với hông
        if (left_shoulder is not None or right_shoulder is not None) and (left_hip is not None or right_hip is not None):
            shoulder_y = left_shoulder[1] if left_shoulder is not None else right_shoulder[1]
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            hip_y = left_hip[1] if left_hip is not None else right_hip[1]
            if left_hip is not None and right_hip is not None:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            
            if shoulder_y > hip_y:
                shoulder_hip_diff = abs(shoulder_y - hip_y) / bbox_height
                if shoulder_hip_diff > thresholds["shoulder_hip_ratio"]:
                    fall_indicators += 1
        
        # 4. Kiểm tra chân so với thân
        if (left_ankle is not None or right_ankle is not None) and (left_hip is not None or right_hip is not None):
            ankle_y = left_ankle[1] if left_ankle is not None else right_ankle[1]
            if left_ankle is not None and right_ankle is not None:
                ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            
            hip_y = left_hip[1] if left_hip is not None else right_hip[1]
            if left_hip is not None and right_hip is not None:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            
            body_center_y = hip_y
            if nose is not None:
                body_center_y = (nose[1] + hip_y) / 2
            
            if ankle_y < body_center_y:  # Chân cao hơn thân
                knee_body_diff = abs(ankle_y - body_center_y) / bbox_height
                if knee_body_diff > thresholds["knee_body_ratio"]:
                    fall_indicators += 1
        
        # 5. Thêm kiểm tra góc nghiêng của thân người
        if (left_shoulder is not None or right_shoulder is not None) and (left_hip is not None or right_hip is not None):
            shoulder_x = left_shoulder[0] if left_shoulder is not None else right_shoulder[0]
            shoulder_y = left_shoulder[1] if left_shoulder is not None else right_shoulder[1]
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            hip_x = left_hip[0] if left_hip is not None else right_hip[0]
            hip_y = left_hip[1] if left_hip is not None else right_hip[1]
            if left_hip is not None and right_hip is not None:
                hip_x = (left_hip[0] + right_hip[0]) / 2
                hip_y = (left_hip[1] + right_hip[1]) / 2
            
            body_angle = abs(np.arctan2(shoulder_y - hip_y, shoulder_x - hip_x))
            # Nếu góc nghiêng > 45 độ (0.785 radian)
            if body_angle > 0.785:
                fall_indicators += 1
        
        # Quyết định dựa trên số lượng chỉ số (cần ít nhất 3 điểm)
        return fall_indicators >= 3
        
    except Exception as e:
        print(f"❌ Lỗi trong analyze_pose_for_fall: {e}")
        return False

def count_people_crossing_line(objects, line_position, frame_width):
    """
    Thay thế: hàm này giờ chuyển thành cập nhật thống kê số người trong frame.
    Args:
        objects: Dictionary của tracked objects
        line_position, frame_width: (không dùng nữa nhưng giữ signature để tương thích)
    """
    global last_frame_stats
    # Số người trong frame = số detections/tracked objects. Ở đây dùng số detections/objects.
    # Caller có thể cập nhật dựa trên rects hoặc objects; chúng ta sẽ tính theo objects length
    n_person = len(objects)
    # Cập nhật last_frame_stats (ip_camera sẽ được cập nhật nơi khởi tạo camera)
    ip_camera = CURRENT_CAMERA_SRC if CURRENT_CAMERA_SRC is not None else RTSP_URL
    last_frame_stats = build_frame_json(ip_camera=ip_camera, n_person=n_person)
    # Debug
    print(f"📊 Frame stats updated: {last_frame_stats}")

def generate_frames() -> Generator[bytes, None, None]:
    """
    Generator để streaming video frames với ThreadedCamera
    """
    global people_count_in, people_count_out, total_people, threaded_camera
    
    # Khởi tạo threaded camera nếu chưa có. Thử RTSP_URL trước nếu có, nếu không thì fallback webcam
    global CURRENT_CAMERA_SRC
    if threaded_camera is None:
        tried_sources = []
        if RTSP_URL:
            tried_sources.append(RTSP_URL)
        # Thử webcam index 0 nếu RTSP không có hoặc lỗi
        tried_sources.append(0)

        for src in tried_sources:
            try:
                print(f"🔄 Thử khởi tạo ThreadedCamera với: {src}")
                threaded_camera = ThreadedCamera(src)
                CURRENT_CAMERA_SRC = src
                print(f"⏳ Đợi camera ổn định...")
                time.sleep(2)
                test_ret, test_frame = threaded_camera.read()
                if test_ret and test_frame is not None:
                    print(f"✅ ThreadedCamera hoạt động bình thường với nguồn {src}, frame size: {test_frame.shape}")
                    break
                else:
                    print(f"⚠️ Nguồn {src} chưa trả frame, thử nguồn kế tiếp")
                    threaded_camera.stop()
                    threaded_camera = None
            except Exception as e:
                print(f"❌ Không thể khởi tạo camera với {src}: {e}")
                threaded_camera = None

        if threaded_camera is None:
            print("❌ Không có nguồn camera nào hoạt động. Kiểm tra RTSP_URL hoặc webcam.")
            return
    
    frame_skip_counter = 0
    video_config = config.get("video_config", {})
    frame_skip = video_config.get("frame_skip", 2)
    
    while True:
        try:
            if not hasattr(threaded_camera, 'read'):
                print("⚠️ ThreadedCamera chưa có method read")
                time.sleep(0.1)
                continue
                
            ret, frame = threaded_camera.read()
            
            if not ret:
                print(f"❌ Không đọc được frame: ret={ret}")
                time.sleep(0.1)
                continue
                
            if frame is None:
                print("❌ Frame là None")
                time.sleep(0.1)
                continue
                
            # Debug: In thông tin frame đầu tiên
            if frame_skip_counter == 0:
                print(f"✅ Đọc frame thành công: {frame.shape}, dtype: {frame.dtype}")
                
        except Exception as e:
            print(f"❌ Lỗi đọc frame: {e}")
            time.sleep(0.1)
            continue
        
        # Skip frames để giảm delay
        frame_skip_counter += 1
        if frame_skip_counter % frame_skip != 0:
            continue
        
        # Phát hiện người
        rects = detect_people(frame)
        
        # Debug: In số lượng người được phát hiện
        if len(rects) > 0:
            print(f"👥 Phát hiện {len(rects)} người: {rects}")
        
        # Cập nhật tracker
        objects = tracker.update(rects)
        
        # Debug: In số lượng objects được track
        if len(objects) > 0:
            print(f"🎯 Tracker đang theo dõi {len(objects)} objects")
        
        # Cập nhật thống kê: số người trong frame (dùng số detections trong rects)
        frame_height, frame_width = frame.shape[:2]
        line_position = frame_width // 2  # giữ biến để hàm cũ có cùng signature
        # Use detections count (rects) as people-in-frame
        n_person = len(rects)
        ip_camera = CURRENT_CAMERA_SRC if CURRENT_CAMERA_SRC is not None else RTSP_URL
        last_frame_stats = build_frame_json(ip_camera=ip_camera, n_person=n_person)

        # Vẽ bounding boxes cho detected people
        ui_config = config.get("ui_config", {})
        bbox_color = tuple(ui_config.get("bbox_color", [0, 255, 0]))
        bbox_thickness = ui_config.get("bbox_thickness", 2)
        text_color = tuple(ui_config.get("text_color", [0, 255, 0]))
        text_scale = ui_config.get("text_scale", 0.5)
        text_thickness = ui_config.get("text_thickness", 2)

        for rect in rects:
            (start_x, start_y, end_x, end_y) = rect
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), bbox_color, bbox_thickness)

        # Vẽ centroids và IDs
        for (object_id, centroid) in objects.items():
            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, text_color, -1)

        # Hiển thị thông tin đếm (số người hiện tại trong frame)
        n_person = last_frame_stats.get('n_person', 0)
        info_text = f"People in frame: {n_person}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Giảm thời gian sleep để tăng responsiveness
        time.sleep(0.01)  # Rất ngắn để giảm delay

def generate_fall_detection_frames() -> Generator[bytes, None, None]:
    """
    Generator để streaming video frames cho fall detection
    """
    global fall_detection_active, threaded_camera
    
    # Khởi tạo threaded camera nếu chưa có
    if threaded_camera is None:
        try:
            print(f"🔄 Bắt đầu khởi tạo ThreadedCamera cho fall detection...")
            # Reuse the same fallback logic: thử RTSP trước rồi webcam
            sources = [RTSP_URL] if RTSP_URL else []
            sources.append(0)
            for src in sources:
                try:
                    threaded_camera = ThreadedCamera(src)
                    CURRENT_CAMERA_SRC = src
                    print(f"✅ Fall detection camera khởi tạo với: {src}")
                    break
                except Exception as e:
                    threaded_camera = None
                    print(f"⚠️ Không thể mở nguồn fall detection với {src}: {e}")
            if threaded_camera is None:
                print("❌ Không có nguồn camera cho fall detection")
                return
            print(f"⏳ Đợi camera ổn định...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Lỗi khởi tạo ThreadedCamera: {e}")
            return
    
    frame_skip_counter = 0
    frame_skip = 3  # Skip nhiều frame hơn cho fall detection
    
    while fall_detection_active:
        try:
            ret, frame = threaded_camera.read()
            
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            # Skip frames để giảm delay
            frame_skip_counter += 1
            if frame_skip_counter % frame_skip != 0:
                continue
            
            # Phát hiện poses và falls
            poses_data, fall_detected, fall_bbox = detect_poses_and_falls(frame)
            
            # Vẽ pose keypoints và bounding boxes
            for pose in poses_data:
                bbox = pose['bbox']
                keypoints = pose['keypoints']
                is_fallen = pose['is_fallen']
                confidence = pose['confidence']
                tracker_id = pose.get('tracker_id', 'N/A')
                
                # Vẽ bounding box
                color = (0, 0, 255) if is_fallen else (0, 255, 0)  # Đỏ nếu ngã, xanh nếu bình thường
                thickness = 3 if is_fallen else 2
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
                
                # Vẽ label với tracker_id
                status = 'FALLEN' if is_fallen else 'NORMAL'
                label = f"ID:{tracker_id} {status} ({confidence:.2f})"
                label_color = (255, 255, 255)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                
                # Vẽ keypoints
                for i, kpt in enumerate(keypoints):
                    if kpt['confidence'] > 0.3:
                        x, y = int(kpt['x']), int(kpt['y'])
                        # Màu khác nhau cho các loại keypoints
                        if i in [0]:  # nose
                            kpt_color = (255, 0, 0)  # Đỏ
                        elif i in [5, 6]:  # shoulders
                            kpt_color = (0, 255, 0)  # Xanh lá
                        elif i in [11, 12]:  # hips
                            kpt_color = (0, 0, 255)  # Xanh dương
                        elif i in [15, 16]:  # ankles
                            kpt_color = (255, 255, 0)  # Vàng
                        else:
                            kpt_color = (255, 255, 255)  # Trắng
                        
                        cv2.circle(frame, (x, y), 4, kpt_color, -1)
                
                # Vẽ skeleton connections (một số kết nối chính)
                connections = [
                    (5, 6),   # shoulders
                    (5, 11),  # left shoulder to left hip
                    (6, 12),  # right shoulder to right hip
                    (11, 12), # hips
                    (11, 13), # left hip to left knee
                    (12, 14), # right hip to right knee
                    (13, 15), # left knee to left ankle
                    (14, 16), # right knee to right ankle
                ]
                
                for start_idx, end_idx in connections:
                    start_kpt = keypoints[start_idx]
                    end_kpt = keypoints[end_idx]
                    
                    if start_kpt['confidence'] > 0.3 and end_kpt['confidence'] > 0.3:
                        start_point = (int(start_kpt['x']), int(start_kpt['y']))
                        end_point = (int(end_kpt['x']), int(end_kpt['y']))
                        line_color = (0, 0, 255) if is_fallen else (0, 255, 0)
                        cv2.line(frame, start_point, end_point, line_color, 2)
            
            # Hiển thị thông tin fall detection
            info_text = f"Fall Detection: {'ON' if fall_detection_active else 'OFF'}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            stats_text = f"People: {people_detected_fall} | Falls: {total_falls}"
            cv2.putText(frame, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if fall_detected:
                alert_text = "FALL DETECTED!"
                cv2.putText(frame, alert_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Hiển thị sensitivity
            sens_text = f"Sensitivity: {fall_sensitivity.upper()}"
            cv2.putText(frame, sens_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.02)
            
        except Exception as e:
            print(f"❌ Lỗi trong generate_fall_detection_frames: {e}")
            time.sleep(0.1)

@app.get("/api")
async def api_info():
    """
    Endpoint API info
    """
    return {"message": "Camera Tracking API", "status": "running"}

@app.get("/health")
async def health_check():
    """
    Endpoint health check cho Docker
    """
    global threaded_camera, yolo_model
    
    # Kiểm tra trạng thái camera
    camera_status = "ok" if threaded_camera and threaded_camera.status else "error"
    
    # Kiểm tra model
    model_status = "ok" if yolo_model else "error"
    
    # Tổng trạng thái
    overall_status = "healthy" if camera_status == "ok" and model_status == "ok" else "unhealthy"
    
    return {
        "status": overall_status,
        "camera": camera_status,
        "model": model_status,
        "timestamp": time.time()
    }

@app.get("/video_feed")
async def video_feed():
    """
    Endpoint để streaming video
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/frame_stats")
async def frame_stats():
    """Return last computed frame stats as JSON"""
    global last_frame_stats
    if not last_frame_stats:
        # If empty, return a minimal object
        return build_frame_json(ip_camera=(CURRENT_CAMERA_SRC or RTSP_URL), n_person=0)
    return last_frame_stats

@app.get("/stats")
async def get_stats():
    global frame_counter, current_fps, detection_counts
    return {
        "total_frames": frame_counter,
        "current_fps": current_fps,
        "detection_counts": detection_counts
    }

# Xóa phần khởi tạo model trong process_video
async def process_video():
    global yolo_model, frame_counter, current_fps, detection_counts
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     results = yolo_model(frame)
    #     ... existing processing code ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
