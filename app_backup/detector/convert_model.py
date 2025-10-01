from ultralytics import YOLO

# Load model YOLOv8n
model = YOLO("yolov8n.pt")

# Export sang ONNX
# model.export(format="onnx", opset=12, simplify=True)
# Export sang TFLite (float32)
model.export(format="tflite", int8=True)