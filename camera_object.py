"""
Real-Time Road Boundary and Object Detection using YOLOv8

- Uses a custom-trained YOLOv8 segmentation model for road boundary detection
- Adds lightweight YOLOv8n-based object detection for enhanced environmental awareness
- Optimized for CPU-based real-time performance (~5 FPS on MacBook with Iriun Webcam)

Author: Paritosh
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# ──────────────────────────────
# Configuration
# ──────────────────────────────

SEG_MODEL_PATH = "/models/daytime.pt"
OBJ_MODEL_PATH = "yolov8n.pt"
DEVICE = "cpu"

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CONFIDENCE_SEG = 0.35
CONFIDENCE_OBJ = 0.4
IMG_SIZE = 640

# ──────────────────────────────
# Load Models
# ──────────────────────────────

seg_model = YOLO(SEG_MODEL_PATH).to(DEVICE)
obj_model = YOLO(OBJ_MODEL_PATH).to(DEVICE)
print(f"[INFO] Models loaded on {DEVICE.upper()}")

# ──────────────────────────────
# Initialize Webcam (Iriun or USB)
# ──────────────────────────────

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ──────────────────────────────
# Inference Loop
# ──────────────────────────────

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    height, width = frame.shape[:2]

    # ─── Road Boundary Segmentation ───
    seg_result = seg_model.predict(
        frame,
        conf=CONFIDENCE_SEG,
        imgsz=IMG_SIZE,
        verbose=False,
        device=DEVICE
    )[0]

    if seg_result.masks is not None:
        for mask_tensor in seg_result.masks.data:
            mask = mask_tensor.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (width, height))

            overlay = frame.copy()
            overlay[mask > 128] = [255, 0, 0]  # Blue overlay
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)  # Red boundaries

    # ─── Object Detection ───
    obj_result = obj_model.predict(
        frame,
        conf=CONFIDENCE_OBJ,
        imgsz=IMG_SIZE,
        verbose=False,
        device=DEVICE
    )[0]

    if obj_result.boxes is not None:
        for box in obj_result.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            c = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{obj_model.names[c]} {conf:.2f}"

            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
            cv2.putText(frame, label, (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ─── FPS Calculation ───
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ─── Display Output ───
    cv2.imshow("Road + Object Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

# ──────────────────────────────
# Cleanup
# ──────────────────────────────

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera feed closed.")
