#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

MODEL_PATH = "public/models/daytime.pt"
FRAME_INTERVAL = 15
CONFIDENCE = 0.3
RESIZED_INPUT = (640, 360)
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS
CAMERA_INDEX = 1
MASK_FADE_ALPHA = 0.25
MASK_DECAY = 0.94

model = YOLO(MODEL_PATH)
model.to("cpu").eval()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ Unable to access camera at index {CAMERA_INDEX}")
    exit(1)

frame_count = 0
cached_result = None
cached_mask = None
last_detect_time = None
prev_display_time = time.time()

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret or frame is None or frame.shape[0] == 0:
        continue

    frame_count += 1

    if frame_count % FRAME_INTERVAL == 0:
        input_frame = cv2.resize(frame, RESIZED_INPUT)
        with torch.no_grad():
            results = model.predict(input_frame, conf=CONFIDENCE, imgsz=640, verbose=False)
            cached_result = results[0]
        last_detect_time = time.time()
        torch.cuda.empty_cache()

        if cached_result and cached_result.masks is not None:
            try:
                mask = cached_result.masks.data[0].cpu().numpy()
                if np.any(mask > 0.5):
                    print("✅ Road boundary detected!")
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                cached_mask = mask.astype(np.float32)
            except Exception as e:
                print(f"⚠️ Mask processing error: {e}")
                cached_mask = None

    if cached_mask is not None:
        faded_mask = (cached_mask * MASK_DECAY).astype(np.uint8)
        overlay = frame.copy()
        overlay[faded_mask > 128] = [255, 0, 0]
        frame = cv2.addWeighted(overlay, MASK_FADE_ALPHA, frame, 1 - MASK_FADE_ALPHA, 0)
        contours, _ = cv2.findContours(faded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
        cached_mask = faded_mask

    curr_time = time.time()
    fps = 1 / (curr_time - prev_display_time)
    prev_display_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if last_detect_time:
        since_detect = curr_time - last_detect_time
        cv2.putText(frame, f"Last detection: {since_detect:.1f}s ago", (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Road Detection", frame)

    elapsed = time.time() - start_time
    delay = FRAME_TIME - elapsed
    if delay > 0:
        time.sleep(delay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
