!pip install ultralytics
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO


model_path = "/content/best.pt" #use the weights from the model generated from model.py
model = YOLO(model_path)

#input directory
input_path = "/content/frame_0001.jpg"


output_folder = "/content/TESTINg/output/"
os.makedirs(output_folder, exist_ok=True)

# Checking if video
if input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    is_video = True
    cap = cv2.VideoCapture(input_path)

    #video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #for unique timestamp
    timestamp = int(time.time())  # Unique timestamp
    output_path = os.path.join(output_folder, f"output_video_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    #for video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, conf=0.05, imgsz=640)

        if results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (width, height))

            # Create transparent blue overlay
            overlay = frame.copy()
            overlay[mask > 128] = [255, 0, 0]  # Blue fill

            # Blend overlay with original frame (Transparency: 0.3)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # Bold red boundary edges
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)  # Red boundary

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved at: {output_path}")
#for image
else:
    is_video = False
    frame = cv2.imread(input_path)
    height, width, _ = frame.shape

    # Run inference
    results = model.predict(frame, conf=0.5, imgsz=640)

    if results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (width, height))

        # Create transparent blue overlay
        overlay = frame.copy()
        overlay[mask > 128] = [255, 0, 0]  # Blue fill

        # Blend overlay with original frame (Transparency: 0.3)
        frame = cv2.addWeighted(overlay, 0.1, frame, 0.7, 0)

        # Bold red boundary edges
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)  # Red boundary


    timestamp = int(time.time())
    output_path = os.path.join(output_folder, f"output_image_{timestamp}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved at: {output_path}")
