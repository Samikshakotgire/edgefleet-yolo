import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import deque
import os

# --- CONFIGURATION ---
MODEL_PATH = 'yolov8n.pt'  
VIDEO_PATH = 'input_video.mp4'  # <--- CHANGE THIS to your filename (e.g., 'cricket_clip.mp4')
OUTPUT_FOLDER = 'results'
ANNOTATION_FOLDER = 'annotations'
BALL_CLASS_ID = 32 

# Create folders if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

# Trajectory settings
pts = deque(maxlen=30)

def process_cricket_video():
    print("Step 1: Loading Model...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file {VIDEO_PATH}. Check the filename!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(f'{OUTPUT_FOLDER}/output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    trajectory_data = []
    frame_idx = 0

    print(f"Step 2: Processing Video ({width}x{height} at {fps} FPS)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run Tracking with lower confidence (conf=0.15) to catch small balls
        results = model.track(frame, persist=True, classes=[BALL_CLASS_ID], conf=0.15, verbose=False)

        x_centroid, y_centroid, visibility = -1.0, -1.0, 0

        if results[0].boxes:
            box = results[0].boxes[0]
            coords = box.xyxy[0].tolist()
            x_centroid = (coords[0] + coords[2]) / 2
            y_centroid = (coords[1] + coords[3]) / 2
            visibility = 1
            pts.appendleft((int(x_centroid), int(y_centroid)))

        # Log data
        trajectory_data.append([frame_idx, round(x_centroid, 1), round(y_centroid, 1), visibility])

        # Draw on frame
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)
        if visibility == 1:
            cv2.circle(frame, (int(x_centroid), int(y_centroid)), 5, (0, 0, 255), -1)

        out.write(frame)
        
        # --- DEBUG MESSAGE ---
        if frame_idx % 20 == 0:
            status = "FOUND BALL" if visibility == 1 else "SEARCHING..."
            print(f"Frame {frame_idx}: {status}")

        frame_idx += 1

    # SAVE THE CSV
    print("Step 3: Saving CSV data...")
    df = pd.DataFrame(trajectory_data, columns=['frame', 'x', 'y', 'visible'])
    df.to_csv(f'{ANNOTATION_FOLDER}/ball_tracking.csv', index=False)
    
    cap.release()
    out.release()
    print(f"SUCCESS! Files saved in /{OUTPUT_FOLDER} and /{ANNOTATION_FOLDER}")

if __name__ == "__main__":
    process_cricket_video()