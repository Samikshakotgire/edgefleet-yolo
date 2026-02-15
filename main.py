import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = 'yolov8n.pt'  # Using the Nano model for speed
VIDEO_PATH = 'D:\EdgeFleetcodes\2.mp4'  # Path to the video from the Drive link
OUTPUT_VIDEO_PATH = 'results/output_with_trajectory.mp4'
OUTPUT_CSV_PATH = 'annotations/ball_tracking.csv'
BALL_CLASS_ID = 32  # COCO class ID for 'sports ball'

# Trajectory settings
TRAJECTORY_LEN = 30  # Number of previous frames to show in the tail
pts = deque(maxlen=TRAJECTORY_LEN)

def process_cricket_video():
    # 1. Load Model
    model = YOLO(MODEL_PATH)
    
    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 3. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    # Data storage for CSV
    trajectory_data = []
    frame_idx = 0

    print("Processing video... This may take a few minutes.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Run YOLOv8 Tracking
        # persist=True tells YOLO to remember objects across frames
        # classes=32 filters for only balls
        results = model.track(frame, persist=True, classes=[BALL_CLASS_ID], verbose=False)

        ball_detected = False
        x_centroid, y_centroid = -1.0, -1.0
        visibility = 0

        # 5. Extract Detection Info
        if results[0].boxes:
            # We take the first detected ball with the highest confidence
            box = results[0].boxes[0]
            # Get coordinates (x1, y1, x2, y2)
            coords = box.xyxy[0].tolist()
            
            # Calculate Centroid
            x_centroid = (coords[0] + coords[2]) / 2
            y_centroid = (coords[1] + coords[3]) / 2
            visibility = 1
            ball_detected = True
            
            # Add to trajectory point list
            pts.appendleft((int(x_centroid), int(y_centroid)))

        # 6. Logging for CSV
        trajectory_data.append([frame_idx, round(x_centroid, 1), round(y_centroid, 1), visibility])

        # 7. Draw Trajectory Overlay
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            # Draw a line between consecutive points (the tail)
            thickness = int(np.sqrt(TRAJECTORY_LEN / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), thickness)
        
        if ball_detected:
            # Draw the current centroid dot
            cv2.circle(frame, (int(x_centroid), int(y_centroid)), 5, (0, 0, 255), -1)

        # Write frame to output video
        out.write(frame)
        frame_idx += 1

    # 8. Clean up
    cap.release()
    out.release()
    
    # 9. Save CSV
    df = pd.DataFrame(trajectory_data, columns=['frame', 'x', 'y', 'visible'])
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Done! Results saved to {OUTPUT_VIDEO_PATH} and {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    process_cricket_video()