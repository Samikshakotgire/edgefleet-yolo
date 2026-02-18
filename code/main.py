"""
Cricket Ball Tracking System

A comprehensive computer vision system to detect and track cricket balls
in videos recorded from a fixed camera using YOLOv8, ByteTrack, and Kalman filtering.

Author: EdgeFleet AI Assessment
Date: February 2026
"""

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import glob
import logging
import json
from pathlib import Path

# Advanced tracking and filtering
import supervision as sv
from filterpy.kalman import KalmanFilter
import wandb
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'yolov8n.pt'  
INPUT_DIR = 'input_videos'     
OUTPUT_DIR = 'results'
ANNOTATION_DIR = 'annotations'
BALL_CLASS_ID = 32  # COCO class ID for ball/sports objects

# WandB Configuration for experiment tracking
WANDB_PROJECT = "cricket-ball-tracker-edgefleet"

# Kalman Filter Parameters
KALMAN_PARAMS = {
    'dim_x': 4,  # State: [x_pos, x_vel, y_pos, y_vel]
    'dim_z': 2,  # Measurement: [x_pos, y_pos]
    'process_noise': 1000,  # Process noise covariance (P)
    'measurement_noise': 5   # Measurement noise covariance (R)
}

# Detection Parameters
DETECTION_PARAMS = {
    'confidence_threshold': 0.15,
    'max_trail_length': 30  # Max points in trajectory trail
}
# Use WandB only if an API key is provided in the environment
USE_WANDB = bool(os.getenv('WANDB_API_KEY'))

# Create folders automatically
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(ANNOTATION_DIR).mkdir(parents=True, exist_ok=True)

logger.info(f"Output directories created: {OUTPUT_DIR}, {ANNOTATION_DIR}")


def initialize_kalman_filter():
    """
    Initialize and configure Kalman Filter for trajectory prediction.
    
    State model: Constant velocity model
    - State: [x_pos, x_vel, y_pos, y_vel]
    - Measurement: [x_pos, y_pos]
    
    Returns:
        KalmanFilter: Configured Kalman filter instance
    """
    kf = KalmanFilter(
        dim_x=KALMAN_PARAMS['dim_x'],
        dim_z=KALMAN_PARAMS['dim_z']
    )
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([
        [1, 1, 0, 0],  # x_pos += x_vel
        [0, 1, 0, 0],  # x_vel unchanged
        [0, 0, 1, 1],  # y_pos += y_vel
        [0, 0, 0, 1]   # y_vel unchanged
    ], dtype=float)
    
    # Measurement matrix (we measure [x, y] from [x, x_vel, y, y_vel])
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=float)
    
    # Noise covariances
    kf.R *= KALMAN_PARAMS['measurement_noise']
    kf.P *= KALMAN_PARAMS['process_noise']
    
    # Initial state
    kf.x = np.zeros(4, dtype=float)
    
    logger.debug("Kalman Filter initialized successfully")
    return kf


def load_model(model_path):
    """
    Load YOLOv8 model for object detection.
    
    Args:
        model_path (str): Path to YOLO model file
        
    Returns:
        YOLO: Loaded YOLO model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)
    logger.info("Model loaded successfully")
    return model


def process_all_videos():
    """
    Main pipeline: Process all input videos for ball detection and tracking.
    
    For each video:
    1. Load frame by frame
    2. Detect ball using YOLOv8
    3. Track with ByteTrack for ID consistency
    4. Predict trajectory with Kalman Filter
    5. Overlay visualizations
    6. Export CSV annotations and processed video
    7. Log metrics to WandB
    """
    global USE_WANDB
    
    try:
        # Initialize WandB for experiment tracking (optional)
        if USE_WANDB:
            try:
                wandb.init(
                    project=WANDB_PROJECT,
                    config={
                        "confidence_threshold": DETECTION_PARAMS['confidence_threshold'],
                        "model": "yolov8n",
                        "tracker": "ByteTrack",
                        "prediction_method": "Kalman Filter"
                    }
                )
                logger.info("WandB initialized")
            except Exception as e:
                logger.warning(f"WandB authentication failed: {str(e)}. Continuing without WandB logging.")
                USE_WANDB = False
        else:
            logger.info("WANDB_API_KEY not set; running without WandB logging")

        # Load AI Model and Trackers
        logger.info("Loading AI Model + Advanced Trackers...")
        model = load_model(MODEL_PATH)
        byte_tracker = sv.ByteTrack()  # For consistent object ID tracking
        kf = initialize_kalman_filter()  # For trajectory prediction

        # Find all video files
        video_files = glob.glob(os.path.join(INPUT_DIR, "*.*"))
        video_files = [
            f for f in video_files 
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        
        if not video_files:
            logger.warning(f"No video files found in {INPUT_DIR}")
            return
        
        logger.info(f"Found {len(video_files)} videos to process")

        # Process each video
        for video_idx, video_path in enumerate(video_files, 1):
            try:
                process_single_video(
                    model, byte_tracker, kf, video_path
                )
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
                continue

        if USE_WANDB:
            try:
                wandb.finish()
            except Exception:
                pass
        logger.info("✅ ALL VIDEOS PROCESSED! Check results/ folder")
        
    except Exception as e:
        logger.error(f"Fatal error in process_all_videos: {str(e)}")
        if USE_WANDB:
            try:
                wandb.finish()
            except Exception:
                pass
        raise


def process_single_video(model, byte_tracker, kf, video_path):
    """
    Process a single video for ball detection and tracking.
    
    Args:
        model (YOLO): YOLOv8 model for detection
        byte_tracker (ByteTrack): Tracker for maintaining ball ID
        kf (KalmanFilter): Kalman filter for trajectory prediction
        video_path (str): Path to input video file
    """
    file_name = os.path.basename(video_path).split('.')[0]
    logger.info(f"--- Starting Video: {file_name} ---")

    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video props: {width}x{height} @ {fps:0.1f} FPS, {total_frames} frames")

        # Prepare output paths
        video_out_path = os.path.join(OUTPUT_DIR, f"{file_name}_processed.mp4")
        csv_out_path = os.path.join(ANNOTATION_DIR, f"{file_name}_data.csv")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        
        # Data storage
        pts = deque(maxlen=DETECTION_PARAMS['max_trail_length'])
        trajectory_data = []
        frame_idx = 0
        detection_count = 0

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detection & Tracking
            results = model.track(
                frame,
                persist=True,
                classes=[BALL_CLASS_ID],
                conf=DETECTION_PARAMS['confidence_threshold'],
                verbose=False
            )
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)

            # Default: ball not detected
            x_centroid, y_centroid, visibility, conf = -1.0, -1.0, 0, 0.0
            
            if len(detections) > 0:
                # Extract best detection
                box = detections.xyxy[0]
                conf = float(detections.confidence[0])
                x_centroid = float((box[0] + box[2]) / 2)
                y_centroid = float((box[1] + box[3]) / 2)
                visibility = 1
                detection_count += 1
                
                # Kalman Filter: Update with measurement & predict next position
                kf.update(np.array([x_centroid, 0, y_centroid, 0]))
                predicted = kf.predict()
                pred_x, pred_y = int(predicted[0]), int(predicted[2])
                
                # Add to trajectory trail
                pts.appendleft((int(x_centroid), int(y_centroid)))
                
                # Draw predicted position (BLUE circle)
                cv2.circle(frame, (pred_x, pred_y), 8, (255, 0, 0), -1)
                cv2.putText(
                    frame, "PREDICTED",
                    (pred_x + 10, pred_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                )

            # Store annotation data
            trajectory_data.append([frame_idx, round(x_centroid, 1), round(y_centroid, 1), visibility])

            # Draw trajectory trail (yellow line)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0, 255, 255), 2)
            
            # Draw detected ball (RED circle) with confidence
            if visibility == 1:
                cv2.circle(frame, (int(x_centroid), int(y_centroid)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"Conf:{conf:.2f}",
                    (int(x_centroid) + 10, int(y_centroid)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )

            # Write frame to output video
            out.write(frame)
            frame_idx += 1
            
            # Log metrics to WandB every 100 frames
            if frame_idx % 100 == 0 and frame_idx > 0:
                detection_rate = detection_count / frame_idx
                if USE_WANDB:
                    wandb.log({
                        "frame": frame_idx,
                        "detection_rate": detection_rate,
                        "video": file_name
                    })
                logger.info(f"  Frame {frame_idx}/{total_frames}, Detection Rate: {detection_rate:.1%}")

        # Save CSV annotation file
        df = pd.DataFrame(
            trajectory_data,
            columns=['frame', 'x', 'y', 'visible']
        )
        df.to_csv(csv_out_path, index=False)
        logger.info(f"✅ Annotation file saved: {csv_out_path}")
        
        # Log final metrics to WandB (optional)
        final_detection_rate = detection_count / frame_idx if frame_idx > 0 else 0
        if USE_WANDB:
            wandb.log({
                "video": wandb.Video(video_out_path),
                "detection_rate_final": final_detection_rate,
                "total_frames": frame_idx,
                "detections": detection_count,
                "video_name": file_name
            })
        
        cap.release()
        out.release()
        logger.info(f"✅ {file_name}: {final_detection_rate:.1%} detection rate")
        
    except Exception as e:
        logger.error(f"Exception in process_single_video for {video_path}: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("CRICKET BALL TRACKING SYSTEM - Starting Pipeline")
        logger.info("=" * 60)
        process_all_videos()
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        exit(1)
