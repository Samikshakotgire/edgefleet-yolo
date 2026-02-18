"""
Configuration Module for Cricket Ball Tracking System

This module centralizes all hyperparameters and configuration values
for the ball detection, tracking, and visualization pipeline.

Usage:
    from config import ModelConfig, DetectionConfig, KalmanConfig
    conf = DetectionConfig()
    print(conf.confidence_threshold)
"""

import os
from pathlib import Path


class PathConfig:
    """File and directory paths configuration."""
    
    # Project root
    ROOT_DIR = Path(__file__).parent.parent
    
    # Directories
    INPUT_DIR = str(ROOT_DIR / 'input_videos')
    OUTPUT_DIR = str(ROOT_DIR / 'results')
    ANNOTATION_DIR = str(ROOT_DIR / 'annotations')
    
    # Model
    MODEL_PATH = str(ROOT_DIR / 'yolov8n.pt')
    
    @classmethod
    def ensure_directories(cls):
        """Create output directories if they don't exist."""
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.ANNOTATION_DIR).mkdir(parents=True, exist_ok=True)


class ModelConfig:
    """YOLOv8 and detection model configuration."""
    
    # Model selection
    MODEL_TYPE = 'yolov8n'  # Options: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
    
    # Ball detection class
    BALL_CLASS_ID = 32  # COCO dataset: class 32 includes sports balls
    
    # Model input settings
    IMGSZ = 640  # Input image size (pixels)
    }


class DetectionConfig:
    """Ball detection configuration."""
    
    # Confidence threshold for detections
    CONFIDENCE_THRESHOLD = 0.15
    
    # IOA threshold for ByteTrack
    IOU_THRESHOLD = 0.5
    
    # Visualization
    TRAIL_MAX_LENGTH = 30  # Maximum trajectory trail points
    CIRCLE_RADIUS = 5  # Detected ball circle radius (pixels)
    PREDICTED_RADIUS = 8  # Predicted position circle radius (pixels)
    
    # Detection is frame interpolation
    INFER_EVERY_N_FRAMES = 1  # Run detection on every N-th frame (1 = every frame)


class KalmanFilterConfig:
    """Kalman Filter configuration for trajectory prediction."""
    
    # State space dimensions
    STATE_DIM = 4  # [x_pos, x_vel, y_pos, y_vel]
    MEASUREMENT_DIM = 2  # [x_pos, y_pos]
    
    # Noise parameters (higher values = more tolerance)
    PROCESS_NOISE_MATRIX = 1000.0  # Model uncertainty (P matrix)
    MEASUREMENT_NOISE_MATRIX = 5.0  # Sensor uncertainty (R matrix)
    
    # Special case: object lost prediction
    # How many frames to predict ahead when ball is not detected
    LOOKAHEAD_FRAMES = 5
    
    @classmethod
    def get_state_transition_matrix(cls):
        """
        Get F matrix for constant velocity model.
        
        State: [x_pos, x_vel, y_pos, y_vel]^T
        Transition: x_pos(t+1) = x_pos(t) + x_vel(t)
        """
        import numpy as np
        return np.array([
            [1, 1, 0, 0],  # x_pos += x_vel
            [0, 1, 0, 0],  # x_vel unchanged
            [0, 0, 1, 1],  # y_pos += y_vel
            [0, 0, 0, 1]   # y_vel unchanged
        ], dtype=float)
    
    @classmethod
    def get_measurement_matrix(cls):
        """
        Get H matrix for position measurement.
        
        We measure [x_pos, y_pos] from state [x_pos, x_vel, y_pos, y_vel]
        """
        import numpy as np
        return np.array([
            [1, 0, 0, 0],  # Measure x_pos
            [0, 0, 1, 0]   # Measure y_pos
        ], dtype=float)


class TrackerConfig:
    """Multi-object tracker configuration."""
    
    # ByteTrack parameters
    TRACK_ACTIVATION_THRESHOLD = 0.25  # Activation threshold for new tracks
    MIN_TRACK_LENGTH = 5  # Minimum frames to maintain track
    TRACK_BUFFER = 30  # Number of frames to buffer lost tracks


class WandBConfig:
    """Weights & Biases experiment tracking configuration."""
    
    # Project and entity
    PROJECT_NAME = "cricket-ball-tracker-edgefleet"
    ENTITY = None  # Set to your W&B team name if using team workspace
    
    # Logging frequency
    LOG_EVERY_N_FRAMES = 100  # Log metrics every N frames
    
    # Config to log
    CONFIG = {
        "model": "yolov8n",
        "tracker": "ByteTrack",
        "prediction_method": "Kalman Filter",
        "ball_class_id": ModelConfig.BALL_CLASS_ID,
        "confidence_threshold": DetectionConfig.CONFIDENCE_THRESHOLD,
    }


class VideoConfig:
    """Video encoding and output configuration."""
    
    # Video codec
    CODEC = 'mp4v'  # MP4 codec
    
    # Supported input formats
    SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    # Video properties
    PRESERVE_FPS = True  # Keep original FPS
    PRESERVE_RESOLUTION = True  # Keep original resolution


class LoggingConfig:
    """Logging configuration."""
    
    LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console output
    ENABLE_CONSOLE = True
    
    # File output
    ENABLE_FILE_LOG = True
    LOG_FILE = 'cricket_tracker.log'


# Export commonly used configs
if __name__ == "__main__":
    # Quick config inspection
    print("=" * 60)
    print("CRICKET BALL TRACKING - CONFIGURATION")
    print("=" * 60)
    
    print("\n📁 Paths:")
    print(f"  Input Dir: {PathConfig.INPUT_DIR}")
    print(f"  Output Dir: {PathConfig.OUTPUT_DIR}")
    print(f"  Model: {PathConfig.MODEL_PATH}")
    
    print("\n🎯 Detection:")
    print(f"  Confidence Threshold: {DetectionConfig.CONFIDENCE_THRESHOLD}")
    print(f"  Trail Length: {DetectionConfig.TRAIL_MAX_LENGTH}")
    
    print("\n🔮 Kalman Filter:")
    print(f"  State Dim: {KalmanFilterConfig.STATE_DIM}")
    print(f"  Process Noise: {KalmanFilterConfig.PROCESS_NOISE_MATRIX}")
    print(f"  Measurement Noise: {KalmanFilterConfig.MEASUREMENT_NOISE_MATRIX}")
    
    print("\n📊 WandB:")
    print(f"  Project: {WandBConfig.PROJECT_NAME}")
    print(f"  Log Frequency: Every {WandBConfig.LOG_EVERY_N_FRAMES} frames")
    
    print("\n✅ Configuration ready!")
