# Cricket Ball Tracking System

A comprehensive **computer vision pipeline** to detect and track cricket balls in videos from a fixed camera. Built for the EdgeFleet AI/ML Assessment.

## 🎯 Objective

Detect the cricket ball **centroid** in each frame where visible, generate per-frame **CSV annotations**, produce an **annotated video** with trajectory overlay, and provide **fully reproducible** code.

## ✨ Key Features

- **YOLOv8 Detection**: Real-time ball detection across video frames
- **ByteTrack**: Maintains consistent object identity across frames
- **Kalman Filtering**: Predicts ball trajectory for smooth motion estimation
- **Comprehensive Logging**: Track pipeline progress and metrics
- **WandB Integration**: Monitor experiments and visualize results
- **Robust Error Handling**: Graceful failure recovery and detailed logging
- **Modular Code**: Clean, documented functions for easy maintenance

## 📋 Requirements

- **Python**: 3.8+
- **OS**: Windows, Linux, macOS
- **GPU** (Optional): Recommended for faster inference

## 🚀 Quick Start

### 1️⃣ Setup Python Environment

```bash
# Clone/enter the repository
cd EdgeFleetcodes

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Input Videos

Place your cricket videos in the `input_videos/` directory:

```
input_videos/
├── match1.mp4
├── match2.mov
└── match3.avi
```

**Supported Formats**: `.mp4`, `.avi`, `.mov`, `.mkv`

### 4️⃣ Download YOLOv8 Model

The model file `yolov8n.pt` should be in the project root. If not:

```bash
# Option 1: Manual download
# Visit: https://github.com/ultralytics/assets/releases

# Option 2: Auto-download via Python
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 5️⃣ Configure WandB (Optional but Recommended)

For experiment tracking and visualization:

```bash
wandb login
# Follow prompts to authenticate with your W&B account
```

### 6️⃣ Run the Pipeline

```bash
cd codes/
python main.py
```

## 📂 Project Structure

```
EdgeFleetcodes/
├── codes/
│   ├── main.py                 # Main detection & tracking pipeline
│   ├── eval.py                 # Evaluation metrics
│   ├── kalman.py               # Kalman filter utilities
│   └── app.py                  # (Optional) Web interface
├── annotations/                # Output CSV files
│   ├── 1_data.csv
│   ├── 2_data.csv
│   └── ...
├── results/                    # Output annotated videos
│   ├── 1_processed.mp4
│   ├── 2_processed.mp4
│   └── ...
├── input_videos/               # Input cricket videos
├── yolov8n.pt                  # YOLOv8 Nano model
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── report.pdf                  # Detailed technical report
```

## 📊 Output Formats

### CSV Annotation File

Each video generates a CSV with per-frame detections:

**File**: `annotations/<video_name>_data.csv`

```csv
frame,x,y,visible
0,512.3,298.1,1
1,518.7,305.4,1
2,-1.0,-1.0,0
3,525.1,312.8,1
...
```

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int | Frame index (0-based) |
| `x` | float | Centroid X coordinate in pixels (-1.0 if not visible) |
| `y` | float | Centroid Y coordinate in pixels (-1.0 if not visible) |
| `visible` | int | 1 if ball detected, 0 otherwise |

### Processed Video

**File**: `results/<video_name>_processed.mp4`

Overlays on the video output:
- 🔴 **RED circle**: Detected ball centroid
- 🔵 **BLUE circle**: Kalman-predicted next position
- 💛 **YELLOW trail**: Historical trajectory (last 30 frames)
- **Confidence score**: Detection confidence (0.0-1.0)

## ⚙️ Configuration & Hyperparameters

Edit these in `codes/main.py`:

```python
# Model & I/O
MODEL_PATH = 'yolov8n.pt'          # Path to YOLO model
INPUT_DIR = 'input_videos'          # Input directory
OUTPUT_DIR = 'results'              # Output directory
ANNOTATION_DIR = 'annotations'      # Annotation directory

# Detection
BALL_CLASS_ID = 32                  # COCO class for ball
DETECTION_PARAMS = {
    'confidence_threshold': 0.15,    # Detection confidence (0-1)
    'max_trail_length': 30           # Trajectory trail length
}

# Kalman Filter
KALMAN_PARAMS = {
    'dim_x': 4,                      # State dimension [x_pos, x_vel, y_pos, y_vel]
    'dim_z': 2,                      # Measurement dimension [x_pos, y_pos]
    'process_noise': 1000,           # Process noise (P)
    'measurement_noise': 5           # Measurement noise (R)
}

# WandB
WANDB_PROJECT = "cricket-ball-tracker-edgefleet"
```

## 📈 Monitoring with WandB

View real-time metrics and visualizations:

```bash
# Your WandB dashboard URL will be printed during execution
# Example: https://wandb.ai/your-username/cricket-ball-tracker-edgefleet
```

**Logged Metrics**:
- Detection rate per video
- Confidence scores
- Frame-by-frame progress
- Processed video artifacts

## 🔍 How It Works

### 1. Frame Detection
- **YOLOv8** detects objects matching COCO class 32 (ball)
- Returns bounding box + confidence score

### 2. Object Tracking
- **ByteTrack** maintains consistent ID across frames
- Prevents ID switches when ball briefly occludes

### 3. Trajectory Prediction
- **Kalman Filter** estimates velocity from motion
- Predicts next position even if ball is occluded
- Constant velocity motion model

### 4. Visualization & Output
- Draws detected position (RED), predicted position (BLUE), trail (YELLOW)
- Exports annotated video + CSV with complete frame-by-frame data

## 🧪 Evaluation & Validation

Run evaluation metrics on processed data:

```bash
cd codes/
python eval.py
```

Outputs performance metrics like detection rate, consistency scores, trajectory smoothness.

## 🛠️ Troubleshooting

### Issue: "Model file not found"
- **Solution**: Ensure `yolov8n.pt` is in the project root
- Download from: https://github.com/ultralytics/assets/releases

### Issue: "No video files found"
- **Solution**: Place videos in `input_videos/` directory
- Check file extensions are `.mp4`, `.avi`, `.mov`, or `.mkv`

### Issue: Low detection rate
- **Lower** `DETECTION_PARAMS['confidence_threshold']` (default: 0.15)
- Ensure good **lighting** in cricket videos
- Ball may be too fast or blurry for detection

### Issue: Kalman predictions are unstable
- **Increase** `KALMAN_PARAMS['process_noise']` (higher = more tolerance)
- **Lower** `KALMAN_PARAMS['measurement_noise']` (lower = trust measurements more)

### Issue: WandB authentication fails
- Run: `wandb login` and follow prompts
- Or set env var: `WANDB_API_KEY=your_key`

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | 8.3.0 | YOLOv8 detection |
| opencv-python | 4.10.0.84 | Video I/O & visualization |
| pandas | latest | CSV handling |
| numpy | latest | Numerical operations |
| supervision | 0.20.0 | Detection utilities |
| filterpy | 1.4.5 | Kalman filtering |
| wandb | 0.17.0 | Experiment tracking |
| matplotlib | latest | Visualization (optional) |

## 📄 License

Assessment project for EdgeFleet AI/ML evaluation.

## 👨‍💼 Author

EdgeFleet AI Assessment  
February 2026

## 📞 Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Review **logs** in console output
3. Check WandB dashboard for metrics
4. Refer to **report.pdf** for technical details

---

**Status**: ✅ Production-ready evaluation pipeline
