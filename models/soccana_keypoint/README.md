---
datasets:
- Adit-jain/Soccana_Keypoint_detection_v1
language:
- en
base_model:
- Ultralytics/YOLO11
pipeline_tag: object-detection
tags:
- soccer
- football
- pitch
- field
- ground
- keypoint
- detection
- camera
- calibration
- homography
---

# ⚽ Soccer Field Keypoint Detection Model

<div align="center">

<p>
<img src="Keypoint_thumbnail.jpg" width="600"/>
</p>

[Demo Link](https://drive.google.com/file/d/1zrQ76-K3Dr0YYS_i3RUxoOGPowcCUiBE/view?usp=sharing)

*Advanced computer vision model for detecting and analyzing soccer field keypoints using YOLOv11 pose estimation*

[![GitHub](https://img.shields.io/badge/GitHub-Soccer_Analysis-blue?logo=github)](https://github.com/Adit-jain/Soccer_Analysis)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green?logo=python)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Pose-orange)](https://github.com/ultralytics/ultralytics)

</div>

## 📖 Introduction

The **Soccer Field Keypoint Detection Model** is a computer vision solution designed specifically for detecting and analyzing soccer field keypoints in video streams and images. Built on the YOLOv11 pose estimation architecture, this model can accurately identify 29 critical keypoints that define the geometry of a soccer field, including corners, penalty areas, goal areas, center circle, and other field markings.

This model is part of the comprehensive [Soccer Analysis](https://github.com/Adit-jain/Soccer_Analysis) project and enables advanced tactical analysis, field coordinate transformations, and homography calculations for professional soccer video analysis.

### 🎯 Key Features

- **29-Point Field Detection**: Comprehensive keypoint coverage including all major field markings
- **Real-Time Performance**: Optimized for live video analysis and streaming applications  
- **High Accuracy**: Robust detection across various camera angles, lighting conditions, and field qualities
- **FIFA Standards Compliant**: Keypoint mapping follows official FIFA field specifications
- **Tactical Analysis Ready**: Direct integration with homography transformations and tactical overlays

The model demonstrates exceptional performance across diverse scenarios:

#### ✅ **Multi-Scenario Detection**
- **Various Camera Angles**: From broadcast to close-up perspectives
- **Different Field Conditions**: Natural grass, artificial turf, various lighting
- **Partial Field Visibility**: Robust detection even with incomplete field views
- **Real-Time Processing**: 30+ FPS on standard hardware

#### 📊 **Detection Examples**
- Corner detection with sub-pixel accuracy
- Penalty area boundary identification  
- Center circle and center line detection
- Goal area precise mapping
- Sideline boundary recognition

## 🏗️ Model Details and Architecture

### Base Architecture
- **Model Type**: YOLOv11 Pose Estimation
- **Input Resolution**: 640×640 pixels (configurable)
- **Keypoint Count**: 29 field-specific keypoints
- **Output Format**: (x, y, visibility) for each keypoint
- **Framework**: Ultralytics YOLOv11 with PyTorch backend

### Keypoint Mapping (29 Points)

The model detects 29 strategically placed keypoints covering all major field elements:

#### 🏁 **Field Boundaries (4 points)**
- `sideline_top_left` (0): Top-left corner of the field
- `sideline_top_right` (16): Top-right corner of the field  
- `sideline_bottom_left` (9): Bottom-left corner of the field
- `sideline_bottom_right` (25): Bottom-right corner of the field

#### ⚽ **Penalty Areas (8 points)**
- Left penalty area: `big_rect_left_*` (1-4)
- Right penalty area: `big_rect_right_*` (17-20)

#### 🥅 **Goal Areas (8 points)**
- Left goal area: `small_rect_left_*` (5-8) 
- Right goal area: `small_rect_right_*` (21-24)

#### 🎯 **Center Elements (9 points)**
- Center line: `center_line_top` (11), `center_line_bottom` (12)
- Center circle: `center_circle_*` (13-14, 27-28)
- Field center: `field_center` (15)
- Semicircles: `left_semicircle_right` (10), `right_semicircle_left` (26)

```python
KEYPOINT_NAMES = {
    0: "sideline_top_left",
    1: "big_rect_left_top_pt1", 
    2: "big_rect_left_top_pt2",
    3: "big_rect_left_bottom_pt1",
    4: "big_rect_left_bottom_pt2",
    5: "small_rect_left_top_pt1",
    6: "small_rect_left_top_pt2", 
    7: "small_rect_left_bottom_pt1",
    8: "small_rect_left_bottom_pt2",
    9: "sideline_bottom_left",
    10: "left_semicircle_right",
    11: "center_line_top",
    12: "center_line_bottom", 
    13: "center_circle_top",
    14: "center_circle_bottom",
    15: "field_center",
    16: "sideline_top_right",
    17: "big_rect_right_top_pt1",
    18: "big_rect_right_top_pt2",
    19: "big_rect_right_bottom_pt1",
    20: "big_rect_right_bottom_pt2",
    21: "small_rect_right_top_pt1",
    22: "small_rect_right_top_pt2",
    23: "small_rect_right_bottom_pt1", 
    24: "small_rect_right_bottom_pt2",
    25: "sideline_bottom_right",
    26: "right_semicircle_left",
    27: "center_circle_left",
    28: "center_circle_right",
}
```

### Technical Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Input Size** | 640×640 | Default input resolution |
| **Batch Size** | 32 | Training batch size |
| **Epochs** | 200 | Default training epochs |
| **Confidence Threshold** | 0.5 | Keypoint visibility threshold |
| **Learning Rate** | 0.01 | Initial learning rate |
| **Dropout** | 0.3 | Regularization dropout rate |
| **Architecture** | YOLOv11n-pose | Efficient pose estimation variant |

## 📤 Model Output

### Detection Format
The model outputs keypoint detections in the following structure:

```python
keypoints: np.ndarray  # Shape: (N, 29, 3)
# N = number of field detections
# 29 = number of keypoints per detection  
# 3 = (x_coordinate, y_coordinate, visibility_confidence)
```

### Visibility Confidence
- **Range**: 0.0 to 1.0
- **Threshold**: 0.5 (configurable)
- **Interpretation**: 
  - `> 0.5`: Keypoint is visible and reliable
  - `≤ 0.5`: Keypoint is occluded or uncertain

### Field Corner Extraction
```python
corners = {
    'top_left': (x, y),      # Field corner coordinates
    'top_right': (x, y),     # in image pixel space
    'bottom_left': (x, y),
    'bottom_right': (x, y)
}
```

### Field Dimensions
```python
dimensions = {
    'width': field_width,    # Calculated field width in pixels
    'height': field_height,  # Calculated field height in pixels  
    'area': field_area       # Total field area
}
```

## 🚀 Usage and Implementation

### Quick Start

```python
from keypoint_detection import load_keypoint_model, get_keypoint_detections
import cv2

# Load the keypoint detection model
model_path = "Models/Trained/yolov11_keypoints_29/First/weights/best.pt"
model = load_keypoint_model(model_path)

# Process a single frame
frame = cv2.imread("soccer_field.jpg")
detections, keypoints = get_keypoint_detections(model, frame)

# Extract field information
from keypoint_detection import extract_field_corners, calculate_field_dimensions
corners = extract_field_corners(keypoints)
dimensions = calculate_field_dimensions(corners)

print(f"Detected {len(detections)} field(s)")
print(f"Field corners: {corners}")
print(f"Field dimensions: {dimensions}")
```

### Pipeline Integration

```python
from pipelines import KeypointPipeline

# Initialize pipeline
pipeline = KeypointPipeline(model_path)

# Process video with keypoint detection
pipeline.detect_in_video(
    video_path="input_match.mp4",
    output_path="output_with_keypoints.mp4", 
    frame_count=1000
)

# Real-time keypoint detection
pipeline.detect_realtime("live_stream.mp4")
```

### Advanced Usage with Tactical Analysis

```python
from pipelines import TacticalPipeline

# Complete tactical analysis with keypoint-based field mapping
tactical_pipeline = TacticalPipeline(
    keypoint_model_path=model_path,
    detection_model_path=detection_model_path
)

# Generate tactical overlay
tactical_pipeline.analyze_video(
    input_path="match.mp4",
    output_path="tactical_analysis.mp4",
    output_mode="overlay"  # Options: "overlay", "side-by-side", "tactical-only"
)
```

### Training Custom Models

```python
from keypoint_detection.training import YOLOKeypointTrainer, TrainingConfig

# Create custom training configuration
config = TrainingConfig(
    dataset_yaml_path="path/to/keypoint_dataset.yaml",
    model_name="custom_keypoint_model",
    epochs=100,
    img_size=640,
    batch_size=16
)

# Initialize trainer and start training
trainer = YOLOKeypointTrainer(config)
results = trainer.train_and_validate()
```

## 📁 GitHub Repository

**Repository**: [Soccer_Analysis](https://github.com/Adit-jain/Soccer_Analysis)

### Project Structure
```
Soccer_Analysis/
├── keypoint_detection/           # Core keypoint detection module
│   ├── detect_keypoints.py       # Core detection functions
│   ├── keypoint_constants.py     # Field specifications & keypoint mapping
│   └── training/                 # Training utilities
│       ├── config.py             # Training configuration
│       ├── trainer.py            # Modular trainer class
│       └── main.py               # Training entry point
├── pipelines/                    # Pipeline coordination
│   ├── keypoint_pipeline.py      # Keypoint detection pipeline
│   └── tactical_pipeline.py      # Tactical analysis with keypoints  
├── tactical_analysis/            # Field coordinate transformations
│   └── homography.py             # Homography calculations using keypoints
└── main.py                       # Multi-analysis entry point
```
---