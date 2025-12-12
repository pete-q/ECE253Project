# ECE 253 Project: Video Enhancement for Object Detection in Adverse Conditions

## Overview

This project evaluates Digital Image Processing (DIP) enhancement techniques for improving YOLO-based vehicle detection and counting under adverse conditions (rain, fog, low light). We apply specialized DIP algorithms to enhance video quality, then compare detection performance on original vs. enhanced videos.

### Key Features

- **6 DIP Enhancement Algorithms**: Rain removal (R1/R2), Fog removal (F1/F2), Low-light enhancement (L1/L2)
- **YOLOv8 Object Detection**: Vehicle detection with built-in tracking ([model from Farzad Nekouee](https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation))
- **Vehicle Counting**: Line-crossing algorithm for traffic analysis
- **Comparison Analysis**: Original vs. enhanced video performance evaluation

## Repository Structure

```
253_Project/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── .gitignore                # Git ignore rules
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── dip/                  # DIP enhancement algorithms
│   │   ├── __init__.py
│   │   └── processors.py     # R1, R2, F1, F2, L1, L2 processors
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── video_utils.py    # Video I/O helpers
│
├── notebooks/                # Jupyter notebooks for analysis
│   ├── fog_yolo_comparison.ipynb
│   ├── lowlight_yolo_comparison.ipynb
│   └── rain_yolo_comparison.ipynb
│
├── scripts/                  # Standalone Python scripts
│   └── process_lowlight_yolo.py
│
├── data/                     # Data files
│   └── ground_truth.csv      # Ground truth vehicle counts
│
├── models/                   # Pretrained YOLO models
│   ├── best.pt               # Fine-tuned YOLOv8 model from Farzad Nekouee
│   └── README.md
│
├── videos/                   # Input videos (gitignored)
│   ├── Rain/
│   ├── Fog/
│   └── low_light/
│
└── output/                   # Generated output videos (gitignored)
```

## Quick Start

### Installation

```bash
# Clone and navigate to repository
git clone <repository-url>
cd 253_Project

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, GPU recommended for faster processing

### Data Setup

1. **Place videos** in `videos/Rain/`, `videos/Fog/`, or `videos/low_light/`
2. **YOLO model** is included at `models/best.pt` (credit: [Farzad Nekouee](https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation))
3. **Ground truth** counts in `data/ground_truth.csv`: LL1_down.mp4 (37), Rain1.mp4 (47), Fog1.mp4 (21)

### Running the Code

**Option 1: Jupyter Notebooks** (Recommended)
```bash
jupyter notebook
# Open: notebooks/lowlight_yolo_comparison.ipynb (or fog/rain variants)
# Run all cells to process videos and see results
```

**Note**: Notebooks include automatic path setup to find the `src` module. If you still get import errors, install the package:
```bash
pip install -e .
```

**Option 2: Python Script**
```bash
cd 253_Project  # Make sure you're in project root
python scripts/process_lowlight_yolo.py
```

**Output**: All processed videos saved to `output/` with YOLO detections, track IDs, and vehicle counts

## DIP Enhancement Algorithms

| Type | Method | Description |
|------|--------|-------------|
| **Rain Removal** | R1 | Temporal median filtering + bilateral filter |
| | R2 | Frequency-based rain attenuation |
| **Fog Removal** | F1 | Dark Channel Prior dehazing |
| | F2 | CLAHE + color correction in LAB space |
| **Low-Light** | L1 | CLAHE in YCrCb color space |
| | L2 | Gamma correction + median blur |

## Vehicle Counting System

Uses YOLOv8 with line-crossing detection:
- **Tracking**: Persistent IDs via YOLOv8 built-in tracker
- **Counting Line**: Horizontal line at 65% frame height
- **Detection**: Counts when vehicle center crosses within 55px of line

**Key Parameters** (adjustable in notebooks):
- `imgsz=640` (inference size: 480=fast, 640=balanced, 960=accurate)
- `conf=0.4` (confidence threshold)
- `device=None` (auto-detect GPU/CPU)

## Performance Tips

**Speed**: Use `imgsz=640`, `half=True` on GPU, `device='cuda'/'mps'`  
**Accuracy**: Use `imgsz=960`, lower `conf` threshold, fine-tuned model  
**Memory**: Clear notebook outputs, process one video at a time

## Results

Each notebook compares vehicle counts between original and enhanced videos:
- **Negative difference**: Under-counting (missed detections)
- **Positive difference**: Over-counting (false detections)
- **Zero difference**: Perfect match with ground truth

Watch output videos in `output/` folder to assess detection quality, tracking stability, and false positives/negatives.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Notebooks auto-fix this. If issue persists: run `pip install -e .` from project root |
| Videos won't display | Wait 30s, clear outputs, or open from `output/` folder |
| GPU out of memory | Reduce `imgsz=480`, use `device='cpu'`, or process shorter videos |
| Slow processing | Enable `half=True`, reduce `imgsz`, use smaller model (yolov8n.pt) |

## Extending the Project

To add new DIP algorithms, edit `src/dip/processors.py`:

```python
# 1. Add function
def process_x1(frame):
    enhanced_frame = ...  # Your enhancement
    return enhanced_frame

# 2. Register in AVAILABLE_PROCESSES
AVAILABLE_PROCESSES = {..., "X1": process_x1}

# 3. Use in notebooks
dip_function = build_DIP_pipeline("X1")
enhanced = run_DIP_pipeline(frame, dip_function)
```

## Credits

- **YOLO Model**: Fine-tuned YOLOv8 model from [Farzad Nekouee's Traffic Density Estimation project](https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation)
- **Project**: ECE 253 coursework on video enhancement for object detection

## License

Educational project for ECE 253 coursework.

---

**Version**: 1.0.0 | **Last Updated**: December 2025
