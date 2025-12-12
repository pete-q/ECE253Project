# Models Directory

## Included Model

- **`best.pt`** - Fine-tuned YOLOv8 model for vehicle detection from aerial views
  - **Source**: [Farzad Nekouee's YOLOv8 Traffic Density Estimation](https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation)
  - **Description**: YOLOv8 model fine-tuned on top-view vehicle detection dataset
  - **Classes**: Vehicles (cars, trucks, buses)
  - **Performance**: Optimized for traffic monitoring and vehicle counting

## Alternative Models

You can also use pretrained YOLOv8 models (automatically downloaded):
- `yolov8n.pt` - Nano (fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

## Note

Model files (`*.pt`) are gitignored to avoid committing large files.
