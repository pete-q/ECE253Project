import os
import cv2
import numpy as np
from ultralytics import YOLO
from DIP import build_DIP_pipeline, run_DIP_pipeline


class CentroidTracker:
    """
    Extremely lightweight centroid-based tracker used to keep IDs consistent
    when running YOLO in predict mode (no built-in tracking IDs).
    """

    def __init__(self, max_distance=75, max_frames_missing=12):
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.next_id = 0
        self.objects = {}      # object_id -> centroid np.array([x, y])
        self.missing = {}      # object_id -> frames since last seen

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.missing[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.missing.pop(object_id, None)

    def update(self, centroids):
        centroids = np.asarray(centroids, dtype=np.float32)

        if len(centroids) == 0:
            # Increment missing counter for all current objects
            for object_id in list(self.missing.keys()):
                self.missing[object_id] += 1
                if self.missing[object_id] > self.max_frames_missing:
                    self._deregister(object_id)
            return dict(self.objects)

        if len(self.objects) == 0:
            for centroid in centroids:
                self._register(centroid)
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[object_id] for object_id in object_ids])

        # Compute pairwise distances (existing objects x new detections)
        distances = np.linalg.norm(object_centroids[:, None, :] - centroids[None, :, :], axis=2)

        rows = distances.min(axis=1).argsort()
        assigned_cols = set()
        assigned_rows = set()

        for row in rows:
            col = distances[row].argmin()
            if col in assigned_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = centroids[col]
            self.missing[object_id] = 0

            assigned_cols.add(col)
            assigned_rows.add(row)

        # Register unmatched detections
        for col, centroid in enumerate(centroids):
            if col not in assigned_cols:
                self._register(centroid)

        # Increase missing count for unmatched existing objects
        for row, object_id in enumerate(object_ids):
            if row not in assigned_rows:
                self.missing[object_id] += 1
                if self.missing[object_id] > self.max_frames_missing:
                    self._deregister(object_id)

        return dict(self.objects)

def process_video_with_dip(input_video_path, output_video_path, dip_process_name):
    """
    Process a video through DIP pipeline and save the result.
    """
    print(f"Processing video with DIP {dip_process_name}...")
    
    # Build DIP pipeline
    dip_function = build_DIP_pipeline(dip_process_name)
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video info: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Create output writer with mp4v codec for smoother output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create writer: {output_video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply DIP processing
        processed_frame = run_DIP_pipeline(frame, dip_function)
        
        # Write processed frame
        out.write(processed_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    print(f"DIP processing complete! Saved to {output_video_path}")
    print(f"Total frames processed: {frame_count}/{total_frames}")


def run_yolo_on_video(input_video_path, output_video_path, model_path, model=None, imgsz=640, device=None, half=False):
    """
    Run YOLO detection on a video and save the result with bounding boxes.
    Counts vehicles that cross a horizontal line positioned in the lower third of the frame.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to save output video
        model_path: Path to YOLO model file
        model: Pre-loaded YOLO model (optional, to avoid reloading)
        imgsz: Image size for inference (default 640, smaller = faster but less accurate)
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        half: Use FP16 half precision (faster on GPU, default False)
    """
    print(f"\nRunning YOLO detection on {os.path.basename(input_video_path)}...")
    
    # Load YOLO model (reuse if provided)
    if model is None:
        model = YOLO(model_path)
    
    # Auto-detect device if not specified
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU
        else:
            device = 'cpu'
    
    print(f"  Using device: {device}, imgsz: {imgsz}, half: {half}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video info: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Create output writer with mp4v codec for smoother output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create writer: {output_video_path}")
    
    # Vehicle counting setup - horizontal line placed lower in the frame
    counting_line_y = int(height * 0.65)  # ~2/3 down from the top
    counted_ids = set()  # Track which vehicle IDs have been counted
    vehicle_count = 0
    threshold = 55  # Pixels threshold for line crossing detection

    tracker = CentroidTracker(max_distance=80, max_frames_missing=15)
    
    frame_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO prediction (optimized settings)
        results = model.predict(
            frame,
            imgsz=imgsz,  # Reduced from 960 for speed
            conf=0.2,
            iou=0.4,
            verbose=False,
            max_det=50,
            device=device,
            half=half,  # FP16 for faster inference on GPU
        )
        
        # Get frame with bounding boxes drawn
        annotated_frame = results[0].plot(line_width=2, conf=True)
        
        # Count detections and check for line crossing
        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0
        total_detections += num_detections

        centroids = []
        if boxes is not None and num_detections > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in xyxy:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centroids.append([cx, cy])

        tracked_objects = tracker.update(centroids)
        
        # Check if vehicles cross the counting line using tracker-provided IDs
        for track_id, centroid in tracked_objects.items():
            center_y = centroid[1]
            if abs(center_y - counting_line_y) < threshold and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            # Visualize tracker IDs
            cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 255), -1)
            cv2.putText(
                annotated_frame,
                f"ID {track_id}",
                (int(centroid[0]) - 20, int(centroid[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        
        # Draw horizontal counting line
        cv2.line(annotated_frame, (0, counting_line_y), (width, counting_line_y), (0, 255, 0), 3)
        
        # Display vehicle count on frame
        count_text = f"Vehicle Count: {vehicle_count}"
        cv2.putText(annotated_frame, count_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Write annotated frame
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    print(f"YOLO detection complete! Saved to {output_video_path}")
    print(f"Total frames: {frame_count}/{total_frames}, Total detections: {total_detections}")
    print(f"Vehicles counted crossing line: {vehicle_count}")
    
    return vehicle_count


def main():
    # Define paths
    project_root = "/Users/pete/Desktop/253_Project"
    input_video = os.path.join(project_root, "videos/low_light/LL1.mp4")
    model_path = os.path.join(project_root, "model/best.pt")
    
    # Create output directory
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths (using mp4 directly for better smoothness)
    dip_enhanced_mp4 = os.path.join(output_dir, "LL1_L1_enhanced.mp4")
    original_yolo_mp4 = os.path.join(output_dir, "LL1_original_yolo.mp4")
    enhanced_yolo_mp4 = os.path.join(output_dir, "LL1_enhanced_yolo.mp4")
    
    # Performance optimization settings
    # Reduce imgsz for speed: 640 is ~2.25x faster than 960, 480 is ~4x faster
    # Trade-off: Lower imgsz = faster but potentially less accurate on small objects
    imgsz = 640  # Change to 480 for even faster processing, or 960 for best accuracy
    device = None  # Auto-detect (cuda/mps/cpu)
    half = False  # Set to True for GPU to use FP16 (faster)
    
    print("=" * 70)
    print("LOW LIGHT VIDEO PROCESSING WITH DIP L1 AND YOLO DETECTION")
    print("=" * 70)
    print(f"\nInput video: {input_video}")
    print(f"YOLO model: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"\nPerformance settings: imgsz={imgsz}, device=auto, half={half}")
    
    # Load model once and reuse (faster than reloading for each video)
    print("\nLoading YOLO model...")
    model = YOLO(model_path)
    
    # Step 1: Process video with DIP L1 (Low Light Enhancement)
    print("\n" + "=" * 70)
    print("STEP 1: APPLYING DIP L1 (LOW LIGHT ENHANCEMENT)")
    print("=" * 70)
    process_video_with_dip(input_video, dip_enhanced_mp4, "L1")
    
    # Step 2: Run YOLO on original video (reuse model)
    print("\n" + "=" * 70)
    print("STEP 2: RUNNING YOLO DETECTION ON ORIGINAL VIDEO")
    print("=" * 70)
    original_count = run_yolo_on_video(input_video, original_yolo_mp4, model_path, 
                                       model=model, imgsz=imgsz, device=device, half=half)
    
    # Step 3: Run YOLO on DIP-enhanced video (reuse model)
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING YOLO DETECTION ON DIP-ENHANCED VIDEO")
    print("=" * 70)
    enhanced_count = run_yolo_on_video(dip_enhanced_mp4, enhanced_yolo_mp4, model_path,
                                       model=model, imgsz=imgsz, device=device, half=half)
    
    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  1. DIP L1 Enhanced Video:        {dip_enhanced_mp4}")
    print(f"  2. Original Video + YOLO:        {original_yolo_mp4}")
    print(f"  3. Enhanced Video + YOLO:        {enhanced_yolo_mp4}")
    print("\nVehicle Count Results:")
    print(f"  Original Video:                  {original_count} vehicles")
    print(f"  DIP-Enhanced Video:              {enhanced_count} vehicles")
    print(f"  Difference:                      {enhanced_count - original_count} vehicles")
    print("\nKey Improvements for Smooth Video Output:")
    print("  - Centroid tracker built on top of YOLO predictions for ID stability")
    print("  - MP4v codec for better quality")
    print("  - All frames processed without skipping")
    print("  - Original FPS preserved")
    print("  - Vehicle counting using line crossing detection")
    print("\nYou can now view these videos to compare:")
    print("  - Original vs DIP-enhanced quality")
    print("  - YOLO detection performance on original vs enhanced video")
    print("  - Vehicle counting displayed on each frame")
    print("=" * 70)


if __name__ == "__main__":
    main()

