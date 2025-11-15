import os
import cv2
from ultralytics import YOLO
from utils.video_utils import open_video, create_writer
from DIP import build_DIP_pipeline, run_DIP_pipeline

def validate_video(
    video_path,
    dip_process=None,
    output_name="output.avi",
    model_path=None,
    polygons=None,
    lane_threshold=None
):

    # Resolve default model path relative to project root
    if model_path is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best.pt"))
    model = YOLO(model_path)

    # Build DIP process
    dip_function = build_DIP_pipeline(dip_process)

    # Open video
    cap, width, height, fps = open_video(video_path)
    out = create_writer(output_name, fps, width, height)

    # Polygon setup
    if polygons is not None:
        left_poly  = polygons["left"]
        right_poly = polygons["right"]

    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply DIP pipeline
        dip_frame = run_DIP_pipeline(frame.copy(), dip_function)

        # Run YOLO
        results = model.predict(dip_frame, imgsz=640, conf=0.4)
        processed_frame = results[0].plot(line_width=1)

        # Count vehicles
        boxes = results[0].boxes.xyxy.cpu().numpy()
        left_count, right_count = 0, 0

        for box in boxes:
            cx = (box[0] + box[2]) / 2
            if lane_threshold is not None:
                if cx < lane_threshold:
                    left_count += 1
                else:
                    right_count += 1

        # Draw polygons
        if polygons is not None:
            cv2.polylines(processed_frame, [left_poly], True, (0,255,0), 2)
            cv2.polylines(processed_frame, [right_poly], True, (255,0,0), 2)

        out.write(processed_frame)
        frame_idx += 1

    cap.release()
    out.release()
