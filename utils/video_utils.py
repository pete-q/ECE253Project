import cv2
from typing import Tuple

def open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, width, height, fps

def create_writer(output_name: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create writer: {output_name}")
    return writer


