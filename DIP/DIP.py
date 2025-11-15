from collections import deque
import cv2
import numpy as np

_RAIN_TEMPORAL_BUFFER = deque(maxlen=5)
_FOG_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_LOW_LIGHT_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def process_r1(frame):
    _RAIN_TEMPORAL_BUFFER.append(frame)
    stack = np.stack(_RAIN_TEMPORAL_BUFFER, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    return cv2.bilateralFilter(median, 9, 75, 75)

def process_r2(frame):
    base = cv2.bilateralFilter(frame, 9, 75, 75).astype(np.float32) / 255.0
    detail = frame.astype(np.float32) / 255.0 - base
    attenuated = base + detail * 0.25
    return np.clip(attenuated * 255.0, 0, 255).astype(np.uint8)

def _dark_channel(image, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(np.min(image, axis=2), kernel)

def _estimate_airlight(image, dark):
    flat_dark = dark.reshape(-1)
    flat_image = image.reshape(-1, 3)
    return flat_image[np.argmax(flat_dark)]

def _estimate_transmission(image, airlight, omega, size):
    norm = image / np.maximum(airlight, 1e-3)
    return 1.0 - omega * _dark_channel(norm, size)

def process_f1(frame, size=15, omega=0.95):
    normalized = frame.astype(np.float32) / 255.0
    dark = _dark_channel(normalized, size)
    airlight = _estimate_airlight(normalized, dark)
    transmission = _estimate_transmission(normalized, airlight, omega, size)
    transmission = np.clip(transmission, 0.1, 1.0)
    restored = (normalized - airlight) / transmission[:, :, None] + airlight
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)

def process_f2(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _FOG_CLAHE.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    for c in range(3):
        channel = enhanced[:, :, c]
        mean = channel.mean() + 1e-5
        enhanced[:, :, c] = channel * (128.0 / mean)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def process_l1(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = _LOW_LIGHT_CLAHE.apply(y)
    merged = cv2.merge((y, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def process_l2(frame, gamma=0.7, kernel=3):
    normalized = frame.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return cv2.medianBlur(corrected, kernel)

AVAILABLE_PROCESSES = {
    "R1": process_r1,
    "R2": process_r2,
    "F1": process_f1,
    "F2": process_f2,
    "L1": process_l1,
    "L2": process_l2,
}

def list_available_processes():
    return list(AVAILABLE_PROCESSES.keys())

def build_DIP_pipeline(process_name):
    if not process_name:
        return None
    if process_name not in AVAILABLE_PROCESSES:
        raise ValueError("Unknown DIP process: {}".format(process_name))
    return AVAILABLE_PROCESSES[process_name]

def run_DIP_pipeline(frame, process_fn):
    if process_fn is None:
        return frame
    return process_fn(frame)



