import pandas as pd
import numpy as np

def load_ground_truth(gt_path='ground_truth.csv'):
    """Loads total vehicle count ground truth."""
    try:
        df = pd.read_csv(gt_path)
        return dict(zip(df.video_name, df.total_count))
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
        return {}

def load_frame_ground_truth(gt_path='frame_ground_truth.csv'):
    """Loads frame-level vehicle count ground truth."""
    try:
        df = pd.read_csv(gt_path)
        # Returns dict: "video_name" -> {frame_num: count, ...}
        gt = {}
        for _, row in df.iterrows():
            vid = row['video_name']
            if vid not in gt:
                gt[vid] = {}
            gt[vid][row['frame_number']] = row['true_count']
        return gt
    except Exception as e:
        print(f"Warning: Could not load frame ground truth: {e}")
        return {}

def calculate_metrics(predicted_total, true_total):
    """Calculates Absolute Error and Relative Error."""
    if true_total == 0:
        return 0, 0.0  # Avoid division by zero if GT is not set
    
    ae = abs(predicted_total - true_total)
    re = ae / true_total
    return ae, re

def calculate_frame_metrics(predicted_frames, true_frames):
    """
    Calculates MAE and MAPE for frame-level counts.
    predicted_frames: dict {frame_num: count}
    true_frames: dict {frame_num: count}
    """
    ae_list = []
    ape_list = []
    
    for frame, true_count in true_frames.items():
        if frame in predicted_frames:
            pred_count = predicted_frames[frame]
            ae = abs(pred_count - true_count)
            ae_list.append(ae)
            
            if true_count > 0:
                ape_list.append(ae / true_count)
    
    if not ae_list:
        return 0, 0.0
        
    mae = np.mean(ae_list)
    mape = np.mean(ape_list) if ape_list else 0.0
    
    return mae, mape

