#!/usr/bin/env python3
"""
YOWO Smart Home Dashboard - Professional Inference & Debug System

Features:
- Temporal smoothing (EMA on confidence and boxes)
- Detection tracking (IoU-based track continuity)
- Multi-panel debug visualization
- Performance profiling
- Confidence distribution analysis
- Export-ready detection logs

Usage:
    python dashboard.py --engine yowo_epoch_9_int8.engine
    python dashboard.py --engine yowo_epoch_9_int8.engine --debug  # Show debug panels
"""

import argparse
import time
import threading
import json
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from flask import Flask, Response, render_template_string, jsonify
# NOTE: pycuda and tensorrt are imported inside the worker thread
# to ensure CUDA context is bound to the correct thread

# ============================================================
# Load Smart Home Classes
# ============================================================
def load_classes():
    obj = [l.strip() for l in open('data/ActionGenome/annotations/object_classes.txt')]
    rel = [l.strip() for l in open('data/ActionGenome/annotations/relationship_classes.txt')]
    with open('config/smart_home_final.json', 'r') as f:
        cfg = json.load(f)
    act = cfg['action_names']
    return obj, act, rel

OBJ_CLASSES, ACT_CLASSES, REL_CLASSES = load_classes()

# Output indices for smart home model: [conf(1), obj(36), act(42), rel(26), box(4)] = 109
OBJ_START, OBJ_END = 1, 37
ACT_START, ACT_END = 37, 79
REL_START, REL_END = 79, 105
BOX_START, BOX_END = 105, 109


# ============================================================
# Detection Data Structures
# ============================================================
@dataclass
class Detection:
    """Single detection with all attributes."""
    box: np.ndarray          # [x1, y1, x2, y2] normalized
    conf: float              # Objectness confidence
    obj_idx: int             # Object class index
    obj_prob: float          # Object class probability
    obj_name: str            # Object class name
    actions: List[Tuple[str, float]]    # [(action_name, prob), ...]
    relations: List[Tuple[str, float]]  # [(relation_name, prob), ...]
    is_person: bool
    track_id: int = -1       # Assigned by tracker
    age: int = 0             # Frames since first seen
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)
    
    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


@dataclass
class Track:
    """Tracked detection over time with smoothing."""
    track_id: int
    detections: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Smoothed values (EMA)
    smooth_box: np.ndarray = None
    smooth_conf: float = 0.0
    smooth_obj_probs: np.ndarray = None
    smooth_act_probs: np.ndarray = None
    smooth_rel_probs: np.ndarray = None
    
    # Track state
    age: int = 0
    hits: int = 0
    misses: int = 0
    is_confirmed: bool = False
    
    # EMA coefficients (higher = more responsive to new detections)
    BOX_EMA = 0.5      
    CONF_EMA = 0.6
    PROB_EMA = 0.5     # Balanced smoothing - not too aggressive
    
    def update(self, det: Detection, raw_probs: dict):
        """Update track with new detection."""
        self.detections.append(det)
        self.age += 1
        self.hits += 1
        self.misses = 0
        
        if self.smooth_box is None:
            # First detection - initialize
            self.smooth_box = det.box.copy()
            self.smooth_conf = det.conf
            self.smooth_obj_probs = raw_probs['obj'].copy()
            self.smooth_act_probs = raw_probs['act'].copy()
            self.smooth_rel_probs = raw_probs['rel'].copy()
        else:
            # EMA update
            self.smooth_box = self.BOX_EMA * det.box + (1 - self.BOX_EMA) * self.smooth_box
            self.smooth_conf = self.CONF_EMA * det.conf + (1 - self.CONF_EMA) * self.smooth_conf
            self.smooth_obj_probs = self.PROB_EMA * raw_probs['obj'] + (1 - self.PROB_EMA) * self.smooth_obj_probs
            self.smooth_act_probs = self.PROB_EMA * raw_probs['act'] + (1 - self.PROB_EMA) * self.smooth_act_probs
            self.smooth_rel_probs = self.PROB_EMA * raw_probs['rel'] + (1 - self.PROB_EMA) * self.smooth_rel_probs
        
        # Confirm track after 3 hits
        if self.hits >= 3:
            self.is_confirmed = True
    
    def predict(self):
        """Mark frame without detection (track continues with prediction)."""
        self.age += 1
        self.misses += 1
        # Could add motion prediction here if needed
    
    def get_smoothed_detection(self) -> Optional[Detection]:
        """Get smoothed detection for display."""
        if not self.is_confirmed or self.misses > 5:
            return None
        
        # Get smoothed class predictions
        obj_idx = np.argmax(self.smooth_obj_probs)
        obj_prob = self.smooth_obj_probs[obj_idx]
        
        # Get top actions (persons only — objects don't have actions)
        actions = []
        if obj_idx == 0:  # person
            for idx in np.argsort(-self.smooth_act_probs)[:5]:
                if self.smooth_act_probs[idx] > 0.05:  # Low threshold for webcam domain
                    actions.append((ACT_CLASSES[idx], float(self.smooth_act_probs[idx])))
        
        # Get top relations (ALL detections — objects have per-object relations)
        relations = []
        for idx in np.argsort(-self.smooth_rel_probs)[:3]:
            if self.smooth_rel_probs[idx] > 0.15:
                relations.append((REL_CLASSES[idx], float(self.smooth_rel_probs[idx])))
        
        return Detection(
            box=self.smooth_box,
            conf=self.smooth_conf,
            obj_idx=obj_idx,
            obj_prob=float(obj_prob),
            obj_name=OBJ_CLASSES[obj_idx],
            actions=actions,
            relations=relations,
            is_person=bool(obj_idx == 0),
            track_id=self.track_id,
            age=self.age
        )


# ============================================================
# IoU-based Tracker
# ============================================================
class Tracker:
    """Simple IoU-based multi-object tracker with temporal smoothing."""
    
    def __init__(self, iou_threshold=0.15, max_age=10):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age
    
    def iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)
    
    def update(self, detections: List[Detection], raw_probs_list: List[dict]) -> List[Detection]:
        """Update tracks with new detections, return smoothed detections."""
        
        # Predict all tracks (increment age)
        for track in self.tracks.values():
            track.predict()
        
        # Match detections to existing tracks using IoU
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        if detections and self.tracks:
            # Build cost matrix
            cost_matrix = np.zeros((len(detections), len(unmatched_tracks)))
            for i, det in enumerate(detections):
                for j, tid in enumerate(unmatched_tracks):
                    track = self.tracks[tid]
                    if track.smooth_box is not None:
                        cost_matrix[i, j] = self.iou(det.box, track.smooth_box)
            
            # Greedy matching (could use Hungarian algorithm for better results)
            matched = []
            while cost_matrix.size > 0 and cost_matrix.max() > self.iou_threshold:
                i, j = np.unravel_index(cost_matrix.argmax(), cost_matrix.shape)
                matched.append((unmatched_dets[i], unmatched_tracks[j]))
                cost_matrix[i, :] = 0
                cost_matrix[:, j] = 0
            
            for det_idx, track_id in matched:
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_id)
        
        # Update matched tracks
        for det_idx, track_id in [(d, t) for d, t in zip(range(len(detections)), list(self.tracks.keys())) 
                                   if d not in unmatched_dets and t not in unmatched_tracks]:
            pass  # Already handled above
        
        # Properly update matched tracks
        matched_pairs = []
        if detections and self.tracks:
            cost_matrix = np.zeros((len(detections), len(self.tracks)))
            track_ids = list(self.tracks.keys())
            for i, det in enumerate(detections):
                for j, tid in enumerate(track_ids):
                    track = self.tracks[tid]
                    if track.smooth_box is not None:
                        cost_matrix[i, j] = self.iou(det.box, track.smooth_box)
            
            used_dets = set()
            used_tracks = set()
            while True:
                if cost_matrix.size == 0:
                    break
                max_iou = cost_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(cost_matrix.argmax(), cost_matrix.shape)
                matched_pairs.append((i, track_ids[j]))
                used_dets.add(i)
                used_tracks.add(track_ids[j])
                cost_matrix[i, :] = 0
                cost_matrix[:, j] = 0
            
            unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
            unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
        
        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            self.tracks[track_id].update(detections[det_idx], raw_probs_list[det_idx])
            detections[det_idx].track_id = track_id
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(track_id=self.next_id)
            new_track.update(det, raw_probs_list[det_idx])
            self.tracks[self.next_id] = new_track
            det.track_id = self.next_id
            self.next_id += 1
        
        # Remove dead tracks
        dead_tracks = [tid for tid, track in self.tracks.items() if track.misses > self.max_age]
        for tid in dead_tracks:
            del self.tracks[tid]
        
        # Return smoothed detections from confirmed tracks
        smoothed = []
        for track in self.tracks.values():
            det = track.get_smoothed_detection()
            if det is not None:
                smoothed.append(det)
        
        return smoothed


# ============================================================
# TensorRT Engine (initialized inside worker thread)
# ============================================================
class Engine:
    def __init__(self, path, cuda_module):
        import tensorrt as trt
        self.cuda = cuda_module
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(path, 'rb') as f:
            engine_data = f.read()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {path}. Check GPU memory.")
        
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.clip_length = self.in_shape[2]
        
        self.d_in = self.cuda.mem_alloc(int(np.prod(self.in_shape) * 4))
        self.d_out = self.cuda.mem_alloc(int(np.prod(self.out_shape) * 4))
        self.h_in = self.cuda.pagelocked_empty(self.in_shape, dtype=np.float32)
        self.h_out = self.cuda.pagelocked_empty(self.out_shape, dtype=np.float32)
        self.stream = self.cuda.Stream()
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        
        print(f"Engine loaded: {path}")
        print(f"  Input: {self.in_shape}, Output: {self.out_shape}")

    def run(self, data):
        np.copyto(self.h_in, data)
        self.cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self.h_out.copy()


# ============================================================
# Post-processing
# ============================================================
def nms(boxes, scores, thresh):
    """Standard NMS on boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= thresh]
    return keep


def class_aware_nms(boxes, scores, class_ids, act_scores, thresh):
    """NMS applied separately per class. For persons, use tighter NMS + action score."""
    if len(boxes) == 0:
        return []
    
    keep = []
    unique_classes = np.unique(class_ids)
    
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_indices = np.where(cls_mask)[0]
        
        if len(cls_indices) == 0:
            continue
        
        cls_boxes = boxes[cls_mask]
        
        # For persons (class 0): use tighter NMS threshold + combined score
        # This aggressively merges overlapping person boxes
        if cls_id == 0:
            cls_conf = scores[cls_mask]
            cls_act = act_scores[cls_mask]
            # Combined score: prioritize detections with good action predictions
            cls_scores_combined = cls_conf * (1.0 + cls_act * 2.0)
            # Use very tight NMS for persons to merge overlapping boxes
            person_nms_thresh = 0.15
            cls_keep = nms(cls_boxes, cls_scores_combined, person_nms_thresh)
        else:
            cls_scores_combined = scores[cls_mask]
            cls_keep = nms(cls_boxes, cls_scores_combined, thresh)
        
        keep.extend(cls_indices[cls_keep])
    
    return keep


def postprocess(out, conf_th=0.08, nms_th=0.45, min_obj_prob=0.15, min_person_prob=0.10, img_size=224):
    """Post-process model output, return detections and raw probabilities.
    
    Args:
        conf_th: Minimum objectness confidence (lowered for better recall)
        nms_th: NMS IoU threshold
        min_obj_prob: Minimum object class probability for non-person objects
        min_person_prob: Minimum probability for person detection (low for webcam domain)
        img_size: Model input size for box normalization
    """
    if out.ndim == 3:
        out = out.squeeze(0)
    
    confs = out[:, 0]
    mask = confs > conf_th
    if not np.any(mask):
        return [], []
    
    filt = out[mask]
    boxes = filt[:, BOX_START:BOX_END] / img_size
    
    # Get class IDs for class-aware NMS
    obj_probs = filt[:, OBJ_START:OBJ_END]
    class_ids = np.argmax(obj_probs, axis=1)
    
    # Get max action probability for each detection (for action-aware NMS)
    act_probs_all = filt[:, ACT_START:ACT_END]
    act_max_scores = np.max(act_probs_all, axis=1)
    
    # Class-aware NMS: for persons, use combined conf+action score
    keep = class_aware_nms(boxes, filt[:, 0], class_ids, act_max_scores, nms_th)
    
    detections = []
    raw_probs_list = []
    
    for i in keep:
        d = filt[i]
        # Normalize and CLIP boxes to [0, 1] range
        box = np.clip(d[BOX_START:BOX_END] / img_size, 0.0, 1.0)
        
        obj_probs = d[OBJ_START:OBJ_END]
        obj_idx = np.argmax(obj_probs)
        obj_prob = obj_probs[obj_idx]
        
        # Use lower threshold for person detection (index 0)
        is_person = bool(obj_idx == 0)
        threshold = min_person_prob if is_person else min_obj_prob
        
        if obj_prob < threshold:
            continue
        
        # Get actions (persons only) and relations (ALL detections)
        actions = []
        relations = []
        act_probs = d[ACT_START:ACT_END]
        rel_probs = d[REL_START:REL_END]
        
        # Actions only for persons (objects don't perform actions)
        if is_person:
            for idx in np.argsort(-act_probs)[:5]:
                if act_probs[idx] > 0.05:  # Low threshold for webcam domain
                    actions.append((ACT_CLASSES[idx], float(act_probs[idx])))
        
        # Relations for ALL detections (each object has its own relation to person)
        for idx in np.argsort(-rel_probs)[:3]:
            if rel_probs[idx] > 0.15:
                relations.append((REL_CLASSES[idx], float(rel_probs[idx])))
        
        det = Detection(
            box=box,
            conf=float(d[0]),
            obj_idx=obj_idx,
            obj_prob=float(obj_prob),
            obj_name=OBJ_CLASSES[obj_idx],
            actions=actions,
            relations=relations,
            is_person=is_person
        )
        detections.append(det)
        
        # Store raw probabilities for tracker
        raw_probs_list.append({
            'obj': obj_probs.copy(),
            'act': act_probs.copy(),
            'rel': rel_probs.copy()
        })
    
    return detections, raw_probs_list


# ============================================================
# Performance Profiler
# ============================================================
class Profiler:
    def __init__(self, window_size=100):
        self.times = defaultdict(lambda: deque(maxlen=window_size))
        self.counts = defaultdict(int)
    
    def log(self, name: str, duration: float):
        self.times[name].append(duration)
        self.counts[name] += 1
    
    def get_stats(self) -> dict:
        stats = {}
        for name, times in self.times.items():
            if times:
                arr = np.array(times)
                stats[name] = {
                    'mean_ms': float(np.mean(arr) * 1000),
                    'std_ms': float(np.std(arr) * 1000),
                    'min_ms': float(np.min(arr) * 1000),
                    'max_ms': float(np.max(arr) * 1000),
                    'count': self.counts[name]
                }
        return stats


# ============================================================
# Detection Logger
# ============================================================
class DetectionLogger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'detections_{int(time.time())}.jsonl')
        self.frame_count = 0
    
    def log(self, detections: List[Detection], timestamp: float):
        self.frame_count += 1
        entry = {
            'frame': self.frame_count,
            'timestamp': timestamp,
            'detections': [
                {
                    'track_id': d.track_id,
                    'box': d.box.tolist(),
                    'conf': d.conf,
                    'obj_name': d.obj_name,
                    'obj_prob': d.obj_prob,
                    'actions': d.actions,
                    'relations': d.relations,
                    'age': d.age
                }
                for d in detections
            ]
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


# ============================================================
# Global State
# ============================================================
app = Flask(__name__)

class State:
    frame = None
    frame_buffer = deque(maxlen=32)
    clip_length = 32
    raw_detections = []      # Before tracking
    tracked_detections = []  # After tracking with smoothing
    inf_fps = 0.0
    total_fps = 0.0
    running = True
    debug_mode = False
    
    # Debug info
    conf_histogram = np.zeros(20)
    detection_counts = deque(maxlen=100)
    track_ages = {}
    
    # Profiler
    profiler = Profiler()

S = State()
state_lock = threading.Lock()
tracker = Tracker(iou_threshold=0.5, max_age=8)
det_logger = None


# ============================================================
# Main Inference Loop
# ============================================================
def main_loop(cam_id, engine_path, enable_logging=False):
    global det_logger
    
    # Initialize CUDA in this thread (CRITICAL: must be done before any CUDA ops)
    import pycuda.driver as cuda
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_ctx = cuda_device.make_context()
    print(f"CUDA initialized in worker thread (device: {cuda_device.name()})")
    
    cap = None
    try:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            S.running = False
            return
        
        engine = Engine(engine_path, cuda)
        S.clip_length = engine.clip_length
        S.frame_buffer = deque(maxlen=S.clip_length)
        
        if enable_logging:
            det_logger = DetectionLogger()
            print(f"Logging detections to: {det_logger.log_file}")
        
        img_size = engine.in_shape[3]  # H from (1, 3, T, H, W)
        clip_buffer = np.zeros((1, 3, S.clip_length, img_size, img_size), dtype=np.float32)
        fps_buf = deque(maxlen=30)
        inf_times = deque(maxlen=30)
        
        print(f"Starting inference (clip_length={S.clip_length})...")
        
        while S.running:
            t_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            with state_lock:
                S.frame = frame.copy()
            
            # Preprocess
            t0 = time.time()
            small = cv2.resize(frame, (img_size, img_size))
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            S.frame_buffer.append(small)
            S.profiler.log('preprocess', time.time() - t0)
            
            if len(S.frame_buffer) >= S.clip_length:
                # Build clip
                t0 = time.time()
                for i, f in enumerate(S.frame_buffer):
                    clip_buffer[0, :, i, :, :] = f.transpose(2, 0, 1)
                S.profiler.log('clip_build', time.time() - t0)
                
                # Inference
                t0 = time.time()
                out = engine.run(clip_buffer)
                inf_time = time.time() - t0
                inf_times.append(inf_time)
                S.profiler.log('inference', inf_time)
                
                # Post-process
                t0 = time.time()
                raw_dets, raw_probs = postprocess(out, img_size=img_size)
                S.profiler.log('postprocess', time.time() - t0)
                
                # Track and smooth
                t0 = time.time()
                tracked_dets = tracker.update(raw_dets, raw_probs)
                S.profiler.log('tracking', time.time() - t0)
                
                # Update debug info
                if S.debug_mode:
                    confs = out[:, 0] if out.ndim == 2 else out[0, :, 0]
                    hist, _ = np.histogram(confs, bins=20, range=(0, 1))
                    S.conf_histogram = hist
                    S.detection_counts.append(len(tracked_dets))
                    S.track_ages = {t.track_id: t.age for t in tracker.tracks.values()}
                
                # Log
                if det_logger:
                    det_logger.log(tracked_dets, time.time())
                
                with state_lock:
                    S.raw_detections = raw_dets
                    S.tracked_detections = tracked_dets
                    S.inf_fps = 1.0 / np.mean(inf_times) if inf_times else 0
            
            # Total FPS
            total_time = time.time() - t_start
            fps_buf.append(1.0 / total_time)
            S.total_fps = np.mean(fps_buf)
    
    finally:
        if cap is not None:
            cap.release()
        cuda_ctx.pop()
        cuda_ctx.detach()
        print("CUDA context cleaned up")


# ============================================================
# Web Dashboard
# ============================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>YOWO Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            background: #0d1117; 
            color: #c9d1d9; 
            font-family: 'Segoe UI', system-ui, sans-serif;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #30363d;
        }
        h1 { color: #58a6ff; font-size: 24px; }
        .stats { display: flex; gap: 20px; }
        .stat { 
            background: #161b22; 
            padding: 10px 20px; 
            border-radius: 6px;
            border: 1px solid #30363d;
        }
        .stat-value { font-size: 24px; color: #7ee787; font-weight: bold; }
        .stat-label { font-size: 12px; color: #8b949e; }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        
        .video-panel {
            background: #161b22;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #30363d;
        }
        .video-panel img {
            width: 100%;
            display: block;
        }
        
        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .panel {
            background: #161b22;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #30363d;
        }
        .panel-title {
            font-size: 14px;
            color: #8b949e;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .detection-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .detection-item {
            padding: 10px;
            margin-bottom: 8px;
            background: #0d1117;
            border-radius: 6px;
            border-left: 3px solid #238636;
        }
        .detection-item.object {
            border-left-color: #f78166;
        }
        .det-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .det-name { font-weight: bold; color: #f0f6fc; }
        .det-conf { color: #7ee787; }
        .det-actions { font-size: 12px; color: #8b949e; }
        
        .perf-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .perf-item {
            background: #0d1117;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
        }
        .perf-value { font-size: 18px; color: #58a6ff; }
        .perf-label { font-size: 11px; color: #8b949e; }
        
        .track-info {
            font-size: 12px;
            color: #8b949e;
        }
        .track-badge {
            display: inline-block;
            padding: 2px 6px;
            background: #238636;
            border-radius: 10px;
            font-size: 10px;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>YOWO Smart Home Dashboard</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="inf-fps">--</div>
                <div class="stat-label">Inference FPS</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="total-fps">--</div>
                <div class="stat-label">Total FPS</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="det-count">--</div>
                <div class="stat-label">Active Tracks</div>
            </div>
        </div>
    </div>
    
    <div class="main-grid">
        <div class="video-panel">
            <img id="stream" src="/video_feed">
        </div>
        
        <div class="side-panel">
            <div class="panel">
                <div class="panel-title">Active Detections</div>
                <div class="detection-list" id="det-list"></div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Performance</div>
                <div class="perf-grid" id="perf-grid"></div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Track Info</div>
                <div class="track-info" id="track-info"></div>
            </div>
        </div>
    </div>
    
    <script>
        function updateStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('inf-fps').textContent = data.inf_fps.toFixed(1);
                    document.getElementById('total-fps').textContent = data.total_fps.toFixed(1);
                    document.getElementById('det-count').textContent = data.det_count;
                    
                    // Update detection list
                    let html = '';
                    data.detections.forEach(d => {
                        const cls = d.is_person ? '' : 'object';
                        html += `<div class="detection-item ${cls}">
                            <div class="det-header">
                                <span class="det-name">
                                    <span class="track-badge">T${d.track_id}</span>
                                    ${d.name}
                                </span>
                                <span class="det-conf">${(d.conf * 100).toFixed(0)}%</span>
                            </div>
                            <div class="det-actions">${d.actions.join(', ') || 'No actions'}</div>
                        </div>`;
                    });
                    document.getElementById('det-list').innerHTML = html || '<div style="color:#8b949e">No detections</div>';
                    
                    // Update performance
                    let perfHtml = '';
                    for (const [name, stats] of Object.entries(data.perf)) {
                        perfHtml += `<div class="perf-item">
                            <div class="perf-value">${stats.mean_ms.toFixed(1)}</div>
                            <div class="perf-label">${name} (ms)</div>
                        </div>`;
                    }
                    document.getElementById('perf-grid').innerHTML = perfHtml;
                    
                    // Update track info
                    let trackHtml = `Active tracks: ${Object.keys(data.tracks).length}<br>`;
                    for (const [tid, age] of Object.entries(data.tracks)) {
                        trackHtml += `<span class="track-badge">T${tid}: ${age}f</span> `;
                    }
                    document.getElementById('track-info').innerHTML = trackHtml;
                });
        }
        
        setInterval(updateStats, 200);
        
        // Handle stream errors
        document.getElementById('stream').onerror = function() {
            setTimeout(() => { this.src = '/video_feed?' + Date.now(); }, 500);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/stats')
def api_stats():
    with state_lock:
        dets = S.tracked_detections[:]
        inf_fps = S.inf_fps
        total_fps = S.total_fps
    
    return jsonify({
        'inf_fps': inf_fps,
        'total_fps': total_fps,
        'det_count': len(dets),
        'detections': [
            {
                'track_id': d.track_id,
                'name': d.obj_name,
                'conf': d.conf,
                'is_person': bool(d.is_person),
                'actions': [f"{a[0]}: {a[1]:.0%}" for a in d.actions[:3]]
            }
            for d in dets
        ],
        'perf': S.profiler.get_stats(),
        'tracks': S.track_ages
    })


def generate_frames():
    while S.running:
        with state_lock:
            if S.frame is None:
                time.sleep(0.01)
                continue
            frame = S.frame.copy()
            dets = S.tracked_detections[:]
        
        h, w = frame.shape[:2]
        
        for d in dets:
            x1, y1, x2, y2 = (d.box * [w, h, w, h]).astype(int)
            
            if d.is_person:
                color = (0, 255, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Track ID and confidence
                label = f"T{d.track_id} PERSON {d.obj_prob:.0%}"
                cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Actions
                y_off = y1 + 18
                for act, score in d.actions[:3]:
                    act_short = act[:25] + "..." if len(act) > 25 else act
                    cv2.putText(frame, f"> {act_short}: {score:.0%}", (x1+5, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    y_off += 15
            else:
                color = (255, 165, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"T{d.track_id} {d.obj_name} {d.obj_prob:.0%}", 
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # FPS overlay
        cv2.putText(frame, f"INF: {S.inf_fps:.1f} | TOT: {S.total_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.016)


@app.route('/video_feed')
def video_feed():
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/favicon.ico')
def favicon():
    return '', 204


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOWO Smart Home Dashboard')
    parser.add_argument('--engine', default='yowo_v2_resnext_yolo26m_multitask_epoch_10_int8.engine', help='TensorRT engine path')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log', action='store_true', help='Enable detection logging')
    args = parser.parse_args()
    
    S.debug_mode = args.debug
    
    main_thread = threading.Thread(target=main_loop, args=(args.camera, args.engine, args.log), daemon=True)
    main_thread.start()
    
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("YOWO Smart Home Dashboard")
    print("=" * 60)
    print(f"Engine: {args.engine}")
    print(f"Debug mode: {args.debug}")
    print(f"Logging: {args.log}")
    print(f"\nOpen: http://localhost:{args.port}")
    print("=" * 60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        S.running = False
