#!/usr/bin/env python3
"""
YOWO TensorRT Demo - Beautiful Dark UI with Prediction History

Features:
- Dark theme with high contrast colors
- Prediction history (last 2 seconds)  
- True inference FPS measurement
- Action timeline visualization
- Clean, modern design
"""

import argparse
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import pycuda.driver as cuda
import tensorrt as trt

# ============================================================================
# CUDA Context
# ============================================================================
cuda.init()
cuda_device = cuda.Device(0)
cuda_ctx = cuda_device.make_context()

# ============================================================================
# Colors - High Contrast for Dark Background
# ============================================================================
COLORS = {
    'person': (0, 255, 128),      # Bright green
    'object': (255, 165, 0),      # Orange
    'action': (0, 255, 255),      # Cyan
    'relation': (255, 100, 255),  # Pink/magenta
    'text': (255, 255, 255),      # White
    'bg_dark': (20, 20, 30),      # Dark blue-black
    'bg_panel': (40, 40, 50),     # Slightly lighter panel
    'highlight': (100, 200, 255), # Light blue highlight
}

# ============================================================================
# Load Class Labels
# ============================================================================
def load_classes():
    obj = [l.strip() for l in open('data/ActionGenome/annotations/object_classes.txt')]
    rel = [l.strip() for l in open('data/ActionGenome/annotations/relationship_classes.txt')]
    act = []
    for l in open('data/ActionGenome/annotations/Charades_v1_classes.txt'):
        parts = l.strip().split(' ', 1)
        if len(parts) > 1: act.append(parts[1])
    return obj, act, rel

OBJ_CLASSES, ACT_CLASSES, REL_CLASSES = load_classes()

@dataclass
class Det:
    box: np.ndarray
    conf: float
    name: str
    actions: List[Tuple[str, float]]
    relations: List[Tuple[str, float]]
    is_person: bool
    timestamp: float = field(default_factory=time.time)

# ============================================================================
# TensorRT Engine
# ============================================================================
class Engine:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        cuda_ctx.push()
        try:
            with open(path, 'rb') as f:
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            self.in_name = self.engine.get_tensor_name(0)
            self.out_name = self.engine.get_tensor_name(1)
            self.in_shape = self.engine.get_tensor_shape(self.in_name)
            self.out_shape = self.engine.get_tensor_shape(self.out_name)
            self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape) * 4))
            self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape) * 4))
            self.h_out = np.empty(self.out_shape, dtype=np.float32)
            self.stream = cuda.Stream()
            self.ctx.set_tensor_address(self.in_name, int(self.d_in))
            self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        finally:
            cuda_ctx.pop()

    def run(self, data):
        cuda_ctx.push()
        try:
            cuda.memcpy_htod_async(self.d_in, data.astype(np.float32), self.stream)
            self.ctx.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            return self.h_out.copy()
        finally:
            cuda_ctx.pop()

# ============================================================================
# Post-Processing
# ============================================================================
def nms(boxes, scores, thresh):
    if len(boxes) == 0: return []
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

def postprocess(out, conf_th=0.15, nms_th=0.3):
    confs = out[:, 0]
    mask = confs > conf_th
    if not np.any(mask): return []
    filt = out[mask]
    boxes = filt[:, 220:224] / 224.0
    keep = nms(boxes, filt[:, 0], nms_th)
    results = []
    now = time.time()
    for i in keep:
        d = filt[i]
        box = d[220:224] / 224.0
        obj_idx = np.argmax(d[1:37])
        is_person = (obj_idx == 0)
        
        # Actions (for persons)
        actions = []
        if is_person:
            act_probs = d[37:194]
            for idx in np.argsort(-act_probs)[:5]:
                if act_probs[idx] > 0.12:
                    actions.append((ACT_CLASSES[idx], act_probs[idx]))
        
        # Relations (for all objects)
        relations = []
        rel_probs = d[194:220]
        for idx in np.argsort(-rel_probs)[:3]:
            if rel_probs[idx] > 0.15:
                relations.append((REL_CLASSES[idx], rel_probs[idx]))
        
        results.append(Det(box, d[0], OBJ_CLASSES[obj_idx], actions, relations, is_person, now))
    return results

# ============================================================================
# Global State
# ============================================================================
app = Flask(__name__)

class State:
    frame = None
    frame_buffer = deque(maxlen=16)  # Direct 16 frames for K=16 model
    detections = []
    prediction_history = deque(maxlen=50)  # Last ~2 seconds of predictions
    inf_fps = 0.0
    cam_fps = 0.0
    running = True
    last_inference_time = 0

S = State()
state_lock = threading.Lock()

# ============================================================================
# Main Loop - Optimized for Maximum FPS
# ============================================================================
def main_loop(cam_id, engine_path):
    # Open camera
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        S.running = False
        return
    
    # Load engine
    engine = Engine(engine_path)
    print(f"Engine loaded: {engine_path}")
    print(f"  Input shape: {engine.in_shape}")
    print(f"  Output shape: {engine.out_shape}")
    
    cam_fps_buf = deque(maxlen=30)
    inf_fps_buf = deque(maxlen=30)
    last_cam_time = time.time()
    
    print("Starting main loop...")
    
    while S.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        
        # Camera FPS tracking
        now = time.time()
        cam_fps_buf.append(1.0 / (now - last_cam_time + 1e-6))
        last_cam_time = now
        S.cam_fps = np.mean(cam_fps_buf)
        
        # Update display frame
        with state_lock:
            S.frame = frame.copy()
        
        # Preprocess and add to buffer
        small = cv2.resize(frame, (224, 224))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        S.frame_buffer.append(small)
        
        # Run inference when buffer is full
        if len(S.frame_buffer) >= 16:
            t0 = time.time()
            
            # Build clip directly from last 16 frames
            clip = np.stack(list(S.frame_buffer), axis=0)  # [16, H, W, 3]
            clip = np.ascontiguousarray(clip.transpose(3, 0, 1, 2)[None])  # [1, 3, 16, H, W]
            
            # Run inference
            out = engine.run(clip)
            dets = postprocess(out)
            
            with state_lock:
                S.detections = dets
                # Add to history
                for d in dets:
                    if d.is_person and d.actions:
                        S.prediction_history.append(d)
            
            dt = time.time() - t0
            inf_fps_buf.append(1.0 / dt)
            S.inf_fps = np.mean(inf_fps_buf)
            S.last_inference_time = dt * 1000  # ms
    
    cap.release()

# ============================================================================
# Drawing Functions
# ============================================================================
def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    cv2.circle(img, (x1+radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x1+radius, y2-radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y2-radius), radius, color, thickness)

def draw_detection(frame, det, h, w):
    """Draw a single detection with beautiful styling"""
    x1, y1, x2, y2 = (det.box * [w, h, w, h]).astype(int)
    
    # Colors based on type
    if det.is_person:
        box_color = COLORS['person']
        text_bg = (0, 100, 50)
    else:
        box_color = COLORS['object']
        text_bg = (100, 60, 0)
    
    # Draw box with glow effect (thicker border underneath)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Shadow
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    
    # Label background
    label = f"{det.name} {det.conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), text_bg, -1)
    cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw actions for persons
    y_off = y1 + 20
    for act, score in det.actions[:3]:
        # Truncate long action names
        act_short = act[:25] + "..." if len(act) > 25 else act
        act_text = f"{act_short}: {score:.0%}"
        
        # Background bar
        bar_width = int(score * 120)
        cv2.rectangle(frame, (x1, y_off-12), (x1 + bar_width, y_off+4), COLORS['action'], -1)
        cv2.rectangle(frame, (x1, y_off-12), (x1 + 120, y_off+4), COLORS['action'], 1)
        
        cv2.putText(frame, act_text, (x1+3, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        y_off += 20

def draw_stats_panel(frame, h, w):
    """Draw stats panel with FPS and detection counts"""
    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (220, 90), COLORS['bg_panel'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Stats text
    cv2.putText(frame, f"INFERENCE: {S.inf_fps:.1f} FPS", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['person'], 2)
    cv2.putText(frame, f"LATENCY: {S.last_inference_time:.0f}ms", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
    cv2.putText(frame, f"CAMERA: {S.cam_fps:.0f} FPS", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['highlight'], 1)

def draw_history_panel(frame, h, w):
    """Draw recent prediction history panel"""
    # Panel on right side
    panel_x = w - 250
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 10), (w-10, 180), COLORS['bg_panel'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "RECENT ACTIONS", (panel_x+10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['highlight'], 1)
    
    # Get recent unique actions (last 2 seconds)
    now = time.time()
    recent = []
    seen = set()
    for d in reversed(list(S.prediction_history)):
        if now - d.timestamp > 2.0:
            break
        for act, score in d.actions[:2]:
            if act not in seen:
                seen.add(act)
                recent.append((act, score, now - d.timestamp))
                if len(recent) >= 6:
                    break
        if len(recent) >= 6:
            break
    
    # Draw actions
    y_off = 50
    for act, score, age in recent:
        # Fade based on age
        alpha = 1.0 - (age / 2.0)
        color = tuple(int(c * alpha) for c in COLORS['action'])
        
        act_short = act[:22] if len(act) <= 22 else act[:20] + ".."
        cv2.putText(frame, f"• {act_short}", (panel_x+10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, f"{score:.0%}", (w-50, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
        y_off += 20

# ============================================================================
# Flask Routes  
# ============================================================================
@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>YOWO Multi-Task Demo</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 50%, #0f0f1f 100%);
            min-height: 100vh;
            font-family: 'Roboto', sans-serif;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        
        h1 {
            font-family: 'Orbitron', sans-serif;
            text-align: center;
            font-size: 2rem;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        }
        
        .video-container {
            position: relative;
            border: 2px solid rgba(0, 255, 136, 0.3);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 0 40px rgba(0, 204, 255, 0.2);
        }
        
        img {
            width: 100%;
            display: block;
        }
        
        .info-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            padding: 15px 20px;
            background: rgba(30, 30, 50, 0.8);
            border-radius: 10px;
            border: 1px solid rgba(100, 200, 255, 0.2);
        }
        
        .info-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .dot { 
            width: 10px; height: 10px; 
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        .dot.green { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .dot.blue { background: #00ccff; box-shadow: 0 0 10px #00ccff; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .footer {
            text-align: center;
            margin-top: 20px;
            color: rgba(255,255,255,0.5);
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 YOWO MULTI-TASK ACTION DETECTION</h1>
        
        <div class="video-container">
            <img id="stream" src="/video_feed">
        </div>
        
        <div class="info-bar">
            <div class="info-item">
                <div class="dot green"></div>
                <span>ResNeXt101 + YOLO11m • TensorRT FP16</span>
            </div>
            <div class="info-item">
                <div class="dot blue"></div>
                <span>Jetson Orin Nano • 224×224 @ 16 frames</span>
            </div>
        </div>
        
        <div class="footer">
            Objects: 36 classes • Actions: 157 classes • Relations: 26 classes
        </div>
    </div>
    
    <script>
        var img = document.getElementById('stream');
        img.onerror = function() {
            setTimeout(function() { img.src = '/video_feed?' + Date.now(); }, 100);
        };
    </script>
</body>
</html>
    """)

def generate_frames():
    """Generate MJPEG frames"""
    target_fps = 30
    frame_time = 1.0 / target_fps
    
    while S.running:
        start = time.time()
        
        with state_lock:
            if S.frame is None:
                time.sleep(0.01)
                continue
            frame = S.frame.copy()
            dets = S.detections[:]
        
        h, w = frame.shape[:2]
        
        # Draw detections
        for d in dets:
            draw_detection(frame, d, h, w)
        
        # Draw panels
        draw_stats_panel(frame, h, w)
        draw_history_panel(frame, h, w)
        
        # Encode
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        
        # Pace
        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

@app.route('/video_feed')
def video_feed():
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

@app.route('/favicon.ico')
def favicon(): return '', 204

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default='yowo_resnext_yolo11m_multitask_fp16.engine')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    main_thread = threading.Thread(target=main_loop, args=(args.camera, args.engine), daemon=True)
    main_thread.start()
    
    time.sleep(2)
    
    try:
        print("\n" + "="*60)
        print("🎬 YOWO Multi-Task Demo Starting...")
        print("="*60)
        print(f"Engine: {args.engine}")
        print("Access at: http://localhost:5000")
        print("="*60 + "\n")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        S.running = False
