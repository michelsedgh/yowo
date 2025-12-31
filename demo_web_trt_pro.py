#!/usr/bin/env python3
"""
YOWO TensorRT Demo - Optimized for Smooth Streaming
Uses single-frame buffer to prevent browser buffering issues.
"""

import argparse
import time
import threading
from collections import deque
from dataclasses import dataclass
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
    is_person: bool

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
    for i in keep:
        d = filt[i]
        box = d[220:224] / 224.0
        obj_idx = np.argmax(d[1:37])
        is_person = (obj_idx == 0)
        actions = []
        if is_person:
            act_probs = d[37:194]
            for idx in np.argsort(-act_probs)[:3]:
                if act_probs[idx] > 0.15:
                    actions.append((ACT_CLASSES[idx], act_probs[idx]))
        results.append(Det(box, d[0], OBJ_CLASSES[obj_idx], actions, is_person))
    return results

# ============================================================================
# Global State - Simple and Clean
# ============================================================================
app = Flask(__name__)

class State:
    frame = None              # Latest camera frame (BGR)
    frame_buffer = deque(maxlen=80)  # For temporal inference
    detections = []           # Latest detections
    inf_fps = 0.0
    running = True

S = State()
state_lock = threading.Lock()

# ============================================================================
# Thread 1: Camera Capture + Inference (Combined for simplicity)
# ============================================================================
def main_loop(cam_id, engine_path):
    # Open camera
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        S.running = False
        return
    
    # Load engine
    engine = Engine(engine_path)
    print("Engine loaded, starting main loop...")
    
    fps_buf = deque(maxlen=30)
    frame_count = 0
    
    while S.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        
        # Update frame for display
        with state_lock:
            S.frame = frame
        
        # Add to temporal buffer
        small = cv2.resize(frame, (224, 224))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        S.frame_buffer.append(small)
        
        frame_count += 1
        
        # Run inference every 5 frames (matches sample_rate=5)
        if len(S.frame_buffer) >= 80 and frame_count % 5 == 0:
            t0 = time.time()
            
            # Build clip from buffer
            clip = np.stack([S.frame_buffer[i*5] for i in range(16)], axis=0)
            clip = np.ascontiguousarray(clip.transpose(3, 0, 1, 2)[None])
            
            # Run inference
            out = engine.run(clip)
            dets = postprocess(out)
            
            with state_lock:
                S.detections = dets
            
            dt = time.time() - t0
            fps_buf.append(1.0 / dt)
            S.inf_fps = np.mean(fps_buf)
    
    cap.release()

# ============================================================================
# Flask Routes
# ============================================================================
@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html><html><head><title>YOWO Demo</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<style>
body{background:#111;color:#0f0;font-family:monospace;text-align:center;padding:20px;margin:0}
img{max-width:100%;border:2px solid #0f0;display:block;margin:0 auto}
h1{letter-spacing:3px;margin-bottom:20px}
</style></head><body>
<h1>YOWO ACTION DETECTION</h1>
<img id="stream" src="/video_feed">
<p>Jetson Orin Nano | TensorRT FP16</p>
<script>
// Force image refresh to prevent caching
var img = document.getElementById('stream');
img.onerror = function() { setTimeout(function() { img.src = '/video_feed?' + Date.now(); }, 100); };
</script>
</body></html>
    """)

def generate_frames():
    """Generate MJPEG frames at a steady rate."""
    target_fps = 25  # Target stream FPS
    frame_time = 1.0 / target_fps
    
    while S.running:
        start = time.time()
        
        # Get current frame and detections
        with state_lock:
            if S.frame is None:
                time.sleep(0.01)
                continue
            frame = S.frame.copy()
            dets = S.detections[:]
            inf_fps = S.inf_fps
        
        # Draw detections
        h, w = frame.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = (d.box * [w, h, w, h]).astype(int)
            color = (0, 255, 0) if d.is_person else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d.name} {d.conf:.0%}"
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_off = y1 + 20
            for act, s in d.actions:
                cv2.putText(frame, f"> {act}: {s:.0%}", (x1, y_off), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                y_off += 18
        
        # Stats overlay
        cv2.putText(frame, f"INF: {inf_fps:.1f} FPS", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Encode to JPEG
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        
        # Pace to target FPS to prevent buffering
        elapsed = time.time() - start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

@app.route('/video_feed')
def video_feed():
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering if present
    return resp

@app.route('/favicon.ico')
def favicon(): return '', 204

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default='yowo_multitask_fp16.engine')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    # Start main processing thread
    main_thread = threading.Thread(target=main_loop, args=(args.camera, args.engine), daemon=True)
    main_thread.start()
    
    # Give it a moment to initialize
    time.sleep(2)
    
    try:
        print("Starting web server...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        S.running = False
