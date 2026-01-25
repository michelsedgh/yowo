#!/usr/bin/env python3
"""
YOWO Smart Home Demo - Fixed for 42-action model
Output format: [conf(1), obj(36), act(42), rel(26), box(4)] = 109
"""

import argparse
import time
import threading
import json
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import pycuda.driver as cuda
import tensorrt as trt

# CUDA Context
cuda.init()
cuda_device = cuda.Device(0)
cuda_ctx = cuda_device.make_context()

# Load Smart Home classes
def load_classes():
    obj = [l.strip() for l in open('data/ActionGenome/annotations/object_classes.txt')]
    rel = [l.strip() for l in open('data/ActionGenome/annotations/relationship_classes.txt')]
    
    # Smart home 42 actions from config
    with open('config/smart_home_final.json', 'r') as f:
        cfg = json.load(f)
    act = cfg['action_names']
    
    return obj, act, rel

OBJ_CLASSES, ACT_CLASSES, REL_CLASSES = load_classes()
print(f"Loaded: {len(OBJ_CLASSES)} objects, {len(ACT_CLASSES)} actions, {len(REL_CLASSES)} relations")

# Output indices for smart home model
# [conf(1), obj(36), act(42), rel(26), box(4)] = 109
OBJ_START, OBJ_END = 1, 37
ACT_START, ACT_END = 37, 79
REL_START, REL_END = 79, 105
BOX_START, BOX_END = 105, 109

# ============================================================
# OBJECT-ACTION BOOST: Fix for broken cascade
# The cascade attention learned uniform weights, so it doesn't 
# properly link objects to actions. This mapping directly boosts
# action probabilities when relevant objects are detected nearby.
# ============================================================
# Object indices (0-indexed): laptop=18, food=16, phone=23, tv=32, book=4, etc.
# Action indices: see config/smart_home_final.json
OBJECT_ACTION_BOOST = {
    # laptop (obj 18) -> "Working/Playing on a laptop" (act 7)
    18: [(7, 0.4)],  # Boost by 0.4 when laptop detected
    # food (obj 16) -> "Holding some food" (act 11), "Taking food" (act 12)
    16: [(11, 0.3), (12, 0.2)],
    # phone/camera (obj 23) -> phone actions (act 4, 5, 6)
    23: [(4, 0.3), (5, 0.25), (6, 0.25)],
    # cup/glass/bottle (obj 10) -> drinking/holding cup (act 19, 20)
    10: [(19, 0.3), (20, 0.25)],
    # television (obj 32) -> watching tv (act 26)
    32: [(26, 0.35)],
    # book (obj 4) -> reading (act 27, 28)
    4: [(27, 0.3), (28, 0.25)],
}
ENABLE_OBJECT_ACTION_BOOST = True  # Set to False to disable

@dataclass
class Det:
    box: np.ndarray
    conf: float
    name: str
    actions: List[Tuple[str, float]]
    relations: List[Tuple[str, float]]
    is_person: bool

# TensorRT Engine
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
            # Extract clip length from input shape [B, C, T, H, W]
            self.clip_length = self.in_shape[2]
            self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape) * 4))
            self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape) * 4))
            self.h_out = np.empty(self.out_shape, dtype=np.float32)
            self.stream = cuda.Stream()
            self.ctx.set_tensor_address(self.in_name, int(self.d_in))
            self.ctx.set_tensor_address(self.out_name, int(self.d_out))
            print(f"Engine: {path}")
            print(f"  Input: {self.in_shape} (clip_length={self.clip_length})")
            print(f"  Output: {self.out_shape}")
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

# NMS
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

DEBUG_OBJECTS = True  # Set to True to see what objects are detected

def postprocess(out, conf_th=0.15, nms_th=0.3):
    confs = out[:, 0]
    mask = confs > conf_th
    if not np.any(mask): return []
    
    filt = out[mask]
    boxes = filt[:, BOX_START:BOX_END] / 224.0
    keep = nms(boxes, filt[:, 0], nms_th)
    
    # First pass: collect all detected objects (non-person) for boost lookup
    detected_objects = []  # List of (obj_idx, box, conf)
    person_indices = []
    for i in keep:
        d = filt[i]
        obj_probs = d[OBJ_START:OBJ_END]
        obj_idx = np.argmax(obj_probs)
        box = d[BOX_START:BOX_END] / 224.0
        if obj_idx == 0:  # person
            person_indices.append(i)
        else:
            detected_objects.append((obj_idx, box, d[0], obj_probs[obj_idx]))
    
    results = []
    for i in keep:
        d = filt[i]
        box = d[BOX_START:BOX_END] / 224.0
        
        # Object prediction (softmax already applied in export)
        obj_probs = d[OBJ_START:OBJ_END]
        obj_idx = np.argmax(obj_probs)
        is_person = (obj_idx == 0)
        
        # DEBUG: Full cascade trace
        if DEBUG_OBJECTS:
            top3_obj = np.argsort(-obj_probs)[:5]
            laptop_prob = obj_probs[18]
            food_prob = obj_probs[16]
            phone_prob = obj_probs[23]
            print(f"\n=== Detection (conf={d[0]:.2f}, {'PERSON' if is_person else OBJ_CLASSES[obj_idx]}) ===")
            print(f"  Top-5 objects: {[(OBJ_CLASSES[j], f'{obj_probs[j]:.2f}') for j in top3_obj]}")
            print(f"  laptop={laptop_prob:.3f}, food={food_prob:.3f}, phone={phone_prob:.3f}")
        
        # Actions - ONLY for persons
        actions = []
        if is_person:
            act_probs = d[ACT_START:ACT_END].copy()  # Make a copy to modify
            
            # OBJECT-ACTION BOOST: Fix broken cascade by directly boosting actions
            if ENABLE_OBJECT_ACTION_BOOST:
                # Method 1: Check THIS detection's secondary object probs
                # Even though person is top class, laptop/food might have non-trivial prob
                for obj_check_idx in OBJECT_ACTION_BOOST.keys():
                    obj_prob_here = obj_probs[obj_check_idx]
                    if obj_prob_here > 0.05:  # Even small probability indicates object presence
                        for act_idx, boost in OBJECT_ACTION_BOOST[obj_check_idx]:
                            old_prob = act_probs[act_idx]
                            # Scale boost by object probability
                            act_probs[act_idx] = min(0.95, act_probs[act_idx] + boost * obj_prob_here * 3)
                            if DEBUG_OBJECTS and obj_prob_here > 0.02:
                                print(f"  BOOST(self): {OBJ_CLASSES[obj_check_idx]}={obj_prob_here:.2f} -> {ACT_CLASSES[act_idx]} {old_prob:.2f} -> {act_probs[act_idx]:.2f}")
                
                # Method 2: Check separately detected objects nearby
                for det_obj_idx, det_box, det_conf, det_obj_prob in detected_objects:
                    if det_obj_idx in OBJECT_ACTION_BOOST and det_obj_prob > 0.3:
                        px1, py1, px2, py2 = box
                        ox1, oy1, ox2, oy2 = det_box
                        p_cx, p_cy = (px1+px2)/2, (py1+py2)/2
                        o_cx, o_cy = (ox1+ox2)/2, (oy1+oy2)/2
                        dist = np.sqrt((p_cx-o_cx)**2 + (p_cy-o_cy)**2)
                        if dist < 0.5:  # Within proximity
                            for act_idx, boost in OBJECT_ACTION_BOOST[det_obj_idx]:
                                old_prob = act_probs[act_idx]
                                act_probs[act_idx] = min(0.95, act_probs[act_idx] + boost * det_obj_prob)
                                if DEBUG_OBJECTS:
                                    print(f"  BOOST(nearby): {OBJ_CLASSES[det_obj_idx]} detected -> {ACT_CLASSES[act_idx]} {old_prob:.2f} -> {act_probs[act_idx]:.2f}")
            
            # DEBUG: Show top-5 action predictions after boost
            if DEBUG_OBJECTS:
                top5_act = np.argsort(-act_probs)[:5]
                print(f"  Action probs (after boost): {[(ACT_CLASSES[j], f'{act_probs[j]:.2f}') for j in top5_act]}")
            for idx in np.argsort(-act_probs)[:5]:
                if act_probs[idx] > 0.12:
                    actions.append((ACT_CLASSES[idx], float(act_probs[idx])))
        
        # Relations - for all detections
        relations = []
        rel_probs = d[REL_START:REL_END]
        if DEBUG_OBJECTS and is_person:
            top3_rel = np.argsort(-rel_probs)[:5]
            # Key relations: holding(14), touching(21), carrying(9)
            print(f"  Relation probs: {[(REL_CLASSES[j], f'{rel_probs[j]:.2f}') for j in top3_rel]}")
            print(f"  holding={rel_probs[14]:.3f}, touching={rel_probs[21]:.3f}, carrying={rel_probs[9]:.3f}")
        for idx in np.argsort(-rel_probs)[:3]:
            if rel_probs[idx] > 0.15:
                relations.append((REL_CLASSES[idx], float(rel_probs[idx])))
        
        results.append(Det(box, float(d[0]), OBJ_CLASSES[obj_idx], actions, relations, is_person))
    
    return results

# Global State
app = Flask(__name__)

class State:
    frame = None
    frame_buffer = deque(maxlen=32)  # 32 consecutive frames for K=32 model
    clip_length = 32  # Will be updated from engine input shape
    detections = []
    inf_fps = 0.0
    running = True

S = State()
state_lock = threading.Lock()

def main_loop(cam_id, engine_path):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        S.running = False
        return
    
    engine = Engine(engine_path)
    
    # Update clip length from engine and resize frame buffer
    S.clip_length = engine.clip_length
    S.frame_buffer = deque(maxlen=S.clip_length)
    
    fps_buf = deque(maxlen=30)
    
    print(f"Starting inference loop (clip_length={S.clip_length})...")
    
    while S.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        
        with state_lock:
            S.frame = frame.copy()
        
        # Preprocess
        small = cv2.resize(frame, (224, 224))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        S.frame_buffer.append(small)
        
        # Run inference when buffer full
        if len(S.frame_buffer) >= S.clip_length:
            t0 = time.time()
            
            clip = np.stack(list(S.frame_buffer), axis=0)
            clip = np.ascontiguousarray(clip.transpose(3, 0, 1, 2)[None])
            
            out = engine.run(clip)
            dets = postprocess(out)
            
            with state_lock:
                S.detections = dets
            
            dt = time.time() - t0
            fps_buf.append(1.0 / dt)
            S.inf_fps = np.mean(fps_buf)
    
    cap.release()

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html><html><head><title>YOWO Smart Home</title>
<style>
body{background:#0a0a12;color:#00ff88;font-family:monospace;text-align:center;padding:20px;margin:0}
img{max-width:100%;border:2px solid #00ff88;display:block;margin:0 auto}
h1{letter-spacing:3px;margin-bottom:20px}
.info{color:#888;margin-top:10px}
</style></head><body>
<h1>YOWO SMART HOME ACTION DETECTION</h1>
<img id="stream" src="/video_feed">
<p class="info">42 Actions | TensorRT FP16 | Orin Nano</p>
<script>
var img = document.getElementById('stream');
img.onerror = function() { setTimeout(function() { img.src = '/video_feed?' + Date.now(); }, 100); };
</script>
</body></html>
    """)

def generate_frames():
    while S.running:
        with state_lock:
            if S.frame is None:
                time.sleep(0.01)
                continue
            frame = S.frame.copy()
            dets = S.detections[:]
            inf_fps = S.inf_fps
        
        h, w = frame.shape[:2]
        
        for d in dets:
            x1, y1, x2, y2 = (d.box * [w, h, w, h]).astype(int)
            
            if d.is_person:
                color = (0, 255, 128)  # Green for person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"PERSON {d.conf:.0%}"
                cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show actions for person
                y_off = y1 + 20
                for act, score in d.actions[:3]:
                    # Shorten action name
                    act_short = act[:30] + "..." if len(act) > 30 else act
                    cv2.putText(frame, f"> {act_short}: {score:.0%}", (x1, y_off), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                    y_off += 18
                
                # Show relations for person (NEW - was missing!)
                for rel, score in d.relations[:2]:
                    cv2.putText(frame, f"  [{rel}: {score:.0%}]", (x1, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,255), 1)
                    y_off += 16
            else:
                color = (255, 165, 0)  # Orange for objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{d.name} {d.conf:.0%}", (x1, y1-8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Stats
        cv2.putText(frame, f"INF: {inf_fps:.1f} FPS", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Detections: {len(dets)}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,200,255), 1)
        
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

@app.route('/favicon.ico')
def favicon(): return '', 204

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default='yowo_resnext_yolo26m_e7_fp16.engine')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    main_thread = threading.Thread(target=main_loop, args=(args.camera, args.engine), daemon=True)
    main_thread.start()
    
    time.sleep(2)
    
    print("\n" + "="*50)
    print("YOWO Smart Home Demo")
    print("="*50)
    print(f"Actions: {len(ACT_CLASSES)}")
    print(f"Objects: {len(OBJ_CLASSES)}")
    print(f"Open: http://localhost:{args.port}")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        S.running = False
