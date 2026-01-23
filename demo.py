#!/usr/bin/env python3
"""
Smart Home Action Detection - Final Demo
=========================================
Clean, focused, professional UI for understanding human activities.

Key Design:
- Video feed with SINGLE clean box per person  
- Large current action display
- Scrolling action log (like terminal output)
- NO NMS (trusting O2O heads, higher conf threshold instead)
"""

import argparse
import json
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import pycuda.driver as cuda
import tensorrt as trt

# CUDA setup
cuda.init()
cuda_ctx = cuda.Device(0).make_context()

# Output format
N_OBJ, N_ACT, N_REL = 36, 42, 26
IDX = {'conf': 0, 'obj': (1, 37), 'act': (37, 79), 'rel': (79, 105), 'box': (105, 109)}

# Load labels
with open('config/smart_home_final.json') as f:
    ACTIONS = json.load(f)['action_names']
OBJECTS = [l.strip() for l in open('data/ActionGenome/annotations/object_classes.txt')]
print(f"Loaded {len(ACTIONS)} actions, {len(OBJECTS)} objects")


class TRTEngine:
    def __init__(self, path):
        cuda_ctx.push()
        try:
            with open(path, 'rb') as f:
                rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = rt.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            
            self.in_name = self.engine.get_tensor_name(0)
            self.out_name = self.engine.get_tensor_name(1)
            in_shape = self.engine.get_tensor_shape(self.in_name)
            out_shape = self.engine.get_tensor_shape(self.out_name)
            
            self.d_in = cuda.mem_alloc(int(np.prod(in_shape) * 4))
            self.d_out = cuda.mem_alloc(int(np.prod(out_shape) * 4))
            self.h_out = np.empty(out_shape, np.float32)
            self.stream = cuda.Stream()
            
            self.ctx.set_tensor_address(self.in_name, int(self.d_in))
            self.ctx.set_tensor_address(self.out_name, int(self.d_out))
            print(f"Engine loaded: in={in_shape}, out={out_shape}")
        finally:
            cuda_ctx.pop()

    def __call__(self, x):
        cuda_ctx.push()
        try:
            cuda.memcpy_htod_async(self.d_in, x.astype(np.float32), self.stream)
            self.ctx.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            return self.h_out.copy()
        finally:
            cuda_ctx.pop()


class State:
    def __init__(self):
        self.frame = None
        self.buffer = deque(maxlen=16)
        self.current_action = None  # (name, confidence)
        self.action_log = deque(maxlen=15)  # [(time, action, conf), ...]
        self.box = None  # Single person box
        self.fps = 0
        self.latency = 0
        self.running = True
        self.lock = threading.Lock()

S = State()


def process_output(out, conf_thresh=0.40, img_size=224):
    """Extract the single best person detection with actions."""
    confs = out[:, IDX['conf']]
    
    # Find person detections only
    persons = []
    for i, row in enumerate(out):
        if confs[i] < conf_thresh:
            continue
        obj_probs = row[IDX['obj'][0]:IDX['obj'][1]]
        if np.argmax(obj_probs) == 0:  # Person class
            persons.append((i, confs[i]))
    
    if not persons:
        return None, None, []
    
    # Take highest confidence person
    best_idx = max(persons, key=lambda x: x[1])[0]
    row = out[best_idx]
    
    # Box
    box = row[IDX['box'][0]:IDX['box'][1]] / img_size
    box = np.clip(box, 0, 1)
    
    # Actions (all above 8%)
    act_probs = row[IDX['act'][0]:IDX['act'][1]]
    actions = [(ACTIONS[i], float(act_probs[i])) 
               for i in np.argsort(-act_probs) if act_probs[i] > 0.08]
    
    return box, float(row[IDX['conf']]), actions


def inference_loop(engine, camera, img_size, conf_thresh):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps_buf = deque(maxlen=30)
    last_action = None
    
    while S.running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        with S.lock:
            S.frame = frame.copy()
        
        # Prepare frame
        small = cv2.resize(frame, (img_size, img_size))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        S.buffer.append(small)
        
        if len(S.buffer) < 16:
            continue
        
        # Inference
        t0 = time.time()
        clip = np.stack(list(S.buffer), axis=0)
        clip = np.ascontiguousarray(clip.transpose(3, 0, 1, 2)[None])
        
        out = engine(clip)
        box, conf, actions = process_output(out, conf_thresh, img_size)
        
        dt = time.time() - t0
        fps_buf.append(1.0 / dt)
        
        with S.lock:
            S.fps = np.mean(fps_buf)
            S.latency = dt * 1000
            S.box = box
            
            if actions:
                top_action = actions[0]
                S.current_action = top_action
                
                # Log if action changed or significant
                if last_action != top_action[0] and top_action[1] > 0.15:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    S.action_log.appendleft((timestamp, top_action[0], top_action[1]))
                    last_action = top_action[0]
            else:
                S.current_action = None
    
    cap.release()


# ============================================================================
# Web Interface
# ============================================================================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Home Activity Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0f0f14;
            color: #fff;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 320px;
            height: 100vh;
        }
        
        /* Video Section */
        .video-section {
            display: flex;
            flex-direction: column;
            background: #0a0a0e;
            padding: 20px;
        }
        
        .video-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #00e676;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            font-size: 0.85rem;
            color: #666;
        }
        .stats span { color: #00bcd4; }
        
        .video-wrapper {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .video-frame {
            max-width: 100%;
            max-height: 100%;
            border-radius: 12px;
            border: 2px solid #1a1a24;
        }
        
        /* Current Action - BIG display */
        .current-action {
            background: linear-gradient(135deg, #1a1a28 0%, #12121a 100%);
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
        }
        
        .current-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #00e676;
            margin-bottom: 8px;
        }
        
        .current-text {
            font-size: 1.4rem;
            font-weight: 600;
            color: #fff;
            min-height: 1.6em;
        }
        
        .current-conf {
            margin-top: 10px;
            height: 6px;
            background: #2a2a3a;
            border-radius: 3px;
            overflow: hidden;
        }
        
        .current-conf-bar {
            height: 100%;
            background: linear-gradient(90deg, #00e676, #00bcd4);
            transition: width 0.3s ease;
        }
        
        /* Sidebar - Action Log */
        .sidebar {
            background: #14141c;
            border-left: 1px solid #1e1e28;
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #1e1e28;
        }
        
        .sidebar-title {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
        }
        
        .log-container {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        
        .log-entry {
            display: flex;
            align-items: flex-start;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            background: #1a1a24;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .log-time {
            font-size: 0.7rem;
            color: #555;
            min-width: 55px;
            font-family: monospace;
        }
        
        .log-content {
            flex: 1;
        }
        
        .log-action {
            font-size: 0.85rem;
            color: #fff;
            margin-bottom: 4px;
        }
        
        .log-conf {
            font-size: 0.7rem;
            color: #00bcd4;
        }
        
        .log-bar {
            width: 100%;
            height: 3px;
            background: #2a2a3a;
            border-radius: 2px;
            margin-top: 6px;
        }
        
        .log-bar-fill {
            height: 100%;
            background: #00bcd4;
            border-radius: 2px;
        }
        
        /* Empty state */
        .empty {
            color: #444;
            text-align: center;
            padding: 40px 20px;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-section">
            <div class="video-header">
                <div class="title">🏠 Smart Home Activity Monitor</div>
                <div class="stats">
                    <div id="fps">--</div>
                    <div id="latency">--</div>
                </div>
            </div>
            
            <div class="video-wrapper">
                <img class="video-frame" id="stream" src="/feed">
            </div>
            
            <div class="current-action">
                <div class="current-label">Current Activity</div>
                <div class="current-text" id="current-action">Waiting for detection...</div>
                <div class="current-conf">
                    <div class="current-conf-bar" id="current-conf" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">📋 Activity Log</div>
            </div>
            <div class="log-container" id="log">
                <div class="empty">Actions will appear here</div>
            </div>
        </div>
    </div>
    
    <script>
        async function update() {
            try {
                const r = await fetch('/state');
                const d = await r.json();
                
                document.getElementById('fps').innerHTML = 
                    `<span>${d.fps.toFixed(1)}</span> FPS`;
                document.getElementById('latency').innerHTML = 
                    `<span>${d.latency.toFixed(0)}</span>ms`;
                
                const actionEl = document.getElementById('current-action');
                const confEl = document.getElementById('current-conf');
                
                if (d.current) {
                    actionEl.textContent = d.current.name;
                    confEl.style.width = (d.current.conf * 100) + '%';
                } else {
                    actionEl.textContent = 'No activity detected';
                    confEl.style.width = '0%';
                }
                
                const logEl = document.getElementById('log');
                if (d.log.length > 0) {
                    logEl.innerHTML = d.log.map(entry => `
                        <div class="log-entry">
                            <div class="log-time">${entry.time}</div>
                            <div class="log-content">
                                <div class="log-action">${entry.action}</div>
                                <div class="log-bar">
                                    <div class="log-bar-fill" style="width: ${entry.conf * 100}%"></div>
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
            } catch(e) {}
        }
        
        setInterval(update, 150);
        update();
        
        document.getElementById('stream').onerror = function() {
            setTimeout(() => this.src = '/feed?' + Date.now(), 100);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/state')
def state():
    with S.lock:
        return {
            'fps': S.fps,
            'latency': S.latency,
            'current': {'name': S.current_action[0], 'conf': S.current_action[1]} 
                       if S.current_action else None,
            'log': [{'time': t, 'action': a, 'conf': c} for t, a, c in S.action_log]
        }

def gen_frames():
    while S.running:
        with S.lock:
            if S.frame is None:
                time.sleep(0.01)
                continue
            frame = S.frame.copy()
            box = S.box
            action = S.current_action
        
        h, w = frame.shape[:2]
        
        if box is not None:
            x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
            
            # Clean single box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 118), 2)
            
            # Action label below box
            if action:
                label = f"{action[0][:30]} ({action[1]:.0%})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y2+2), (x1+tw+10, y2+th+12), (0, 230, 118), -1)
                cv2.putText(frame, label, (x1+5, y2+th+8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.033)

@app.route('/feed')
def feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/favicon.ico')
def fav(): return '', 204


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--engine', default='yowo_smart_home_yolo26m_e1_fp16.engine')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--conf', type=float, default=0.40)
    args = p.parse_args()
    
    engine = TRTEngine(args.engine)
    
    t = threading.Thread(target=inference_loop, 
                        args=(engine, args.camera, args.size, args.conf), 
                        daemon=True)
    t.start()
    time.sleep(2)
    
    print("\n" + "="*50)
    print("🏠 Smart Home Activity Monitor")
    print("="*50)
    print(f"Confidence threshold: {args.conf}")
    print(f"Access: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
