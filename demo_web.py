#!/usr/bin/env python3
"""
YOWO TensorRT Web Demo

Streams live webcam inference to a web browser.
Access from laptop: http://<orin-nano-ip>:5000

Usage:
    python demo_web.py --engine yowo_trtexec_fp16.engine --camera 0
"""

import argparse
import os
import time
import numpy as np
import cv2
from PIL import Image
from collections import deque
from flask import Flask, Response, render_template_string
import threading

import tensorrt as trt
import pycuda.driver as cuda

# Initialize CUDA context at module level
cuda.init()
cuda_device = cuda.Device(0)
cuda_ctx = cuda_device.make_context()

import torch  # Import after CUDA init
from dataset.transforms import BaseTransform
from utils.nms import nms  # For NMS post-processing

# ============================================================================
# CLASS LABELS
# ============================================================================

OBJECT_CLASSES = [
    'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
    'closetcabinet', 'clothes', 'cupglassbottle', 'dish', 'door', 'doorknob',
    'doorway', 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
    'mirror', 'papernotebook', 'phonecamera', 'picture', 'pillow', 'refrigerator',
    'sandwich', 'shelf', 'shoe', 'sofacouch', 'table', 'television', 'towel',
    'vacuum', 'window'
]

ACTION_CLASSES = [
    'Holding clothes', 'Putting clothes', 'Taking clothes', 'Throwing clothes',
    'Tidying clothes', 'Washing clothes', 'Closing door', 'Fixing door',
    'Opening door', 'Put on table', 'Sitting on table', 'Sitting at table',
    'Tidying table', 'Washing table', 'Working at table', 'Holding phone',
    'Playing w/ phone', 'Putting phone', 'Taking phone', 'Talking on phone',
    'Holding bag', 'Opening bag', 'Putting bag', 'Taking bag', 'Throwing bag',
    'Closing book', 'Holding book', 'Opening book', 'Putting book', 'Smiling at book',
    'Taking book', 'Throwing book', 'Reading book', 'Holding towel', 'Putting towel',
    'Taking towel', 'Throwing towel', 'Tidying towel', 'Washing w/ towel',
    'Closing box', 'Holding box', 'Opening box', 'Putting box', 'Taking box',
    'Taking from box', 'Throwing box', 'Closing laptop', 'Holding laptop',
    'Opening laptop', 'Putting laptop', 'Taking laptop', 'Watching laptop',
    'Working on laptop', 'Holding shoe', 'Putting shoe', 'Putting on shoe',
    'Taking shoe', 'Taking off shoe', 'Throwing shoe', 'Sitting in chair',
    'Standing on chair', 'Holding food', 'Putting food', 'Taking food',
    'Throwing food', 'Eating sandwich', 'Making sandwich', 'Holding sandwich',
    'Putting sandwich', 'Taking sandwich', 'Holding blanket', 'Putting blanket',
    'Snuggling blanket', 'Taking blanket', 'Throwing blanket', 'Tidying blanket',
    'Holding pillow', 'Putting pillow', 'Snuggling pillow', 'Taking pillow',
    'Throwing pillow', 'Put on shelf', 'Tidying shelf', 'Grabbing picture',
    'Holding picture', 'Laughing at pic', 'Putting picture', 'Taking picture',
    'Looking at pic', 'Closing window', 'Opening window', 'Washing window',
    'Looking outside', 'Holding mirror', 'Smiling mirror', 'Washing mirror',
    'Looking mirror', 'Walking doorway', 'Holding broom', 'Putting broom',
    'Taking broom', 'Throwing broom', 'Tidying broom', 'Fixing light',
    'On light', 'Off light', 'Drink cup', 'Holding cup',
    'Pour cup', 'Putting cup', 'Taking cup', 'Washing cup',
    'Closing closet', 'Opening closet', 'Tidying closet', 'Holding paper',
    'Putting paper', 'Taking paper', 'Holding dish', 'Putting dish', 'Taking dish',
    'Washing dish', 'Lying sofa', 'Sitting sofa', 'Lying floor',
    'Sitting floor', 'Throw floor', 'Tidy floor', 'Holding meds',
    'Taking meds', 'Put groceries', 'Laugh TV', 'Watch TV',
    'Wake bed', 'Lying bed', 'Sitting bed', 'Fix vacuum',
    'Holding vacuum', 'Taking vacuum', 'Wash hands', 'Fix doorknob',
    'Grasp doorknob', 'Close fridge', 'Open fridge', 'Fix hair',
    'Work paper', 'Waking', 'Cooking', 'Dressing', 'Laughing',
    'Running', 'Sit down', 'Smiling', 'Sneezing', 'Stand up', 'Undress', 'Eating'
]


class TRTInference:
    """TensorRT inference wrapper with proper CUDA context handling."""
    
    def __init__(self, engine_path):
        cuda_ctx.push()
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            
            print(f"Loading TensorRT engine: {engine_path}")
            with open(engine_path, 'rb') as f:
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # Get I/O info
            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)
            self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
            self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
            
            print(f"  Input: {self.input_name} {self.input_shape}")
            print(f"  Output: {self.output_name} {self.output_shape}")
            
            # Allocate buffers
            self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * 4))
            self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * 4))
            self.h_output = np.empty(self.output_shape, dtype=np.float32)
            self.stream = cuda.Stream()
            
            # Set addresses
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            
            print("Engine loaded successfully!")
        finally:
            cuda_ctx.pop()
    
    def infer(self, input_data):
        cuda_ctx.push()
        try:
            input_data = input_data.astype(np.float32)
            cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
            self.context.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            return self.h_output.copy()
        finally:
            cuda_ctx.pop()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def post_process(predictions, img_size=224, conf_thresh=0.4, nms_thresh=0.5, topk=40):
    """
    Post-process TensorRT output with proper NMS.
    
    This mirrors the PyTorch model's post_process_multi_hot() function.
    """
    preds = predictions[0]  # [N, 225]
    
    # Parse: [conf(1), obj(36), act(157), rel(26), box(4), interact(1)]
    conf = sigmoid(preds[:, 0])      # [N]
    obj = softmax(preds[:, 1:37])    # [N, 36]
    act = sigmoid(preds[:, 37:194])  # [N, 157]
    box = preds[:, 220:224]          # [N, 4] - pixel coords
    
    # Step 1: Top-K filtering (like PyTorch model)
    if len(conf) > topk:
        top_indices = np.argsort(-conf)[:topk]
        conf = conf[top_indices]
        obj = obj[top_indices]
        act = act[top_indices]
        box = box[top_indices]
    
    # Step 2: Confidence threshold filtering
    keep = conf > conf_thresh
    if not np.any(keep):
        return []
    
    conf = conf[keep]
    obj = obj[keep]
    act = act[keep]
    box = box[keep]
    
    # Normalize boxes to [0, 1]
    box = box / img_size
    box = np.clip(box, 0, 1)
    
    # Step 3: NMS using class-agnostic approach
    # (simpler and works well for multi-task)
    if len(conf) > 0:
        keep_nms = nms(box, conf, nms_thresh)
        conf = conf[keep_nms]
        obj = obj[keep_nms]
        act = act[keep_nms]
        box = box[keep_nms]
    
    # Build detections
    detections = []
    for i in range(len(conf)):
        obj_idx = int(np.argmax(obj[i]))
        obj_score = obj[i, obj_idx]
        combined = float(np.sqrt(conf[i] * obj_score))
        
        det = {
            'box': box[i],
            'conf': combined,
            'obj_idx': obj_idx,
            'obj_name': OBJECT_CLASSES[obj_idx],
            'actions': []
        }
        
        # Get actions for person
        if obj_idx == 0:
            top_acts = np.argsort(-act[i])[:3]
            for act_idx in top_acts:
                if act[i, act_idx] > 0.3 and act_idx < len(ACTION_CLASSES):
                    det['actions'].append((ACTION_CLASSES[act_idx], float(act[i, act_idx])))
        
        detections.append(det)
    
    return detections


def draw_detections(frame, detections):
    """Draw detections on frame."""
    h, w = frame.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)
        
        is_person = (det['obj_idx'] == 0)
        color = (0, 255, 0) if is_person else (0, 165, 255)
        
        cv2.rectangle(frame, (x1_px, y1_px), (x2_px, y2_px), color, 2)
        
        # Label
        label = f"{det['obj_name']}: {det['conf']:.2f}"
        cv2.putText(frame, label, (x1_px, y1_px - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Actions
        for i, (action, score) in enumerate(det['actions']):
            cv2.putText(frame, f"  {action}: {score:.2f}", 
                        (x1_px, y1_px + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame


# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)

# Global state
class State:
    engine = None
    transform = None
    camera = None
    frame_buffer = None
    latest_frame = None
    fps = 0
    running = True
    len_clip = 16
    sample_rate = 5

state = State()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOWO Live Demo</title>
    <style>
        body { 
            background: #1a1a2e; 
            color: #eee; 
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0; padding: 20px;
            text-align: center;
        }
        h1 { color: #4ecca3; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; }
        .video-container {
            display: inline-block;
            background: #16213e;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        img { 
            border-radius: 5px;
            max-width: 100%;
        }
        .stats {
            margin-top: 15px;
            font-size: 14px;
            color: #4ecca3;
        }
        .legend {
            margin-top: 20px;
            text-align: left;
            display: inline-block;
        }
        .legend-item { margin: 5px 0; }
        .box { display: inline-block; width: 20px; height: 20px; margin-right: 10px; vertical-align: middle; }
        .person { background: #00ff00; }
        .object { background: #ffa500; }
    </style>
</head>
<body>
    <h1>🎬 YOWO Multi-Task Demo</h1>
    <p class="subtitle">Real-time Action Detection on Orin Nano (TensorRT FP16)</p>
    
    <div class="video-container">
        <img src="/video_feed" alt="Video Stream">
    </div>
    
    <div class="stats">
        Model: X3D-M + YOLO11m | Clip: 16 frames @ 5 sample rate
    </div>
    
    <div class="legend">
        <div class="legend-item"><span class="box person"></span> Person (with actions)</div>
        <div class="legend-item"><span class="box object"></span> Objects</div>
    </div>
    
    <p style="margin-top: 30px; color: #666;">
        Press Ctrl+C on server to stop
    </p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


def generate_frames():
    """Generate MJPEG frames for streaming."""
    while state.running:
        if state.latest_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', state.latest_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30fps max stream rate


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def inference_loop():
    """
    Main inference loop - matches training configuration.
    
    Training: clip_length=16, sample_rate=5
    - Buffer holds 80 frames (16 * 5)
    - We sample every 5th frame to get 16 frames
    - We run inference every 5 new frames (not every frame)
    
    At 30fps camera: 6 inferences per second, each seeing 2.67s of context
    """
    buffer_size = state.len_clip * state.sample_rate  # 16 * 5 = 80
    state.frame_buffer = deque(maxlen=buffer_size)
    
    fps_times = deque(maxlen=30)
    frame_count = 0
    last_detections = []
    
    print(f"Starting inference loop...")
    print(f"  Buffer: {buffer_size} frames")
    print(f"  Inference every: {state.sample_rate} frames")
    print(f"  Clip length: {state.len_clip} frames sampled")
    
    while state.running:
        ret, frame = state.camera.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        # Convert to PIL and add to buffer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        state.frame_buffer.append(frame_pil)
        frame_count += 1
        
        # Wait for buffer to fill
        if len(state.frame_buffer) < buffer_size:
            cv2.putText(frame, f"Buffering: {len(state.frame_buffer)}/{buffer_size}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            state.latest_frame = frame
            continue
        
        # Only run inference every sample_rate frames
        # This matches training where each sampled position shifts by 1
        if frame_count % state.sample_rate == 0:
            t_start = time.time()
            
            # Sample clip: take every sample_rate-th frame from buffer
            clip = [state.frame_buffer[i * state.sample_rate] for i in range(state.len_clip)]
            
            # Transform
            x, _ = state.transform(clip)
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).numpy()
            
            # Inference
            output = state.engine.infer(x)
            
            # Post-process
            last_detections = post_process(output)
            
            # Track FPS
            fps_times.append(time.time() - t_start)
        
        # Draw detections (use last detections for frames between inferences)
        frame = draw_detections(frame, last_detections)
        
        # FPS display (inference FPS, not camera FPS)
        if fps_times:
            inf_fps = 1.0 / np.mean(fps_times)
            cv2.putText(frame, f"Inf: {inf_fps:.1f} FPS | Dets: {len(last_detections)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        state.latest_frame = frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True, help='TensorRT engine path')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    args = parser.parse_args()
    
    print("="*60)
    print("YOWO TensorRT Web Demo")
    print("="*60)
    
    # Load engine
    state.engine = TRTInference(args.engine)
    state.transform = BaseTransform(img_size=224)
    
    # Open camera with reduced resolution for faster processing
    print(f"Opening camera {args.camera}...")
    state.camera = cv2.VideoCapture(args.camera)
    if not state.camera.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    # Force 640x480 for faster frame reads (instead of 1920x1080)
    state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    state.camera.set(cv2.CAP_PROP_FPS, 30)
    
    w = int(state.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(state.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")
    
    # Start inference thread
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    inference_thread.start()
    
    # Get local IP
    import socket
    hostname = socket.gethostname()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print(f"\n{'='*60}")
    print(f"Open in browser: http://{local_ip}:{args.port}")
    print(f"{'='*60}\n")
    
    # Run Flask
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        state.camera.release()
        cuda_ctx.pop()
        print("\nDemo stopped.")


if __name__ == '__main__':
    main()
