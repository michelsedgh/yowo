#!/usr/bin/env python3
"""
YOWO TensorRT Inference Demo

Runs the TensorRT-optimized YOWO model on webcam or video file.
Outputs to video file (no display required).

Usage:
    # Run on webcam with FP16 engine
    python demo_tensorrt.py --engine yowo_v2_x3d_m_yolo11m_multitask_epoch_5_fp16.engine --camera 0
    
    # Run on video file
    python demo_tensorrt.py --engine yowo_v2_x3d_m_yolo11m_multitask_epoch_5_fp16.engine --video input.mp4
"""

import argparse
import os
import time
import numpy as np
import cv2
from PIL import Image
from collections import deque

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from dataset.transforms import BaseTransform
from utils.nms import multiclass_nms


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
    'Snuggling w/ blanket', 'Taking blanket', 'Throwing blanket', 'Tidying blanket',
    'Holding pillow', 'Putting pillow', 'Snuggling w/ pillow', 'Taking pillow',
    'Throwing pillow', 'Put on shelf', 'Tidying shelf', 'Grabbing picture',
    'Holding picture', 'Laughing at picture', 'Putting picture', 'Taking picture',
    'Looking at picture', 'Closing window', 'Opening window', 'Washing window',
    'Looking out window', 'Holding mirror', 'Smiling in mirror', 'Washing mirror',
    'Looking in mirror', 'Walking thru doorway', 'Holding broom', 'Putting broom',
    'Taking broom', 'Throwing broom', 'Tidying w/ broom', 'Fixing light',
    'Turning on light', 'Turning off light', 'Drinking from cup', 'Holding cup',
    'Pouring into cup', 'Putting cup', 'Taking cup', 'Washing cup',
    'Closing closet', 'Opening closet', 'Tidying closet', 'Holding paper',
    'Putting paper', 'Taking paper', 'Holding dish', 'Putting dish', 'Taking dish',
    'Washing dish', 'Lying on sofa', 'Sitting on sofa', 'Lying on floor',
    'Sitting on floor', 'Throwing on floor', 'Tidying floor', 'Holding medicine',
    'Taking medicine', 'Putting groceries', 'Laughing at TV', 'Watching TV',
    'Awakening in bed', 'Lying on bed', 'Sitting in bed', 'Fixing vacuum',
    'Holding vacuum', 'Taking vacuum', 'Washing hands', 'Fixing doorknob',
    'Grasping doorknob', 'Closing fridge', 'Opening fridge', 'Fixing hair',
    'Working on paper', 'Awakening', 'Cooking', 'Dressing', 'Laughing',
    'Running', 'Sit down', 'Smiling', 'Sneezing', 'Standing up', 'Undressing', 'Eating'
]

RELATION_CLASSES = [
    'lookingat', 'notlookingat', 'unsure', 'above', 'beneath', 'infrontof',
    'behind', 'onthesideof', 'in', 'carrying', 'coveredby', 'drinkingfrom',
    'eating', 'haveitontheback', 'holding', 'leaningon', 'lyingon', 'notcontacting',
    'otherrelationship', 'sittingon', 'standingon', 'touching', 'twisting',
    'wearing', 'wiping', 'writingon'
]


class TensorRTInference:
    """TensorRT inference engine wrapper."""
    
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name} {self.output_shape}")
        
        # Allocate buffers
        self.input_size = int(np.prod(self.input_shape)) * 4
        self.output_size = int(np.prod(self.output_shape)) * 4
        
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.h_output = np.empty(self.output_shape, dtype=np.float32)
        
        self.stream = cuda.Stream()
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
    
    def infer(self, input_data):
        """Run inference on input data."""
        # Ensure correct dtype
        input_data = input_data.astype(np.float32)
        
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # Execute
        self.context.execute_async_v3(self.stream.handle)
        
        # Copy output to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        return self.h_output.copy()


def post_process(predictions, img_size, conf_thresh=0.1, nms_thresh=0.5):
    """
    Post-process TensorRT output.
    
    predictions: [1, N, 225] where 225 = [conf(1), obj(36), act(157), rel(26), box(4), interact(1)]
    """
    preds = predictions[0]  # Remove batch dim: [N, 225]
    
    # Parse predictions
    conf = preds[:, 0:1]           # [N, 1]
    obj = preds[:, 1:37]           # [N, 36]
    act = preds[:, 37:194]         # [N, 157]
    rel = preds[:, 194:220]        # [N, 26]
    box = preds[:, 220:224]        # [N, 4]
    interact = preds[:, 224:225]   # [N, 1]
    
    # Apply activations
    conf = 1 / (1 + np.exp(-conf))  # Sigmoid
    obj = np.exp(obj) / np.sum(np.exp(obj), axis=-1, keepdims=True)  # Softmax
    act = 1 / (1 + np.exp(-act))    # Sigmoid
    rel = 1 / (1 + np.exp(-rel))    # Sigmoid
    interact = 1 / (1 + np.exp(-interact))  # Sigmoid
    
    # Confidence filtering
    conf_flat = conf.flatten()
    keep = conf_flat > conf_thresh
    
    if not np.any(keep):
        return np.zeros((0, 225))
    
    conf = conf[keep]
    obj = obj[keep]
    act = act[keep]
    rel = rel[keep]
    box = box[keep]
    interact = interact[keep]
    
    # Normalize boxes
    box = box / img_size
    box = np.clip(box, 0, 1)
    
    # NMS using object class
    obj_labels = np.argmax(obj, axis=-1)
    scores, labels, box = multiclass_nms(
        conf.flatten(), obj_labels, box, nms_thresh, 36, False)
    
    # Combine output
    if len(box) > 0:
        # Reapply filtering to other predictions
        keep_idx = np.arange(len(obj))[:len(box)]
        out = np.concatenate([
            box,                          # [0:4]
            scores[:, None],              # [4]
            interact[keep_idx],           # [5]
            obj[keep_idx],                # [6:42]
            act[keep_idx],                # [42:199]
            rel[keep_idx],                # [199:225]
        ], axis=-1)
    else:
        out = np.zeros((0, 225))
    
    return out


def visualize(frame, detections, vis_thresh=0.3, max_actions=3):
    """Visualize detections on frame."""
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        conf = det[4]
        interact = det[5]
        obj_probs = det[6:42]
        act_probs = det[42:199]
        rel_probs = det[199:225]
        
        h, w = frame.shape[:2]
        x1_px = int(x1 * w)
        y1_px = int(y1 * h)
        x2_px = int(x2 * w)
        y2_px = int(y2 * h)
        
        obj_idx = int(np.argmax(obj_probs))
        obj_name = OBJECT_CLASSES[obj_idx]
        obj_score = obj_probs[obj_idx]
        
        combined = np.sqrt(conf * obj_score)
        if combined < vis_thresh:
            continue
        
        is_person = (obj_idx == 0)
        color = (0, 255, 0) if is_person else (255, 165, 0)
        
        cv2.rectangle(frame, (x1_px, y1_px), (x2_px, y2_px), color, 2)
        
        # Labels
        labels = [f"{obj_name}: {combined:.2f}"]
        
        if is_person:
            top_acts = np.argsort(-act_probs)[:max_actions]
            for act_idx in top_acts:
                if act_probs[act_idx] > vis_thresh and act_idx < len(ACTION_CLASSES):
                    labels.append(f"  {ACTION_CLASSES[act_idx]}: {act_probs[act_idx]:.2f}")
        
        # Draw labels
        for i, label in enumerate(labels):
            y = y1_px - 5 - i * 18
            if y < 15:
                y = y2_px + 15 + i * 18
            cv2.putText(frame, label, (x1_px, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO TensorRT Demo')
    
    parser.add_argument('--engine', required=True, type=str,
                        help='Path to TensorRT engine')
    parser.add_argument('--camera', default=None, type=int,
                        help='Camera device ID')
    parser.add_argument('--video', default=None, type=str,
                        help='Input video path')
    parser.add_argument('--output', default='output_tensorrt.mp4', type=str,
                        help='Output video path')
    
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--len_clip', default=16, type=int)
    parser.add_argument('--sample_rate', default=5, type=int)
    
    parser.add_argument('--conf_thresh', default=0.1, type=float)
    parser.add_argument('--vis_thresh', default=0.3, type=float)
    parser.add_argument('--nms_thresh', default=0.5, type=float)
    
    parser.add_argument('--max_frames', default=500, type=int,
                        help='Max frames to process (0 for unlimited)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("YOWO TensorRT Demo")
    print("="*60)
    
    # Load engine
    trt_engine = TensorRTInference(args.engine)
    
    # Input source
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        print(f"Using camera: {args.camera}")
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Using video: {args.video}")
    else:
        print("Error: Specify --camera or --video")
        return
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"Input: {width}x{height} @ {fps:.1f} FPS")
    
    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"Output: {args.output}")
    
    # Transform
    transform = BaseTransform(img_size=args.img_size)
    
    # Frame buffer
    buffer_size = args.len_clip * args.sample_rate
    frame_buffer = deque(maxlen=buffer_size)
    
    # Stats
    frame_count = 0
    inference_times = []
    
    print(f"\nProcessing (clip={args.len_clip}, rate={args.sample_rate})...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
        
        frame_count += 1
        
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_buffer.append(frame_pil)
        
        # Wait for enough frames
        if len(frame_buffer) < buffer_size:
            out.write(frame)
            continue
        
        # Sample frames
        clip = [frame_buffer[i * args.sample_rate] for i in range(args.len_clip)]
        
        # Transform - returns list of torch tensors [T, 3, H, W]
        x, _ = transform(clip)
        # Stack: [T, 3, H, W] -> [3, T, H, W]
        import torch
        x = torch.stack(x, dim=1)  # [3, T, H, W]
        x = x.unsqueeze(0).numpy()  # [1, 3, T, H, W]
        
        # Inference
        t_start = time.perf_counter()
        output = trt_engine.infer(x)
        t_infer = (time.perf_counter() - t_start) * 1000
        inference_times.append(t_infer)
        
        # Post-process
        detections = post_process(output, args.img_size, 
                                  args.conf_thresh, args.nms_thresh)
        
        # Visualize
        frame = visualize(frame, detections, args.vis_thresh)
        
        # Add info overlay
        cv2.putText(frame, f"Frame: {frame_count} | Infer: {t_infer:.1f}ms", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        out.write(frame)
        
        if frame_count % 50 == 0:
            avg_time = np.mean(inference_times[-50:])
            print(f"  Frame {frame_count}: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
    
    cap.release()
    out.release()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Frames processed: {frame_count}")
    if inference_times:
        print(f"Inference time: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms")
        print(f"FPS: {1000 / np.mean(inference_times):.1f}")
    print(f"Output saved: {args.output}")


if __name__ == '__main__':
    main()
