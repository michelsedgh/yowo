#!/usr/bin/env python3
"""
Live Webcam Demo for YOWO Multi-Task Model (Charades AG)

This script runs the trained YOWO multi-task model on a live webcam feed.
It displays:
- Object detection (36 classes): What objects are detected
- Action detection (157 classes): What actions are being performed (person only)
- Relation detection (26 classes): How objects are being interacted with

Usage:
    conda activate yowov2
    python demo_live_charades_ag.py --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_5.pth --cuda

Requirements:
    - OpenCV with webcam support
    - PyTorch with CUDA (for Orin Nano)
    - Trained YOWO multi-task checkpoint
"""

import argparse
import cv2
import os
import time
import numpy as np
import torch
from PIL import Image
from collections import deque

from dataset.transforms import BaseTransform
from utils.misc import load_weight
from config import build_dataset_config, build_model_config
from models import build_model


# ============================================================================
# CLASS LABELS (from Action Genome + Charades)
# ============================================================================

OBJECT_CLASSES = [
    'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
    'closetcabinet', 'clothes', 'cupglassbottle', 'dish', 'door', 'doorknob',
    'doorway', 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
    'mirror', 'papernotebook', 'phonecamera', 'picture', 'pillow', 'refrigerator',
    'sandwich', 'shelf', 'shoe', 'sofacouch', 'table', 'television', 'towel',
    'vacuum', 'window'
]

RELATION_CLASSES = [
    'lookingat', 'notlookingat', 'unsure', 'above', 'beneath', 'infrontof',
    'behind', 'onthesideof', 'in', 'carrying', 'coveredby', 'drinkingfrom',
    'eating', 'haveitontheback', 'holding', 'leaningon', 'lyingon', 'notcontacting',
    'otherrelationship', 'sittingon', 'standingon', 'touching', 'twisting',
    'wearing', 'wiping', 'writingon'
]

# Simplified action names (removing prefixes like "Holding a", "Putting", etc.)
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


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO Live Webcam Demo (Charades AG)')

    # Basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='Input frame size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA')
    parser.add_argument('--camera', default=0, type=int,
                        help='Camera device ID')
    parser.add_argument('-vs', '--vis_thresh', default=0.35, type=float,
                        help='Threshold for visualization')

    # Model
    parser.add_argument('-v', '--version', default='yowo_v2_x3d_m_yolo11m_multitask', type=str,
                        help='Model version')
    parser.add_argument('-d', '--dataset', default='charades_ag', type=str,
                        help='Dataset config to use')
    parser.add_argument('--weight', default='yowo_v2_x3d_m_yolo11m_multitask_epoch_5.pth',
                        type=str, help='Path to trained weights')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='Confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='Top-k detections per scale')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='Video clip length')
    parser.add_argument('--sample_rate', default=5, type=int,
                        help='Frame sampling rate (matches training)')
    
    # Display options
    parser.add_argument('--show_fps', action='store_true', default=True,
                        help='Show FPS on screen')
    parser.add_argument('--max_actions', default=3, type=int,
                        help='Max actions to display per detection')
    parser.add_argument('--max_relations', default=2, type=int,
                        help='Max relations to display per detection')

    return parser.parse_args()


def visualize_multitask_detection(frame, out_bboxes, orig_w, orig_h, args):
    """
    Visualize multi-task detections on the frame.
    
    out_bboxes format: [x1, y1, x2, y2, conf, interact, obj_36, act_157, rel_26]
    - indices [0:4]: bbox coordinates (normalized 0-1)
    - index [4]: detection confidence
    - index [5]: interaction score
    - indices [6:42]: object class probabilities (36)
    - indices [42:199]: action class probabilities (157)
    - indices [199:225]: relation class probabilities (26)
    """
    for bbox in out_bboxes:
        # Parse bbox components
        x1, y1, x2, y2 = bbox[:4]
        det_conf = float(bbox[4])
        interact_score = float(bbox[5])
        
        # Get class probabilities
        obj_probs = bbox[6:42]  # 36 object classes
        act_probs = bbox[42:199]  # 157 action classes
        rel_probs = bbox[199:225]  # 26 relation classes
        
        # Rescale bbox to original size
        x1_px = int(x1 * orig_w)
        y1_px = int(y1 * orig_h)
        x2_px = int(x2 * orig_w)
        y2_px = int(y2 * orig_h)
        
        # Get top object (softmax, so just argmax)
        obj_idx = int(np.argmax(obj_probs))
        obj_name = OBJECT_CLASSES[obj_idx]
        obj_score = float(obj_probs[obj_idx])
        
        # Skip low confidence detections
        combined_score = np.sqrt(det_conf * obj_score)
        if combined_score < args.vis_thresh:
            continue
        
        # Color based on object type
        is_person = (obj_idx == 0)
        if is_person:
            color = (0, 255, 0)  # Green for person
        else:
            color = (255, 165, 0)  # Orange for objects
        
        # Draw bounding box
        cv2.rectangle(frame, (x1_px, y1_px), (x2_px, y2_px), color, 2)
        
        # Prepare text labels
        labels = []
        
        # Object label with confidence
        labels.append(f"{obj_name}: {combined_score:.2f}")
        
        # Actions (only for person detections)
        if is_person:
            # Get top actions above threshold
            act_indices = np.where(act_probs > args.vis_thresh)[0]
            act_scores = act_probs[act_indices]
            # Sort by score
            sorted_idx = np.argsort(-act_scores)[:args.max_actions]
            for idx in sorted_idx:
                act_idx = act_indices[idx]
                act_score = act_scores[idx]
                if act_idx < len(ACTION_CLASSES):
                    labels.append(f"  {ACTION_CLASSES[act_idx]}: {act_score:.2f}")
        
        # Relations (for all detections if interaction score is high)
        if interact_score > 0.3:
            rel_indices = np.where(rel_probs > args.vis_thresh)[0]
            rel_scores = rel_probs[rel_indices]
            sorted_idx = np.argsort(-rel_scores)[:args.max_relations]
            for idx in sorted_idx:
                rel_idx = rel_indices[idx]
                rel_score = rel_scores[idx]
                if rel_idx < len(RELATION_CLASSES):
                    labels.append(f"  [{RELATION_CLASSES[rel_idx]}]")
        
        # Draw labels with background
        y_offset = y1_px - 5
        for i, label in enumerate(labels):
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Text background
            text_y = y_offset - i * 18
            if text_y < 15:
                text_y = y2_px + 15 + i * 18  # Put below bbox if no space above
            
            cv2.rectangle(frame,
                          (x1_px, text_y - 12),
                          (x1_px + text_size[0] + 2, text_y + 2),
                          color, -1)
            cv2.putText(frame, label, (x1_px + 1, text_y - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


@torch.no_grad()
def run_live_demo(args, model, device, transform):
    """Run live webcam demo with the model."""
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return
    
    # Get camera properties
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {cam_width}x{cam_height}")
    
    # Frame buffer for temporal context
    # We sample every nth frame based on sample_rate
    # For len_clip=16 and sample_rate=5, we need 16 frames sampled at rate 5
    frame_buffer = deque(maxlen=args.len_clip * args.sample_rate)
    
    # FPS tracking
    fps_history = deque(maxlen=30)
    frame_count = 0
    
    print("\n" + "="*60)
    print("YOWO Multi-Task Live Demo")
    print("="*60)
    print(f"Model: {args.version}")
    print(f"Clip Length: {args.len_clip} | Sample Rate: {args.sample_rate}")
    print(f"Effective temporal span: {args.len_clip * args.sample_rate} frames")
    print(f"Press 'q' to quit")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        t_start = time.time()
        
        # Convert to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Add to buffer
        frame_buffer.append(frame_pil)
        frame_count += 1
        
        # Wait until we have enough frames
        if len(frame_buffer) < args.len_clip * args.sample_rate:
            # Show "Buffering..." message
            cv2.putText(frame, f"Buffering: {len(frame_buffer)}/{args.len_clip * args.sample_rate}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('YOWO Multi-Task Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Sample frames from buffer at the correct rate
        # Get frames at indices: 0, sample_rate, 2*sample_rate, ..., (len_clip-1)*sample_rate
        video_clip = [frame_buffer[i * args.sample_rate] for i in range(args.len_clip)]
        
        # Get original size
        orig_h, orig_w = frame.shape[:2]
        
        # Transform clip
        x, _ = transform(video_clip)
        # List [T, 3, H, W] -> [3, T, H, W]
        x = torch.stack(x, dim=1)
        x = x.unsqueeze(0).to(device)  # [B, 3, T, H, W], B=1
        
        # Inference
        t_infer = time.time()
        outputs = model(x)
        infer_time = time.time() - t_infer
        
        # Get detections (batch size = 1)
        if isinstance(outputs, list):
            bboxes = outputs[0]  # First (and only) batch
        else:
            bboxes = outputs
        
        # Visualize detections
        frame = visualize_multitask_detection(
            frame, bboxes, orig_w, orig_h, args)
        
        # FPS calculation
        total_time = time.time() - t_start
        fps_history.append(1.0 / total_time if total_time > 0 else 0)
        avg_fps = np.mean(fps_history)
        
        # Display FPS and inference time
        if args.show_fps:
            info_text = f"FPS: {avg_fps:.1f} | Inference: {infer_time*1000:.1f}ms"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Detection count
            det_count = len(bboxes) if isinstance(bboxes, np.ndarray) else 0
            cv2.putText(frame, f"Detections: {det_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('YOWO Multi-Task Demo', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            args.vis_thresh = min(0.9, args.vis_thresh + 0.05)
            print(f"Visualization threshold: {args.vis_thresh:.2f}")
        elif key == ord('-'):
            args.vis_thresh = max(0.1, args.vis_thresh - 0.05)
            print(f"Visualization threshold: {args.vis_thresh:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


if __name__ == '__main__':
    args = parse_args()
    
    # Set device
    if args.cuda and torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    
    # Build configs
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)
    
    # Verify this is a multi-task setup
    print(f"\nDataset: {args.dataset}")
    print(f"  Objects: {d_cfg.get('num_objects', 36)}")
    print(f"  Actions: {d_cfg.get('num_actions', 157)}")
    print(f"  Relations: {d_cfg.get('num_relations', 26)}")
    
    # Calculate total classes for compatibility
    num_classes = d_cfg.get('valid_num_classes', 219)
    
    # Transform
    basetransform = BaseTransform(img_size=args.img_size)
    
    # Build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=False
    )
    
    # Load weights
    print(f"\nLoading weights from: {args.weight}")
    model = load_weight(model=model, path_to_ckpt=args.weight)
    
    # Move to device and set eval mode
    model = model.to(device).eval()
    print("Model loaded successfully!")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Run demo
    run_live_demo(
        args=args,
        model=model,
        device=device,
        transform=basetransform
    )
