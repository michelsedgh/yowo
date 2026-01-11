#!/usr/bin/env python3
"""
Quick Evaluation Script with Readable Output

Usage:
    python eval_readable.py -v yowo_v2_resnext_yolo11m_multitask --resume weights/best.pth

This gives you CLEAR, UNDERSTANDABLE metrics instead of mAP.
"""

import argparse
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.dataset_config import dataset_config
from config.yowo_v2_config import yowo_v2_config
from evaluator.readable_evaluator import build_readable_evaluator
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Readable Evaluation for YOWO Multi-Task')
    
    # Model
    parser.add_argument('-v', '--version', default='yowo_v2_resnext_yolo11m_multitask',
                        help='Model version')
    parser.add_argument('--resume', '-r', required=True, type=str,
                        help='Path to checkpoint')
    
    # Data
    parser.add_argument('--root', default='data/', type=str,
                        help='Data root (contains ActionGenome/)')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='Clip length')
    
    # Eval settings
    parser.add_argument('--max_samples', default=None, type=int,
                        help='Max samples to evaluate (for quick testing)')
    parser.add_argument('--conf_thresh', default=0.3, type=float,
                        help='Confidence threshold')
    parser.add_argument('--test_batch_size', default=8, type=int,
                        help='Test batch size')
    
    # Device
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA')
    
    return parser.parse_args()


def collate_fn(batch):
    """Collate function for dataloader."""
    frame_ids, video_clips, targets = zip(*batch)
    video_clips = torch.stack(video_clips)
    return frame_ids, video_clips, targets


def main():
    args = parse_args()
    
    print("="*70)
    print("🔍 READABLE MODEL EVALUATION")
    print("="*70)
    print(f"Model: {args.version}")
    print(f"Checkpoint: {args.resume}")
    print("="*70)
    
    # Device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Config
    d_cfg = dataset_config['charades_ag']
    m_cfg = yowo_v2_config[args.version]
    
    # Dummy args for model building
    class ModelArgs:
        def __init__(self):
            self.conf_thresh = args.conf_thresh
            self.nms_thresh = 0.5
            self.topk = 50
            self.center_sampling_radius = 2.5
            self.topk_candicate = 10
            self.loss_conf_weight = 1.0
            self.loss_cls_weight = 1.0
            self.loss_reg_weight = 5.0
    
    model_args = ModelArgs()
    
    # Build model
    print("\n📦 Loading model...")
    model, _ = build_model(
        args=model_args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=d_cfg['valid_num_classes'],
        trainable=False,
        resume=args.resume
    )
    model = model.to(device)
    model.eval()
    
    # Build evaluator
    print("\n📊 Building evaluator...")
    evaluator = build_readable_evaluator(
        args=args,
        d_cfg=d_cfg,
        img_size=d_cfg['test_size'],
        sampling_rate=d_cfg['sampling_rate'],
        collate_fn=collate_fn
    )
    
    # Run evaluation
    print("\n🚀 Running evaluation...")
    results = evaluator.evaluate(
        model=model,
        epoch=1,
        max_samples=args.max_samples,
        verbose=True
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
