#!/usr/bin/env python
"""
Charades-AG Evaluation Script

Evaluates a trained YOWO multi-task model on Charades-AG test set.

Usage:
    python eval_charades_ag.py \
        -v yowo_v2_x3d_m_yolo11m_multitask \
        --weight path/to/checkpoint.pth \
        --root ./data \
        -K 16 \
        --cuda

For quick testing (first 100 samples):
    python eval_charades_ag.py ... --max_samples 100
"""

import argparse
import torch
import os

from evaluator.charades_ag_evaluator import CharadesAGEvaluator
from dataset.transforms import BaseTransform
from utils.misc import load_weight, CollateFunc
from config import build_dataset_config, build_model_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Charades-AG Evaluation')
    
    # Basic
    parser.add_argument('-bs', '--batch_size', default=8, type=int,
                        help='test batch size')
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--save_path', default='./evaluator/eval_results/',
                        type=str, help='Path to save evaluation results')
    
    # Dataset
    parser.add_argument('-d', '--dataset', default='charades_ag',
                        help='dataset name')
    parser.add_argument('--root', default='./data',
                        help='data root directory')
    
    # Model
    parser.add_argument('-v', '--version', default='yowo_v2_x3d_m_yolo11m_multitask',
                        type=str, help='model version')
    parser.add_argument('--weight', default=None, type=str,
                        help='path to trained checkpoint')
    parser.add_argument('-ct', '--conf_thresh', default=0.01, type=float,
                        help='confidence threshold for detections')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk prediction candidates')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length')
    parser.add_argument('-m', '--memory', action='store_true', default=False,
                        help='memory propagation (not used)')
    
    # Evaluation
    parser.add_argument('--iou_thresh', default=0.5, type=float,
                        help='IoU threshold for mAP calculation')
    parser.add_argument('--max_samples', default=None, type=int,
                        help='max samples to evaluate (for quick testing)')
    parser.add_argument('--epoch', default=1, type=int,
                        help='epoch number (for logging)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Dataset must be charades_ag
    assert args.dataset == 'charades_ag', "This script only supports charades_ag dataset"
    
    # Device
    if args.cuda and torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device("cuda")
    else:
        print('Using CPU')
        device = torch.device("cpu")
    
    # Config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)
    
    # Number of classes for multi-task model
    num_objects = d_cfg.get('num_objects', 36)
    num_actions = d_cfg.get('num_actions', 157)
    num_relations = d_cfg.get('num_relations', 26)
    num_classes = num_objects + num_actions + num_relations
    
    print(f"\n{'='*60}")
    print(f"Charades-AG Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.version}")
    print(f"Weight: {args.weight}")
    print(f"Device: {device}")
    print(f"Classes: {num_objects} objects, {num_actions} actions, {num_relations} relations")
    print(f"{'='*60}\n")
    
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
    if args.weight is not None:
        model = load_weight(model=model, path_to_ckpt=args.weight)
        print(f"Loaded weights from: {args.weight}")
    else:
        print("WARNING: No weights loaded! Using random initialization.")
    
    # Move to device and set eval mode
    model = model.to(device).eval()
    
    # Transform
    transform = BaseTransform(img_size=args.img_size)
    
    # Collate function
    collate_fn = CollateFunc()
    
    # Sampling rate from config
    sampling_rate = d_cfg.get('sampling_rate', 5)
    
    # Create evaluator
    evaluator = CharadesAGEvaluator(
        d_cfg=d_cfg,
        data_root=args.root,
        img_size=args.img_size,
        len_clip=args.len_clip,
        sampling_rate=sampling_rate,
        batch_size=args.batch_size,
        transform=transform,
        collate_fn=collate_fn,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        save_path=args.save_path
    )
    
    # Run evaluation
    results = evaluator.evaluate_frame_map(
        model=model,
        epoch=args.epoch,
        max_samples=args.max_samples
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Object mAP:   {results['object_mAP']*100:.2f}%")
    print(f"Action mAP:   {results['action_mAP']*100:.2f}%")
    print(f"Relation mAP: {results['relation_mAP']*100:.2f}%")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()
