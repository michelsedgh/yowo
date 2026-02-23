import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import sys
import time
import argparse
import torch
import torch.backends.cudnn as cudnn

from config import build_dataset_config, build_model_config
from models import build_model
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer

# Pull the parser from train.py to ensure identical environment
from train import parse_args

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Device: {device}")
    
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)
    
    print("Building dataset...")
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=True)
    dataloader = build_dataloader(args, dataset, args.batch_size, CollateFunc(), is_train=True)
    
    print("Building model...")
    model, criterion = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=True,
        resume=None
    )
    model = model.to(device)
    model.train()
    
    print("Building optimizer...")
    optimizer, _ = build_optimizer(d_cfg, model, args.base_lr, None)
    
    print("\n[TEST] Grabbing a single batch...")
    batch = next(iter(dataloader))
    frame_ids, video_clips, targets = batch
    
    video_clips = video_clips.to(device)
    
    print("\n[TEST] Starting 100-Iteration Overfit Sequence...")
    for iter_i in range(101):
        outputs = model(video_clips)
        loss_dict = criterion(outputs, targets)
        losses = loss_dict['losses']
        
        if torch.isnan(losses):
            print("ERROR: Loss exploded to NaN! The mathematical logic failed.")
            return
            
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        if iter_i % 10 == 0:
            print(f"Iter {iter_i:3d}: Total={losses.item():.4f} | " + 
                  f"Obj={loss_dict.get('loss_obj', 0):.4f} | " +
                  f"Rel={loss_dict.get('loss_rel', 0):.4f} | " +
                  f"Act={loss_dict.get('loss_act', 0):.4f} | " +
                  f"Box={loss_dict.get('loss_box', 0):.4f} | " +
                  f"Conf={loss_dict.get('loss_conf', 0):.4f}")
            
    print("\n[✅ PASS] Overfit test completed! If the total loss is near 0.00, your logic is flawless.")

if __name__ == '__main__':
    main()
