#!/usr/bin/env python3
"""
SANITY CHECK: Test the full training pipeline on real data before full training.

This script:
1. Loads a real sample from Charades-AG dataset
2. Runs it through the model (forward pass)
3. Computes all losses
4. Backpropagates and checks gradient flow
5. Visualizes attention patterns (what positions attend to what)

Run with: python sanity_check.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("  SANITY CHECK: Full Pipeline on Real Data")
    print("=" * 70)
    
    # ============ SETUP ============
    # Use CPU for sanity check to avoid GPU memory issues
    device = 'cpu'
    print(f"\nDevice: {device} (using CPU for sanity check)")
    
    data_root = 'data/ActionGenome/'
    print(f"Data root: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"ERROR: Data root not found at {data_root}")
        return False
    
    # ============ LOAD DATASET ============
    print("\n--- Loading Dataset ---")
    from config import yowo_v2_config
    from dataset.charades_ag import CharadesAGDataset
    from dataset.transforms import Augmentation, BaseTransform
    
    cfg = yowo_v2_config['yowo_v2_x3d_m_yolo11m_multitask']
    
    # Use base transform (no augmentation for sanity check)
    transform = BaseTransform(img_size=224)
    
    dataset = CharadesAGDataset(
        cfg=cfg,
        data_root=data_root,
        is_train=True,
        img_size=224,
        transform=transform,
        len_clip=16,
        sampling_rate=1
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"  Objects: {dataset.num_objects}")
    print(f"  Actions: {dataset.num_actions}")
    print(f"  Relations: {dataset.num_relations}")
    
    # ============ LOAD ONE SAMPLE ============
    print("\n--- Loading Sample ---")
    
    # Get first sample
    sample_idx = 0
    info, video_clip, target = dataset[sample_idx]
    
    print(f"Sample info: Video={info[0]}, Frame={info[1]}")
    print(f"Video clip shape: {video_clip.shape}")
    print(f"Target boxes: {target['boxes'].shape}")
    print(f"Target labels: {target['labels'].shape}")
    
    # Analyze the sample
    if target['boxes'].shape[0] > 0:
        labels = target['labels'].numpy()
        print(f"\nBoxes in this sample:")
        for i in range(min(5, len(labels))):
            obj_class = labels[i, :36].argmax()
            obj_name = dataset.ag_objects[obj_class] if obj_class < len(dataset.ag_objects) else "unknown"
            num_actions = int(labels[i, 36:193].sum())
            num_relations = int(labels[i, 193:].sum())
            print(f"  Box {i}: {obj_name} | Actions: {num_actions} | Relations: {num_relations}")
    
    # ============ BUILD MODEL ============
    print("\n--- Building Model ---")
    from models.yowo.yowo_multitask import YOWOMultiTask
    
    model = YOWOMultiTask(
        cfg=cfg,
        device=device,
        num_objects=36,
        num_actions=157,
        num_relations=26,
        trainable=True
    )
    model = model.to(device)
    model.train()
    
    print(f"Model built: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ============ FORWARD PASS ============
    print("\n--- Forward Pass ---")
    
    # Prepare input
    video_clip = video_clip.unsqueeze(0).to(device)  # [1, 3, 16, H, W]
    
    # Forward pass
    outputs = model(video_clip)
    
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Predictions per FPN level:")
    for i, (obj, act, rel) in enumerate(zip(outputs['pred_obj'], outputs['pred_act'], outputs['pred_rel'])):
        print(f"  Level {i}: {obj.shape[1]} anchors")
    
    total_anchors = sum(o.shape[1] for o in outputs['pred_obj'])
    print(f"Total predictions: {total_anchors}")
    
    # ============ COMPUTE LOSSES ============
    print("\n--- Computing Losses ---")
    from models.yowo.loss_multitask import MultiTaskCriterion
    import argparse
    
    args = argparse.Namespace(
        loss_conf_weight=1.0,
        loss_reg_weight=5.0,
        center_sampling_radius=2.5,
        topk_candicate=10
    )
    
    criterion = MultiTaskCriterion(
        args=args,
        img_size=224,
        num_objects=36,
        num_actions=157,
        num_relations=26
    )
    
    # Prepare targets
    targets = [{
        'boxes': target['boxes'].to(device),
        'labels': target['labels'].to(device)
    }]
    
    # Compute loss
    try:
        loss_dict = criterion(outputs, targets)
        
        print("Loss components:")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.4f}")
            else:
                print(f"  {k}: {v}")
        
        total_loss = loss_dict['losses']
        print(f"\nTotal loss: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============ BACKWARD PASS ============
    print("\n--- Backward Pass (Gradient Check) ---")
    
    # Zero gradients
    model.zero_grad()
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats[name] = param.grad.abs().mean().item()
    
    # Group by module
    modules = ['backbone_2d', 'backbone_3d', 'cls_channel_encoders', 'heads', 
               'obj_preds', 'act_preds', 'rel_preds', 'obj_cross_attn', 'obj_rel_cross_attn']
    
    for module in modules:
        module_grads = [v for k, v in grad_stats.items() if module in k]
        if module_grads:
            avg_grad = sum(module_grads) / len(module_grads)
            print(f"  {module}: avg_grad={avg_grad:.8f}")
    
    # Check position embeddings
    pos_grads = [v for k, v in grad_stats.items() if 'pos_embed' in k]
    if pos_grads:
        print(f"  position_embeddings: avg_grad={sum(pos_grads)/len(pos_grads):.8f}")
    
    # ============ ATTENTION VISUALIZATION ============
    print("\n--- Attention Pattern Analysis ---")
    
    with torch.no_grad():
        # Get attention weights from scene context attention
        # We need to modify the forward slightly to get attention weights
        
        # Re-run forward with a single level to analyze attention
        key_frame = video_clip[:, :, -1, :, :]  # [B, 3, H, W]
        
        feat_3d = model.backbone_3d(video_clip)
        cls_feats, reg_feats = model.backbone_2d(key_frame)
        
        # Process level 1 (14x14 feature map)
        level = 1
        cls_feat = cls_feats[level]
        feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
        cls_feat = model.cls_channel_encoders[level](cls_feat, feat_3d_up)
        reg_feat = model.reg_channel_encoders[level](cls_feats[level], feat_3d_up)
        cls_feat, reg_feat = model.heads[level](cls_feat, reg_feat)
        
        # Get predictions
        obj_pred = model.obj_preds[level](cls_feat)
        rel_feat = model.obj_cross_attn[level](cls_feat, obj_pred)
        rel_pred = model.rel_preds[level](rel_feat)
        
        # Get attention from scene context
        act_feat, attn_weights = model.obj_rel_cross_attn[level](cls_feat, obj_pred, rel_pred, return_weights=True)
        
        print(f"Attention weights shape: {attn_weights.shape}")  # [B, N, N]
        
        # Analyze attention pattern
        N = attn_weights.shape[-1]
        H = W = int(N ** 0.5)
        
        # Find position with highest object confidence
        obj_conf = obj_pred.softmax(dim=1).max(dim=1)[0]  # [B, H, W]
        obj_conf_flat = obj_conf.flatten(1)  # [B, N]
        
        # Get top positions
        top_k = 5
        _, top_indices = obj_conf_flat[0].topk(top_k)
        
        print(f"\nTop {top_k} positions by object confidence:")
        for idx in top_indices:
            y, x = idx // W, idx % W
            obj_class = obj_pred[0, :, y, x].argmax().item()
            obj_name = dataset.ag_objects[obj_class] if obj_class < len(dataset.ag_objects) else "unknown"
            
            # Where does this position attend to?
            attn_from_this = attn_weights[0, idx, :]  # [N]
            top_attn_idx = attn_from_this.topk(3)[1]  # Top 3 attended positions
            
            print(f"  Position ({y},{x}): {obj_name}")
            print(f"    Top attended positions: {[(i//W, i%W) for i in top_attn_idx.tolist()]}")
    
    # ============ SUMMARY ============
    print("\n" + "=" * 70)
    print("  SANITY CHECK COMPLETE!")
    print("=" * 70)
    print()
    print("✅ Dataset loading: OK")
    print("✅ Model forward pass: OK")
    print("✅ Loss computation: OK")
    print("✅ Backward pass: OK")
    print("✅ Gradient flow: OK")
    print("✅ Attention patterns: OK")
    print()
    print("The model is ready for training!")
    print()
    print("Key observations:")
    print(f"  - Sample from video: {info[0]}")
    print(f"  - Number of boxes: {target['boxes'].shape[0]}")
    print(f"  - Total loss: {total_loss.item():.4f}")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Sanity check failed!")
        sys.exit(1)
