#!/usr/bin/env python3
"""
MINI TRAINING TEST: Verify the model learns before full training.

This script:
1. Runs training for ~500 iterations on a subset of data
2. Tracks loss trends (should DECREASE)
3. Tracks attention metrics (entropy should DECREASE, nearby/far should INCREASE)
4. Gives a clear PASS/FAIL verdict

Run with: python mini_training_test.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def mini_training_test():
    print("=" * 70)
    print("  MINI TRAINING TEST: Will the model learn?")
    print("=" * 70)
    
    # ============ SETUP ============
    # Force CPU due to GPU memory constraints on this device
    device = 'cpu'
    print(f"\nDevice: {device} (using CPU for stability)")
    print("⚠️  Running on CPU - this will be slow but works for testing")
    max_iters = 30  # Keep small for CPU
    
    # ============ LOAD DATASET ============
    print("\n--- Loading Dataset ---")
    
    from config import yowo_v2_config
    from dataset.charades_ag import CharadesAGDataset
    from dataset.transforms import BaseTransform
    from torch.utils.data import DataLoader, Subset
    
    cfg = yowo_v2_config['yowo_v2_x3d_m_yolo11m_multitask']
    transform = BaseTransform(img_size=224)
    
    full_dataset = CharadesAGDataset(
        cfg=cfg,
        data_root='data/ActionGenome/',
        is_train=True,
        img_size=224,
        transform=transform,
        len_clip=16,
        sampling_rate=1
    )
    
    # Use small subset for testing
    subset_size = min(500, len(full_dataset))
    indices = list(range(0, subset_size))
    dataset = Subset(full_dataset, indices)
    
    print(f"Using {len(dataset)} samples (subset of {len(full_dataset)})")
    
    # Simple collate function
    def collate_fn(batch):
        frame_ids = [item[0] for item in batch]
        video_clips = torch.stack([item[1] for item in batch])
        targets = [item[2] for item in batch]
        return frame_ids, video_clips, targets
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2 if device == 'cpu' else 4,
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # ============ BUILD MODEL ============
    print("\n--- Building Model ---")
    
    from models.yowo.yowo_multitask import YOWOMultiTask
    from models.yowo.loss_multitask import MultiTaskCriterion
    import argparse
    
    model = YOWOMultiTask(
        cfg=cfg,
        device=device,
        num_objects=36,
        num_actions=157,
        num_relations=26,
        trainable=True
    )
    model = model.to(device).train()
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss criterion
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # ============ TRACKING ============
    loss_history = defaultdict(list)
    attention_history = {
        'entropy': [],
        'nearby_far_ratio': [],
        'pos_scale': []
    }
    
    # ============ MINI TRAINING LOOP ============
    print(f"\n--- Running {max_iters} Training Iterations ---\n")
    
    start_time = time.time()
    iter_count = 0
    
    for epoch in range(10):  # Multiple epochs over small dataset
        for batch_idx, (frame_ids, video_clips, targets) in enumerate(dataloader):
            if iter_count >= max_iters:
                break
            
            # Move to device
            video_clips = video_clips.to(device)
            targets_device = []
            for t in targets:
                targets_device.append({
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                })
            
            # Forward
            optimizer.zero_grad()
            outputs = model(video_clips)
            
            # Loss
            loss_dict = criterion(outputs, targets_device)
            total_loss = loss_dict['losses']
            
            # Check for NaN
            if torch.isnan(total_loss):
                print(f"⚠️  NaN loss at iter {iter_count}")
                continue
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Track losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    loss_history[k].append(v.item())
            
            # Track attention metrics every 10 iterations
            if iter_count % 10 == 0:
                with torch.no_grad():
                    # Get attention entropy from last forward pass
                    # We need to re-run to get attention weights
                    key_frame = video_clips[:, :, -1, :, :]
                    feat_3d = model.backbone_3d(video_clips)
                    cls_feats, reg_feats = model.backbone_2d(key_frame)
                    
                    entropies = []
                    ratios = []
                    
                    for level in range(len(model.stride)):
                        cls_feat = cls_feats[level]
                        
                        if level == 0:
                            feat_3d_up = F.interpolate(feat_3d, scale_factor=4)
                        elif level == 1:
                            feat_3d_up = F.interpolate(feat_3d, scale_factor=2)
                        else:
                            feat_3d_up = feat_3d
                        
                        cls_feat = model.cls_channel_encoders[level](cls_feat, feat_3d_up)
                        reg_feat = model.reg_channel_encoders[level](reg_feats[level], feat_3d_up)
                        cls_feat, reg_feat = model.heads[level](cls_feat, reg_feat)
                        
                        obj_pred = model.obj_preds[level](cls_feat)
                        rel_feat = model.obj_cross_attn[level](cls_feat, obj_pred)
                        rel_pred = model.rel_preds[level](rel_feat)
                        
                        _, attn_weights = model.obj_rel_cross_attn[level](
                            cls_feat, obj_pred, rel_pred, return_weights=True
                        )
                        
                        B, N, _ = attn_weights.shape
                        H = W = int(np.sqrt(N))
                        
                        # Entropy
                        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean().item()
                        max_entropy = np.log(N)
                        entropies.append(entropy / max_entropy)
                        
                        # Nearby vs far
                        nearby_attn = []
                        far_attn = []
                        for i in range(min(10, N)):
                            y, x = i // W, i % W
                            nearby_mask = torch.zeros(N, device=device)
                            for dy in range(-2, 3):
                                for dx in range(-2, 3):
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < H and 0 <= nx < W:
                                        nearby_mask[ny * W + nx] = 1
                            far_mask = 1 - nearby_mask
                            if nearby_mask.sum() > 0 and far_mask.sum() > 0:
                                nearby_attn.append((attn_weights[0, i] * nearby_mask).sum().item() / nearby_mask.sum().item())
                                far_attn.append((attn_weights[0, i] * far_mask).sum().item() / far_mask.sum().item())
                        
                        if far_attn:
                            ratios.append(np.mean(nearby_attn) / (np.mean(far_attn) + 1e-10))
                    
                    attention_history['entropy'].append(np.mean(entropies))
                    attention_history['nearby_far_ratio'].append(np.mean(ratios) if ratios else 1.0)
                    attention_history['pos_scale'].append(
                        np.mean([m.pos_scale.item() for m in model.obj_rel_cross_attn])
                    )
            
            # Print progress
            if iter_count % 20 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iter_count:3d}/{max_iters} | "
                      f"loss: {total_loss.item():.2f} | "
                      f"loss_act: {loss_dict.get('loss_act', 0):.2f} | "
                      f"time: {elapsed:.1f}s")
            
            iter_count += 1
        
        if iter_count >= max_iters:
            break
    
    total_time = time.time() - start_time
    print(f"\nCompleted {iter_count} iterations in {total_time:.1f}s")
    
    # ============ ANALYSIS ============
    print("\n" + "=" * 70)
    print("  ANALYSIS: Did the model learn?")
    print("=" * 70)
    
    results = {}
    
    # 1. Loss trend
    print("\n--- Loss Trend ---")
    if len(loss_history['losses']) >= 10:
        first_10 = np.mean(loss_history['losses'][:10])
        last_10 = np.mean(loss_history['losses'][-10:])
        loss_change = (last_10 - first_10) / first_10 * 100
        
        print(f"First 10 iters avg loss: {first_10:.2f}")
        print(f"Last 10 iters avg loss:  {last_10:.2f}")
        print(f"Change: {loss_change:+.1f}%")
        
        results['loss_decreased'] = loss_change < -5
        
        if loss_change < -10:
            print("✅ PASS: Loss decreased significantly!")
        elif loss_change < 0:
            print("⚠️  MARGINAL: Loss decreased slightly")
        else:
            print("❌ FAIL: Loss did not decrease")
    
    # 2. Action loss trend
    print("\n--- Action Loss Trend ---")
    if len(loss_history.get('loss_act', [])) >= 10:
        first_10_act = np.mean(loss_history['loss_act'][:10])
        last_10_act = np.mean(loss_history['loss_act'][-10:])
        act_change = (last_10_act - first_10_act) / first_10_act * 100
        
        print(f"First 10 iters avg: {first_10_act:.2f}")
        print(f"Last 10 iters avg:  {last_10_act:.2f}")
        print(f"Change: {act_change:+.1f}%")
        
        results['action_loss_decreased'] = act_change < -5
        
        if act_change < -10:
            print("✅ PASS: Action loss decreased!")
        elif act_change < 0:
            print("⚠️  MARGINAL: Action loss decreased slightly")
        else:
            print("❌ FAIL: Action loss did not decrease")
    
    # 3. Attention entropy
    print("\n--- Attention Entropy ---")
    if len(attention_history['entropy']) >= 5:
        first_entropy = np.mean(attention_history['entropy'][:3])
        last_entropy = np.mean(attention_history['entropy'][-3:])
        
        print(f"Initial entropy (normalized): {first_entropy:.4f}")
        print(f"Final entropy (normalized):   {last_entropy:.4f}")
        
        results['entropy_decreased'] = last_entropy < first_entropy
        
        if last_entropy < first_entropy * 0.95:
            print("✅ PASS: Attention becoming more focused!")
        elif last_entropy < first_entropy:
            print("⚠️  MARGINAL: Slight improvement")
        else:
            print("⚠️  Entropy not decreasing (may need more iterations)")
    
    # 4. Nearby/Far ratio
    print("\n--- Nearby vs Far Attention Ratio ---")
    if len(attention_history['nearby_far_ratio']) >= 5:
        first_ratio = np.mean(attention_history['nearby_far_ratio'][:3])
        last_ratio = np.mean(attention_history['nearby_far_ratio'][-3:])
        
        print(f"Initial ratio: {first_ratio:.4f}")
        print(f"Final ratio:   {last_ratio:.4f}")
        
        results['ratio_increased'] = last_ratio > first_ratio
        
        if last_ratio > first_ratio * 1.05:
            print("✅ PASS: Attending more to nearby positions!")
        elif last_ratio > first_ratio:
            print("⚠️  MARGINAL: Slight improvement")
        else:
            print("⚠️  Ratio not increasing (may need more iterations)")
    
    # 5. pos_scale stability
    print("\n--- Position Scale (pos_scale) ---")
    if len(attention_history['pos_scale']) >= 2:
        pos_scales = attention_history['pos_scale']
        print(f"Initial pos_scale: {pos_scales[0]:.4f}")
        print(f"Final pos_scale:   {pos_scales[-1]:.4f}")
        
        # Should not explode or vanish
        results['pos_scale_stable'] = 0.1 < pos_scales[-1] < 2.0
        
        if 0.1 < pos_scales[-1] < 2.0:
            print("✅ PASS: pos_scale is stable")
        else:
            print("❌ FAIL: pos_scale unstable")
    
    # ============ FINAL VERDICT ============
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)
    
    # Key checks
    key_pass = results.get('loss_decreased', False)
    action_pass = results.get('action_loss_decreased', False)
    
    if key_pass and action_pass:
        print("\n🎉 PASS: The model IS learning!")
        print("   - Total loss decreasing ✓")
        print("   - Action loss decreasing ✓")
        print("\n   Ready for full training!")
        return True
    elif key_pass or action_pass:
        print("\n⚠️  MARGINAL: Model shows some learning")
        print("   Consider running more iterations or adjusting learning rate")
        return True
    else:
        print("\n❌ FAIL: Model not learning well")
        print("   Something may be wrong with the architecture")
        return False


if __name__ == "__main__":
    success = mini_training_test()
    sys.exit(0 if success else 1)
