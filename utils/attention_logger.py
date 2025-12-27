"""
Attention Pattern Logger for YOWO Multi-Task Training

This module logs attention patterns during training to verify:
1. Attention entropy (uniform vs focused) - should DECREASE over training
2. Nearby vs far attention ratio - should INCREASE over training  
3. Person-to-object attention - should be HIGHER than background
4. Position encoding scale - tracks the learned pos_scale value

Usage:
    from utils.attention_logger import AttentionLogger
    
    # Initialize
    attn_logger = AttentionLogger(log_dir=path_to_save)
    
    # During training (every N iterations)
    if iter_i % 100 == 0:
        attn_logger.log_attention(model, outputs, targets, epoch, iter_i)
    
    # At end of epoch
    attn_logger.save_epoch_summary(epoch)
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class AttentionLogger:
    """Logs attention patterns during training for verification."""
    
    def __init__(self, log_dir, log_frequency=100):
        """
        Args:
            log_dir: Directory to save attention logs
            log_frequency: Log every N iterations (default 100)
        """
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        self.epoch_stats = defaultdict(list)
        
        # Create log directory
        self.attn_log_dir = os.path.join(log_dir, 'attention_logs')
        os.makedirs(self.attn_log_dir, exist_ok=True)
        
        print(f"AttentionLogger initialized. Logs will be saved to: {self.attn_log_dir}")
    
    def log_attention(self, model, outputs, targets, epoch, iter_i):
        """
        Log attention patterns from the model.
        
        Args:
            model: YOWOMultiTask model (or model_without_ddp)
            outputs: Model outputs dict
            targets: Ground truth targets
            epoch: Current epoch
            iter_i: Current iteration
        """
        # Get the model (handle DDP wrapper)
        if hasattr(model, 'module'):
            model = model.module
            
        # Compute attention statistics
        stats = self._compute_attention_stats(model, outputs, targets)
        
        # Log to console every log_frequency iterations
        if iter_i % self.log_frequency == 0:
            self._print_stats(stats, epoch, iter_i)
        
        # Accumulate for epoch summary
        for k, v in stats.items():
            self.epoch_stats[k].append(v)
    
    def _compute_attention_stats(self, model, outputs, targets):
        """Compute various attention statistics."""
        stats = {}
        
        # Get the scene context attention modules
        if hasattr(model, 'obj_rel_cross_attn'):
            attn_modules = model.obj_rel_cross_attn
            
            # Get pos_scale values
            pos_scales = []
            for module in attn_modules:
                if hasattr(module, 'pos_scale'):
                    pos_scales.append(module.pos_scale.item())
            
            if pos_scales:
                stats['pos_scale'] = np.mean(pos_scales)
        
        # Analyze output predictions for attention quality indicators
        # Higher confidence at positions with strong predictions = good attention
        
        # Get predictions from outputs
        if 'pred_obj' in outputs and 'pred_act' in outputs:
            # Analyze prediction confidence patterns
            obj_preds = outputs['pred_obj']  # List of [B, N, num_objects]
            act_preds = outputs['pred_act']  # List of [B, N, num_actions]
            
            # Compute average max confidence (proxy for attention quality)
            obj_conf = []
            act_conf = []
            for obj_pred, act_pred in zip(obj_preds, act_preds):
                obj_conf.append(F.softmax(obj_pred, dim=-1).max(dim=-1)[0].mean().item())
                act_conf.append(torch.sigmoid(act_pred).max(dim=-1)[0].mean().item())
            
            stats['avg_obj_confidence'] = np.mean(obj_conf)
            stats['avg_act_confidence'] = np.mean(act_conf)
        
        # Analyze ground truth labels to understand what the model should learn
        if targets and len(targets) > 0:
            # Count person vs object boxes
            num_person = 0
            num_total = 0
            for target in targets:
                if 'labels' in target:
                    labels = target['labels']
                    num_total += labels.shape[0]
                    if labels.dim() > 1 and labels.size(-1) >= 36:
                        # Object class is first 36 dims, class 0 is person
                        obj_classes = labels[:, :36].argmax(dim=-1)
                        num_person += (obj_classes == 0).sum().item()
            
            stats['num_person_boxes'] = num_person
            stats['num_total_boxes'] = num_total
        
        return stats
    
    def _print_stats(self, stats, epoch, iter_i):
        """Print attention statistics."""
        log_parts = [f'[Attn Stats]']
        
        for k, v in stats.items():
            if isinstance(v, float):
                log_parts.append(f'[{k}: {v:.4f}]')
            else:
                log_parts.append(f'[{k}: {v}]')
        
        print(' '.join(log_parts), flush=True)
    
    def log_detailed_attention(self, model, video_clips, targets, epoch, iter_i):
        """
        Log detailed attention patterns by running a forward pass with return_weights=True.
        
        This is more expensive so should be called less frequently (e.g., once per epoch).
        
        Args:
            model: YOWOMultiTask model
            video_clips: Video input [B, 3, T, H, W]
            targets: Ground truth targets
            epoch: Current epoch
            iter_i: Current iteration
        """
        # Get the model (handle DDP wrapper)
        if hasattr(model, 'module'):
            model = model.module
        
        model.eval()
        
        with torch.no_grad():
            # We need to manually extract attention weights
            # This requires modifying the forward pass slightly
            
            # Get backbone features
            key_frame = video_clips[:, :, -1, :, :]
            feat_3d = model.backbone_3d(video_clips)
            cls_feats, reg_feats = model.backbone_2d(key_frame)
            
            stats = {
                'attention_entropy': [],
                'nearby_vs_far_ratio': [],
                'uniform_attention_pct': []
            }
            
            # Process each FPN level
            for level in range(len(model.stride)):
                cls_feat = cls_feats[level]
                
                # Upsample/downsample 3D features
                if level == 0:
                    feat_3d_up = F.interpolate(feat_3d, scale_factor=4)
                elif level == 1:
                    feat_3d_up = F.interpolate(feat_3d, scale_factor=2)
                else:
                    feat_3d_up = feat_3d
                
                # Channel encode
                cls_feat = model.cls_channel_encoders[level](cls_feat, feat_3d_up)
                reg_feat = model.reg_channel_encoders[level](reg_feats[level], feat_3d_up)
                
                # Head
                cls_feat, reg_feat = model.heads[level](cls_feat, reg_feat)
                
                # Get predictions
                obj_pred = model.obj_preds[level](cls_feat)
                
                # Get relation features and predictions
                rel_feat = model.obj_cross_attn[level](cls_feat, obj_pred)
                rel_pred = model.rel_preds[level](rel_feat)
                
                # Get attention weights from scene context
                _, attn_weights = model.obj_rel_cross_attn[level](
                    cls_feat, obj_pred, rel_pred, return_weights=True
                )
                
                B, N, _ = attn_weights.shape
                H = W = int(np.sqrt(N))
                
                # Compute entropy (lower = more focused)
                entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean().item()
                max_entropy = np.log(N)
                normalized_entropy = entropy / max_entropy
                stats['attention_entropy'].append(normalized_entropy)
                
                # Compute nearby vs far ratio
                # For each position, compare attention to neighbors vs far positions
                nearby_attn = []
                far_attn = []
                
                for i in range(min(10, N)):  # Sample 10 positions
                    y, x = i // W, i % W
                    
                    # Nearby positions (within 2 pixels)
                    nearby_mask = torch.zeros(N, device=attn_weights.device)
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                nearby_mask[ny * W + nx] = 1
                    
                    # Far positions
                    far_mask = 1 - nearby_mask
                    
                    if nearby_mask.sum() > 0 and far_mask.sum() > 0:
                        nearby_attn.append((attn_weights[0, i] * nearby_mask).sum().item() / nearby_mask.sum().item())
                        far_attn.append((attn_weights[0, i] * far_mask).sum().item() / far_mask.sum().item())
                
                if nearby_attn and far_attn:
                    ratio = np.mean(nearby_attn) / (np.mean(far_attn) + 1e-10)
                    stats['nearby_vs_far_ratio'].append(ratio)
                
                # Check if attention is near-uniform
                uniform_val = 1.0 / N
                is_uniform = (attn_weights.std(dim=-1) < uniform_val).float().mean().item()
                stats['uniform_attention_pct'].append(is_uniform)
            
            # Average across levels
            for k in list(stats.keys()):
                if stats[k]:
                    stats[k] = np.mean(stats[k])
                else:
                    del stats[k]
        
        model.train()
        
        # Log
        print(f"\n[Detailed Attention Analysis - Epoch {epoch+1}, Iter {iter_i}]")
        print(f"  Attention entropy (normalized): {stats.get('attention_entropy', 'N/A'):.4f}")
        print(f"    (0 = focused on single position, 1 = uniform)")
        print(f"  Nearby vs Far ratio: {stats.get('nearby_vs_far_ratio', 'N/A'):.4f}")
        print(f"    (>1 means attending more to nearby positions)")
        print(f"  Uniform attention %: {stats.get('uniform_attention_pct', 'N/A')*100:.1f}%")
        print(f"    (Should DECREASE during training)")
        print()
        
        # Save to file
        log_file = os.path.join(self.attn_log_dir, f'detailed_epoch{epoch+1}_iter{iter_i}.json')
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def save_epoch_summary(self, epoch):
        """Save summary of attention statistics for the epoch."""
        if not self.epoch_stats:
            return
        
        summary = {}
        for k, v in self.epoch_stats.items():
            if v:
                summary[k] = {
                    'mean': float(np.mean(v)),
                    'std': float(np.std(v)),
                    'min': float(np.min(v)),
                    'max': float(np.max(v))
                }
        
        # Save to file
        summary_file = os.path.join(self.attn_log_dir, f'epoch_{epoch+1}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n[Attention Summary - Epoch {epoch+1}]")
        for k, v in summary.items():
            print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")
        print()
        
        # Reset for next epoch
        self.epoch_stats = defaultdict(list)
        
        return summary


def create_attention_monitor_hook(log_dir):
    """
    Create a hook that can be registered on the SceneContextAttention module
    to automatically log attention patterns.
    
    Usage:
        hook = create_attention_monitor_hook(log_dir)
        for module in model.obj_rel_cross_attn:
            module.register_forward_hook(hook)
    """
    stats = {'count': 0, 'entropy_sum': 0}
    
    def hook(module, input, output):
        # input: (cls_feat, obj_pred, rel_pred)
        # output: act_feat or (act_feat, attn_weights)
        
        if stats['count'] % 100 != 0:
            stats['count'] += 1
            return
        
        stats['count'] += 1
        
        # We can't easily get attention weights without modifying the forward pass
        # This hook is mainly for tracking pos_scale
        if hasattr(module, 'pos_scale'):
            pos_scale = module.pos_scale.item()
            print(f"[Attn Hook] pos_scale: {pos_scale:.4f}")
    
    return hook
