"""
Cross-Attention Monitor for YOWO Multi-Task Training

This module provides comprehensive monitoring of the cross-attention mechanism
to verify that object and relation context is being used for action detection.

Key Metrics Tracked:
1. Attention Entropy - Should DECREASE as model learns to focus
2. Object Prediction Confidence - Should INCREASE over training
3. Context Contribution - How much context changes features
4. Gradient Flow - Verifies gradients reach cross-attention modules

Usage:
    from utils.cross_attention_monitor import CrossAttentionMonitor
    
    # Initialize in train.py
    monitor = CrossAttentionMonitor(model, log_dir=path_to_save, log_interval=100)
    
    # In training loop
    for iter_i, (video_clips, targets) in enumerate(dataloader):
        # ... forward, loss, backward ...
        monitor.log_step(video_clips, targets, loss_dict, epoch, iter_i)
    
    # At end of epoch
    monitor.save_epoch_summary(epoch)
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


class CrossAttentionMonitor:
    """
    Comprehensive monitor for cross-attention mechanism in YOWO multi-task model.
    
    This tracks whether the model is learning to use object/relation context
    for action prediction.
    """
    
    def __init__(self, model, log_dir: str, log_interval: int = 100, device: str = 'cpu'):
        """
        Args:
            model: YOWOMultiTask model (or model_without_ddp)
            log_dir: Directory to save logs
            log_interval: How often to print metrics (in iterations)
            device: Device for computation
        """
        self.model = model
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.device = device
        self.iteration = 0
        
        # Create log directory
        self.attn_log_dir = os.path.join(log_dir, 'cross_attention_logs')
        os.makedirs(self.attn_log_dir, exist_ok=True)
        
        # Accumulate stats for epoch summary
        self.epoch_stats = defaultdict(list)
        
        # Track trends
        self.history = {
            'attention_entropy': [],
            'object_confidence': [],
            'context_contribution': [],
            'sca_gradient': [],
            'ocm_gradient': []
        }
        
        print(f"[CrossAttentionMonitor] Initialized. Logs: {self.attn_log_dir}")
    
    def _get_model(self):
        """Get the underlying model (handle DDP wrapper)."""
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        return model
    
    @torch.no_grad()
    def compute_attention_metrics(self, video_clips: torch.Tensor, level: int = 2) -> Dict[str, float]:
        """
        Compute cross-attention metrics for a given input.
        
        Args:
            video_clips: [B, 3, T, H, W] video tensor
            level: FPN level to analyze (default 2, coarsest)
        
        Returns:
            Dict with attention metrics
        """
        model = self._get_model()
        was_training = model.training
        model.eval()
        
        try:
            device = video_clips.device
            
            # Forward through backbone
            key_frame = video_clips[:, :, -1, :, :]
            feat_3d = model.backbone_3d(video_clips)
            cls_feats, reg_feats = model.backbone_2d(key_frame)
            
            # Process selected level
            if level == 0:
                feat_3d_up = F.interpolate(feat_3d, scale_factor=4)
            elif level == 1:
                feat_3d_up = F.interpolate(feat_3d, scale_factor=2)
            else:
                feat_3d_up = feat_3d
            
            cls_feat = model.cls_channel_encoders[level](cls_feats[level], feat_3d_up)
            reg_feat = model.reg_channel_encoders[level](reg_feats[level], feat_3d_up)
            cls_feat, reg_feat = model.heads[level](cls_feat, reg_feat)
            
            # Object prediction
            obj_pred = model.obj_preds[level](cls_feat)
            obj_probs = F.softmax(obj_pred, dim=1)
            
            # Relation prediction
            rel_feat = model.obj_cross_attn[level](cls_feat, obj_pred)
            rel_pred = model.rel_preds[level](rel_feat)
            rel_probs = torch.sigmoid(rel_pred)
            
            # Action prediction with attention weights
            act_feat, attn_weights = model.obj_rel_cross_attn[level](
                cls_feat, obj_pred, rel_pred, return_weights=True
            )
            
            # Metric 1: Attention entropy (lower = more focused)
            N = attn_weights.shape[-1]
            entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean()
            max_entropy = np.log(N)
            normalized_entropy = (entropy / max_entropy).item()
            
            # Metric 2: Object prediction confidence
            max_obj_prob = obj_probs.max(dim=1)[0].mean().item()
            
            # Metric 3: How much does context change features?
            feature_change = (act_feat - cls_feat).abs().mean()
            context_contribution = (feature_change / cls_feat.abs().mean().clamp(min=0.01)).item()
            
            # Metric 4: Attention focus (std of attention weights)
            attn_std = attn_weights.std().item()
            
            # Metric 5: Top object predictions
            obj_class_preds = obj_probs.argmax(dim=1).flatten()
            unique_classes = len(obj_class_preds.unique())
            
            # Metric 6: Relation activity
            rel_active = (rel_probs > 0.5).float().sum(dim=1).mean().item()
            
            metrics = {
                'attention_entropy': normalized_entropy,
                'attention_std': attn_std,
                'object_confidence': max_obj_prob,
                'context_contribution': context_contribution,
                'unique_obj_classes': unique_classes,
                'active_relations_avg': rel_active,
            }
            
            return metrics
            
        finally:
            if was_training:
                model.train()
    
    def compute_gradient_metrics(self, level: int = 2) -> Dict[str, float]:
        """Compute gradient magnitudes for cross-attention components."""
        model = self._get_model()
        metrics = {}
        
        # SceneContextAttention gradients
        if hasattr(model, 'obj_rel_cross_attn'):
            sca = model.obj_rel_cross_attn[level]
            if hasattr(sca, 'key_proj') and sca.key_proj[0].weight.grad is not None:
                metrics['sca_key_grad'] = sca.key_proj[0].weight.grad.abs().mean().item()
            else:
                metrics['sca_key_grad'] = 0.0
        
        # ObjectContextModule gradients
        if hasattr(model, 'obj_cross_attn'):
            ocm = model.obj_cross_attn[level]
            if hasattr(ocm, 'context_proj') and ocm.context_proj[0].weight.grad is not None:
                metrics['ocm_ctx_grad'] = ocm.context_proj[0].weight.grad.abs().mean().item()
            else:
                metrics['ocm_ctx_grad'] = 0.0
        
        # Action prediction layer gradients
        if hasattr(model, 'act_preds') and model.act_preds[level].weight.grad is not None:
            metrics['act_pred_grad'] = model.act_preds[level].weight.grad.abs().mean().item()
        else:
            metrics['act_pred_grad'] = 0.0
        
        return metrics
    
    def log_step(self, video_clips: torch.Tensor, targets: List, 
                 loss_dict: Optional[Dict], epoch: int, iter_i: int):
        """
        Log metrics at regular intervals.
        
        Call this after loss.backward() but before optimizer.step().
        """
        self.iteration += 1
        
        # Only log at specified intervals
        if self.iteration % self.log_interval != 0:
            return
        
        # Compute attention metrics
        attn_metrics = self.compute_attention_metrics(video_clips)
        
        # Compute gradient metrics
        grad_metrics = self.compute_gradient_metrics()
        
        # Combine metrics
        metrics = {**attn_metrics, **grad_metrics}
        
        # Store in epoch stats
        for k, v in metrics.items():
            self.epoch_stats[k].append(v)
        
        # Store in history
        self.history['attention_entropy'].append(metrics['attention_entropy'])
        self.history['object_confidence'].append(metrics['object_confidence'])
        self.history['context_contribution'].append(metrics['context_contribution'])
        self.history['sca_gradient'].append(metrics.get('sca_key_grad', 0))
        self.history['ocm_gradient'].append(metrics.get('ocm_ctx_grad', 0))
        
        # Count targets
        num_persons = 0
        num_total_boxes = 0
        for target in targets:
            if isinstance(target, dict) and 'labels' in target:
                labels = target['labels']
                if labels.dim() >= 2:
                    obj_classes = labels[:, :36].argmax(dim=-1)
                    num_persons += (obj_classes == 0).sum().item()
                    num_total_boxes += labels.shape[0]
        
        # Print log
        print(f"\n[CrossAttn] Epoch {epoch+1}, Iter {iter_i}")
        print(f"  Attention entropy:    {metrics['attention_entropy']:.4f} " + 
              f"(1.0=uniform, <0.9=focused)")
        print(f"  Object confidence:    {metrics['object_confidence']:.4f} " +
              f"(should increase over training)")
        print(f"  Context contribution: {metrics['context_contribution']:.2f}x")
        print(f"  Gradients - SCA: {metrics.get('sca_key_grad', 0):.6f}, " +
              f"OCM: {metrics.get('ocm_ctx_grad', 0):.6f}")
        
        if loss_dict:
            loss_act = loss_dict.get('loss_act', 0)
            if isinstance(loss_act, torch.Tensor):
                loss_act = loss_act.item()
            print(f"  Losses - act: {loss_act:.4f}")
        
        print(f"  Batch: {num_persons}/{num_total_boxes} person boxes")
    
    def save_epoch_summary(self, epoch: int):
        """Save summary of metrics for the epoch."""
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
        print(f"\n{'='*60}")
        print(f"[CrossAttn Summary - Epoch {epoch+1}]")
        print(f"{'='*60}")
        
        # Key metrics
        if 'attention_entropy' in summary:
            ent = summary['attention_entropy']['mean']
            status = "✅ focusing" if ent < 0.98 else "⏳ still uniform"
            print(f"  Attention entropy: {ent:.4f} ({status})")
        
        if 'object_confidence' in summary:
            conf = summary['object_confidence']['mean']
            status = "✅ learning" if conf > 0.05 else "⏳ still uncertain"
            print(f"  Object confidence: {conf:.4f} ({status})")
        
        if 'sca_key_grad' in summary:
            grad = summary['sca_key_grad']['mean']
            status = "✅ gradients flowing" if grad > 0 else "⚠️ no gradients"
            print(f"  SCA gradient: {grad:.6f} ({status})")
        
        print(f"{'='*60}\n")
        
        # Reset for next epoch
        self.epoch_stats = defaultdict(list)
        
        return summary
    
    def get_training_summary(self) -> str:
        """Get a summary of cross-attention behavior over all training."""
        if len(self.history['attention_entropy']) < 10:
            return "Not enough data for summary yet."
        
        n = len(self.history['attention_entropy'])
        first_n = max(1, n // 10)
        last_n = max(1, n // 10)
        
        first_entropy = np.mean(self.history['attention_entropy'][:first_n])
        last_entropy = np.mean(self.history['attention_entropy'][-last_n:])
        
        first_conf = np.mean(self.history['object_confidence'][:first_n])
        last_conf = np.mean(self.history['object_confidence'][-last_n:])
        
        return f"""
Cross-Attention Training Summary
================================
Total logged steps: {n}

Attention Entropy (lower = more focused):
  First 10%: {first_entropy:.4f}
  Last 10%:  {last_entropy:.4f}
  Change:    {last_entropy - first_entropy:+.4f} {'(improving!)' if last_entropy < first_entropy else '(still uniform)'}

Object Confidence (higher = more certain):
  First 10%: {first_conf:.4f}
  Last 10%:  {last_conf:.4f}
  Change:    {last_conf - first_conf:+.4f} {'(improving!)' if last_conf > first_conf else '(still uncertain)'}

Interpretation:
- Attention entropy < 0.95 = model is focusing attention ✅
- Object confidence > 0.3 = model is detecting objects ✅
- Both improving = cross-attention is working as intended!
"""


def integrate_monitor_to_train(model, train_script_path: str, log_dir: str):
    """
    Instructions for integrating the monitor into train.py.
    
    Add these lines to train.py:
    
    1. At the top of the file:
        from utils.cross_attention_monitor import CrossAttentionMonitor
    
    2. After model creation (around line 183):
        # Initialize cross-attention monitor for multi-task models
        ca_monitor = None
        if 'multitask' in args.version.lower():
            ca_monitor = CrossAttentionMonitor(
                model_without_ddp, 
                log_dir=path_to_save, 
                log_interval=100
            )
    
    3. Inside training loop, after loss.backward() (around line 288):
        # Log cross-attention metrics
        if ca_monitor is not None:
            ca_monitor.log_step(video_clips, targets, loss_dict, epoch, iter_i)
    
    4. At end of each epoch (around line 311):
        # Save cross-attention summary
        if ca_monitor is not None:
            ca_monitor.save_epoch_summary(epoch)
    """
    print(integrate_monitor_to_train.__doc__)
