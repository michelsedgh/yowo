#!/usr/bin/env python3
"""
Cross-Attention Monitoring Module

This module provides functions to monitor whether the cross-attention 
mechanism is being used effectively during training.

Add these to your training loop to track:
1. Attention entropy (should decrease over time)
2. Object prediction confidence (should increase over time)
3. Context contribution to features (should be non-zero)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class CrossAttentionMonitor:
    """
    Monitor cross-attention behavior during training.
    
    Usage:
        monitor = CrossAttentionMonitor(model)
        
        # In training loop:
        outputs = model(video_clip)
        metrics = monitor.compute_metrics(video_clip)
        monitor.log_metrics(epoch, step, metrics)
    """
    
    def __init__(self, model, log_interval: int = 100):
        """
        Args:
            model: YOWOMultiTask model
            log_interval: How often to print metrics (in steps)
        """
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        
        # Track metrics over time
        self.history = {
            'attention_entropy': [],
            'object_confidence': [],
            'context_contribution': [],
            'gradient_magnitude': []
        }
    
    @torch.no_grad()
    def compute_metrics(self, video_clip: torch.Tensor, level: int = 2) -> Dict[str, float]:
        """
        Compute cross-attention metrics for a given input.
        
        Args:
            video_clip: [B, 3, T, H, W] video tensor
            level: FPN level to analyze (0, 1, or 2)
        
        Returns:
            Dict with metrics
        """
        was_training = self.model.training
        self.model.eval()
        
        try:
            B = video_clip.shape[0]
            device = video_clip.device
            
            # Forward through backbone
            key_frame = video_clip[:, :, -1, :, :]
            feat_3d = self.model.backbone_3d(video_clip)
            cls_feats, reg_feats = self.model.backbone_2d(key_frame)
            
            # Process selected level
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            cls_feat = self.model.cls_channel_encoders[level](cls_feats[level], feat_3d_up)
            reg_feat = self.model.reg_channel_encoders[level](reg_feats[level], feat_3d_up)
            cls_feat, reg_feat = self.model.heads[level](cls_feat, reg_feat)
            
            # Object prediction
            obj_pred = self.model.obj_preds[level](cls_feat)
            obj_probs = F.softmax(obj_pred, dim=1)
            
            # Relation prediction 
            rel_feat = self.model.obj_cross_attn[level](cls_feat, obj_pred)
            rel_pred = self.model.rel_preds[level](rel_feat)
            rel_probs = torch.sigmoid(rel_pred)
            
            # Action prediction with attention
            act_feat, attn_weights = self.model.obj_rel_cross_attn[level](
                cls_feat, obj_pred, rel_pred, return_weights=True
            )
            
            # Metric 1: Attention entropy (lower = more focused)
            N = attn_weights.shape[-1]
            entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean()
            max_entropy = np.log(N)
            normalized_entropy = (entropy / max_entropy).item()
            
            # Metric 2: Object prediction confidence
            max_obj_prob = obj_probs.max(dim=1)[0].mean().item()
            
            # Metric 3: Context contribution to features
            # How much does the output differ from input?
            feature_change = (act_feat - cls_feat).abs().mean()
            context_contribution = (feature_change / cls_feat.abs().mean()).item()
            
            # Metric 4: Top object class distribution
            obj_class_counts = obj_probs.argmax(dim=1).flatten()
            unique_classes = len(obj_class_counts.unique())
            
            metrics = {
                'attention_entropy': normalized_entropy,
                'object_confidence': max_obj_prob,
                'context_contribution': context_contribution,
                'unique_obj_classes': unique_classes,
                'max_rel_prob': rel_probs.max().item(),
            }
            
            return metrics
            
        finally:
            if was_training:
                self.model.train()
    
    def compute_gradient_metrics(self, level: int = 2) -> Dict[str, float]:
        """Compute gradient magnitudes for cross-attention components."""
        sca = self.model.obj_rel_cross_attn[level]
        ocm = self.model.obj_cross_attn[level]
        
        metrics = {}
        
        # SceneContextAttention gradients
        if sca.key_proj[0].weight.grad is not None:
            metrics['sca_key_grad'] = sca.key_proj[0].weight.grad.abs().mean().item()
        else:
            metrics['sca_key_grad'] = 0.0
            
        # ObjectContextModule gradients
        if ocm.context_proj[0].weight.grad is not None:
            metrics['ocm_ctx_grad'] = ocm.context_proj[0].weight.grad.abs().mean().item()
        else:
            metrics['ocm_ctx_grad'] = 0.0
        
        # Action prediction gradients
        if self.model.act_preds[level].weight.grad is not None:
            metrics['act_pred_grad'] = self.model.act_preds[level].weight.grad.abs().mean().item()
        else:
            metrics['act_pred_grad'] = 0.0
            
        return metrics
    
    def log_step(self, video_clip: torch.Tensor, epoch: int, step: int, 
                 loss_dict: Optional[Dict] = None, force: bool = False):
        """
        Log metrics at regular intervals.
        
        Args:
            video_clip: Current training batch
            epoch: Current epoch
            step: Current step within epoch
            loss_dict: Loss dictionary from criterion
            force: Force logging regardless of interval
        """
        self.step_count += 1
        
        if not force and self.step_count % self.log_interval != 0:
            return
        
        # Compute metrics
        metrics = self.compute_metrics(video_clip)
        grad_metrics = self.compute_gradient_metrics()
        
        # Store in history
        self.history['attention_entropy'].append(metrics['attention_entropy'])
        self.history['object_confidence'].append(metrics['object_confidence'])
        self.history['context_contribution'].append(metrics['context_contribution'])
        self.history['gradient_magnitude'].append(grad_metrics.get('sca_key_grad', 0))
        
        # Print log
        print(f"\n[CrossAttn Monitor] Epoch {epoch}, Step {step}")
        print(f"  Attention entropy:    {metrics['attention_entropy']:.4f} (1.0=uniform, lower=focused)")
        print(f"  Object confidence:    {metrics['object_confidence']:.4f} (higher=more confident)")
        print(f"  Context contribution: {metrics['context_contribution']:.4f} (how much context changes features)")
        print(f"  Unique obj classes:   {metrics['unique_obj_classes']}")
        print(f"  Gradients - SCA: {grad_metrics['sca_key_grad']:.6f}, OCM: {grad_metrics['ocm_ctx_grad']:.6f}")
        
        if loss_dict:
            print(f"  Losses - act: {loss_dict.get('loss_act', 0):.4f}, obj: {loss_dict.get('loss_obj', 0):.4f}")
    
    def get_summary(self) -> str:
        """Get a summary of cross-attention behavior over training."""
        if not self.history['attention_entropy']:
            return "No data collected yet."
        
        # Compare first 10% vs last 10% of training
        n = len(self.history['attention_entropy'])
        first_n = max(1, n // 10)
        last_n = max(1, n // 10)
        
        first_entropy = np.mean(self.history['attention_entropy'][:first_n])
        last_entropy = np.mean(self.history['attention_entropy'][-last_n:])
        
        first_conf = np.mean(self.history['object_confidence'][:first_n])
        last_conf = np.mean(self.history['object_confidence'][-last_n:])
        
        summary = f"""
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
- If entropy is decreasing → model is learning to focus attention
- If object confidence is increasing → model is learning object detection
- Once objects are detected confidently, attention should become more focused
"""
        return summary


def add_monitoring_to_trainer(trainer_module):
    """
    Add cross-attention monitoring to an existing trainer.
    
    Example usage in train.py:
        from cross_attention_monitor import CrossAttentionMonitor
        
        monitor = CrossAttentionMonitor(model, log_interval=100)
        
        for epoch in range(epochs):
            for step, (video_clip, targets) in enumerate(dataloader):
                # ... forward, loss, backward, step ...
                
                # Add this line:
                monitor.log_step(video_clip, epoch, step, loss_dict)
            
            # At end of epoch:
            print(monitor.get_summary())
    """
    pass


if __name__ == "__main__":
    # Test the monitor
    print("Testing CrossAttentionMonitor...")
    
    from models.yowo.yowo_multitask import YOWOMultiTask
    from config import yowo_v2_config
    
    cfg = yowo_v2_config['yowo_v2_x3d_m_yolo11m_multitask']
    model = YOWOMultiTask(cfg=cfg, device='cpu', trainable=True)
    
    monitor = CrossAttentionMonitor(model, log_interval=1)
    
    # Dummy input
    video_clip = torch.randn(1, 3, 16, 224, 224)
    
    # Compute metrics
    metrics = monitor.compute_metrics(video_clip)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ Monitor working correctly!")
