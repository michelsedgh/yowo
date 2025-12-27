"""
Cross-Attention Monitor for YOWO Multi-Task Training

Monitors the GlobalSceneContext and ObjectContextModule to verify
that object and relation context is being learned for action detection.

Key Metrics:
1. Object Confidence - Are we detecting objects with certainty?
2. Context Scale - Is the model learning to weight context?
3. Feature Change - How much does context change the features?
4. Gradient Flow - Are all components receiving gradients?
5. Action/Object/Relation Loss Trends - Core training progress
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
    Monitors GlobalSceneContext and ObjectContextModule during training.
    """
    
    def __init__(self, model, log_dir: str, log_interval: int = 100, device: str = 'cpu'):
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
        
        # Loss history for trend analysis
        self.loss_history = {
            'loss_act': [],
            'loss_obj': [],
            'loss_rel': [],
            'loss_conf': [],
        }
        
        print(f"[CrossAttentionMonitor] Initialized. Logs: {self.attn_log_dir}")
    
    def _get_model(self):
        """Get the underlying model (handle DDP wrapper)."""
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        return model
    
    @torch.no_grad()
    def compute_metrics(self, video_clips: torch.Tensor, level: int = 2) -> Dict[str, float]:
        """Compute meaningful metrics for context learning."""
        model = self._get_model()
        was_training = model.training
        model.eval()
        
        try:
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
            
            # Action features with context
            act_feat = model.obj_rel_cross_attn[level](cls_feat, obj_pred, rel_pred)
            
            metrics = {}
            
            # 1. Object detection confidence (should increase)
            metrics['obj_confidence'] = obj_probs.max(dim=1)[0].mean().item()
            
            # 2. Person detection rate (class 0 confidence)
            metrics['person_confidence'] = obj_probs[:, 0].mean().item()
            
            # 3. Active relations per position (should start low, may increase)
            metrics['relations_active'] = (rel_probs > 0.5).float().mean().item()
            
            # 4. Context contribution (how much features change)
            feature_change = (act_feat - cls_feat).abs().mean()
            feat_magnitude = cls_feat.abs().mean().clamp(min=0.01)
            metrics['context_contribution'] = (feature_change / feat_magnitude).item()
            
            # 5. Context scale (learnable parameter in GlobalSceneContext)
            if hasattr(model.obj_rel_cross_attn[level], 'context_scale'):
                metrics['context_scale'] = model.obj_rel_cross_attn[level].context_scale.item()
            
            return metrics
            
        finally:
            if was_training:
                model.train()
    
    def compute_gradient_metrics(self, level: int = 2) -> Dict[str, float]:
        """Compute actual gradient magnitudes for context modules."""
        model = self._get_model()
        metrics = {}
        
        # GlobalSceneContext (obj_rel_cross_attn) - check context_proj
        if hasattr(model, 'obj_rel_cross_attn'):
            sca = model.obj_rel_cross_attn[level]
            if hasattr(sca, 'context_proj'):
                layer = sca.context_proj[0]  # First conv/linear in the projection
                if layer.weight.grad is not None:
                    metrics['scene_ctx_grad'] = layer.weight.grad.abs().mean().item()
                else:
                    metrics['scene_ctx_grad'] = 0.0
        
        # ObjectContextModule (obj_cross_attn) - check context_proj
        if hasattr(model, 'obj_cross_attn'):
            ocm = model.obj_cross_attn[level]
            if hasattr(ocm, 'context_proj'):
                layer = ocm.context_proj[0]
                if layer.weight.grad is not None:
                    metrics['obj_ctx_grad'] = layer.weight.grad.abs().mean().item()
                else:
                    metrics['obj_ctx_grad'] = 0.0
        
        # Action head gradient (most critical)
        if hasattr(model, 'act_preds') and model.act_preds[level].weight.grad is not None:
            metrics['act_head_grad'] = model.act_preds[level].weight.grad.abs().mean().item()
        else:
            metrics['act_head_grad'] = 0.0
        
        return metrics
    
    def log_step(self, video_clips: torch.Tensor, targets: List, 
                 loss_dict: Optional[Dict], epoch: int, iter_i: int):
        """Log metrics at regular intervals."""
        self.iteration += 1
        
        # Track loss history every step (for trend analysis)
        if loss_dict:
            for key in ['loss_act', 'loss_obj', 'loss_rel', 'loss_conf']:
                if key in loss_dict:
                    val = loss_dict[key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    self.loss_history[key].append(val)
        
        # Only print at specified intervals
        if self.iteration % self.log_interval != 0:
            return
        
        # Compute attention metrics
        metrics = self.compute_metrics(video_clips)
        grad_metrics = self.compute_gradient_metrics()
        metrics.update(grad_metrics)
        
        # Store in epoch stats
        for k, v in metrics.items():
            self.epoch_stats[k].append(v)
        
        # Count targets
        num_persons = 0
        num_total_boxes = 0
        for target in targets:
            if isinstance(target, dict) and 'labels' in target:
                labels = target['labels']
                if labels.dim() >= 2 and labels.size(-1) >= 36:
                    obj_classes = labels[:, :36].argmax(dim=-1)
                    num_persons += (obj_classes == 0).sum().item()
                    num_total_boxes += labels.shape[0]
        
        # Get loss values
        loss_act = loss_dict.get('loss_act', 0) if loss_dict else 0
        loss_obj = loss_dict.get('loss_obj', 0) if loss_dict else 0
        loss_rel = loss_dict.get('loss_rel', 0) if loss_dict else 0
        if isinstance(loss_act, torch.Tensor): loss_act = loss_act.item()
        if isinstance(loss_obj, torch.Tensor): loss_obj = loss_obj.item()
        if isinstance(loss_rel, torch.Tensor): loss_rel = loss_rel.item()
        
        # Print compact, useful log
        print(f"\n[Context Monitor] Epoch {epoch+1}, Iter {iter_i}")
        print(f"  📦 Obj conf: {metrics['obj_confidence']:.3f} | "
              f"Person: {metrics['person_confidence']:.3f} | "
              f"Relations: {metrics['relations_active']:.3f}")
        print(f"  🔗 Context contribution: {metrics['context_contribution']:.2f}x | "
              f"Scale: {metrics.get('context_scale', 1.0):.3f}")
        print(f"  📈 Gradients: SceneCtx={metrics.get('scene_ctx_grad', 0):.4f}, "
              f"ObjCtx={metrics.get('obj_ctx_grad', 0):.4f}, "
              f"ActHead={metrics.get('act_head_grad', 0):.4f}")
        print(f"  📊 Losses: act={loss_act:.2f}, obj={loss_obj:.2f}, rel={loss_rel:.2f}")
        print(f"  👥 Batch: {num_persons}/{num_total_boxes} person boxes")
    
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
        
        # Calculate loss trends
        if len(self.loss_history['loss_act']) > 100:
            first_100 = np.mean(self.loss_history['loss_act'][:100])
            last_100 = np.mean(self.loss_history['loss_act'][-100:])
            summary['loss_act_trend'] = {
                'first_100_avg': float(first_100),
                'last_100_avg': float(last_100),
                'improvement': float(first_100 - last_100)
            }
        
        # Save to file
        summary_file = os.path.join(self.attn_log_dir, f'epoch_{epoch+1}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch+1} CONTEXT LEARNING SUMMARY")
        print(f"{'='*70}")
        
        if 'obj_confidence' in summary:
            conf = summary['obj_confidence']['mean']
            status = "✅ good" if conf > 0.15 else "📈 learning" if conf > 0.05 else "⏳ early"
            print(f"  Object Confidence: {conf:.4f} ({status})")
        
        if 'context_contribution' in summary:
            contrib = summary['context_contribution']['mean']
            status = "✅ active" if contrib > 1.0 else "⚠️ weak"
            print(f"  Context Contribution: {contrib:.2f}x ({status})")
        
        if 'scene_ctx_grad' in summary:
            grad = summary['scene_ctx_grad']['mean']
            status = "✅ flowing" if grad > 0.0001 else "⚠️ weak"
            print(f"  Scene Context Gradient: {grad:.6f} ({status})")
        
        if 'obj_ctx_grad' in summary:
            grad = summary['obj_ctx_grad']['mean']
            status = "✅ flowing" if grad > 0.0001 else "⚠️ weak"
            print(f"  Object Context Gradient: {grad:.6f} ({status})")
        
        if 'loss_act_trend' in summary:
            trend = summary['loss_act_trend']
            improvement = trend['improvement']
            status = "✅ improving" if improvement > 0 else "⚠️ check"
            print(f"  Action Loss: {trend['first_100_avg']:.2f} → {trend['last_100_avg']:.2f} ({status})")
        
        print(f"{'='*70}\n")
        
        # Reset for next epoch
        self.epoch_stats = defaultdict(list)
        
        return summary
