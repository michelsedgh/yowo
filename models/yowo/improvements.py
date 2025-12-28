"""
YOWO Improvements Module

This module contains optional improvements for better action detection:
1. Soft-Argmax for differentiable position extraction
2. Learnable Temporal Attention for X3D
3. Action-Object Co-occurrence Matrix
4. Class-weighted Focal Loss utilities

All improvements are designed to be modular and easy to integrate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# 1. SOFT-ARGMAX: Differentiable Position Extraction
# =============================================================================

def soft_argmax_2d(probs: torch.Tensor, H: int, W: int, temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute soft (differentiable) argmax coordinates.
    
    Instead of returning the index of the max value (non-differentiable),
    this returns the expected position using softmax-weighted coordinates.
    
    Args:
        probs: Probability tensor of shape [B, C, H*W]
        H: Height of the spatial grid
        W: Width of the spatial grid  
        temperature: Lower = sharper (more like hard argmax), Higher = softer
                    Recommended: 0.1 for training, can decrease for inference
    
    Returns:
        y: Expected Y coordinates [B, C] in range [0, 1]
        x: Expected X coordinates [B, C] in range [0, 1]
    
    Why this helps:
        - With hard argmax, the position is just a number with no gradient
        - With soft argmax, gradients can flow through: "if you increase the 
          probability at position (0.3, 0.7), the expected position moves there"
        - This allows the model to learn position-action associations directly
    """
    B, C, HW = probs.shape
    device = probs.device
    
    # Create normalized coordinate grids [0, 1]
    y_coords = torch.arange(H, device=device).float() / max(H - 1, 1)
    x_coords = torch.arange(W, device=device).float() / max(W - 1, 1)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    y_flat = y_grid.flatten()  # [HW]
    x_flat = x_grid.flatten()  # [HW]
    
    # Sharpen with temperature and normalize to get attention weights
    # Lower temperature = more peaked distribution (closer to argmax)
    weights = F.softmax(probs / temperature, dim=-1)  # [B, C, HW]
    
    # Compute expected (weighted average) coordinates
    y = (weights * y_flat).sum(dim=-1)  # [B, C]
    x = (weights * x_flat).sum(dim=-1)  # [B, C]
    
    return y, x


# =============================================================================
# 2. LEARNABLE TEMPORAL ATTENTION FOR X3D
# =============================================================================

class TemporalAttentionPooling(nn.Module):
    """
    Learnable temporal attention pooling for X3D.
    
    Instead of simple mean/max pooling over time, this module learns
    which timesteps are most important for the current prediction.
    
    The X3D paper uses global spatiotemporal averaging in the head, but
    for action detection, learnable attention can be beneficial because:
    - Some actions are defined by the end state (e.g., "opening door" = door open at end)
    - Some actions need full context (e.g., "walking" = movement over time)
    - Model can learn task-specific temporal weighting
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Temporal attention: compute importance score for each timestep
        # Uses squeeze-excitation style attention
        self.temporal_squeeze = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool spatial, keep temporal
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-scale temporal pooling weights (learnable)
        # Combines: recent (last), mid-term, and overall context
        self.scale_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  # recent, mid, overall
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] - 3D features from X3D backbone
            
        Returns:
            out: [B, C, H, W] - Temporally pooled features
        """
        B, C, T, H, W = x.shape
        
        if T == 1:
            return x.squeeze(2)
        
        # Method 1: Attention-weighted pooling
        # Compute per-timestep attention scores
        squeezed = self.temporal_squeeze(x)  # [B, C, T, 1, 1]
        squeezed = squeezed.view(B, C, T)     # [B, C, T]
        attn = self.temporal_attention(squeezed)  # [B, C, T]
        attn = attn.view(B, C, T, 1, 1)       # [B, C, T, 1, 1]
        
        # Apply attention and sum over time
        attended = (x * attn).sum(dim=2)  # [B, C, H, W]
        
        # Method 2: Multi-scale temporal features
        recent = x[:, :, -1]                          # [B, C, H, W] - last frame
        mid = x[:, :, T//2:].mean(dim=2)              # [B, C, H, W] - second half
        overall = x.mean(dim=2)                       # [B, C, H, W] - all frames
        
        # Normalize scale weights
        weights = F.softmax(self.scale_weights, dim=0)
        multi_scale = weights[0] * recent + weights[1] * mid + weights[2] * overall
        
        # Combine both methods (attended + multi-scale)
        out = 0.5 * attended + 0.5 * multi_scale
        
        return out


# =============================================================================
# 3. ACTION-OBJECT CO-OCCURRENCE MATRIX
# =============================================================================

class ActionObjectCooccurrence(nn.Module):
    """
    Explicit action-object co-occurrence modeling.
    
    This module learns which objects commonly co-occur with which actions:
    - "typing" strongly associated with "laptop", "keyboard"
    - "eating" strongly associated with "food", "fork", "spoon"
    - "sitting" strongly associated with "chair", "couch", "bed"
    
    The matrix is initialized to zeros (no prior knowledge) and learned from data.
    It provides an additional signal to the action head based on detected objects.
    
    Integration:
        This is applied AFTER the GlobalSceneContext, adding object-based
        action priors to the action logits.
    """
    
    def __init__(self, num_actions: int = 157, num_objects: int = 36, 
                 hidden_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.num_actions = num_actions
        self.num_objects = num_objects
        self.temperature = temperature
        
        # Learnable co-occurrence embeddings
        # Instead of raw matrix, use low-rank factorization for efficiency
        # M = action_embed @ object_embed.T
        self.action_embed = nn.Parameter(torch.randn(num_actions, hidden_dim) * 0.01)
        self.object_embed = nn.Parameter(torch.randn(num_objects, hidden_dim) * 0.01)
        
        # Scale factor for co-occurrence contribution
        self.cooc_scale = nn.Parameter(torch.tensor(0.1))
        
    def get_cooccurrence_matrix(self) -> torch.Tensor:
        """Get the full co-occurrence matrix (for visualization/analysis)."""
        return torch.matmul(self.action_embed, self.object_embed.T)  # [num_actions, num_objects]
    
    def forward(self, action_logits: torch.Tensor, object_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply object-based action priors.
        
        Args:
            action_logits: [B, num_actions, H, W] - raw action predictions
            object_probs: [B, num_objects, H, W] - softmax object probabilities
            
        Returns:
            enhanced_logits: [B, num_actions, H, W] - action logits with object priors
        """
        B, A, H, W = action_logits.shape
        
        # Compute co-occurrence matrix
        cooc = torch.matmul(self.action_embed, self.object_embed.T)  # [A, O]
        
        # For each spatial position, compute object-weighted action prior
        # Global object presence (max-pool across space)
        obj_global = object_probs.max(dim=-1)[0].max(dim=-1)[0]  # [B, O]
        
        # Object-to-action contribution: which actions are likely given these objects?
        action_prior = torch.matmul(obj_global, cooc.T)  # [B, A]
        
        # Broadcast to spatial dimensions
        action_prior = action_prior.unsqueeze(-1).unsqueeze(-1)  # [B, A, 1, 1]
        action_prior = action_prior.expand(-1, -1, H, W)  # [B, A, H, W]
        
        # Add scaled prior to logits
        enhanced_logits = action_logits + self.cooc_scale * action_prior
        
        return enhanced_logits


# =============================================================================
# 4. CLASS-WEIGHTED FOCAL LOSS
# =============================================================================

class ClassWeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class frequency weighting.
    
    Combines two strategies for handling class imbalance:
    1. Focal Loss: Down-weights easy (well-classified) examples
    2. Class weighting: Up-weights rare classes
    
    For 157 action classes with highly imbalanced frequencies, this helps
    the model learn rare but important actions like "playing saxophone"
    instead of only focusing on common actions like "standing".
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 class_weights: Optional[torch.Tensor] = None,
                 reduction: str = 'none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Class weights: higher weight for rare classes
        # Should be computed from dataset: weight[i] = 1 / frequency[i]
        # Normalized so mean weight = 1
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted focal loss.
        
        Args:
            logits: [B, C, ...] - raw predictions (before sigmoid)
            targets: [B, C, ...] - binary targets
            
        Returns:
            loss: Weighted focal loss
        """
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal term: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for positive/negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * ce_loss
        
        # Apply per-class weights if provided
        if self.class_weights is not None:
            # Weights shape: [C] -> broadcast to loss shape
            weight_shape = [1] * len(loss.shape)
            weight_shape[1] = -1  # Class dimension
            weights = self.class_weights.view(*weight_shape)
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_class_weights_from_dataset(dataset, num_classes: int, 
                                       class_offset: int = 36) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from dataset.
    
    Args:
        dataset: CharadesAGDataset or similar
        num_classes: Number of action classes (157)
        class_offset: Offset in label vector where actions start (36 for objects)
        
    Returns:
        weights: [num_classes] tensor of per-class weights
    """
    # Count occurrences of each class
    class_counts = torch.zeros(num_classes)
    
    for i in range(len(dataset)):
        _, _, target = dataset[i]
        if 'labels' in target and len(target['labels']) > 0:
            labels = target['labels']
            # Get action labels (after object classes)
            action_labels = labels[:, class_offset:class_offset + num_classes]
            # Sum across boxes (if any action is present in any box)
            class_counts += action_labels.max(dim=0)[0]
    
    # Compute inverse frequency weights
    # Add smoothing to avoid division by zero
    weights = 1.0 / (class_counts + 1)
    
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    
    # Clip extreme weights
    weights = weights.clamp(min=0.1, max=10.0)
    
    return weights


# =============================================================================
# 5. LABEL SMOOTHING
# =============================================================================

def apply_label_smoothing(labels: torch.Tensor, smoothing: float = 0.1, 
                          num_classes: Optional[int] = None) -> torch.Tensor:
    """
    Apply label smoothing to reduce overconfidence.
    
    For multi-hot labels (like actions), smoothing works as:
    - Positive labels: 1.0 -> 1.0 - smoothing + smoothing/num_classes
    - Negative labels: 0.0 -> smoothing/num_classes
    
    This prevents the model from becoming overconfident on training data,
    which can improve generalization.
    
    Args:
        labels: [B, C, ...] - original labels
        smoothing: Smoothing factor (0.1 = 10% smoothing)
        num_classes: Number of classes (inferred from labels if not provided)
        
    Returns:
        smoothed_labels: Labels with smoothing applied
    """
    if num_classes is None:
        num_classes = labels.shape[1] if len(labels.shape) > 1 else labels.shape[0]
    
    # For multi-hot: shift labels toward uniform
    smoothed = labels * (1 - smoothing) + smoothing / num_classes
    
    return smoothed


# =============================================================================
# UTILITY: Integration into existing architecture
# =============================================================================

def integrate_soft_argmax_into_global_scene_context(module):
    """
    Monkey-patch GlobalSceneContext to use soft-argmax.
    
    This replaces the hard argmax position extraction with differentiable
    soft argmax, enabling gradient flow through position computations.
    
    Usage:
        from models.yowo.improvements import integrate_soft_argmax_into_global_scene_context
        for m in model.obj_rel_cross_attn:
            integrate_soft_argmax_into_global_scene_context(m)
    """
    original_forward = module.forward
    
    def patched_forward(cls_feat, obj_pred, rel_pred, return_weights=False):
        B, C, H, W = cls_feat.shape
        
        obj_probs = F.softmax(obj_pred, dim=1)
        rel_probs = torch.sigmoid(rel_pred)
        
        # Global confidences (unchanged)
        obj_global = obj_probs.max(dim=-1)[0].max(dim=-1)[0]
        rel_global = rel_probs.max(dim=-1)[0].max(dim=-1)[0]
        
        # SOFT position extraction (DIFFERENTIABLE!)
        obj_flat = obj_probs.view(B, module.num_objects, -1)
        obj_y, obj_x = soft_argmax_2d(obj_flat, H, W, temperature=0.1)
        
        rel_flat = rel_probs.view(B, module.num_relations, -1)
        rel_y, rel_x = soft_argmax_2d(rel_flat, H, W, temperature=0.1)
        
        # Relative positions
        person_y = obj_y[:, 0:1]
        person_x = obj_x[:, 0:1]
        obj_rel_y = obj_y - person_y
        obj_rel_x = obj_x - person_x
        rel_rel_y = rel_y - person_y
        rel_rel_x = rel_x - person_x
        
        # Combine
        global_context = torch.cat([
            obj_global, rel_global,
            obj_y, obj_x, rel_y, rel_x,
            obj_rel_y, obj_rel_x, rel_rel_y, rel_rel_x
        ], dim=1)
        
        # Rest of forward (context projection, fusion, etc.)
        context_embed = module.context_proj(global_context)
        context_broadcast = context_embed.unsqueeze(-1).unsqueeze(-1)
        context_broadcast = context_broadcast.expand(-1, -1, H, W)
        
        feat_scale = cls_feat.abs().mean().clamp(min=0.01)
        ctx_scale = context_broadcast.abs().mean().clamp(min=0.01)
        context_scaled = context_broadcast * (feat_scale / ctx_scale)
        context_scaled = context_scaled * module.context_scale.abs().clamp(min=0.1, max=2.0)
        
        combined = torch.cat([cls_feat, context_scaled], dim=1)
        delta = module.fusion(combined)
        out = module.norm(cls_feat + delta)
        
        if return_weights:
            dummy_weights = torch.ones(B, H*W, H*W, device=cls_feat.device) / (H*W)
            return out, dummy_weights
        return out
    
    module.forward = patched_forward
    print(f"✅ Integrated soft-argmax into {module.__class__.__name__}")
