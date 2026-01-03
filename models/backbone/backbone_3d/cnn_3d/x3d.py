"""
X3D Backbone for YOWO with DUAL-PATHWAY TEMPORAL POOLING

X3D is a family of efficient video networks designed by Facebook AI Research.

=== DUAL-PATHWAY TEMPORAL POOLING (Key Innovation) ===

Problem: Different action types need different temporal pooling strategies:
- TRANSIENT actions (throwing, taking): Need MAX pooling to capture peak moments
- ORDER-SENSITIVE actions (opening vs closing): Need attention pooling for direction
- STATIC actions (sitting, holding): Either works fine

Solution: Use BOTH pooling methods with SE-style adaptive fusion:

    X3D Features [B, C, T, H, W]
           ↓
    ┌──────┴──────┐
    ↓              ↓
  MAX Pool    Attention Pool
    ↓              ↓
 [B, C, H, W]   [B, C, H, W]
    └──────┬──────┘
           ↓
      Concatenate [B, 2C, H, W]
           ↓
      SE-style Channel Attention
      (learns which channels need max vs attention)
           ↓
      1x1 Conv Fusion [B, C, H, W]

The SE-style attention learns per-channel importance, allowing the model
to dynamically select max features for transient actions and attention
features for order-sensitive actions.

Reference:
    "X3D: Expanding Architectures for Efficient Video Recognition"
    https://arxiv.org/abs/2004.04730
"""

import torch
import torch.nn as nn


# Feature dimensions for each X3D variant
X3D_FEATURE_DIMS = {
    'x3d_xs': 192,
    'x3d_s': 192,
    'x3d_m': 192,
    'x3d_l': 192,
}


class SEFusion(nn.Module):
    """
    Squeeze-and-Excitation style fusion for dual-pathway features.
    
    This module takes concatenated features from two pathways [B, 2C, H, W]
    and learns to weight each channel adaptively before fusion.
    
    The SE mechanism:
    1. Global average pooling to get channel statistics
    2. MLP to learn channel importance weights
    3. Sigmoid to get gate values
    4. Apply gates and fuse to final output
    """
    
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels  # 2 * C
        self.out_channels = out_channels  # C
        
        # Squeeze: Global average pooling (happens in forward)
        
        # Excitation: MLP to learn channel weights
        mid_channels = max(in_channels // reduction, 32)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid()
        )
        
        # Fusion: 1x1 conv to combine channels
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: Concatenated dual-pathway features [B, 2C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Squeeze: Global average pooling
        squeezed = x.mean(dim=[2, 3])  # [B, 2C]
        
        # Excitation: Learn channel weights
        weights = self.excitation(squeezed)  # [B, 2C]
        weights = weights.view(B, C, 1, 1)  # [B, 2C, 1, 1]
        
        # Apply channel weights
        x = x * weights
        
        # Fusion
        x = self.fusion(x)
        
        return x


class X3DBackbone(nn.Module):
    """
    X3D backbone wrapper for YOWO with DUAL-PATHWAY TEMPORAL POOLING.
    
    Key Features:
    1. DUAL-PATHWAY POOLING: Both max and attention pooling in parallel
       - Max pooling: Captures transient action peaks (throwing, taking)
       - Attention pooling: Preserves temporal order (opening vs closing)
    
    2. SE-STYLE ADAPTIVE FUSION: Channel attention learns optimal combination
       - Transient actions can emphasize max-pooled channels
       - Order-sensitive actions can emphasize attention channels
    
    3. MULTI-SCALE TEMPORAL: Different windows for different action durations
    """
    
    def __init__(self, model_name='x3d_s', pretrained=True):
        super().__init__()
        self.model_name = model_name
        
        # Load the full X3D model from torch hub
        print(f'Loading X3D model: {model_name} (pretrained={pretrained})')
        full_model = torch.hub.load(
            'facebookresearch/pytorchvideo', 
            model_name, 
            pretrained=pretrained
        )
        
        # Extract only the backbone (blocks 0-4)
        self.backbone = nn.ModuleList([full_model.blocks[i] for i in range(5)])
        
        # Get feature dimension
        self.feat_dim = X3D_FEATURE_DIMS[model_name]
        
        # ============ PATHWAY 1: ATTENTION-BASED POOLING ============
        # For ORDER-SENSITIVE actions (opening/closing, direction-aware)
        
        # Attention MLP: learns per-timestep importance
        self.attention_mlp = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.feat_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feat_dim // 4, self.feat_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-scale weights for attention pathway
        self.attention_scale_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # ============ PATHWAY 2: MAX POOLING ============
        # For TRANSIENT actions (throwing, taking, putting - brief peak moments)
        
        # Multi-scale max pooling weights
        self.max_scale_weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]))
        
        # ============ SE-STYLE ADAPTIVE FUSION ============
        # Learns per-channel which pathway is more important
        self.se_fusion = SEFusion(
            in_channels=self.feat_dim * 2,  # Concatenated pathways
            out_channels=self.feat_dim,
            reduction=4
        )
        
    def forward(self, x):
        """
        Forward pass through X3D backbone with dual-pathway pooling and SE fusion.
        
        Args:
            x: Input video tensor [B, C, T, H, W]
               
        Returns:
            Feature tensor [B, feat_dim, H', W']
        """
        # Pass through backbone blocks
        for block in self.backbone:
            x = block(x)
        
        # x: [B, C, T', H, W] where C=192
        B, C, T, H, W = x.shape
        
        if T > 1:
            # ============ PATHWAY 1: ATTENTION-BASED POOLING ============
            
            # Compute per-timestep attention
            squeezed = x.mean(dim=[3, 4])  # [B, C, T]
            attn_weights = self.attention_mlp(squeezed)  # [B, C, T]
            attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
            
            # Attention-weighted sum
            attended = (x * attn_weights).sum(dim=2)  # [B, C, H, W]
            
            # Multi-scale temporal views
            recent_attn = x[:, :, -1, :, :]  # Last frame
            overall_attn = x.mean(dim=2)      # Mean
            
            # Weighted combination
            attn_w = torch.softmax(self.attention_scale_weights, dim=0)
            attention_pooled = (attn_w[0] * attended + 
                               attn_w[1] * recent_attn + 
                               attn_w[2] * overall_attn)
            
            # ============ PATHWAY 2: MAX POOLING ============
            
            # Multi-scale max pooling
            t_recent = max(1, T // 4)  # Last 25%
            recent_max = x[:, :, -t_recent:].max(dim=2)[0]
            
            t_mid = max(1, T // 2)  # Last 50%
            mid_max = x[:, :, -t_mid:].max(dim=2)[0]
            
            full_max = x.max(dim=2)[0]  # Full clip
            
            # Weighted combination
            max_w = torch.softmax(self.max_scale_weights, dim=0)
            max_pooled = (max_w[0] * recent_max + 
                         max_w[1] * mid_max + 
                         max_w[2] * full_max)
            
            # ============ SE-STYLE ADAPTIVE FUSION ============
            # Concatenate both pathways
            dual_features = torch.cat([attention_pooled, max_pooled], dim=1)  # [B, 2C, H, W]
            
            # SE-style fusion learns optimal channel weights
            x = self.se_fusion(dual_features)  # [B, C, H, W]
            
        else:
            x = x.squeeze(2)
        
        return x
    
    def get_pathway_weights(self):
        """Get current pathway scale weights for monitoring."""
        return {
            'attention_weight': 0.5,  # Both pathways have equal architectural weight
            'max_weight': 0.5,        # The SE fusion learns the actual balance
            'attention_scales': torch.softmax(self.attention_scale_weights, dim=0).tolist(),
            'max_scales': torch.softmax(self.max_scale_weights, dim=0).tolist()
        }
    
    def get_se_channel_stats(self):
        """Get SE fusion statistics for analysis (after forward pass)."""
        # This could be extended to track which channels prefer which pathway
        return {
            'se_fusion_params': sum(p.numel() for p in self.se_fusion.parameters())
        }


def build_x3d_3d(model_name='x3d_s', pretrained=True):
    """
    Build X3D 3D backbone with dual-pathway temporal pooling.
    
    Args:
        model_name: One of 'x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l'
        pretrained: Whether to load pretrained weights from Kinetics
        
    Returns:
        model: X3D backbone model
        feat_dims: Output feature dimension (192)
    """
    if model_name not in X3D_FEATURE_DIMS:
        raise ValueError(f"Unknown X3D model: {model_name}. "
                        f"Available: {list(X3D_FEATURE_DIMS.keys())}")
    
    model = X3DBackbone(model_name=model_name, pretrained=pretrained)
    feat_dims = model.feat_dim
    
    print(f"  Dual-pathway temporal pooling: ENABLED")
    print(f"  - Pathway 1: Attention pooling (order-sensitive actions)")
    print(f"  - Pathway 2: Multi-scale max pooling (transient actions)")
    print(f"  - Fusion: SE-style channel attention (learns optimal blend)")
    
    return model, feat_dims
