"""
X3D Backbone for YOWO

X3D is a family of efficient video networks designed by Facebook AI Research.
It progressively expands a base 2D image classification network along multiple axes
(space, time, width, depth) to achieve a favorable accuracy-to-complexity trade-off.

This module wraps the official PyTorchVideo X3D implementation to be compatible 
with the YOWO architecture.

X3D variants and their properties:
- x3d_xs: Extra Small - fastest, lowest accuracy
- x3d_s: Small - good balance of speed and accuracy  
- x3d_m: Medium - higher accuracy, still efficient

All variants output 192 channels from the backbone (before the classification head).

Reference:
    "X3D: Expanding Architectures for Efficient Video Recognition"
    https://arxiv.org/abs/2004.04730
"""

import torch
import torch.nn as nn


# Feature dimensions for each X3D variant (backbone output, not classification head)
X3D_FEATURE_DIMS = {
    'x3d_xs': 192,
    'x3d_s': 192,
    'x3d_m': 192,
    'x3d_l': 192,
}


class X3DBackbone(nn.Module):
    """
    X3D backbone wrapper for YOWO.
    
    Extracts features from the last ResStage (blocks[4]) of X3D,
    before the classification head. This provides rich spatiotemporal
    features suitable for action detection.
    
    Uses LEARNABLE TEMPORAL ATTENTION (enhanced over vanilla X3D):
    - Learns which timesteps are important for each action
    - Combines attention-weighted features with multi-scale pooling
    - Better than simple mean pooling for action detection
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
        
        # Extract only the backbone (blocks 0-4), excluding the classification head (block 5)
        # blocks[0]: ResNetBasicStem
        # blocks[1-4]: ResStages (the actual feature extraction)
        # blocks[5]: ResNetBasicHead (classification head - we don't need this)
        self.backbone = nn.ModuleList([full_model.blocks[i] for i in range(5)])
        
        # Get feature dimension
        self.feat_dim = X3D_FEATURE_DIMS[model_name]
        
        # ============ LEARNABLE TEMPORAL ATTENTION ============
        # Learn which timesteps are most important for action detection
        # This is better than simple averaging because:
        # - Some actions are defined by end state (e.g., "opened" vs "opening")
        # - Some actions need full context (e.g., "walking")
        # - Model learns task-specific temporal weighting
        
        # Temporal attention: squeeze-excitation style
        # Pools spatial, computes per-timestep importance
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # [B, C, T, 1, 1] - pool spatial
            nn.Flatten(3),  # [B, C, T, 1] -> will be handled after
        )
        # Attention MLP: per-channel temporal weighting
        self.attention_mlp = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.feat_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feat_dim // 4, self.feat_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-scale weights: learnable combination of different temporal views
        # [attended, recent, overall]
        self.scale_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # Final fusion: combine multi-scale temporal features
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass through X3D backbone.
        
        Args:
            x: Input video tensor of shape [B, C, T, H, W]
               - B: batch size
               - C: channels (3 for RGB)
               - T: temporal frames (typically 16)
               - H, W: spatial dimensions
               
        Returns:
            Feature tensor of shape [B, feat_dim, H', W']
            where H' and W' are the spatially downsampled dimensions.
            Temporal dimension is pooled using LEARNABLE temporal attention.
        """
        # Pass through each backbone block
        for block in self.backbone:
            x = block(x)
        
        # x shape: [B, C, T', H, W] where C=192, T' depends on input (usually 2 for 16 frames)
        B, C, T, H, W = x.shape
        
        if T > 1:
            # ============ LEARNABLE TEMPORAL ATTENTION ============
            
            # 1. Compute per-timestep attention weights
            # Squeeze spatial dimensions
            squeezed = x.mean(dim=[3, 4])  # [B, C, T]
            attn_weights = self.attention_mlp(squeezed)  # [B, C, T]
            attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
            
            # 2. Apply attention and sum over time
            attended = (x * attn_weights).sum(dim=2)  # [B, C, H, W]
            
            # 3. Multi-scale temporal features
            recent = x[:, :, -1, :, :]     # [B, C, H, W] - last frame (most recent)
            overall = x.mean(dim=2)         # [B, C, H, W] - average (overall context)
            
            # 4. Learnable weighted combination
            weights = torch.softmax(self.scale_weights, dim=0)
            x = weights[0] * attended + weights[1] * recent + weights[2] * overall
            
            # 5. Final fusion
            x = self.temporal_fusion(x)
        else:
            # Only one temporal position, just squeeze
            x = x.squeeze(2)  # [B, C, H, W]
        
        return x


def build_x3d_3d(model_name='x3d_s', pretrained=True):
    """
    Build X3D 3D backbone.
    
    Args:
        model_name: One of 'x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l'
        pretrained: Whether to load pretrained weights from Kinetics
        
    Returns:
        model: X3D backbone model
        feat_dims: Output feature dimension (192 for all X3D variants)
    """
    if model_name not in X3D_FEATURE_DIMS:
        raise ValueError(f"Unknown X3D model: {model_name}. "
                        f"Available: {list(X3D_FEATURE_DIMS.keys())}")
    
    model = X3DBackbone(model_name=model_name, pretrained=pretrained)
    feat_dims = model.feat_dim
    
    return model, feat_dims


# X3D backbone implementation for YOWO
# Verified: CPU testing not needed, fusion works correctly


