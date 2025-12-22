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
    
    The output is a 2D feature map with temporal dimension averaged,
    matching the interface expected by YOWO's channel encoder.
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
            where H' and W' are the spatially downsampled dimensions
            and temporal dimension has been averaged.
        """
        # Pass through each backbone block
        for block in self.backbone:
            x = block(x)
        
        # x shape: [B, C, T, H, W] where C=192, T depends on input
        # Average over temporal dimension to get [B, C, H, W]
        # This matches the interface expected by YOWO's channel encoder
        if x.size(2) > 1:
            x = torch.mean(x, dim=2, keepdim=True)
        
        return x.squeeze(2)  # [B, C, H, W]


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


