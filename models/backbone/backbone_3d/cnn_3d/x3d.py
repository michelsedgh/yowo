"""
X3D Backbone for YOWO with 4-POOL TEMPORAL AWARENESS

X3D is a family of efficient video networks designed by Facebook AI Research.

=== 4-POOL TEMPORAL ARCHITECTURE ===

Instead of collapsing 16 timesteps into 1 (losing temporal order),
we pool into 4 temporal segments and CONCATENATE:

   Frames 1-4   → pool → early      [192, 7, 7]
   Frames 5-8   → pool → early_mid  [192, 7, 7]
   Frames 9-12  → pool → late_mid   [192, 7, 7]
   Frames 13-16 → pool → late       [192, 7, 7]
   ───────────────────────────────────────────────
   CONCATENATE  →                   [768, 7, 7]

This preserves temporal order:
- The heads can learn "for standing up, focus on late channels"
- The heads can learn "for sitting down, focus on early channels"

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

# Output dimension after 4-pool concatenation
X3D_OUTPUT_DIMS = {
    'x3d_xs': 768,  # 192 × 4
    'x3d_s': 768,
    'x3d_m': 768,
    'x3d_l': 768,
}


class X3DBackbone(nn.Module):
    """
    X3D backbone with 4-POOL temporal awareness.
    
    Key Features:
    - Pools 16 frames into 4 temporal segments (early, early_mid, late_mid, late)
    - Concatenates into 768 channels (preserves temporal order!)
    - Position encoding for attention-based refinement
    - Attention mechanism with temporal context (k=3)
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
        
        # Extract backbone (blocks 0-4)
        self.backbone = nn.ModuleList([full_model.blocks[i] for i in range(5)])
        
        # Feature dimensions
        self.base_feat_dim = X3D_FEATURE_DIMS[model_name]  # 192
        self.feat_dim = X3D_OUTPUT_DIMS[model_name]  # 768 (4 × 192)
        
        # Number of temporal pools
        self.num_pools = 4
        
        # Position encoding for attention (16 temporal positions)
        self.max_temporal_positions = 16
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.base_feat_dim, self.max_temporal_positions) * 0.02
        )
        
        # Attention MLP with temporal context (k=3)
        self.attention_mlp = nn.Sequential(
            nn.Conv1d(self.base_feat_dim, self.base_feat_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.base_feat_dim // 4, self.base_feat_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Optional: Temporal fusion layer to refine concatenated features
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass with 4-pool temporal awareness.
        
        Args:
            x: Input video tensor [B, C, T, H, W]
               
        Returns:
            Feature tensor [B, 768, H', W']
            768 = 192 × 4 (early + early_mid + late_mid + late)
        """
        # Pass through X3D backbone
        for block in self.backbone:
            x = block(x)
        
        # x: [B, 192, T', H, W] where T' depends on input (usually 16 for 16 frames)
        B, C, T, H, W = x.shape
        
        if T >= 4:
            # ============ 4-POOL TEMPORAL ARCHITECTURE ============
            
            # Calculate segment boundaries
            seg_size = T // 4
            
            # Pool each temporal segment (with attention weighting)
            segments = []
            for i in range(4):
                start_t = i * seg_size
                end_t = (i + 1) * seg_size if i < 3 else T  # Last segment gets remainder
                
                seg_features = x[:, :, start_t:end_t]  # [B, C, seg_size, H, W]
                
                # Apply attention within segment
                seg_T = seg_features.shape[2]
                if seg_T > 1:
                    # Squeeze spatial for attention
                    squeezed = seg_features.mean(dim=[3, 4])  # [B, C, seg_T]
                    
                    # Add position encoding for this segment
                    pos_start = start_t
                    pos_end = min(start_t + seg_T, self.max_temporal_positions)
                    pos_len = pos_end - pos_start
                    if pos_len > 0:
                        squeezed[:, :, :pos_len] = squeezed[:, :, :pos_len] + self.pos_embed[:, :, pos_start:pos_end]
                    
                    # Compute attention weights
                    attn_weights = self.attention_mlp(squeezed)  # [B, C, seg_T]
                    attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, seg_T, 1, 1]
                    
                    # Weighted sum over time
                    pooled = (seg_features * attn_weights).sum(dim=2)  # [B, C, H, W]
                else:
                    pooled = seg_features.squeeze(2)  # [B, C, H, W]
                
                segments.append(pooled)
            
            # Concatenate all 4 segments: [B, 768, H, W]
            x = torch.cat(segments, dim=1)  # [B, 192*4, H, W] = [B, 768, H, W]
            
            # Optional refinement
            x = self.temporal_fusion(x)
            
        elif T > 1:
            # Fallback for very few frames: simple mean then replicate
            x = x.mean(dim=2)  # [B, C, H, W]
            x = x.repeat(1, 4, 1, 1)  # [B, 768, H, W]
            x = self.temporal_fusion(x)
        else:
            # Single frame: replicate
            x = x.squeeze(2)  # [B, C, H, W]
            x = x.repeat(1, 4, 1, 1)  # [B, 768, H, W]
            x = self.temporal_fusion(x)
        
        return x
    
    def get_temporal_info(self):
        """Get info about temporal processing for debugging."""
        return {
            'num_temporal_pools': self.num_pools,
            'base_feat_dim': self.base_feat_dim,
            'output_feat_dim': self.feat_dim,
            'temporal_segments': ['early (1-4)', 'early_mid (5-8)', 'late_mid (9-12)', 'late (13-16)'],
        }


def build_x3d_3d(model_name='x3d_s', pretrained=True):
    """
    Build X3D 3D backbone with 4-pool temporal awareness.
    
    Args:
        model_name: One of 'x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l'
        pretrained: Whether to load pretrained weights from Kinetics
        
    Returns:
        model: X3D backbone model
        feat_dims: Output feature dimension (768 = 192 × 4 temporal pools)
    """
    if model_name not in X3D_FEATURE_DIMS:
        raise ValueError(f"Unknown X3D model: {model_name}. "
                        f"Available: {list(X3D_FEATURE_DIMS.keys())}")
    
    model = X3DBackbone(model_name=model_name, pretrained=pretrained)
    feat_dims = model.feat_dim  # 768
    
    print(f"  4-POOL temporal architecture: ENABLED")
    print(f"  - 4 temporal segments: early, early_mid, late_mid, late")
    print(f"  - Base features: {model.base_feat_dim} per segment")
    print(f"  - Output features: {feat_dims} (concatenated)")
    print(f"  - Position encoding: YES")
    print(f"  - Attention context: k=3")
    
    return model, feat_dims
