"""
YOLO11 Backbone Integration for YOWOv2 - Option B (Native Dimensions)

This implementation preserves YOLO11's native feature dimensions [256, 512, 512]
to avoid information loss from projection, while still providing decoupled
heads for cls/reg feature separation.

Design Decision:
- FreeYOLO projects [256, 512, 1024] → [256, 256, 256] before decoupled heads
- Option A: Project [256, 512, 512] → [256, 256, 256] (matches FreeYOLO)
- Option B (THIS): Keep [256, 512, 512], use scale-specific decoupled heads

Option B Advantages:
1. No information loss from projection bottleneck
2. Preserves YOLO11's native feature representation
3. More capacity at P4/P5 (512 vs 256 channels)
4. YOWO's channel encoders handle non-uniform dims natively

YOLO11m Architecture Reference (from yolo11.yaml):
    - Layer 16: P3/8-small, 256 channels (C3k2 output)
    - Layer 19: P4/16-medium, 512 channels (C3k2 output)
    - Layer 22: P5/32-large, 512 channels (capped by max_channels)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO


# ============================================================================
# Activation and Normalization helpers
# ============================================================================

def get_activation(act_type='silu'):
    """Get activation function by name."""
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    else:
        return nn.SiLU(inplace=True)


def get_norm(norm_type, dim):
    """Get normalization layer by name."""
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    else:
        return nn.BatchNorm2d(dim)


# ============================================================================
# Basic Conv block (matching FreeYOLO's Conv)
# ============================================================================

class Conv(nn.Module):
    """Standard Conv block: Conv2d + Norm + Activation"""
    def __init__(self, c1, c2, k=1, p=0, s=1, act_type='silu', norm_type='BN'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            get_norm(norm_type, c2),
            get_activation(act_type),
        )
    
    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Decoupled Head (scale-specific version for Option B)
# ============================================================================

class DecoupledHead(nn.Module):
    """
    Decoupled Head for separating classification and regression features.
    
    This version accepts ANY input dimension (scale-specific).
    Input dim = output dim (no projection within head).
    """
    def __init__(self, feat_dim, num_cls_heads=2, num_reg_heads=2, 
                 act_type='silu', norm_type='BN'):
        super().__init__()
        
        self.feat_dim = feat_dim
        
        # Classification feature branch
        self.cls_feats = nn.Sequential(*[
            Conv(feat_dim, feat_dim, k=3, p=1, s=1,
                 act_type=act_type, norm_type=norm_type)
            for _ in range(num_cls_heads)
        ])
        
        # Regression feature branch  
        self.reg_feats = nn.Sequential(*[
            Conv(feat_dim, feat_dim, k=3, p=1, s=1,
                 act_type=act_type, norm_type=norm_type)
            for _ in range(num_reg_heads)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, feat_dim, H, W]
        Returns:
            cls_feats: Classification features [B, feat_dim, H, W]
            reg_feats: Regression features [B, feat_dim, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)
        return cls_feats, reg_feats


# ============================================================================
# Main YOLO11 Backbone Class - Option B (Native Dimensions)
# ============================================================================

class YOLO11Backbone(nn.Module):
    """
    YOLO11 backbone + PANet neck for YOWOv2 - Option B.
    
    This implementation:
    1. Extracts FPN features from YOLO11 (layers 16, 19, 22)
    2. Keeps NATIVE dimensions [256, 512, 512] (no projection!)
    3. Applies scale-specific decoupled heads for cls/reg separation
    4. Returns (cls_feats, reg_feats) compatible with YOWOv2
    
    YOWO's channel encoders are designed to handle non-uniform dims:
        encoder_i takes bk_dim_2d[i] + bk_dim_3d as input
    So we don't need to project to uniform dimensions.
    """
    
    # Configuration matching FreeYOLO's style
    NUM_CLS_HEADS = 2
    NUM_REG_HEADS = 2
    ACT_TYPE = 'silu'
    NORM_TYPE = 'BN'
    
    def __init__(self, model_name='yolo11m.pt', pretrained=True):
        super().__init__()
        
        # ============ Load YOLO11 Model ============
        print('=' * 40)
        print(f'Loading YOLO11: {model_name}')
        yolo = YOLO(model_name)
        
        # Get the model layers
        self.model = yolo.model.model
        
        # FPN output layers
        self.feature_indices = [16, 19, 22]
        self.stop_layer = 23
        
        # Get which outputs to save for later layers
        self.save = list(yolo.model.save)
        for idx in self.feature_indices:
            if idx not in self.save:
                self.save.append(idx)
        self.save.sort()
        
        # ============ Get Native Feature Dimensions ============
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            raw_feats = self._extract_raw_features(dummy)
            self.fpn_dims = [f.shape[1] for f in raw_feats]
        
        print(f'  Native FPN dimensions: {self.fpn_dims}')
        print(f'  Keeping native dims (Option B - no projection)')
        
        # ============ Scale-Specific Decoupled Heads ============
        # Each scale has its OWN decoupled head with DIFFERENT input dim
        self.decoupled_heads = nn.ModuleList([
            DecoupledHead(
                feat_dim=dim,  # USE NATIVE DIM FOR EACH SCALE
                num_cls_heads=self.NUM_CLS_HEADS,
                num_reg_heads=self.NUM_REG_HEADS,
                act_type=self.ACT_TYPE,
                norm_type=self.NORM_TYPE
            )
            for dim in self.fpn_dims
        ])
        
        print(f'  Decoupled heads (scale-specific):')
        for i, dim in enumerate(self.fpn_dims):
            print(f'    P{i+3}: {dim}ch → cls_head + reg_head → {dim}ch')
        
        # ============ Enable Gradients for Fine-tuning ============
        for param in self.model.parameters():
            param.requires_grad = True
        
        print('=' * 40)
    
    def _extract_raw_features(self, x):
        """Extract raw FPN features from YOLO11."""
        y = []
        outputs = []
        
        for i, m in enumerate(self.model):
            if i >= self.stop_layer:
                break
            
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)
            y.append(x if i in self.save else None)
            
            if i in self.feature_indices:
                outputs.append(x)
        
        return outputs
    
    def forward(self, x):
        """
        Forward pass through YOLO11 backbone with decoupled heads.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            (cls_feats, reg_feats): Tuple of feature lists
                cls_feats: [P3, P4, P5] with dims [256, 512, 512]
                reg_feats: [P3, P4, P5] with dims [256, 512, 512]
        """
        # Extract raw FPN features
        raw_feats = self._extract_raw_features(x)
        
        # Apply scale-specific decoupled heads
        all_cls_feats = []
        all_reg_feats = []
        
        for feat, head in zip(raw_feats, self.decoupled_heads):
            cls_feat, reg_feat = head(feat)
            all_cls_feats.append(cls_feat)
            all_reg_feats.append(reg_feat)
        
        return all_cls_feats, all_reg_feats


# ============================================================================
# Builder Function
# ============================================================================

def build_yolo_11(model_name='yolo11m.pt', pretrained=True):
    """
    Build YOLO11 backbone for YOWOv2 (Option B - Native Dimensions).
    
    Args:
        model_name: YOLO11 variant ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', etc.)
        pretrained: Whether to use pretrained weights
        
    Returns:
        model: YOLO11Backbone instance
        feat_dims: List of feature dimensions [256, 512, 512] for yolo11m
                   NOTE: Non-uniform! YOWO handles this natively.
    """
    model = YOLO11Backbone(model_name, pretrained)
    
    # Return NATIVE dimensions (not uniform like FreeYOLO)
    feat_dims = model.fpn_dims
    
    print(f'\nYOLO11 integrated for YOWOv2 (Option B):')
    print(f'  Feature dimensions (native): {feat_dims}')
    print(f'  Feature strides: [8, 16, 32]')
    print(f'  Output format: (cls_feats, reg_feats) - decoupled!')
    print(f'  Note: YOWO handles non-uniform dims via per-level encoders')
    
    return model, feat_dims


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('Testing YOLO11 Backbone - Option B (Native Dimensions)')
    print('=' * 60 + '\n')
    
    # Build model
    model, feat_dims = build_yolo_11(model_name='yolo11m.pt', pretrained=True)
    model.eval()
    
    print(f'\nFeature dimensions (native): {feat_dims}')
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    cls_feats, reg_feats = model(x)
    
    print('\nOutput shapes:')
    for i, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
        print(f'  P{i+3}: cls={list(cls_feat.shape)}, reg={list(reg_feat.shape)}')
    
    # Verify dimensions match
    assert [f.shape[1] for f in cls_feats] == feat_dims, "Dimension mismatch!"
    
    print('\n✅ YOLO11 Option B test passed!')
    print('   Native dimensions preserved, decoupled heads working.')
