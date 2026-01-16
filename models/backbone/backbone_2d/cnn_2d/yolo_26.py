"""
YOLO26 Backbone Integration for YOWOv2

This implementation extracts FPN features from YOLO26, preserving native 
dimensions for use in YOWO's multi-task architecture.

Based on yolo_11.py - YOLO26 has the same interface via ultralytics,
just with improved backbone weights and training.

YOLO26 key features (vs YOLO11):
    - NMS-free native design (we only use backbone features, not detection head)
    - DFL removal (simpler box regression - not relevant for us)
    - Improved small object detection (ProgLoss + STAL during training)
    - 43% faster CPU inference

For YOWO, we only care about the backbone FPN features, which have the same
structure as YOLO11: P3 [256], P4 [512], P5 [512] for the 'l' variant.
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
# Basic Conv block
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
# Decoupled Head (for cls/reg feature separation)
# ============================================================================

class DecoupledHead(nn.Module):
    """
    Decoupled Head for separating classification and regression features.
    Accepts any input dimension and outputs same dimension.
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
# Main YOLO26 Backbone Class
# ============================================================================

class YOLO26Backbone(nn.Module):
    """
    YOLO26 backbone + PANet neck for YOWOv2.
    
    This implementation:
    1. Extracts FPN features from YOLO26 (layers 16, 19, 22)
    2. Keeps NATIVE dimensions (no projection)
    3. Applies scale-specific decoupled heads for cls/reg separation
    4. Returns (cls_feats, reg_feats) compatible with YOWOv2
    """
    
    # Configuration
    NUM_CLS_HEADS = 2
    NUM_REG_HEADS = 2
    ACT_TYPE = 'silu'
    NORM_TYPE = 'BN'
    
    def __init__(self, model_name='yolo26l.pt', pretrained=True):
        super().__init__()
        
        # ============ Load YOLO26 Model ============
        print('=' * 40)
        print(f'Loading YOLO26: {model_name}')
        yolo = YOLO(model_name)
        
        # Get the model layers
        self.model = yolo.model.model
        
        # FPN output layers (same structure as YOLO11)
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
        print(f'  (P3: {self.fpn_dims[0]}, P4: {self.fpn_dims[1]}, P5: {self.fpn_dims[2]})')
        
        # ============ Scale-Specific Decoupled Heads ============
        self.decoupled_heads = nn.ModuleList([
            DecoupledHead(
                feat_dim=dim,
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
        """Extract raw FPN features from YOLO26."""
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
        Forward pass through YOLO26 backbone with decoupled heads.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            (cls_feats, reg_feats): Tuple of feature lists
                cls_feats: [P3, P4, P5] classification features
                reg_feats: [P3, P4, P5] regression features
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

def build_yolo_26(model_name='yolo26l.pt', pretrained=True):
    """
    Build YOLO26 backbone for YOWOv2.
    
    Args:
        model_name: YOLO26 variant ('yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 
                                    'yolo26l.pt', 'yolo26x.pt')
        pretrained: Whether to use pretrained weights
        
    Returns:
        model: YOLO26Backbone instance
        feat_dims: List of feature dimensions (e.g., [256, 512, 512] for yolo26l)
    """
    model = YOLO26Backbone(model_name, pretrained)
    feat_dims = model.fpn_dims
    
    print(f'\nYOLO26 integrated for YOWOv2:')
    print(f'  Model: {model_name}')
    print(f'  Feature dimensions: {feat_dims}')
    print(f'  Feature strides: [8, 16, 32]')
    print(f'  Output format: (cls_feats, reg_feats) - decoupled!')
    
    return model, feat_dims


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('Testing YOLO26 Backbone')
    print('=' * 60 + '\n')
    
    # Build model
    model, feat_dims = build_yolo_26(model_name='yolo26l.pt', pretrained=True)
    model.eval()
    
    print(f'\nFeature dimensions: {feat_dims}')
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    cls_feats, reg_feats = model(x)
    
    print('\nOutput shapes:')
    for i, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
        print(f'  P{i+3}: cls={list(cls_feat.shape)}, reg={list(reg_feat.shape)}')
    
    # Verify dimensions match
    assert [f.shape[1] for f in cls_feats] == feat_dims, "Dimension mismatch!"
    
    print('\n✅ YOLO26 backbone test passed!')
    print('   Ready for integration with YOWO.')
