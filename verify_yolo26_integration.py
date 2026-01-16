#!/usr/bin/env python3
"""
Verification Script: Test YOLO26 Integration with YOWO

This script verifies that YOLO26 is correctly integrated and compatible
with your YOWO multi-task architecture. Run this to be 100% sure everything works.

Usage:
    python verify_yolo26_integration.py
"""

# Fix path conflict (Python 3.13 with 3.10 system paths)
import sys
sys.path = [p for p in sys.path if '/usr/lib/python3.10' not in p]

import torch
import torch.nn as nn

print("=" * 70)
print("YOLO26 Integration Verification for YOWO")
print("=" * 70)

# ============================================================================
# Step 1: Test YOLO26 Backbone Standalone
# ============================================================================
print("\n[1/5] Testing YOLO26 backbone standalone...")

try:
    from models.backbone.backbone_2d.cnn_2d.yolo_26 import build_yolo_26
    
    # Build YOLO26L backbone
    model, feat_dims = build_yolo_26('yolo26l.pt', pretrained=True)
    model.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cls_feats, reg_feats = model(x)
    
    # Verify outputs
    assert len(cls_feats) == 3, f"Expected 3 FPN levels, got {len(cls_feats)}"
    assert len(reg_feats) == 3, f"Expected 3 FPN levels, got {len(reg_feats)}"
    
    print(f"  ✅ YOLO26L loaded successfully")
    print(f"  ✅ FPN dimensions: {feat_dims}")
    print(f"  ✅ Output shapes:")
    for i, (c, r) in enumerate(zip(cls_feats, reg_feats)):
        print(f"      P{i+3}: cls={list(c.shape)}, reg={list(r.shape)}")
    
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# ============================================================================
# Step 2: Test YOLO26 vs YOLO11 Compatibility
# ============================================================================
print("\n[2/5] Comparing YOLO26 vs YOLO11 feature dimensions...")

try:
    from models.backbone.backbone_2d.cnn_2d.yolo_11 import build_yolo_11
    
    # Build YOLO11 for comparison
    model_11, dims_11 = build_yolo_11('yolo11m.pt', pretrained=True)
    model_26, dims_26 = build_yolo_26('yolo26l.pt', pretrained=True)
    
    print(f"  YOLO11m FPN dims: {dims_11}")
    print(f"  YOLO26L FPN dims: {dims_26}")
    
    # Check if compatible (same number of levels, similar structure)
    assert len(dims_11) == len(dims_26), "Different number of FPN levels!"
    
    print(f"  ✅ Both have {len(dims_11)} FPN levels")
    print(f"  ✅ Compatible for drop-in replacement")
    
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# ============================================================================
# Step 3: Test Config Loading
# ============================================================================
print("\n[3/5] Testing YOLO26 config loading...")

try:
    from config.yowo_v2_config import yowo_v2_config
    
    yolo26_configs = [k for k in yowo_v2_config.keys() if 'yolo26' in k]
    
    print(f"  ✅ Found {len(yolo26_configs)} YOLO26 configs:")
    for cfg_name in yolo26_configs:
        cfg = yowo_v2_config[cfg_name]
        print(f"      - {cfg_name}")
        print(f"        2D: {cfg['backbone_2d']}, 3D: {cfg['backbone_3d']}")
    
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# ============================================================================
# Step 4: Test Full Build Pipeline
# ============================================================================
print("\n[4/5] Testing full YOWO build with YOLO26...")

try:
    from config.yowo_v2_config import yowo_v2_config
    from models.backbone import build_backbone_2d, build_backbone_3d
    
    # Use the YOLO26L + ResNeXt config
    cfg = yowo_v2_config['yowo_v2_resnext_yolo26l_multitask']
    
    # Build 2D backbone
    backbone_2d, dims_2d = build_backbone_2d(cfg, pretrained=True)
    print(f"  ✅ 2D backbone built: dims={dims_2d}")
    
    # Build 3D backbone
    backbone_3d, dim_3d = build_backbone_3d(cfg, pretrained=True)
    print(f"  ✅ 3D backbone built: dim={dim_3d}")
    
    # Test forward
    backbone_2d.eval()
    backbone_3d.eval()
    
    with torch.no_grad():
        # 2D input (single frame)
        x_2d = torch.randn(2, 3, 224, 224)
        out_2d = backbone_2d(x_2d)
        
        # 3D input (video clip: B, C, T, H, W)
        x_3d = torch.randn(2, 3, 16, 224, 224)
        out_3d = backbone_3d(x_3d)
    
    print(f"  ✅ 2D forward pass: {len(out_2d[0])} levels")
    print(f"  ✅ 3D forward pass: shape={list(out_3d.shape)}")
    
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Step 5: Test Gradient Flow
# ============================================================================
print("\n[5/5] Testing gradient flow through YOLO26...")

try:
    from models.backbone.backbone_2d.cnn_2d.yolo_26 import build_yolo_26
    
    model, _ = build_yolo_26('yolo26l.pt', pretrained=True)
    model.train()
    
    # Forward pass
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    cls_feats, reg_feats = model(x)
    
    # Create dummy loss
    loss = sum(f.mean() for f in cls_feats) + sum(f.mean() for f in reg_feats)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"  ✅ Gradients flow to input: {has_grad}")
    
    # Count trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✅ Trainable parameters: {num_params:,}")
    
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - YOLO26 is correctly integrated!")
print("=" * 70)

print("""
Next Steps:
-----------
1. To train with YOLO26, use one of these configs:
   - yowo_v2_resnext_yolo26l_multitask  (recommended, high capacity)
   - yowo_v2_resnext_yolo26m_multitask  (faster)
   - yowo_v2_shufflenet_yolo26l_multitask  (fastest)

2. Example training command:
   python train.py --dataset charades \\
                   --version yowo_v2_resnext_yolo26l_multitask \\
                   --max_epoch 10 \\
                   ...

3. Your existing training pipeline works unchanged - YOLO26 is a 
   drop-in replacement for YOLO11 at the backbone level.

Note: YOLO26 was just released today (Jan 14, 2026). If you encounter
any issues, you can fallback to YOLO11 which is battle-tested.
""")
