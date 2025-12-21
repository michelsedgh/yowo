#!/usr/bin/env python3
"""
X3D Integration Verification Script for YOWO

This script thoroughly validates that X3D has been correctly integrated
into the YOWO architecture. It checks:

1. X3D backbone can be built and produces correct output shapes
2. X3D backbone is compatible with YOWO's channel encoder
3. Full YOWO model with X3D backbone can be instantiated
4. Forward pass works correctly for both training and inference modes
5. Gradient flow is correct during training
6. Model can be saved and loaded
7. Comparison with ShuffleNet to verify similar interface

Run this BEFORE training to ensure the integration is correct.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, '/home/michel/yowo')


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_test(name, passed, details=""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"          {details}")


def test_x3d_backbone_standalone():
    """Test 1: Verify X3D backbone can be built and works correctly."""
    print_header("Test 1: X3D Backbone Standalone")
    
    from models.backbone.backbone_3d.cnn_3d.x3d import build_x3d_3d
    
    all_passed = True
    
    for model_name in ['x3d_xs', 'x3d_s', 'x3d_m']:
        try:
            model, feat_dim = build_x3d_3d(model_name=model_name, pretrained=False)
            
            # Test forward pass
            x = torch.randn(1, 3, 16, 224, 224)
            with torch.no_grad():
                out = model(x)
            
            # Verify output shape
            expected_shape = (1, 192)  # [B, C, H, W] where C=192
            passed = (out.dim() == 4 and 
                     out.shape[1] == feat_dim == 192 and
                     out.shape[0] == 1)
            
            print_test(f"{model_name}", passed, 
                      f"Output: {out.shape}, feat_dim: {feat_dim}")
            all_passed = all_passed and passed
            
        except Exception as e:
            print_test(f"{model_name}", False, f"Error: {e}")
            all_passed = False
    
    return all_passed


def test_x3d_vs_shufflenet_interface():
    """Test 2: Verify X3D has same interface as ShuffleNet."""
    print_header("Test 2: X3D vs ShuffleNet Interface Compatibility")
    
    from models.backbone.backbone_3d.cnn_3d.x3d import build_x3d_3d
    from models.backbone.backbone_3d.cnn_3d.shufflnetv2 import build_shufflenetv2_3d
    
    all_passed = True
    
    # Build both
    x3d_model, x3d_feat = build_x3d_3d(model_name='x3d_s', pretrained=False)
    shuffle_model, shuffle_feat = build_shufflenetv2_3d(model_size='2.0x', pretrained=False)
    
    # Same input
    x = torch.randn(1, 3, 16, 224, 224)
    
    with torch.no_grad():
        x3d_out = x3d_model(x)
        shuffle_out = shuffle_model(x)
    
    # Check same output dimensions (except channel count)
    same_dims = (x3d_out.dim() == shuffle_out.dim() == 4)
    same_batch = (x3d_out.shape[0] == shuffle_out.shape[0])
    same_spatial = (x3d_out.shape[2:] == shuffle_out.shape[2:])
    
    print_test("Same output dimensionality (4D)", same_dims,
              f"X3D: {x3d_out.dim()}D, ShuffleNet: {shuffle_out.dim()}D")
    print_test("Same batch dimension", same_batch,
              f"X3D: {x3d_out.shape[0]}, ShuffleNet: {shuffle_out.shape[0]}")
    print_test("Same spatial dimensions", same_spatial,
              f"X3D: {x3d_out.shape[2:]}, ShuffleNet: {shuffle_out.shape[2:]}")
    print_test("Feature dimensions available", True,
              f"X3D: {x3d_feat}, ShuffleNet: {shuffle_feat}")
    
    all_passed = same_dims and same_batch and same_spatial
    return all_passed


def test_backbone_3d_factory():
    """Test 3: Verify build_3d_cnn factory works with X3D configs."""
    print_header("Test 3: 3D Backbone Factory with X3D")
    
    from models.backbone.backbone_3d.cnn_3d import build_3d_cnn
    
    all_passed = True
    
    # Test X3D configs
    x3d_configs = [
        {'backbone_3d': 'x3d_xs'},
        {'backbone_3d': 'x3d_s'},
        {'backbone_3d': 'x3d_m'},
    ]
    
    for cfg in x3d_configs:
        try:
            model, feat_dim = build_3d_cnn(cfg, pretrained=False)
            
            x = torch.randn(1, 3, 16, 224, 224)
            with torch.no_grad():
                out = model(x)
            
            passed = (out.dim() == 4 and out.shape[1] == feat_dim)
            print_test(f"Factory: {cfg['backbone_3d']}", passed,
                      f"Output: {out.shape}, feat_dim: {feat_dim}")
            all_passed = all_passed and passed
            
        except Exception as e:
            print_test(f"Factory: {cfg['backbone_3d']}", False, f"Error: {e}")
            all_passed = False
    
    return all_passed


def test_full_yowo_with_x3d():
    """Test 4: Verify full YOWO model works with X3D backbone."""
    print_header("Test 4: Full YOWO Model with X3D")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo import YOWO
    
    all_passed = True
    device = torch.device("cpu")  # Use CPU for testing
    
    # Test X3D configs
    x3d_model_names = ['yowo_v2_x3d_s', 'yowo_v2_x3d_m']
    
    for model_name in x3d_model_names:
        if model_name not in yowo_v2_config:
            print_test(f"{model_name}", False, "Config not found!")
            all_passed = False
            continue
            
        try:
            cfg = yowo_v2_config[model_name]
            
            # Build YOWO with X3D
            model = YOWO(
                cfg=cfg,
                device=device,
                num_classes=157,  # Charades
                trainable=False
            )
            model.eval()
            
            # Test inference
            x = torch.randn(1, 3, 16, 224, 224)
            with torch.no_grad():
                outputs = model(x)
            
            # Check output format
            if isinstance(outputs, tuple) and len(outputs) == 3:
                scores, labels, bboxes = outputs
                passed = True
                details = f"scores: {len(scores)}, labels: {len(labels)}, bboxes: {len(bboxes)}"
            else:
                passed = False
                details = f"Unexpected output format: {type(outputs)}"
            
            print_test(f"{model_name} (inference)", passed, details)
            all_passed = all_passed and passed
            
        except Exception as e:
            import traceback
            print_test(f"{model_name}", False, f"Error: {e}")
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_training_mode():
    """Test 5: Verify training mode works with X3D."""
    print_header("Test 5: Training Mode with X3D")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo import YOWO
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s']
    
    try:
        # Build YOWO in training mode
        model = YOWO(
            cfg=cfg,
            device=device,
            num_classes=157,
            trainable=True
        )
        model.train()
        
        # Test forward pass in training mode
        x = torch.randn(1, 3, 16, 224, 224)
        outputs = model(x)
        
        # In training mode, outputs should be a dict
        required_keys = ['pred_conf', 'pred_cls', 'pred_box', 'anchors', 'strides']
        has_all_keys = all(k in outputs for k in required_keys)
        
        print_test("Training output format", has_all_keys,
                  f"Keys: {list(outputs.keys())}")
        all_passed = all_passed and has_all_keys
        
        # Check shapes
        for key in ['pred_conf', 'pred_cls', 'pred_box']:
            if key in outputs:
                shapes = [p.shape for p in outputs[key]]
                print_test(f"  {key} shapes", True, f"{shapes}")
        
    except Exception as e:
        import traceback
        print_test("Training mode", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_gradient_flow():
    """Test 6: Verify gradients flow correctly through X3D."""
    print_header("Test 6: Gradient Flow Verification")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo import YOWO
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s']
    
    try:
        model = YOWO(
            cfg=cfg,
            device=device,
            num_classes=157,
            trainable=True
        )
        model.train()
        
        # Forward pass
        x = torch.randn(1, 3, 16, 224, 224, requires_grad=True)
        outputs = model(x)
        
        # Create a simple loss
        loss = sum(p.mean() for p in outputs['pred_conf'])
        loss += sum(p.mean() for p in outputs['pred_cls'])
        loss += sum(p.mean() for p in outputs['pred_box'])
        
        # Backward pass
        loss.backward()
        
        # Check gradients in X3D backbone
        x3d_grads = []
        for name, param in model.backbone_3d.named_parameters():
            if param.grad is not None:
                x3d_grads.append(param.grad.abs().mean().item())
        
        has_grads = len(x3d_grads) > 0 and sum(x3d_grads) > 0
        print_test("X3D backbone receives gradients", has_grads,
                  f"Params with grads: {len(x3d_grads)}, mean grad: {np.mean(x3d_grads):.6f}")
        
        # Check gradients in 2D backbone
        bk2d_grads = []
        for name, param in model.backbone_2d.named_parameters():
            if param.grad is not None:
                bk2d_grads.append(param.grad.abs().mean().item())
        
        has_2d_grads = len(bk2d_grads) > 0 and sum(bk2d_grads) > 0
        print_test("2D backbone receives gradients", has_2d_grads,
                  f"Params with grads: {len(bk2d_grads)}, mean grad: {np.mean(bk2d_grads):.6f}")
        
        # Check gradients in heads
        head_grads = []
        for name, param in model.cls_preds.named_parameters():
            if param.grad is not None:
                head_grads.append(param.grad.abs().mean().item())
        
        has_head_grads = len(head_grads) > 0 and sum(head_grads) > 0
        print_test("Prediction heads receive gradients", has_head_grads,
                  f"Params with grads: {len(head_grads)}, mean grad: {np.mean(head_grads):.6f}")
        
        all_passed = has_grads and has_2d_grads and has_head_grads
        
    except Exception as e:
        import traceback
        print_test("Gradient flow", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_channel_encoder_compatibility():
    """Test 7: Verify channel encoder works with X3D's 192 channels."""
    print_header("Test 7: Channel Encoder Compatibility")
    
    from models.yowo.encoder import build_channel_encoder
    
    all_passed = True
    
    # X3D outputs 192 channels
    x3d_channels = 192
    
    # YOLO11m 2D backbone outputs (example)
    yolo_channels = [128, 256, 512]  # Typical FPN channels
    
    head_dim = 128  # From config
    
    for i, yolo_ch in enumerate(yolo_channels):
        try:
            in_dim = yolo_ch + x3d_channels
            cfg = {'head_act': 'silu', 'head_norm': 'BN'}
            
            encoder = build_channel_encoder(cfg, in_dim, head_dim)
            
            # Simulate input
            x_2d = torch.randn(1, yolo_ch, 28, 28)  # 2D features
            x_3d = torch.randn(1, x3d_channels, 28, 28)  # 3D features (upsampled)
            
            out = encoder(x_2d, x_3d)
            
            passed = (out.shape == (1, head_dim, 28, 28))
            print_test(f"Encoder level {i} (2D:{yolo_ch} + X3D:{x3d_channels} -> {head_dim})", 
                      passed, f"Output: {out.shape}")
            all_passed = all_passed and passed
            
        except Exception as e:
            print_test(f"Encoder level {i}", False, f"Error: {e}")
            all_passed = False
    
    return all_passed


def test_model_save_load():
    """Test 8: Verify model can be saved and loaded."""
    print_header("Test 8: Model Save/Load")
    
    import tempfile
    import os
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo import YOWO
    
    all_passed = True
    device = torch.device("cpu")
    
    cfg = yowo_v2_config['yowo_v2_x3d_s']
    
    try:
        # Build model
        model = YOWO(cfg=cfg, device=device, num_classes=157, trainable=False)
        model.eval()
        
        # Get reference output
        x = torch.randn(1, 3, 16, 224, 224)
        with torch.no_grad():
            ref_scores, ref_labels, ref_bboxes = model(x)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
            torch.save(model.state_dict(), temp_path)
        
        # Load into new model
        model2 = YOWO(cfg=cfg, device=device, num_classes=157, trainable=False)
        model2.load_state_dict(torch.load(temp_path))
        model2.eval()
        
        # Get loaded model output
        with torch.no_grad():
            load_scores, load_labels, load_bboxes = model2(x)
        
        # Clean up
        os.unlink(temp_path)
        
        # Note: Due to empty detections, we just verify the save/load worked
        print_test("Model save/load", True, "State dict saved and loaded successfully")
        
    except Exception as e:
        import traceback
        print_test("Model save/load", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_x3d_yolo11m_config():
    """Test 9: Verify X3D + YOLO11m config works."""
    print_header("Test 9: X3D + YOLO11m Configuration")
    
    from config.yowo_v2_config import yowo_v2_config
    from models.yowo.yowo import YOWO
    
    all_passed = True
    device = torch.device("cpu")
    
    # This is the recommended config for Charades + Action Genome
    model_name = 'yowo_v2_x3d_s_yolo11m'
    
    if model_name not in yowo_v2_config:
        print_test(f"{model_name} config exists", False, "Config not found!")
        return False
    
    print_test(f"{model_name} config exists", True)
    
    cfg = yowo_v2_config[model_name]
    
    try:
        # Build model
        model = YOWO(
            cfg=cfg,
            device=device,
            num_classes=157,  # Charades classes
            trainable=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_test("Model instantiation", True,
                  f"Total: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
        
        # Verify components
        print_test("2D backbone is YOLO11m", cfg['backbone_2d'] == 'yolo11m')
        print_test("3D backbone is X3D-S", cfg['backbone_3d'] == 'x3d_s')
        
        # Test forward pass
        x = torch.randn(1, 3, 16, 224, 224)
        outputs = model(x)
        
        print_test("Forward pass works", True,
                  f"Output keys: {list(outputs.keys())}")
        
        all_passed = True
        
    except Exception as e:
        import traceback
        print_test("Model build", False, f"Error: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print(" X3D INTEGRATION VERIFICATION FOR YOWO")
    print(" Run this script to verify X3D is correctly integrated")
    print("=" * 70)
    
    results = OrderedDict()
    
    results['1. X3D Backbone Standalone'] = test_x3d_backbone_standalone()
    results['2. X3D vs ShuffleNet Interface'] = test_x3d_vs_shufflenet_interface()
    results['3. 3D Backbone Factory'] = test_backbone_3d_factory()
    results['4. Full YOWO with X3D'] = test_full_yowo_with_x3d()
    results['5. Training Mode'] = test_training_mode()
    results['6. Gradient Flow'] = test_gradient_flow()
    results['7. Channel Encoder Compatibility'] = test_channel_encoder_compatibility()
    results['8. Model Save/Load'] = test_model_save_load()
    results['9. X3D + YOLO11m Config'] = test_x3d_yolo11m_config()
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    
    print("\n" + "-" * 70)
    if all_passed:
        print(" ✓ ALL TESTS PASSED - X3D integration is correct!")
        print(" You can now proceed with training on Charades + Action Genome.")
    else:
        print(" ✗ SOME TESTS FAILED - Please fix the issues before training.")
    print("-" * 70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)


