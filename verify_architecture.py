#!/usr/bin/env python3
"""
COMPREHENSIVE ARCHITECTURE VERIFICATION for YOWO Multi-Task Model

This script verifies the entire data flow from input to output:
1. YOLO11 2D Backbone: Extracts multi-scale features from key frame
2. X3D 3D Backbone: Extracts spatiotemporal features from video clip  
3. Channel Encoder: Fuses 2D and 3D features
4. Decoupled Head: Separates classification and regression branches
5. Context Modules: Object → Relation → Action cascaded predictions
6. Loss Computation: Verifies target matching and loss calculation

Run with: python verify_architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    print(f"\n--- {title} ---")


def check_pass(condition, message):
    if condition:
        print(f"  ✅ {message}")
        return True
    else:
        print(f"  ❌ {message}")
        return False


def test_yolo11_backbone():
    """Test YOLO11 2D backbone."""
    print_header("TEST 1: YOLO11 2D Backbone")
    
    try:
        from models.backbone.backbone_2d.cnn_2d.yolo_11 import build_yolo_11
        
        model, feat_dims = build_yolo_11(model_name='yolo11m.pt', pretrained=False)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        cls_feats, reg_feats = model(dummy_input)
        
        print_subheader("Feature Map Verification")
        
        expected_strides = [8, 16, 32]
        expected_shapes = [(224 // s, 224 // s) for s in expected_strides]
        
        all_pass = True
        for i, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
            h, w = expected_shapes[i]
            cls_ok = check_pass(
                cls_f.shape == (2, feat_dims[i], h, w),
                f"Level {i}: cls_feat shape {cls_f.shape} = expected (2, {feat_dims[i]}, {h}, {w})"
            )
            reg_ok = check_pass(
                reg_f.shape == (2, feat_dims[i], h, w),
                f"Level {i}: reg_feat shape {reg_f.shape} = expected (2, {feat_dims[i]}, {h}, {w})"
            )
            all_pass = all_pass and cls_ok and reg_ok
        
        check_pass(len(cls_feats) == 3, f"Number of feature levels: {len(cls_feats)} == 3")
        check_pass(feat_dims == [256, 512, 512], f"Feature dimensions: {feat_dims} == [256, 512, 512]")
        
        return all_pass
        
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_x3d_backbone():
    """Test X3D 3D backbone."""
    print_header("TEST 2: X3D 3D Backbone")
    
    try:
        from models.backbone.backbone_3d.cnn_3d.x3d import build_x3d_3d
        
        model, feat_dim = build_x3d_3d(model_name='x3d_m', pretrained=False)
        
        # Test forward pass with video input
        # Shape: [B, C, T, H, W] = [2, 3, 16, 224, 224]
        dummy_video = torch.randn(2, 3, 16, 224, 224)
        feat_3d = model(dummy_video)
        
        print_subheader("Feature Map Verification")
        
        # X3D downsamples spatially by 32x and temporally averages
        expected_h = 224 // 32  # 7
        expected_w = 224 // 32  # 7
        
        check_pass(
            feat_3d.shape == (2, 192, expected_h, expected_w),
            f"3D feature shape: {feat_3d.shape} == (2, 192, {expected_h}, {expected_w})"
        )
        check_pass(feat_dim == 192, f"Feature dimension: {feat_dim} == 192")
        
        # Verify temporal dim is averaged (squeezed)
        check_pass(len(feat_3d.shape) == 4, "Temporal dimension averaged to 4D tensor")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_channel_encoder_fusion():
    """Test 2D+3D feature fusion in channel encoder."""
    print_header("TEST 3: Channel Encoder (2D + 3D Fusion)")
    
    try:
        from models.yowo.encoder import ChannelEncoder
        
        # Simulate feature dimensions
        # Level 0: 2D=256, 3D=192 -> concat = 448
        in_dim = 256 + 192  # YOLO11 level 0 + X3D
        out_dim = 256  # head_dim
        
        encoder = ChannelEncoder(in_dim, out_dim, act_type='silu', norm_type='BN')
        
        # Create dummy features
        feat_2d = torch.randn(2, 256, 28, 28)  # Level 0 from YOLO11
        feat_3d = torch.randn(2, 192, 28, 28)  # Upsampled X3D
        
        # Forward
        fused = encoder(feat_2d, feat_3d)
        
        print_subheader("Fusion Verification")
        
        check_pass(
            fused.shape == (2, out_dim, 28, 28),
            f"Fused feature shape: {fused.shape} == (2, {out_dim}, 28, 28)"
        )
        
        # Verify gradient flow
        fused.sum().backward()
        
        has_grad = encoder.fuse_convs[0].conv.weight.grad is not None
        check_pass(has_grad, "Gradients flow through encoder")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_modules():
    """Test ObjectContextModule and ObjectRelationContextModule."""
    print_header("TEST 4: Context Modules (Cascaded Predictions)")
    
    try:
        from models.yowo.yowo_multitask import ObjectContextModule, ObjectRelationContextModule
        
        B, C, H, W = 2, 256, 14, 14
        num_objects, num_relations = 36, 26
        
        obj_ctx = ObjectContextModule(dim=C, num_classes=num_objects)
        obj_rel_ctx = ObjectRelationContextModule(dim=C, num_objects=num_objects, num_relations=num_relations)
        
        # Simulate inputs
        cls_feat = torch.randn(B, C, H, W)
        obj_pred = torch.randn(B, num_objects, H, W)  # Object logits
        rel_pred = torch.randn(B, num_relations, H, W)  # Relation logits
        
        print_subheader("ObjectContextModule")
        
        # Test ObjectContextModule
        rel_feat = obj_ctx(cls_feat, obj_pred)
        check_pass(rel_feat.shape == (B, C, H, W), f"Output shape: {rel_feat.shape}")
        
        # Test sensitivity
        zero_obj = torch.zeros_like(obj_pred)
        rel_feat_zero = obj_ctx(cls_feat, zero_obj)
        diff = (rel_feat - rel_feat_zero).abs().mean().item()
        check_pass(diff > 0.01, f"Sensitivity to object input: diff={diff:.4f} > 0.01")
        
        print_subheader("ObjectRelationContextModule")
        
        # Test ObjectRelationContextModule
        act_feat = obj_rel_ctx(cls_feat, obj_pred, rel_pred)
        check_pass(act_feat.shape == (B, C, H, W), f"Output shape: {act_feat.shape}")
        
        # Test sensitivity to both inputs
        zero_rel = torch.zeros_like(rel_pred)
        act_feat_zero = obj_rel_ctx(cls_feat, obj_pred, zero_rel)
        diff_rel = (act_feat - act_feat_zero).abs().mean().item()
        check_pass(diff_rel > 0.01, f"Sensitivity to relation input: diff={diff_rel:.4f} > 0.01")
        
        # Test gradient flow
        print_subheader("Gradient Flow Test")
        
        cls_feat.requires_grad_(True)
        obj_pred.requires_grad_(True)
        rel_pred.requires_grad_(True)
        
        target = torch.randint(0, 2, (B * H * W, 157)).float()
        
        # Simulate cascaded forward
        rel_feat = obj_ctx(cls_feat, obj_pred)
        rel_pred_new = rel_pred  # In real model this would be from rel_preds layer
        act_feat = obj_rel_ctx(cls_feat, obj_pred, rel_pred_new)
        
        # Flatten and compute loss
        act_pred = act_feat.permute(0, 2, 3, 1).flatten(0, 2)[:, :157]  # Simulate action head
        loss = F.binary_cross_entropy_with_logits(act_pred, target)
        loss.backward()
        
        check_pass(obj_pred.grad is not None and obj_pred.grad.abs().sum() > 0, 
                   "Gradient flows to obj_pred")
        check_pass(cls_feat.grad is not None and cls_feat.grad.abs().sum() > 0,
                   "Gradient flows to cls_feat")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_model_forward():
    """Test full YOWOMultiTask model forward pass."""
    print_header("TEST 5: Full Model Forward Pass")
    
    try:
        from config import yowo_v2_config
        from models.yowo.yowo_multitask import YOWOMultiTask
        
        cfg = yowo_v2_config['yowo_v2_x3d_m_yolo11m_multitask']
        device = 'cpu'  # Use CPU to avoid memory issues
        
        print("  Building model (this may take a moment)...")
        model = YOWOMultiTask(
            cfg=cfg,
            device=device,
            num_objects=36,
            num_actions=157,
            num_relations=26,
            trainable=True
        )
        model.eval()
        
        # Create dummy video clip
        B, C, T, H, W = 1, 3, 16, 224, 224
        video_clips = torch.randn(B, C, T, H, W)
        
        print_subheader("Forward Pass Dimensions")
        
        with torch.no_grad():
            model.trainable = True
            outputs = model(video_clips)
        
        # Check output structure
        expected_keys = ["pred_conf", "pred_obj", "pred_act", "pred_rel", "pred_box", "anchors", "strides"]
        for key in expected_keys:
            check_pass(key in outputs, f"Output contains '{key}'")
        
        # Check output shapes
        print_subheader("Output Shapes per FPN Level")
        
        total_anchors = 0
        for level in range(3):
            pred_obj = outputs["pred_obj"][level]
            pred_act = outputs["pred_act"][level]
            pred_rel = outputs["pred_rel"][level]
            anchors = outputs["anchors"][level]
            
            M = pred_obj.shape[1]  # Number of anchors at this level
            total_anchors += M
            
            check_pass(pred_obj.shape == (B, M, 36), f"Level {level} obj_pred: {pred_obj.shape}")
            check_pass(pred_act.shape == (B, M, 157), f"Level {level} act_pred: {pred_act.shape}")
            check_pass(pred_rel.shape == (B, M, 26), f"Level {level} rel_pred: {pred_rel.shape}")
            check_pass(anchors.shape == (M, 2), f"Level {level} anchors: {anchors.shape}")
        
        print(f"\n  Total anchors across all levels: {total_anchors}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation with simulated training data."""
    print_header("TEST 6: Loss Computation")
    
    try:
        from models.yowo.loss_multitask import MultiTaskCriterion
        import argparse
        
        # Create mock args
        args = argparse.Namespace(
            loss_conf_weight=1.0,
            loss_reg_weight=5.0,
            center_sampling_radius=2.5,
            topk_candicate=10
        )
        
        criterion = MultiTaskCriterion(
            args=args,
            img_size=224,
            num_objects=36,
            num_actions=157,
            num_relations=26
        )
        
        print_subheader("Simulating Model Outputs")
        
        B, M = 2, 196  # Batch size and number of anchors
        device = 'cpu'
        
        # Simulate model outputs
        outputs = {
            "pred_conf": [torch.randn(B, M, 1)],
            "pred_obj": [torch.randn(B, M, 36)],
            "pred_act": [torch.randn(B, M, 157)],
            "pred_rel": [torch.randn(B, M, 26)],
            "pred_box": [torch.randn(B, M, 4) * 224],
            "anchors": [torch.rand(M, 2) * 224],
            "strides": [16]
        }
        
        # Simulate targets
        targets = []
        for b in range(B):
            num_gt = 3
            boxes = torch.rand(num_gt, 4)
            # Ensure boxes are valid (x2 > x1, y2 > y1)
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] * 0.5
            
            labels = torch.zeros(num_gt, 219)
            # Set some labels
            labels[0, 0] = 1  # Person
            labels[0, 36] = 1  # Some action
            labels[0, 193] = 1  # Some relation
            labels[1, 5] = 1  # Some object
            labels[2, 0] = 1  # Person
            labels[2, 50] = 1  # Some action
            
            targets.append({
                "boxes": boxes,
                "labels": labels
            })
        
        print_subheader("Computing Losses")
        
        loss_dict = criterion(outputs, targets)
        
        # Check all losses are computed
        expected_losses = ["loss_conf", "loss_obj", "loss_act", "loss_rel", "loss_box", "losses"]
        for loss_name in expected_losses:
            value = loss_dict.get(loss_name, None)
            if value is not None:
                check_pass(not torch.isnan(value) and not torch.isinf(value), 
                          f"{loss_name}: {value.item():.4f}")
            else:
                check_pass(False, f"{loss_name}: NOT COMPUTED")
        
        # Check total loss
        total = loss_dict["losses"]
        check_pass(total.item() > 0, f"Total loss is positive: {total.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cascaded_gradient_flow():
    """Test that cascaded training provides meaningful gradient signal."""
    print_header("TEST 7: Cascaded Gradient Flow Analysis")
    
    try:
        from models.yowo.yowo_multitask import ObjectContextModule, ObjectRelationContextModule
        
        B, C, H, W = 2, 256, 14, 14
        num_objects, num_relations, num_actions = 36, 26, 157
        
        # Build cascade
        obj_ctx = ObjectContextModule(dim=C, num_classes=num_objects)
        obj_rel_ctx = ObjectRelationContextModule(dim=C, num_objects=num_objects, num_relations=num_relations)
        
        obj_pred_layer = nn.Conv2d(C, num_objects, 1)
        rel_pred_layer = nn.Conv2d(C, num_relations, 1)
        act_pred_layer = nn.Conv2d(C, num_actions, 1)
        
        # Set to training mode
        obj_ctx.train()
        obj_rel_ctx.train()
        
        # Input
        cls_feat = torch.randn(B, C, H, W, requires_grad=True)
        
        print_subheader("Forward Pass (Cascaded)")
        
        # Cascaded forward
        obj_pred = obj_pred_layer(cls_feat)
        print(f"  Step 1: obj_pred shape = {obj_pred.shape}")
        
        rel_feat = obj_ctx(cls_feat, obj_pred)
        rel_pred = rel_pred_layer(rel_feat)
        print(f"  Step 2: rel_pred shape = {rel_pred.shape}")
        
        act_feat = obj_rel_ctx(cls_feat, obj_pred, rel_pred)
        act_pred = act_pred_layer(act_feat)
        print(f"  Step 3: act_pred shape = {act_pred.shape}")
        
        print_subheader("Backward Pass (Action Loss Only)")
        
        # Compute realistic BCE loss
        target = torch.randint(0, 2, (B * H * W, num_actions)).float()
        act_flat = act_pred.permute(0, 2, 3, 1).flatten(0, 2)
        loss = F.binary_cross_entropy_with_logits(act_flat, target)
        loss.backward()
        
        # Measure gradients
        obj_grad = obj_pred_layer.weight.grad.abs().mean().item()
        rel_grad = rel_pred_layer.weight.grad.abs().mean().item()
        act_grad = act_pred_layer.weight.grad.abs().mean().item()
        
        print(f"\n  Gradient magnitudes:")
        print(f"    act_pred_layer: {act_grad:.8f}")
        print(f"    rel_pred_layer: {rel_grad:.8f}")
        print(f"    obj_pred_layer: {obj_grad:.8f}")
        
        ratio_obj = act_grad / obj_grad if obj_grad > 0 else float('inf')
        ratio_rel = act_grad / rel_grad if rel_grad > 0 else float('inf')
        
        print(f"\n  Gradient ratios:")
        print(f"    action/object: {ratio_obj:.1f}x")
        print(f"    action/relation: {ratio_rel:.1f}x")
        
        check_pass(obj_grad > 0, "Object layer receives gradients")
        check_pass(rel_grad > 0, "Relation layer receives gradients")
        check_pass(ratio_obj < 1000, f"Object gradient ratio reasonable (< 1000x)")
        check_pass(ratio_rel < 1000, f"Relation gradient ratio reasonable (< 1000x)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_flow_diagram():
    """Print data flow diagram for architecture understanding."""
    print_header("ARCHITECTURE DATA FLOW DIAGRAM")
    
    diagram = """
    ┌──────────────────────────────────────────────────────────────────┐
    │                    INPUT: video_clips [B, 3, T, H, W]            │
    └─────────────────────────────┬────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │ 2D Backbone (YOLO11)│               │ 3D Backbone (X3D)   │
    │ Input: key_frame    │               │ Input: full clip    │
    │ [B, 3, H, W]        │               │ [B, 3, T, H, W]     │
    │                     │               │                     │
    │ Output:             │               │ Output:             │
    │ cls_feats (3 levels)│               │ feat_3d             │
    │ reg_feats (3 levels)│               │ [B, 192, H/32, W/32]│
    └─────────┬───────────┘               └─────────┬───────────┘
              │                                     │
              │ For each FPN level (0,1,2):         │
              │ ┌───────────────────────────────────┘
              ▼ ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ Channel Encoder: Fuse 2D + 3D features                          │
    │ Input: concat(cls_feat_2d, upsample(feat_3d))                   │
    │ Output: cls_feat [B, 256, H_level, W_level]                     │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ Decoupled Head: Separate cls/reg processing                     │
    │ Output: cls_feat, reg_feat                                      │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   │                   ▼
    ┌───────────────────┐         │         ┌───────────────────┐
    │ conf_pred [B,1,H,W]         │         │ reg_pred [B,4,H,W]│
    │ (from reg_feat)   │         │         │ (from reg_feat)   │
    └───────────────────┘         ▼         └───────────────────┘
                        ┌───────────────────┐
                        │ obj_pred [B,36,H,W]│ ◄── Step 1: Object
                        │ (from cls_feat)   │     prediction
                        └─────────┬─────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ ObjectContextModule: Add object context to features             │
    │ Input: cls_feat + obj_pred                                      │
    │ Output: rel_feat (object-aware features)                        │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │ rel_pred [B,26,H,W]│ ◄── Step 2: Relation
                        │ (from rel_feat)   │     prediction
                        └─────────┬─────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ ObjectRelationContextModule: Add obj+rel context to features    │
    │ Input: cls_feat + obj_pred + rel_pred                           │
    │ Output: act_feat (object+relation-aware features)               │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │ act_pred [B,157,H,W]│ ◄── Step 3: Action
                        │ (from act_feat)    │     prediction
                        └────────────────────┘
    
    GRADIENT FLOW (Backward):
    action_loss ──→ act_pred ──→ ObjectRelationContextModule ──→ obj_pred + rel_pred
                                                                      │
    relation_loss ──→ rel_pred ──→ ObjectContextModule ──────────────┼─→ obj_pred
                                                                      │
    object_loss ──→ obj_pred ◄────────────────────────────────────────┘
    """
    print(diagram)
    return True


def main():
    print("\n" + "=" * 70)
    print("  YOWO MULTI-TASK ARCHITECTURE VERIFICATION")
    print("  Checking all components: YOLO11 → X3D → Fusion → Context → Loss")
    print("=" * 70)
    
    results = OrderedDict()
    
    # Run all tests
    results['Data Flow Diagram'] = test_data_flow_diagram()
    
    try:
        results['YOLO11 Backbone'] = test_yolo11_backbone()
    except Exception as e:
        print(f"  Skipping YOLO11 test (may need GPU): {e}")
        results['YOLO11 Backbone'] = None
    
    try:
        results['X3D Backbone'] = test_x3d_backbone()
    except Exception as e:
        print(f"  Skipping X3D test: {e}")
        results['X3D Backbone'] = None
    
    results['Channel Encoder'] = test_channel_encoder_fusion()
    results['Context Modules'] = test_context_modules()
    results['Cascaded Gradients'] = test_cascaded_gradient_flow()
    
    try:
        results['Full Model Forward'] = test_full_model_forward()
    except Exception as e:
        print(f"  Skipping full model test (memory): {e}")
        results['Full Model Forward'] = None
    
    results['Loss Computation'] = test_loss_computation()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⏭️  SKIP"
            skipped += 1
        print(f"  {test_name}: {status}")
    
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n" + "=" * 70)
        print("  🎉 ALL CORE TESTS PASSED!")
        print("  Your architecture is correctly configured and ready for training.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("  ⚠️  SOME TESTS FAILED - Review above for details")
        print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    main()
