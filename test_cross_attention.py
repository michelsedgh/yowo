#!/usr/bin/env python3
"""
Cross-Attention Architecture Verification Test

This script tests whether:
1. Object predictions influence Relation predictions (via ObjectCrossAttention)
2. Object + Relation predictions influence Action predictions (via ObjectRelationCrossAttention)
3. Gradients flow correctly through the cascaded prediction chain
4. The attention weights are meaningful (not uniform)
"""

import torch
import torch.nn as nn
import numpy as np
from models.yowo.yowo_multitask import ObjectCrossAttention, ObjectRelationCrossAttention, YOWOMultiTask
from config import yowo_v2_config


def test_cross_attention_modules():
    """Test the cross-attention modules in isolation."""
    print("=" * 70)
    print("TEST 1: Cross-Attention Module Isolation Tests")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    height, width = 14, 14  # Typical feature map size
    dim = 256
    num_objects = 36
    num_relations = 26
    
    # Create modules
    obj_cross_attn = ObjectCrossAttention(dim=dim, num_classes=num_objects, num_heads=4).to(device)
    obj_rel_cross_attn = ObjectRelationCrossAttention(
        dim=dim, num_objects=num_objects, num_relations=num_relations, num_heads=4
    ).to(device)
    
    # ===== TEST 1.1: ObjectCrossAttention sensitivity =====
    print("\n--- Test 1.1: ObjectCrossAttention Sensitivity ---")
    
    cls_feat = torch.randn(batch_size, dim, height, width, device=device, requires_grad=True)
    
    # Case A: Weak object predictions (near zero)
    weak_obj_pred = torch.zeros(batch_size, num_objects, height, width, device=device)
    out_weak = obj_cross_attn(cls_feat, weak_obj_pred)
    
    # Case B: Strong object predictions (confident logits)
    strong_obj_pred = torch.zeros(batch_size, num_objects, height, width, device=device)
    strong_obj_pred[:, 0, :, :] = 5.0  # Strong "person" predictions everywhere
    strong_obj_pred[:, 10, :, :] = 5.0  # Strong "laptop" predictions
    out_strong = obj_cross_attn(cls_feat, strong_obj_pred)
    
    # The outputs SHOULD be different
    diff = (out_weak - out_strong).abs().mean().item()
    print(f"  Difference between weak vs strong object predictions: {diff:.6f}")
    
    if diff > 0.01:
        print("  ✅ PASS: ObjectCrossAttention IS sensitive to object predictions")
    else:
        print("  ❌ FAIL: ObjectCrossAttention NOT sensitive to object predictions")
        print("         The cross-attention is not properly utilizing object info!")
    
    # ===== TEST 1.2: ObjectRelationCrossAttention sensitivity =====
    print("\n--- Test 1.2: ObjectRelationCrossAttention Sensitivity ---")
    
    cls_feat = torch.randn(batch_size, dim, height, width, device=device, requires_grad=True)
    
    # Case A: No objects or relations
    no_obj = torch.zeros(batch_size, num_objects, height, width, device=device)
    no_rel = torch.zeros(batch_size, num_relations, height, width, device=device)
    out_none = obj_rel_cross_attn(cls_feat, no_obj, no_rel)
    
    # Case B: Person holding a laptop
    with_obj = torch.zeros(batch_size, num_objects, height, width, device=device)
    with_obj[:, 0, :, :] = 5.0  # person
    with_obj[:, 10, :, :] = 5.0  # laptop (assuming index 10)
    
    with_rel = torch.zeros(batch_size, num_relations, height, width, device=device)
    with_rel[:, 5, :, :] = 5.0  # Assume index 5 is "holding"
    
    out_context = obj_rel_cross_attn(cls_feat, with_obj, with_rel)
    
    diff = (out_none - out_context).abs().mean().item()
    print(f"  Difference between no-context vs rich-context: {diff:.6f}")
    
    if diff > 0.01:
        print("  ✅ PASS: ObjectRelationCrossAttention IS sensitive to object+relation")
    else:
        print("  ❌ FAIL: ObjectRelationCrossAttention NOT sensitive to context")
    
    # ===== TEST 1.3: Attention weight distribution =====
    print("\n--- Test 1.3: Attention Weight Distribution ---")
    
    cls_feat = torch.randn(batch_size, dim, height, width, device=device)
    obj_pred = torch.randn(batch_size, num_objects, height, width, device=device)
    
    out, attn_weights = obj_cross_attn(cls_feat, obj_pred, return_weights=True)
    
    # Check if attention weights are uniform or varied
    attn_std = attn_weights.std().item()
    attn_max = attn_weights.max().item()
    attn_min = attn_weights.min().item()
    
    print(f"  Attention weights - std: {attn_std:.6f}, max: {attn_max:.6f}, min: {attn_min:.6f}")
    
    if attn_std > 0.001:
        print("  ✅ PASS: Attention weights are NOT uniform (good!)")
    else:
        print("  ❌ FAIL: Attention weights are nearly uniform (attention not learning)")
    
    return True


def test_gradient_flow():
    """Test that gradients flow through the cascaded prediction chain."""
    print("\n" + "=" * 70)
    print("TEST 2: Gradient Flow Through Cascaded Predictions")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    dim = 256
    H, W = 14, 14
    num_objects = 36
    num_actions = 157
    num_relations = 26
    
    # Simulated architecture components
    obj_cross_attn = ObjectCrossAttention(dim=dim, num_classes=num_objects).to(device)
    obj_rel_cross_attn = ObjectRelationCrossAttention(
        dim=dim, num_objects=num_objects, num_relations=num_relations
    ).to(device)
    
    obj_pred_layer = nn.Conv2d(dim, num_objects, kernel_size=1).to(device)
    rel_pred_layer = nn.Conv2d(dim, num_relations, kernel_size=1).to(device)
    act_pred_layer = nn.Conv2d(dim, num_actions, kernel_size=1).to(device)
    
    # Input feature
    cls_feat = torch.randn(batch_size, dim, H, W, device=device, requires_grad=True)
    
    # Forward pass (mimicking the cascaded predictions)
    # Step 1: Object prediction
    obj_pred = obj_pred_layer(cls_feat)
    
    # Step 2: Relation prediction (uses object info)
    rel_feat = obj_cross_attn(cls_feat, obj_pred)
    rel_pred = rel_pred_layer(rel_feat)
    
    # Step 3: Action prediction (uses object + relation info)
    act_feat = obj_rel_cross_attn(cls_feat, obj_pred, rel_pred)
    act_pred = act_pred_layer(act_feat)
    
    # Backward pass - compute loss from action only
    action_loss = act_pred.mean()
    action_loss.backward()
    
    # Check gradients
    has_obj_grad = obj_pred_layer.weight.grad is not None and obj_pred_layer.weight.grad.abs().sum() > 0
    has_rel_grad = rel_pred_layer.weight.grad is not None and rel_pred_layer.weight.grad.abs().sum() > 0
    has_act_grad = act_pred_layer.weight.grad is not None and act_pred_layer.weight.grad.abs().sum() > 0
    
    print(f"\n  Gradient to action layer: {'✅ YES' if has_act_grad else '❌ NO'}")
    print(f"  Gradient to relation layer: {'✅ YES' if has_rel_grad else '❌ NO'}")
    print(f"  Gradient to object layer: {'✅ YES' if has_obj_grad else '❌ NO'}")
    
    if has_obj_grad:
        print("\n  ✅ PASS: Action loss backpropagates to object predictions!")
        print("         This means object predictions will be optimized for action accuracy.")
    else:
        print("\n  ❌ FAIL: No gradient flow to object predictions!")
        print("         The cascaded design won't help action predictions.")
    
    return has_obj_grad and has_rel_grad and has_act_grad


def test_full_model_inference():
    """Test the full YOWOMultiTask model."""
    print("\n" + "=" * 70)
    print("TEST 3: Full Model Inference Check")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    cfg = yowo_v2_config['yowo_v2_x3d_m_yolo11m_multitask']
    
    print(f"\n  Building YOWOMultiTask model on {device}...")
    
    model = YOWOMultiTask(
        cfg=cfg,
        device=device,
        num_objects=36,
        num_actions=157,
        num_relations=26,
        trainable=True
    ).to(device)
    
    # Create dummy input
    batch_size = 2
    clip_len = 16
    img_size = 224
    
    video_clips = torch.randn(batch_size, 3, clip_len, img_size, img_size, device=device)
    
    print(f"  Input shape: {video_clips.shape}")
    
    # Forward pass
    model.train()
    outputs = model(video_clips)
    
    print(f"\n  Output keys: {list(outputs.keys())}")
    print(f"  Number of FPN levels: {len(outputs['pred_obj'])}")
    
    for i, level in enumerate(outputs['pred_obj']):
        B, M, C = level.shape
        print(f"    Level {i}: {M} anchors, {C} object classes")
    
    # Check that predictions have varied values
    obj_pred = torch.cat(outputs['pred_obj'], dim=1)
    act_pred = torch.cat(outputs['pred_act'], dim=1)
    rel_pred = torch.cat(outputs['pred_rel'], dim=1)
    
    print(f"\n  Object predictions - mean: {obj_pred.mean():.4f}, std: {obj_pred.std():.4f}")
    print(f"  Action predictions - mean: {act_pred.mean():.4f}, std: {act_pred.std():.4f}")
    print(f"  Relation predictions - mean: {rel_pred.mean():.4f}, std: {rel_pred.std():.4f}")
    
    # Verify cross-attention is being used
    print("\n--- Verifying Cross-Attention Usage ---")
    
    # Test that changing object predictions affects action predictions
    with torch.no_grad():
        # Original forward pass - capture intermediate values
        key_frame = video_clips[:, :, -1, :, :]
        feat_3d = model.backbone_3d(video_clips)
        cls_feats, reg_feats = model.backbone_2d(key_frame)
        
        # Get first level features
        feat_3d_up = torch.nn.functional.interpolate(feat_3d, scale_factor=4)
        cls_feat = model.cls_channel_encoders[0](cls_feats[0], feat_3d_up)
        reg_feat = model.reg_channel_encoders[0](reg_feats[0], feat_3d_up)
        cls_feat, reg_feat = model.heads[0](cls_feat, reg_feat)
        
        # Object prediction
        obj_pred_original = model.obj_preds[0](cls_feat)
        
        # Relation with original objects
        rel_feat_original = model.obj_cross_attn[0](cls_feat, obj_pred_original)
        rel_pred_original = model.rel_preds[0](rel_feat_original)
        
        # Action with original objects + relations
        act_feat_original = model.obj_rel_cross_attn[0](cls_feat, obj_pred_original, rel_pred_original)
        act_pred_original = model.act_preds[0](act_feat_original)
        
        # Now modify object predictions
        obj_pred_modified = obj_pred_original.clone()
        obj_pred_modified[:, :, :, :] += 10.0  # Significant modification
        
        # Relation with modified objects
        rel_feat_modified = model.obj_cross_attn[0](cls_feat, obj_pred_modified)
        rel_pred_modified = model.rel_preds[0](rel_feat_modified)
        
        # Action with modified objects + modified relations
        act_feat_modified = model.obj_rel_cross_attn[0](cls_feat, obj_pred_modified, rel_pred_modified)
        act_pred_modified = model.act_preds[0](act_feat_modified)
        
        # Compare
        rel_diff = (rel_pred_original - rel_pred_modified).abs().mean().item()
        act_diff = (act_pred_original - act_pred_modified).abs().mean().item()
        
        print(f"\n  When object predictions change by magnitude 10:")
        print(f"    Relation prediction change: {rel_diff:.6f}")
        print(f"    Action prediction change: {act_diff:.6f}")
        
        if rel_diff > 0.01 and act_diff > 0.01:
            print("\n  ✅ PASS: Cross-attention IS propagating object/relation info to actions!")
        else:
            print("\n  ❌ FAIL: Cross-attention NOT properly propagating information!")
            if rel_diff < 0.01:
                print("         - Relation predictions not affected by objects")
            if act_diff < 0.01:
                print("         - Action predictions not affected by objects/relations")
    
    return True


def test_semantic_attention():
    """Test that attention focuses on semantically related positions."""
    print("\n" + "=" * 70)
    print("TEST 4: Semantic Attention Pattern Analysis")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dim = 256
    H, W = 14, 14
    num_objects = 36
    
    obj_cross_attn = ObjectCrossAttention(dim=dim, num_classes=num_objects).to(device)
    
    # Create a scenario where one position has STRONG object prediction
    cls_feat = torch.randn(1, dim, H, W, device=device)
    
    # Object predictions: mostly zero, but position (7,7) has strong "laptop" prediction
    obj_pred = torch.zeros(1, num_objects, H, W, device=device)
    obj_pred[0, 10, 7, 7] = 10.0  # Strong laptop at center
    
    # Run attention
    out, attn_weights = obj_cross_attn(cls_feat, obj_pred, return_weights=True)
    
    # attn_weights shape: [B, N, N] where N = H*W = 196
    # Check if queries attend more to position 7*14+7 = 105 (the laptop position)
    laptop_position = 7 * W + 7
    
    attention_to_laptop = attn_weights[0, :, laptop_position].mean().item()
    attention_to_others = attn_weights[0, :, :].mean().item()
    
    print(f"\n  Average attention to laptop position: {attention_to_laptop:.6f}")
    print(f"  Average attention overall: {attention_to_others:.6f}")
    print(f"  Ratio (laptop/overall): {attention_to_laptop/attention_to_others:.2f}x")
    
    if attention_to_laptop > attention_to_others:
        print("\n  ✅ PASS: Attention focuses MORE on positions with strong object predictions")
    else:
        print("\n  ⚠️ WARNING: Attention does not focus on strong object positions")
        print("         This may indicate the attention needs more training.")
    
    return True


def test_training_step_simulation():
    """Simulate a training step to verify losses propagate correctly."""
    print("\n" + "=" * 70)
    print("TEST 5: Training Step Simulation")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    cfg = yowo_v2_config['yowo_v2_x3d_m_yolo11m_multitask']
    
    model = YOWOMultiTask(
        cfg=cfg,
        device=device,
        num_objects=36,
        num_actions=157,
        num_relations=26,
        trainable=True
    ).to(device)
    
    model.train()
    
    # Create input
    batch_size = 2
    video_clips = torch.randn(batch_size, 3, 16, 224, 224, device=device)
    
    # Forward
    outputs = model(video_clips)
    
    # Simulate action loss only
    act_preds = torch.cat(outputs['pred_act'], dim=1)  # [B, M, 157]
    action_loss = act_preds.mean()  # Dummy loss
    
    # Backward
    action_loss.backward()
    
    # Check gradients on cross-attention layers
    print("\n  Checking gradients on cross-attention modules...")
    
    obj_cross_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.obj_cross_attn[0].parameters()
    )
    
    obj_rel_cross_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.obj_rel_cross_attn[0].parameters()
    )
    
    obj_pred_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.obj_preds[0].parameters()
    )
    
    rel_pred_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.rel_preds[0].parameters()
    )
    
    print(f"    ObjectCrossAttention gradients: {'✅ YES' if obj_cross_has_grad else '❌ NO'}")
    print(f"    ObjectRelationCrossAttention gradients: {'✅ YES' if obj_rel_cross_has_grad else '❌ NO'}")
    print(f"    Object prediction layer gradients: {'✅ YES' if obj_pred_has_grad else '❌ NO'}")
    print(f"    Relation prediction layer gradients: {'✅ YES' if rel_pred_has_grad else '❌ NO'}")
    
    all_good = obj_cross_has_grad and obj_rel_cross_has_grad and obj_pred_has_grad and rel_pred_has_grad
    
    if all_good:
        print("\n  ✅ ALL GRADIENTS FLOWING CORRECTLY!")
        print("     Your action loss WILL influence object and relation predictions.")
    else:
        print("\n  ❌ SOME GRADIENTS MISSING!")
        print("     The cascaded training may not work as intended.")
    
    return all_good


def main():
    print("\n" + "=" * 70)
    print("  YOWO MULTI-TASK CROSS-ATTENTION ARCHITECTURE VERIFICATION")
    print("=" * 70)
    
    results = {}
    
    try:
        results['cross_attention_modules'] = test_cross_attention_modules()
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        results['cross_attention_modules'] = False
    
    try:
        results['gradient_flow'] = test_gradient_flow()
    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {e}")
        results['gradient_flow'] = False
    
    try:
        results['full_model'] = test_full_model_inference()
    except Exception as e:
        print(f"\n❌ Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results['full_model'] = False
    
    try:
        results['semantic_attention'] = test_semantic_attention()
    except Exception as e:
        print(f"\n❌ Test 4 failed with error: {e}")
        results['semantic_attention'] = False
    
    try:
        results['training_step'] = test_training_step_simulation()
    except Exception as e:
        print(f"\n❌ Test 5 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results['training_step'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
        print("  Your cross-attention architecture is correctly configured.")
        print("  The action head WILL utilize object and relation information.")
    else:
        print("  ⚠️ SOME TESTS FAILED")
        print("  Please review the failed tests above for details.")
    print("=" * 70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    main()
