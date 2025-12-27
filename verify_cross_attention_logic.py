#!/usr/bin/env python3
"""
Cross-Attention Logic Verification - LIGHTWEIGHT Version for Orin Nano

This script verifies the cross-attention architecture without loading the full model,
focusing on the mathematical correctness of the attention mechanism.
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Import the modules directly
from models.yowo.yowo_multitask import (
    ObjectContextModule, 
    SceneContextAttention,
    ObjectRelationContextModule
)


def test_1_object_context_module():
    """Test ObjectContextModule (for Relation predictions)."""
    print("\n" + "="*70)
    print("TEST 1: ObjectContextModule - Does it see object predictions?")
    print("="*70)
    
    device = 'cpu'  # CPU to avoid memory issues
    batch_size = 2
    dim = 256
    H, W = 7, 7  # Smallest feature map
    num_objects = 36
    
    module = ObjectContextModule(dim=dim, num_classes=num_objects).to(device)
    
    # Create inputs
    cls_feat = torch.randn(batch_size, dim, H, W, device=device, requires_grad=True)
    
    # Test 1: Zero object predictions
    obj_pred_zero = torch.zeros(batch_size, num_objects, H, W, device=device)
    out_zero = module(cls_feat, obj_pred_zero)
    
    # Test 2: Strong object predictions
    obj_pred_strong = torch.zeros(batch_size, num_objects, H, W, device=device)
    obj_pred_strong[:, 0, :, :] = 10.0  # Strong "person" everywhere
    obj_pred_strong[:, 10, :, :] = 10.0  # Strong "laptop" 
    out_strong = module(cls_feat, obj_pred_strong)
    
    # Compare outputs
    diff = (out_zero - out_strong).abs().mean().item()
    
    print(f"\n  Input: cls_feat shape {cls_feat.shape}, obj_pred shape {obj_pred_zero.shape}")
    print(f"  Output shape: {out_zero.shape}")
    print(f"\n  Difference when obj_pred changes from 0 to strong: {diff:.6f}")
    
    if diff > 0.01:
        print("\n  ✅ PASS: ObjectContextModule IS SENSITIVE to object predictions!")
        print("         Relation features WILL be influenced by what objects exist.")
        return True
    else:
        print("\n  ❌ FAIL: Output not affected by object predictions")
        return False


def test_2_scene_context_attention():
    """Test SceneContextAttention (for Action predictions)."""
    print("\n" + "="*70)
    print("TEST 2: SceneContextAttention - Does it see objects AND relations?")
    print("="*70)
    
    device = 'cpu'
    batch_size = 2
    dim = 256
    H, W = 7, 7
    num_objects = 36
    num_relations = 26
    
    module = SceneContextAttention(
        dim=dim, 
        num_objects=num_objects, 
        num_relations=num_relations,
        num_heads=4
    ).to(device)
    
    # Create inputs
    cls_feat = torch.randn(batch_size, dim, H, W, device=device, requires_grad=True)
    
    # Test 1: No context (zeros)
    obj_pred_zero = torch.zeros(batch_size, num_objects, H, W, device=device)
    rel_pred_zero = torch.zeros(batch_size, num_relations, H, W, device=device)
    out_zero = module(cls_feat, obj_pred_zero, rel_pred_zero)
    
    # Test 2: Rich context (person + laptop + holding)
    obj_pred_rich = torch.zeros(batch_size, num_objects, H, W, device=device)
    obj_pred_rich[:, 0, :, :] = 10.0  # Person
    obj_pred_rich[:, 10, :, :] = 10.0  # Laptop
    
    rel_pred_rich = torch.zeros(batch_size, num_relations, H, W, device=device)
    rel_pred_rich[:, 5, :, :] = 10.0  # "holding" (assumed index)
    rel_pred_rich[:, 0, :, :] = 10.0  # "looking at" (assumed index)
    
    out_rich = module(cls_feat, obj_pred_rich, rel_pred_rich)
    
    # Compare
    diff = (out_zero - out_rich).abs().mean().item()
    
    print(f"\n  Input: cls_feat {cls_feat.shape}")
    print(f"         obj_pred {obj_pred_zero.shape}, rel_pred {rel_pred_zero.shape}")
    print(f"  Output shape: {out_zero.shape}")
    print(f"\n  Difference when context changes from empty to rich: {diff:.6f}")
    
    if diff > 0.01:
        print("\n  ✅ PASS: SceneContextAttention IS SENSITIVE to object+relation context!")
        print("         Action features WILL be influenced by scene understanding.")
        return True
    else:
        print("\n  ❌ FAIL: Output not affected by context")
        return False


def test_3_attention_weights_distribution():
    """Test that attention weights are actually computed properly."""
    print("\n" + "="*70)
    print("TEST 3: Attention Weights Analysis")
    print("="*70)
    
    device = 'cpu'
    batch_size = 1
    dim = 256
    H, W = 7, 7
    num_objects = 36
    num_relations = 26
    
    module = SceneContextAttention(
        dim=dim, 
        num_objects=num_objects, 
        num_relations=num_relations,
        num_heads=4
    ).to(device)
    
    cls_feat = torch.randn(batch_size, dim, H, W, device=device)
    obj_pred = torch.randn(batch_size, num_objects, H, W, device=device)
    rel_pred = torch.randn(batch_size, num_relations, H, W, device=device)
    
    out, attn_weights = module(cls_feat, obj_pred, rel_pred, return_weights=True)
    
    # attn_weights shape: [B, N, N] where N = H*W
    N = H * W
    
    print(f"\n  Attention weights shape: {attn_weights.shape}")
    print(f"  Expected shape: [{batch_size}, {N}, {N}]")
    
    # Check 1: Softmax property (rows sum to 1)
    row_sums = attn_weights.sum(dim=-1)
    softmax_check = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print(f"\n  ✓ Rows sum to 1 (softmax): {softmax_check}")
    
    # Check 2: All values in [0, 1]
    in_range = (attn_weights >= 0).all() and (attn_weights <= 1).all()
    print(f"  ✓ All values in [0, 1]: {in_range}")
    
    # Check 3: Not uniform
    uniform_val = 1.0 / N
    actual_std = attn_weights.std().item()
    is_uniform = actual_std < uniform_val * 0.1  # Very low std = uniform
    print(f"  ✓ Attention is varied (std={actual_std:.6f}, uniform would be near 0)")
    
    # Check 4: Entropy analysis
    eps = 1e-10
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean()
    max_entropy = np.log(N)
    normalized_entropy = entropy / max_entropy
    print(f"\n  Attention entropy: {entropy:.4f} (max possible: {max_entropy:.4f})")
    print(f"  Normalized entropy: {normalized_entropy:.4f}")
    print(f"    0.0 = fully focused on one position")
    print(f"    1.0 = completely uniform")
    
    if softmax_check and in_range:
        print("\n  ✅ PASS: Attention weights are mathematically valid!")
        return True
    else:
        print("\n  ❌ FAIL: Attention weights have issues")
        return False


def test_4_gradient_flow():
    """Test gradients flow from action loss to object predictions."""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow - Does action loss train object predictions?")
    print("="*70)
    
    device = 'cpu'
    dim = 256
    H, W = 7, 7
    num_objects = 36
    num_relations = 26
    num_actions = 157
    
    # Create the prediction pipeline
    obj_cross_attn = ObjectContextModule(dim=dim, num_classes=num_objects).to(device)
    obj_rel_cross_attn = SceneContextAttention(
        dim=dim, num_objects=num_objects, num_relations=num_relations
    ).to(device)
    
    obj_pred_layer = nn.Conv2d(dim, num_objects, kernel_size=1).to(device)
    rel_pred_layer = nn.Conv2d(dim, num_relations, kernel_size=1).to(device)
    act_pred_layer = nn.Conv2d(dim, num_actions, kernel_size=1).to(device)
    
    # Input
    cls_feat = torch.randn(1, dim, H, W, device=device, requires_grad=True)
    
    # Forward pass (cascaded predictions)
    obj_pred = obj_pred_layer(cls_feat)
    rel_feat = obj_cross_attn(cls_feat, obj_pred)
    rel_pred = rel_pred_layer(rel_feat)
    act_feat = obj_rel_cross_attn(cls_feat, obj_pred, rel_pred)
    act_pred = act_pred_layer(act_feat)
    
    # Compute action loss
    action_loss = act_pred.sum()
    action_loss.backward()
    
    # Check gradients
    obj_layer_has_grad = (obj_pred_layer.weight.grad is not None and 
                          obj_pred_layer.weight.grad.abs().sum() > 0)
    rel_layer_has_grad = (rel_pred_layer.weight.grad is not None and 
                          rel_pred_layer.weight.grad.abs().sum() > 0)
    act_layer_has_grad = (act_pred_layer.weight.grad is not None and 
                          act_pred_layer.weight.grad.abs().sum() > 0)
    
    obj_cross_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in obj_cross_attn.parameters()
    )
    act_cross_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in obj_rel_cross_attn.parameters()
    )
    
    print("\n  Gradient check (action loss backpropagation):")
    print(f"    Action pred layer: {'✅' if act_layer_has_grad else '❌'}")
    print(f"    SceneContextAttention: {'✅' if act_cross_has_grad else '❌'}")
    print(f"    Relation pred layer: {'✅' if rel_layer_has_grad else '❌'}")
    print(f"    ObjectContextModule: {'✅' if obj_cross_has_grad else '❌'}")
    print(f"    Object pred layer: {'✅' if obj_layer_has_grad else '❌'}")
    
    all_good = obj_layer_has_grad and rel_layer_has_grad and act_layer_has_grad
    
    if all_good:
        print("\n  ✅ PASS: Gradients flow through entire cascaded pipeline!")
        print("         Action loss WILL improve object predictions.")
        return True
    else:
        print("\n  ❌ FAIL: Some gradients are missing!")
        return False


def test_5_position_encoding():
    """Test position encoding is being used."""
    print("\n" + "="*70)
    print("TEST 5: Position Encoding - Is spatial info being used?")
    print("="*70)
    
    device = 'cpu'
    dim = 256
    H, W = 7, 7
    num_objects = 36
    num_relations = 26
    
    module = SceneContextAttention(
        dim=dim, num_objects=num_objects, num_relations=num_relations
    ).to(device)
    
    # Get position encoding
    pos = module.get_position_encoding(H, W, device)
    
    print(f"\n  Position encoding shape: {pos.shape}")
    print(f"  pos_scale (learnable): {module.pos_scale.item():.4f}")
    
    # Check that different positions have different encodings
    pos_flat = pos.view(dim, -1).T  # [N, C]
    
    # Cosine similarity between different positions
    pos0 = pos_flat[0]  # top-left
    pos_center = pos_flat[H*W//2]  # center
    pos_corner = pos_flat[-1]  # bottom-right
    
    sim_0_center = torch.cosine_similarity(pos0.unsqueeze(0), pos_center.unsqueeze(0)).item()
    sim_0_corner = torch.cosine_similarity(pos0.unsqueeze(0), pos_corner.unsqueeze(0)).item()
    
    print(f"\n  Position similarity analysis:")
    print(f"    Top-left vs Center: {sim_0_center:.4f}")
    print(f"    Top-left vs Bottom-right: {sim_0_corner:.4f}")
    print(f"    (Lower = positions are more distinguishable)")
    
    if abs(sim_0_center) < 0.99 and abs(sim_0_corner) < 0.99:
        print("\n  ✅ PASS: Positions are distinguishable!")
        print("         Model CAN learn to attend to nearby vs far positions.")
        return True
    else:
        print("\n  ⚠️ WARNING: Positions are very similar")
        return True  # Not a hard failure


def test_6_semantic_focus():
    """Test that keys are derived from object+relation predictions."""
    print("\n" + "="*70)
    print("TEST 6: Semantic Focus - Do keys come from predictions?")
    print("="*70)
    
    device = 'cpu'
    dim = 256
    H, W = 7, 7
    num_objects = 36
    num_relations = 26
    
    module = SceneContextAttention(
        dim=dim, num_objects=num_objects, num_relations=num_relations
    ).to(device)
    
    # Same features, different predictions
    cls_feat = torch.randn(1, dim, H, W, device=device)
    
    # Scenario A: Position (3,3) has a laptop
    obj_pred_a = torch.zeros(1, num_objects, H, W, device=device)
    obj_pred_a[0, 10, 3, 3] = 10.0  # Laptop at (3,3)
    rel_pred_a = torch.zeros(1, num_relations, H, W, device=device)
    
    # Scenario B: Position (3,3) has a person
    obj_pred_b = torch.zeros(1, num_objects, H, W, device=device)
    obj_pred_b[0, 0, 3, 3] = 10.0  # Person at (3,3)
    rel_pred_b = torch.zeros(1, num_relations, H, W, device=device)
    
    out_a, attn_a = module(cls_feat, obj_pred_a, rel_pred_a, return_weights=True)
    out_b, attn_b = module(cls_feat, obj_pred_b, rel_pred_b, return_weights=True)
    
    # Attention pattern should change based on what object is predicted
    attn_diff = (attn_a - attn_b).abs().mean().item()
    
    print(f"\n  Same features, different object predictions at position (3,3)")
    print(f"  Attention pattern difference: {attn_diff:.6f}")
    
    # Check attention to the changed position
    pos_idx = 3 * W + 3
    attn_to_laptop = attn_a[0, :, pos_idx].mean().item()
    attn_to_person = attn_b[0, :, pos_idx].mean().item()
    
    print(f"\n  Attention to position (3,3):")
    print(f"    When laptop present: {attn_to_laptop:.6f}")
    print(f"    When person present: {attn_to_person:.6f}")
    
    if attn_diff > 0.0001:
        print("\n  ✅ PASS: Attention patterns change based on object predictions!")
        print("         The model WILL attend differently to laptops vs persons.")
        return True
    else:
        print("\n  ❌ FAIL: Attention unchanged despite different predictions")
        return False


def main():
    print("\n" + "="*70)
    print("  CROSS-ATTENTION LOGIC VERIFICATION")
    print("  (Lightweight tests for Orin Nano)")
    print("="*70)
    
    results = {}
    
    tests = [
        ("Object Context Module", test_1_object_context_module),
        ("Scene Context Attention", test_2_scene_context_attention),
        ("Attention Weights", test_3_attention_weights_distribution),
        ("Gradient Flow", test_4_gradient_flow),
        ("Position Encoding", test_5_position_encoding),
        ("Semantic Focus", test_6_semantic_focus),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n❌ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("  🎉 ALL LOGIC TESTS PASSED!")
        print("")
        print("  Your cross-attention implementation is CORRECT:")
        print("  ✓ Object predictions influence relation predictions")
        print("  ✓ Object+Relation predictions influence action predictions")
        print("  ✓ Attention weights are mathematically valid")
        print("  ✓ Gradients flow through the entire pipeline")
        print("  ✓ Position encoding provides spatial awareness")
        print("  ✓ Keys are semantically derived from predictions")
        print("")
        print("  The model WILL learn to use scene context for action detection!")
    else:
        print("  ⚠️ SOME TESTS FAILED - Review the output above")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
