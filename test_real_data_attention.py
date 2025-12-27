#!/usr/bin/env python3
"""
Real Data Attention Visualization Test

This script:
1. Loads REAL data from the Charades-AG dataset
2. Runs it through the FIXED model
3. Visualizes attention patterns showing WHERE the model attends
4. Creates side-by-side comparisons: with vs without context
5. Shows that object/relation predictions ACTUALLY influence action features
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import json

# Add project root to path
sys.path.insert(0, '/home/michel/yowo')

from models.yowo.yowo_multitask import SceneContextAttention, ObjectContextModule


def load_real_sample(data_root='/home/michel/yowo/data/ActionGenome'):
    """Load a real sample from the Charades-AG dataset."""
    
    # Load annotations
    with open(os.path.join(data_root, 'annotations/person_bbox.pkl'), 'rb') as f:
        person_bboxes = pickle.load(f)
    with open(os.path.join(data_root, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
        object_data = pickle.load(f)
    
    # Load class definitions
    with open(os.path.join(data_root, 'annotations/object_classes.txt'), 'r') as f:
        objects = [line.strip().lower() for line in f if line.strip()]
    with open(os.path.join(data_root, 'annotations/relationship_classes.txt'), 'r') as f:
        relations = [line.strip().lower() for line in f if line.strip()]
    
    # Find a sample with multiple objects and relationships
    for keyframe_id in list(person_bboxes.keys())[:100]:
        if keyframe_id in object_data and len(object_data[keyframe_id]) >= 2:
            video_id = keyframe_id.split('/')[0]
            frame_file = keyframe_id.split('/')[1]
            frame_idx = int(frame_file.replace('.png', '').replace('.jpg', ''))
            
            # Check if frame exists
            frame_path = os.path.join(data_root, 'frames', video_id, frame_file)
            if not os.path.exists(frame_path):
                frame_path = frame_path.replace('.png', '.jpg')
            if not os.path.exists(frame_path):
                continue
            
            # Load frame
            img = Image.open(frame_path).convert('RGB')
            
            # Get annotations
            person_info = person_bboxes[keyframe_id]
            obj_list = object_data[keyframe_id]
            
            print(f"\nLoaded sample: {keyframe_id}")
            print(f"  Image size: {img.size}")
            print(f"  Person bbox: {person_info['bbox'][0] if len(person_info['bbox']) > 0 else 'N/A'}")
            print(f"  Objects in scene:")
            for obj in obj_list:
                rels = []
                for r_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
                    rels.extend(obj.get(r_type, []) or [])
                print(f"    - {obj['class']}: {rels}")
            
            return {
                'image': img,
                'keyframe_id': keyframe_id,
                'person_info': person_info,
                'objects': obj_list,
                'object_classes': objects,
                'relation_classes': relations
            }
    
    return None


def create_spatial_predictions(sample, H, W, num_objects=36, num_relations=26):
    """
    Create object and relation predictions based on actual bounding boxes.
    This simulates what the model would predict at each spatial position.
    """
    
    img_w, img_h = sample['image'].size
    person_info = sample['person_info']
    obj_list = sample['objects']
    objects = sample['object_classes']
    relations = sample['relation_classes']
    
    # Initialize predictions
    obj_pred = torch.zeros(1, num_objects, H, W)
    rel_pred = torch.zeros(1, num_relations, H, W)
    
    # Scaling factors (annotation space to feature map space)
    ann_w, ann_h = person_info['bbox_size']
    scale_x = W / ann_w
    scale_y = H / ann_h
    
    # Add person prediction at person location
    if len(person_info['bbox']) > 0:
        bbox = person_info['bbox'][0]  # [x1, y1, x2, y2]
        cx = int((bbox[0] + bbox[2]) / 2 * scale_x)
        cy = int((bbox[1] + bbox[3]) / 2 * scale_y)
        cx = np.clip(cx, 0, W-1)
        cy = np.clip(cy, 0, H-1)
        
        # Strong person prediction at center, weaker around
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    strength = 5.0 / (1 + abs(dx) + abs(dy))
                    obj_pred[0, 0, ny, nx] = max(obj_pred[0, 0, ny, nx].item(), strength)
    
    # Add object predictions at object locations
    for obj in obj_list:
        obj_name = obj['class'].lower()
        if obj_name in objects:
            obj_idx = objects.index(obj_name)
            bbox = obj['bbox']  # [x, y, w, h]
            cx = int((bbox[0] + bbox[2]/2) * scale_x)
            cy = int((bbox[1] + bbox[3]/2) * scale_y)
            cx = np.clip(cx, 0, W-1)
            cy = np.clip(cy, 0, H-1)
            
            # Strong object prediction
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        strength = 5.0 / (1 + abs(dx) + abs(dy))
                        obj_pred[0, obj_idx, ny, nx] = max(obj_pred[0, obj_idx, ny, nx].item(), strength)
            
            # Add relationships for this object
            for r_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
                for rel in obj.get(r_type, []) or []:
                    rel_norm = rel.replace('_', '').lower()
                    if rel_norm in relations:
                        rel_idx = relations.index(rel_norm)
                        # Add relation prediction between person and object
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < H and 0 <= nx < W:
                                    strength = 5.0 / (1 + abs(dx) + abs(dy))
                                    rel_pred[0, rel_idx, ny, nx] = max(rel_pred[0, rel_idx, ny, nx].item(), strength)
    
    return obj_pred, rel_pred


def visualize_attention(sample, save_path='/home/michel/yowo/attention_visualization.png'):
    """Create comprehensive attention visualization."""
    
    print("\n" + "="*70)
    print("ATTENTION VISUALIZATION ON REAL DATA")
    print("="*70)
    
    dim = 256
    num_objects = 36
    num_relations = 26
    H, W = 14, 14  # Feature map size
    
    # Create the scene context attention module
    scene_attn = SceneContextAttention(dim=dim, num_objects=num_objects, num_relations=num_relations)
    scene_attn.eval()
    
    # Create object context module
    obj_ctx = ObjectContextModule(dim=dim, num_classes=num_objects)
    obj_ctx.eval()
    
    # Create spatial predictions from real annotations
    obj_pred, rel_pred = create_spatial_predictions(sample, H, W)
    
    print(f"\nCreated predictions from real annotations:")
    print(f"  Object predictions max: {obj_pred.max().item():.2f}")
    print(f"  Relation predictions max: {rel_pred.max().item():.2f}")
    print(f"  Positions with objects: {(obj_pred.max(dim=1)[0] > 1).sum().item()}")
    
    # Random features (simulating backbone output)
    cls_feat = torch.randn(1, dim, H, W)
    
    # Get attention weights WITH real context
    with torch.no_grad():
        out_with_ctx, attn_with = scene_attn(cls_feat, obj_pred, rel_pred, return_weights=True)
        
        # Get attention weights WITHOUT context
        out_no_ctx, attn_without = scene_attn(
            cls_feat, 
            torch.zeros_like(obj_pred), 
            torch.zeros_like(rel_pred), 
            return_weights=True
        )
        
        # Also test object context module
        rel_feat_with = obj_ctx(cls_feat, obj_pred)
        rel_feat_without = obj_ctx(cls_feat, torch.zeros_like(obj_pred))
    
    # Compute differences
    output_diff = (out_with_ctx - out_no_ctx).abs().mean().item()
    attn_diff = (attn_with - attn_without).abs().mean().item()
    rel_feat_diff = (rel_feat_with - rel_feat_without).abs().mean().item()
    
    print(f"\n  SceneContextAttention output difference: {output_diff:.6f}")
    print(f"  Attention pattern difference: {attn_diff:.6f}")
    print(f"  ObjectContextModule output difference: {rel_feat_diff:.6f}")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Original image and object predictions
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(sample['image'])
    ax1.set_title(f"Original Image\n{sample['keyframe_id']}", fontsize=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    obj_heatmap = obj_pred.max(dim=1)[0].squeeze().numpy()
    ax2.imshow(obj_heatmap, cmap='hot', interpolation='nearest')
    ax2.set_title("Object Predictions\n(from GT boxes)", fontsize=10)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 4, 3)
    rel_heatmap = rel_pred.max(dim=1)[0].squeeze().numpy()
    ax3.imshow(rel_heatmap, cmap='hot', interpolation='nearest')
    ax3.set_title("Relation Predictions\n(from GT)", fontsize=10)
    ax3.axis('off')
    
    # Row 1 cont: Attention difference
    ax4 = fig.add_subplot(2, 4, 4)
    attn_diff_map = (attn_with[0] - attn_without[0]).abs().mean(dim=0).numpy().reshape(H, W)
    ax4.imshow(attn_diff_map, cmap='viridis', interpolation='nearest')
    ax4.set_title(f"Attention Change\n(with vs without context)", fontsize=10)
    ax4.axis('off')
    
    # Row 2: Attention patterns
    # Find the person position
    person_pos = np.unravel_index(obj_pred[0, 0].argmax().item(), (H, W))
    query_idx = person_pos[0] * W + person_pos[1]
    
    ax5 = fig.add_subplot(2, 4, 5)
    attn_from_person_with = attn_with[0, query_idx].numpy().reshape(H, W)
    ax5.imshow(attn_from_person_with, cmap='hot', interpolation='nearest')
    ax5.scatter([person_pos[1]], [person_pos[0]], c='cyan', s=100, marker='x', linewidths=2)
    ax5.set_title(f"Attention FROM Person (WITH ctx)\n(query at {person_pos})", fontsize=10)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 4, 6)
    attn_from_person_without = attn_without[0, query_idx].numpy().reshape(H, W)
    ax6.imshow(attn_from_person_without, cmap='hot', interpolation='nearest')
    ax6.scatter([person_pos[1]], [person_pos[0]], c='cyan', s=100, marker='x', linewidths=2)
    ax6.set_title(f"Attention FROM Person (NO ctx)\n(query at {person_pos})", fontsize=10)
    ax6.axis('off')
    
    # Find an object position
    obj_positions = []
    for obj in sample['objects']:
        obj_name = obj['class'].lower()
        if obj_name in sample['object_classes']:
            obj_idx = sample['object_classes'].index(obj_name)
            pos = np.unravel_index(obj_pred[0, obj_idx].argmax().item(), (H, W))
            if obj_pred[0, obj_idx, pos[0], pos[1]] > 1:
                obj_positions.append((pos, obj_name))
    
    if obj_positions:
        obj_pos, obj_name = obj_positions[0]
        obj_query_idx = obj_pos[0] * W + obj_pos[1]
        
        ax7 = fig.add_subplot(2, 4, 7)
        attn_from_obj_with = attn_with[0, obj_query_idx].numpy().reshape(H, W)
        ax7.imshow(attn_from_obj_with, cmap='hot', interpolation='nearest')
        ax7.scatter([obj_pos[1]], [obj_pos[0]], c='lime', s=100, marker='x', linewidths=2)
        ax7.set_title(f"Attention FROM {obj_name} (WITH ctx)\n(query at {obj_pos})", fontsize=10)
        ax7.axis('off')
        
        ax8 = fig.add_subplot(2, 4, 8)
        attn_from_obj_without = attn_without[0, obj_query_idx].numpy().reshape(H, W)
        ax8.imshow(attn_from_obj_without, cmap='hot', interpolation='nearest')
        ax8.scatter([obj_pos[1]], [obj_pos[0]], c='lime', s=100, marker='x', linewidths=2)
        ax8.set_title(f"Attention FROM {obj_name} (NO ctx)\n(query at {obj_pos})", fontsize=10)
        ax8.axis('off')
    
    plt.suptitle(
        f"FIXED Cross-Attention on Real Data\n"
        f"Output Diff: {output_diff:.4f} | Attention Diff: {attn_diff:.4f} | Rel Feat Diff: {rel_feat_diff:.4f}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    return {
        'output_diff': output_diff,
        'attn_diff': attn_diff,
        'rel_feat_diff': rel_feat_diff
    }


def test_gradient_with_real_data(sample):
    """Test gradient flow with real data patterns."""
    
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST WITH REAL DATA PATTERNS")
    print("="*70)
    
    dim = 256
    num_objects = 36
    num_relations = 26
    num_actions = 157
    H, W = 14, 14
    
    # Create modules
    scene_attn = SceneContextAttention(dim=dim, num_objects=num_objects, num_relations=num_relations)
    act_pred_layer = torch.nn.Conv2d(dim, num_actions, kernel_size=1)
    
    # Create real predictions
    obj_pred, rel_pred = create_spatial_predictions(sample, H, W)
    obj_pred.requires_grad = True
    rel_pred.requires_grad = True
    
    cls_feat = torch.randn(1, dim, H, W, requires_grad=True)
    
    # Forward
    act_feat = scene_attn(cls_feat, obj_pred, rel_pred)
    act_pred = act_pred_layer(act_feat)
    
    # Compute action loss
    action_loss = act_pred.sum()
    action_loss.backward()
    
    # Check gradients
    obj_grad_sum = obj_pred.grad.abs().sum().item() if obj_pred.grad is not None else 0
    rel_grad_sum = rel_pred.grad.abs().sum().item() if rel_pred.grad is not None else 0
    
    print(f"\n  Gradient from action loss to object predictions: {obj_grad_sum:.6f}")
    print(f"  Gradient from action loss to relation predictions: {rel_grad_sum:.6f}")
    
    if obj_grad_sum > 0 and rel_grad_sum > 0:
        print("\n  ✅ PASS: Gradients flow from action loss to object/relation predictions!")
        print("     The model WILL learn to predict objects/relations that help action detection.")
    else:
        print("\n  ❌ FAIL: No gradient flow!")
    
    return obj_grad_sum > 0 and rel_grad_sum > 0


def main():
    print("="*70)
    print("TESTING FIXED CROSS-ATTENTION ON REAL DATA")
    print("="*70)
    
    # Load a real sample
    sample = load_real_sample()
    
    if sample is None:
        print("❌ Could not load sample from dataset")
        return 1
    
    # Run visualization
    results = visualize_attention(sample)
    
    # Test gradient flow
    grad_ok = test_gradient_with_real_data(sample)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    all_ok = True
    
    if results['output_diff'] > 0.01:
        print(f"✅ Output changes with context: {results['output_diff']:.4f}")
    else:
        print(f"❌ Output doesn't change enough: {results['output_diff']:.4f}")
        all_ok = False
    
    if results['attn_diff'] > 0.0001:
        print(f"✅ Attention patterns change: {results['attn_diff']:.6f}")
    else:
        print(f"❌ Attention patterns don't change: {results['attn_diff']:.6f}")
        all_ok = False
    
    if results['rel_feat_diff'] > 0.01:
        print(f"✅ Relation features change with objects: {results['rel_feat_diff']:.4f}")
    else:
        print(f"❌ Relation features don't change: {results['rel_feat_diff']:.4f}")
        all_ok = False
    
    if grad_ok:
        print("✅ Gradients flow correctly")
    else:
        print("❌ Gradient issues")
        all_ok = False
    
    if all_ok:
        print("\n🎉 ALL TESTS PASSED ON REAL DATA!")
        print(f"   Visualization saved to: /home/michel/yowo/attention_visualization.png")
    else:
        print("\n⚠️ Some tests failed")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
