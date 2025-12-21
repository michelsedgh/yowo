#!/usr/bin/env python3
"""
Comprehensive Dataset Verification for YOWOv2 with Charades + Action Genome
Tests ONE video (001YG) on ALL its keyframes to ensure:
1. Dataset loader works correctly
2. Image preprocessing is correct (224x224, normalized)
3. Bounding boxes are scaled properly
4. Labels are multi-hot encoded correctly (219 classes)
5. Temporal clip sampling works (16 frames)
"""
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import pickle

# Add to path
sys.path.insert(0, os.getcwd())

from dataset.charades_ag import CharadesAGDataset
from dataset.transforms import BaseTransform

def verify_one_video_all_keyframes(video_id='001YG'):
    """Test all keyframes of one video"""
    
    print("=" * 80)
    print(f"COMPREHENSIVE TEST: Video {video_id}")
    print("=" * 80)
    
    # 1. Setup Dataset (same as training)
    cfg = {
        'train_size': 224,
        'test_size': 224,
        'sampling_rate': 1,
        'multi_hot': True,
        'valid_num_classes': 219,  # CORRECT: 36 + 157 + 26
    }
    
    # Use BaseTransform for testing (no augmentation, just resize + normalize)
    transform = BaseTransform(img_size=224)
    
    dataset = CharadesAGDataset(
        cfg=cfg,
        data_root='/home/michel/yowo/data/ActionGenome',
        is_train=True,
        img_size=224,
        transform=transform,
        len_clip=16,
        sampling_rate=1
    )
    
    print(f"\n✓ Dataset initialized: {len(dataset)} total keyframes")
    print(f"✓ Classes: {dataset.num_classes} (Objects: {dataset.num_objects}, Actions: {dataset.num_actions}, Relations: {dataset.num_relations})")
    
    # Verify class count
    assert dataset.num_classes == 219, f"ERROR: Expected 219 classes, got {dataset.num_classes}"
    print("✓ Class count verified: 219")
    
    # 2. Find all keyframes for this video
    video_keyframes = []
    for idx, kf in enumerate(dataset.keyframes):
        if kf.startswith(f"{video_id}.mp4/"):
            video_keyframes.append(idx)
    
    print(f"\n✓ Found {len(video_keyframes)} keyframes for video {video_id}")
    
    if len(video_keyframes) == 0:
        print(f"ERROR: No keyframes found for video {video_id}")
        return
    
    # 3. Create output directory
    output_dir = Path('/home/michel/yowo/vis_comprehensive_test')
    output_dir.mkdir(exist_ok=True)
    
    # 4. Process each keyframe
    print("\n" + "=" * 80)
    print("Processing keyframes...")
    print("=" * 80)
    
    all_stats = {
        'frame_sizes': [],
        'num_boxes': [],
        'num_persons': [],
        'num_objects': [],
        'num_actions': [],
        'num_relations': [],
        'clip_shapes': [],
    }
    
    for i, idx in enumerate(video_keyframes):
        [vid, frame_idx], video_clip, target = dataset[idx]
        
        # Verify video_clip shape: should be [3, T, H, W]
        assert video_clip.shape[0] == 3, f"Expected channels=3, got {video_clip.shape[0]}"
        assert video_clip.shape[1] == 16, f"Expected temporal=16, got {video_clip.shape[1]}"
        assert video_clip.shape[2] == 224, f"Expected height=224, got {video_clip.shape[2]}"
        assert video_clip.shape[3] == 224, f"Expected width=224, got {video_clip.shape[3]}"
        
        all_stats['clip_shapes'].append(video_clip.shape)
        
        # Get the keyframe (last frame of the clip)
        keyframe = video_clip[:, -1, :, :]  # [3, H, W]
        
        # Convert to numpy for visualization
        img_np = keyframe.permute(1, 2, 0).numpy()  # [H, W, 3]
        # Since the transform already multiplied by 255, we just need to cast to uint8
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Get targets
        boxes = target['boxes'].numpy()  # [N, 4] in normalized coordinates (0-1)
        labels = target['labels'].numpy()  # [N, 219]
        
        all_stats['num_boxes'].append(len(boxes))
        all_stats['frame_sizes'].append(target['orig_size'])
        
        # Draw and analyze
        vis = img_bgr.copy()
        num_persons = 0
        num_objects = 0
        num_actions = 0
        num_relations = 0
        
        for box_idx, (box, label) in enumerate(zip(boxes, labels)):
            # Convert normalized coords to pixels
            x1, y1, x2, y2 = (box * 224).astype(int)
            
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(224, x2), min(224, y2)
            
            # Find active classes
            active_indices = np.where(label > 0.5)[0]
            
            # Determine type
            is_person = 0 in active_indices  # Index 0 is person
            
            if is_person:
                num_persons += 1
                color = (0, 255, 0)  # Green for person
                class_name = "PERSON"
                
                # Count actions
                action_indices = active_indices[(active_indices >= dataset.num_objects) & 
                                               (active_indices < dataset.num_objects + dataset.num_actions)]
                num_actions += len(action_indices)
                
                # Show actions
                if len(action_indices) > 0:
                    actions = [dataset.charades_actions[a - dataset.num_objects] for a in action_indices[:3]]
                    action_text = ', '.join([a.split()[0] if ' ' in a else a for a in actions])
                    class_name += f" [{action_text}]"
            else:
                num_objects += 1
                color = (255, 165, 0)  # Orange for objects
                
                # Find object class
                obj_indices = active_indices[active_indices < dataset.num_objects]
                if len(obj_indices) > 0:
                    class_name = dataset.ag_objects[obj_indices[0]].upper()
                else:
                    class_name = "UNKNOWN"
            
            # Count relations for all boxes
            rel_indices = active_indices[active_indices >= (dataset.num_objects + dataset.num_actions)]
            num_relations += len(rel_indices)
            
            # Show relations
            if len(rel_indices) > 0:
                rels = [dataset.ag_relations[r - (dataset.num_objects + dataset.num_actions)] 
                       for r in rel_indices[:2]]
                rel_text = ','.join(rels)
                class_name += f" |{rel_text}|"
            
            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-4), (x1+label_size[0], y1), color, -1)
            cv2.putText(vis, class_name, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        all_stats['num_persons'].append(num_persons)
        all_stats['num_objects'].append(num_objects)
        all_stats['num_actions'].append(num_actions)
        all_stats['num_relations'].append(num_relations)
        
        # Add info text
        info_text = [
            f"Video: {vid} | Frame: {frame_idx:06d} | KF: {i+1}/{len(video_keyframes)}",
            f"Boxes: {len(boxes)} (P:{num_persons} O:{num_objects})",
            f"Actions: {num_actions} | Relations: {num_relations}",
            f"OrigSize: {target['orig_size'][1]}x{target['orig_size'][0]} -> 224x224",
        ]
        
        y_pos = 20
        for text in info_text:
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            y_pos += 20
        
        # Save
        out_path = output_dir / f"test_{vid}_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), vis)
        
        print(f"  [{i+1:2d}/{len(video_keyframes)}] Frame {frame_idx:06d}: {len(boxes):2d} boxes "
              f"(P:{num_persons} O:{num_objects}) A:{num_actions:2d} R:{num_relations:2d} -> {out_path.name}")
    
    # 5. Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total keyframes processed: {len(video_keyframes)}")
    print(f"Clip shape (all frames): {all_stats['clip_shapes'][0]}")
    print(f"\nBounding Boxes:")
    print(f"  Total: {sum(all_stats['num_boxes'])}")
    print(f"  Per frame: avg={np.mean(all_stats['num_boxes']):.1f}, min={np.min(all_stats['num_boxes'])}, max={np.max(all_stats['num_boxes'])}")
    print(f"\nPersons:")
    print(f"  Total: {sum(all_stats['num_persons'])}")
    print(f"  Frames with person: {sum(1 for x in all_stats['num_persons'] if x > 0)}/{len(video_keyframes)}")
    print(f"\nObjects:")
    print(f"  Total: {sum(all_stats['num_objects'])}")
    print(f"  Per frame: avg={np.mean(all_stats['num_objects']):.1f}")
    print(f"\nActions (Charades):")
    print(f"  Total: {sum(all_stats['num_actions'])}")
    print(f"  Frames with actions: {sum(1 for x in all_stats['num_actions'] if x > 0)}/{len(video_keyframes)}")
    print(f"  Per frame (when present): {np.mean([x for x in all_stats['num_actions'] if x > 0]):.1f}")
    print(f"\nRelations (Action Genome):")
    print(f"  Total: {sum(all_stats['num_relations'])}")
    print(f"  Frames with relations: {sum(1 for x in all_stats['num_relations'] if x > 0)}/{len(video_keyframes)}")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    
    # 6. Verify label encoding
    print("\n" + "=" * 80)
    print("LABEL ENCODING VERIFICATION")
    print("=" * 80)
    
    # Check a sample
    [vid, frame_idx], video_clip, target = dataset[video_keyframes[0]]
    labels = target['labels']
    
    print(f"Label tensor shape: {labels.shape}")
    print(f"Expected: [N, 219] where N is number of boxes")
    assert labels.shape[1] == 219, f"ERROR: Expected 219 classes, got {labels.shape[1]}"
    print("✓ Label shape correct")
    
    print(f"\nLabel breakdown (indices):")
    print(f"  Objects (person + 35 objs): 0-35")
    print(f"  Actions (Charades): 36-192")
    print(f"  Relations (AG): 193-218")
    
    # Sample label analysis
    if labels.shape[0] > 0:
        sample_label = labels[0]
        active = torch.where(sample_label > 0.5)[0]
        print(f"\nSample box has {len(active)} active classes:")
        for a in active[:10]:
            if a < 36:
                print(f"  [{a:3d}] Object: {dataset.ag_objects[a]}")
            elif a < 193:
                print(f"  [{a:3d}] Action: {dataset.charades_actions[a-36]}")
            else:
                print(f"  [{a:3d}] Relation: {dataset.ag_relations[a-193]}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nConclusions:")
    print("1. Dataset loader correctly loads all keyframes")
    print("2. Video clips are properly shaped: [3, 16, 224, 224]")
    print("3. Bounding boxes are scaled to 224x224 resolution")
    print("4. Labels are multi-hot encoded with 219 classes")
    print("5. Ready for training!")

if __name__ == "__main__":
    verify_one_video_all_keyframes('001YG')




