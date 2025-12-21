#!/usr/bin/env python3
"""
Final test: Verify dataset with transforms works correctly
"""
import torch
from dataset.charades_ag import CharadesAGDataset
from dataset.transforms import Augmentation

def test_with_transforms():
    print("=" * 80)
    print("TESTING CHARADES AG DATASET WITH TRANSFORMS")
    print("=" * 80)
    
    # Config
    cfg = {
        'train_size': 224,
        'sampling_rate': 1
    }
    
    # Transforms
    transform = Augmentation(
        img_size=224,
        jitter=0.2,
        hue=0.1,
        saturation=1.5,
        exposure=1.5
    )
    
    # Dataset
    dataset = CharadesAGDataset(
        cfg=cfg,
        data_root='/home/michel/yowo/data/ActionGenome',
        is_train=True,
        img_size=224,
        transform=transform,
        len_clip=16,
        sampling_rate=1
    )
    
    print(f"\n✓ Dataset loaded: {len(dataset)} keyframes")
    print(f"✓ Num classes: {dataset.num_classes}")
    print(f"  - Objects: {dataset.num_objects}")
    print(f"  - Actions: {dataset.num_actions}")
    print(f"  - Relations: {dataset.num_relations}")
    
    # Test a few samples
    print("\nTesting samples...")
    for i in [0, 100, 1000]:
        [video_id, frame_idx], video_clip, target = dataset[i]
        
        print(f"\n  Sample {i}: {video_id} frame {frame_idx}")
        print(f"    Video clip shape: {video_clip.shape}")  # Should be [3, T, H, W]
        print(f"    Boxes shape: {target['boxes'].shape}")
        print(f"    Labels shape: {target['labels'].shape}")
        print(f"    Orig size: {target['orig_size']}")
        
        # Verify shapes
        assert video_clip.shape[0] == 3, f"Expected 3 channels, got {video_clip.shape[0]}"
        assert video_clip.shape[1] == 16, f"Expected 16 frames, got {video_clip.shape[1]}"
        assert video_clip.shape[2] == 224, f"Expected height 224, got {video_clip.shape[2]}"
        assert video_clip.shape[3] == 224, f"Expected width 224, got {video_clip.shape[3]}"
        assert target['labels'].shape[1] == 219, f"Expected 219 classes, got {target['labels'].shape[1]}"
        assert target['boxes'].shape[0] == target['labels'].shape[0], "Boxes and labels count mismatch"
        
        # Check multi-hot
        active_per_box = target['labels'].sum(dim=1)
        print(f"    Active classes per box: {active_per_box.tolist()}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("✓ Dataset is ready for training!")
    print("=" * 80)

if __name__ == "__main__":
    test_with_transforms()



