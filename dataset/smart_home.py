"""
Smart Home Dataset - Filtered Charades-AG for Smart Home Actions

This dataset wraps CharadesAGDataset to:
1. Filter keyframes to only those with smart home actions
2. Remap action indices from 157 to our 42 smart home classes
3. Keep all objects (36) and relations (26) unchanged

Usage:
    from dataset.smart_home import SmartHomeDataset
    dataset = SmartHomeDataset(cfg, data_root, is_train=True, ...)
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .charades_ag import CharadesAGDataset


class SmartHomeDataset(Dataset):
    """
    Filtered dataset for smart home action detection.
    
    Wraps CharadesAGDataset and:
    - Only returns keyframes that have at least one smart home action
    - Remaps 157 action classes to 42 smart home action classes
    - Keeps objects and relations unchanged
    """
    
    def __init__(self, cfg, data_root, is_train=False, img_size=224, 
                 transform=None, len_clip=16, sampling_rate=1, 
                 negative_ratio=0.12):
        """
        Args:
            negative_ratio: Fraction of positive frames to add as no-action negatives.
                           0.12 = add 12% extra frames with no smart-home actions.
                           Set to 0 to disable (old behavior).
        """
        # Load smart home config
        config_path = os.path.join(os.path.dirname(__file__), 
                                   '../config/smart_home_final.json')
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.num_smart_home_actions = self.config['num_actions']
        self.action_indices = set(self.config['action_indices'])
        self.old_to_new = {int(k): v for k, v in self.config['old_to_new'].items()}
        self.is_train = is_train
        # Include negatives in both train and eval to properly measure false positives
        self.negative_ratio = negative_ratio
        
        # Load base dataset
        self.base_dataset = CharadesAGDataset(
            cfg, data_root, is_train, img_size, transform, len_clip, sampling_rate
        )
        
        # Get dimensions from base dataset
        self.num_objects = self.base_dataset.num_objects  # 36
        self.num_relations = self.base_dataset.num_relations  # 26
        
        # Total classes based on config
        self.num_classes = self.num_objects + self.num_smart_home_actions + self.num_relations
        
        # Filter keyframes: positives (have smart home actions) + sampled negatives
        self.positive_indices, self.negative_indices = self._filter_keyframes()
        
        # Sample negatives: ~negative_ratio of positive count
        num_negatives = int(len(self.positive_indices) * self.negative_ratio)
        if num_negatives > 0 and len(self.negative_indices) > 0:
            sampled_negatives = random.sample(
                self.negative_indices, 
                min(num_negatives, len(self.negative_indices))
            )
        else:
            sampled_negatives = []
        
        # Combined indices: all positives + sampled negatives
        self.filtered_indices = self.positive_indices + sampled_negatives
        # Track which are negatives (for __getitem__ to zero out actions)
        self.negative_set = set(sampled_negatives)
        
        print(f"SmartHomeDataset ({'train' if is_train else 'val'}):")
        print(f"  Positive frames: {len(self.positive_indices)}")
        print(f"  Negative frames: {len(sampled_negatives)} ({100*len(sampled_negatives)/max(1,len(self.positive_indices)):.1f}% of positives)")
        print(f"  Total frames: {len(self.filtered_indices)}")
        print(f"  Actions: {self.num_smart_home_actions}")
        print(f"  Objects: {self.num_objects}")
        print(f"  Relations: {self.num_relations}")
        print(f"  Total classes: {self.num_classes}")
    
    def _filter_keyframes(self):
        """Separate keyframes into positives (have smart home actions) and negatives (no smart home actions).
        
        Uses caching to avoid re-filtering on every run (288K+ keyframes is slow).
        """
        import hashlib
        
        # Create cache key based on: keyframe count, action indices, split
        cache_key = f"{len(self.base_dataset)}_{sorted(self.action_indices)}_{self.is_train}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = os.path.join(os.path.dirname(__file__), '../.cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'smart_home_filter_{cache_hash}.npz')
        
        # Try loading from cache
        if os.path.exists(cache_file):
            try:
                cached = np.load(cache_file)
                print(f"  Loaded filtered indices from cache ({cache_file})")
                return cached['positives'].tolist(), cached['negatives'].tolist()
            except:
                pass  # Cache corrupted, recompute
        
        # Compute from scratch with progress
        print(f"  Filtering {len(self.base_dataset)} keyframes (one-time, will be cached)...")
        positives = []
        negatives = []
        
        # Pre-fetch frequently accessed data for speed
        keyframes = self.base_dataset.keyframes
        video_fps = self.base_dataset.video_fps
        video_actions = self.base_dataset.video_actions
        action_indices = self.action_indices
        
        total = len(self.base_dataset)
        last_pct = -1
        
        for idx in range(total):
            # Progress every 10%
            pct = (idx * 100) // total
            if pct >= last_pct + 10:
                print(f"    {pct}% ({idx}/{total})")
                last_pct = pct
            
            keyframe_id = keyframes[idx]
            slash_pos = keyframe_id.index('/')
            video_id_full = keyframe_id[:slash_pos]
            video_id = video_id_full[:-4] if video_id_full.endswith('.mp4') else video_id_full
            frame_file = keyframe_id[slash_pos+1:]
            
            # Parse frame index (remove .png or .jpg)
            dot_pos = frame_file.rfind('.')
            frame_idx = int(frame_file[:dot_pos])
            
            fps = video_fps.get(video_id_full, 24.0)
            time_sec = (frame_idx - 1) / fps
            
            # Check if any smart home action is active at this time
            has_smart_home = False
            for cls_idx, start, end in video_actions.get(video_id, []):
                if start <= time_sec <= end and cls_idx in action_indices:
                    has_smart_home = True
                    break
            
            if has_smart_home:
                positives.append(idx)
            else:
                negatives.append(idx)
        
        # Save to cache
        try:
            np.savez(cache_file, positives=np.array(positives), negatives=np.array(negatives))
            print(f"  Cached filtered indices to {cache_file}")
        except Exception as e:
            print(f"  Warning: Could not cache filtered indices: {e}")
        
        return positives, negatives
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # Get base dataset item
        base_idx = self.filtered_indices[idx]
        is_negative = base_idx in self.negative_set
        info, video_clip, target = self.base_dataset[base_idx]
        
        # Remap labels from 219 (36+157+26) to num_classes
        if target['labels'].shape[0] > 0:
            old_labels = target['labels']  # [N, 219]
            new_labels = torch.zeros(old_labels.shape[0], self.num_classes, 
                                    dtype=old_labels.dtype)
            
            # Copy objects (0:36) -> (0:36) - unchanged
            new_labels[:, :self.num_objects] = old_labels[:, :self.num_objects]
            
            if not is_negative:
                # Remap actions (36:193) -> (36:num_smart_home_actions)
                # For merged actions (multiple old indices -> same new index),
                # use logical OR so ANY source action being active makes the merged action active
                for old_idx, new_idx in self.old_to_new.items():
                    old_pos = self.num_objects + old_idx
                    new_pos = self.num_objects + new_idx
                    # Use maximum to OR the values (handles merged actions correctly)
                    new_labels[:, new_pos] = torch.maximum(new_labels[:, new_pos], old_labels[:, old_pos])
            # else: actions stay all zeros for negative frames
            
            # Copy relations (193:219) -> shifted position - unchanged but shifted
            old_rel_start = self.num_objects + 157  # 36 + 157 = 193
            new_rel_start = self.num_objects + self.num_smart_home_actions
            new_labels[:, new_rel_start:] = old_labels[:, old_rel_start:]
            
            target['labels'] = new_labels
        else:
            target['labels'] = torch.zeros((0, self.num_classes), dtype=torch.float32)
        
        return info, video_clip, target


def build_smart_home_dataset(cfg, args, is_train=False):
    """Build smart home dataset."""
    from dataset.transforms import TrainTransform, ValTransform
    
    if is_train:
        transform = TrainTransform(args.img_size)
    else:
        transform = ValTransform(args.img_size)
    
    return SmartHomeDataset(
        cfg=cfg,
        data_root=args.data_root,
        is_train=is_train,
        img_size=args.img_size,
        transform=transform,
        len_clip=args.len_clip,
        sampling_rate=args.sampling_rate
    )
