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
                 transform=None, len_clip=16, sampling_rate=1):
        
        # Load smart home config
        config_path = os.path.join(os.path.dirname(__file__), 
                                   '../config/smart_home_final.json')
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.num_smart_home_actions = self.config['num_actions']
        self.action_indices = set(self.config['action_indices'])
        self.old_to_new = {int(k): v for k, v in self.config['old_to_new'].items()}
        
        # Load base dataset
        self.base_dataset = CharadesAGDataset(
            cfg, data_root, is_train, img_size, transform, len_clip, sampling_rate
        )
        
        # Get dimensions from base dataset
        self.num_objects = self.base_dataset.num_objects  # 36
        self.num_relations = self.base_dataset.num_relations  # 26
        
        # New num_classes: 36 objects + 42 actions + 26 relations = 104
        self.num_classes = self.num_objects + self.num_smart_home_actions + self.num_relations
        
        # Filter keyframes to only those with smart home actions
        self.filtered_indices = self._filter_keyframes()
        
        print(f"SmartHomeDataset: {len(self.filtered_indices)}/{len(self.base_dataset)} "
              f"keyframes ({100*len(self.filtered_indices)/len(self.base_dataset):.1f}%)")
        print(f"  Actions: {self.num_smart_home_actions} (was 157)")
        print(f"  Objects: {self.num_objects}")
        print(f"  Relations: {self.num_relations}")
        print(f"  Total classes: {self.num_classes}")
    
    def _filter_keyframes(self):
        """Find indices of keyframes that have at least one smart home action."""
        filtered = []
        
        for idx in range(len(self.base_dataset)):
            keyframe_id = self.base_dataset.keyframes[idx]
            video_id_full = keyframe_id.split('/')[0]
            video_id = video_id_full.replace('.mp4', '')
            frame_file = keyframe_id.split('/')[1]
            frame_idx = int(frame_file.replace('.png', '').replace('.jpg', ''))
            
            fps = self.base_dataset.video_fps.get(video_id_full, 24.0)
            time_sec = (frame_idx - 1) / fps
            
            # Check if any smart home action is active at this time
            has_smart_home = False
            for cls_idx, start, end in self.base_dataset.video_actions.get(video_id, []):
                if start <= time_sec <= end and cls_idx in self.action_indices:
                    has_smart_home = True
                    break
            
            if has_smart_home:
                filtered.append(idx)
        
        return filtered
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # Get base dataset item
        base_idx = self.filtered_indices[idx]
        info, video_clip, target = self.base_dataset[base_idx]
        
        # Remap labels from 219 (36+157+26) to 104 (36+42+26)
        if target['labels'].shape[0] > 0:
            old_labels = target['labels']  # [N, 219]
            new_labels = torch.zeros(old_labels.shape[0], self.num_classes, 
                                    dtype=old_labels.dtype)
            
            # Copy objects (0:36) -> (0:36) - unchanged
            new_labels[:, :self.num_objects] = old_labels[:, :self.num_objects]
            
            # Remap actions (36:193) -> (36:78)
            # Only copy the smart home actions, remapped to new indices
            for old_idx, new_idx in self.old_to_new.items():
                old_pos = self.num_objects + old_idx
                new_pos = self.num_objects + new_idx
                new_labels[:, new_pos] = old_labels[:, old_pos]
            
            # Copy relations (193:219) -> (78:104) - unchanged but shifted
            old_rel_start = self.num_objects + 157  # 36 + 157 = 193
            new_rel_start = self.num_objects + self.num_smart_home_actions  # 36 + 42 = 78
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
