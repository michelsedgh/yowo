#!/usr/bin/env python3
"""
Final Verification: Using CharadesAGDataset to visualize.
This proves the loader is correct before training.
"""
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from dataset.charades_ag import CharadesAGDataset
from dataset.transforms import BaseTransform

def final_verify():
    # Mock config
    cfg = {
        'train_size': 224,
        'sampling_rate': 1
    }
    
    # Use transform to convert images to tensors
    transform = BaseTransform(img_size=224)
    
    # Init Dataset
    dataset = CharadesAGDataset(
        cfg=cfg,
        data_root='/home/michel/yowo/data/ActionGenome',
        is_train=True,
        img_size=224,
        transform=transform,
        len_clip=16,
        sampling_rate=1
    )
    
    output_dir = Path('/home/michel/yowo/vis_final_verify')
    output_dir.mkdir(exist_ok=True)
    
    # Pick a few indices
    # We want frames we know have interactions
    test_indices = [0, 100, 500, 1000]
    
    for idx in test_indices:
        [video_id, frame_idx], video_clip, target = dataset[idx]
        
        # Last frame is the keyframe - video_clip is [3, T, H, W]
        keyframe = video_clip[:, -1, :, :]  # [3, H, W]
        img_np = keyframe.permute(1, 2, 0).numpy()  # [H, W, 3]
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        
        boxes = target['boxes'].numpy()  # normalized coordinates
        labels = target['labels'].numpy()
        
        vis = img.copy()
        
        for i, box in enumerate(boxes):
            # Convert normalized coords to pixels
            x1, y1, x2, y2 = (box * 224).astype(int)
            label = labels[i]
            
            # Find active classes
            active_indices = np.where(label > 0.5)[0]
            
            # Determine if human or object
            is_human = active_indices[0] == 0 # Index 0 is person
            
            color = (0, 255, 0) if is_human else (255, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Class name
            if is_human:
                class_name = "PERSON"
            else:
                obj_idx = active_indices[0]
                class_name = dataset.ag_objects[obj_idx]
            
            # Relationships
            rel_indices = active_indices[active_indices >= (dataset.num_objects + dataset.num_actions)]
            rels = [dataset.ag_relations[r - (dataset.num_objects + dataset.num_actions)] for r in rel_indices]
            
            # Actions (only for person)
            act_indices = active_indices[(active_indices >= dataset.num_objects) & (active_indices < (dataset.num_objects + dataset.num_actions))]
            acts = [dataset.charades_actions[a - dataset.num_objects] for a in act_indices]
            
            label_text = f"{class_name}"
            if acts: label_text += f" | ACT: {','.join(acts[:2])}"
            if rels: label_text += f" | REL: {','.join(rels[:2])}"
            
            cv2.putText(vis, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save
        out_path = output_dir / f"verify_{video_id}_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved verification image to {out_path}")

if __name__ == "__main__":
    final_verify()

