import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import csv
import json

class CharadesAGDataset(Dataset):
    def __init__(self, cfg, data_root, is_train=False, img_size=224, transform=None, len_clip=16, sampling_rate=1):
        self.data_root = data_root
        self.is_train = is_train
        self.img_size = img_size
        self.transform = transform
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.seq_len = self.len_clip * self.sampling_rate
        
        # 1. Load Class Definitions
        # AG Object Classes (36: person + 35 objects)
        self.ag_objects = self._load_simple_list(os.path.join(data_root, 'annotations/object_classes.txt'))
        self.num_objects = len(self.ag_objects) # 36
        
        # Charades Action Classes (157)
        self.charades_actions = self._load_charades_list(os.path.join(data_root, 'annotations/Charades_v1_classes.txt'))
        self.num_actions = len(self.charades_actions) # 157
        
        # AG Relationship Classes (26)
        self.ag_relations = self._load_simple_list(os.path.join(data_root, 'annotations/relationship_classes.txt'))
        self.num_relations = len(self.ag_relations) # 26
        
        self.num_classes = self.num_objects + self.num_actions + self.num_relations # 36 + 157 + 26 = 219
        
        # 2. Load Annotations
        self._load_data()
        
        # 3. Load FPS Mapping
        fps_path = os.path.join(data_root, 'annotations/video_fps.json')
        if os.path.exists(fps_path):
            with open(fps_path, 'r') as f:
                self.video_fps = json.load(f)
        else:
            self.video_fps = {}

    def _load_simple_list(self, path):
        with open(path, 'r') as f:
            return [line.strip().lower() for line in f if line.strip()]

    def _load_charades_list(self, path):
        # Format: "c001 Opening a door"
        lines = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) > 1:
                        lines.append(parts[1].lower())
                    else:
                        lines.append(line.lower())
        return lines

    def _load_data(self):
        # Action Genome Annotations
        with open(os.path.join(self.data_root, 'annotations/person_bbox.pkl'), 'rb') as f:
            self.person_bboxes = pickle.load(f)
        with open(os.path.join(self.data_root, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
            self.object_data = pickle.load(f)
            
        # Charades Actions (95/5 split - only videos with AG keyframe annotations)
        csv_name = 'Charades_v1_train_95.csv' if self.is_train else 'Charades_v1_test_5.csv'
        csv_path = os.path.join(self.data_root, 'annotations', csv_name)
        self.video_actions = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['id']
                actions = []
                if row['actions']:
                    for a in row['actions'].split(';'):
                        parts = a.split(' ')
                        if len(parts) == 3:
                            cls_idx = int(parts[0][1:]) # c092 -> 92
                            start, end = float(parts[1]), float(parts[2])
                            actions.append((cls_idx, start, end))
                self.video_actions[vid] = actions
        
        # Keyframes - filter to only include videos in current split
        split_video_ids = set(self.video_actions.keys())
        self.keyframes = sorted([
            kf for kf in self.person_bboxes.keys() 
            if kf.split('/')[0].replace('.mp4', '') in split_video_ids
        ])
        print(f"Loaded {len(self.keyframes)} keyframes for {'train' if self.is_train else 'val'}")

    def __len__(self):
        return len(self.keyframes)

    def _normalize_rel(self, rel):
        return rel.replace('_', '').lower()

    def __getitem__(self, idx):
        keyframe_id = self.keyframes[idx] # e.g. '001YG.mp4/000089.png'
        video_id_full = keyframe_id.split('/')[0] # '001YG.mp4'
        video_id = video_id_full.replace('.mp4', '')
        frame_file = keyframe_id.split('/')[1] # '000089.png' or '000089.jpg'
        frame_idx = int(frame_file.replace('.png', '').replace('.jpg', ''))
        
        # 1. Load video clip (len_clip frames)
        # We sample frames ending at the keyframe
        video_clip = []
        for i in range(self.len_clip):
            f = frame_idx - (self.len_clip - 1 - i) * self.sampling_rate
            f_clamped = max(1, f)
            
            # Check for JPG first (since that's our new standard), then fallback to PNG
            img_path = os.path.join(self.data_root, 'frames', video_id_full, f"{f_clamped:06d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_root, 'frames', video_id_full, f"{f_clamped:06d}.png")
            
            # If still not found, use the keyframe itself as a fallback
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_root, 'frames', video_id_full, f"{frame_idx:06d}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.data_root, 'frames', video_id_full, f"{frame_idx:06d}.png")
            
            try:
                frame = Image.open(img_path).convert('RGB')
            except:
                frame = Image.new('RGB', (self.img_size, self.img_size))
            video_clip.append(frame)
            
        ow, oh = video_clip[-1].size
        
        # 2. Build Targets
        boxes = []
        labels = []
        
        person_info = self.person_bboxes[keyframe_id]
        obj_info_list = self.object_data.get(keyframe_id, [])
        
        # Scaling factors
        ann_w, ann_h = person_info['bbox_size']
        sx, sy = ow / ann_w, oh / ann_h
        
        # A. Add Person Box
        if len(person_info['bbox']) > 0:
            # Format is [x1, y1, x2, y2]
            p_bbox = person_info['bbox'][0].copy()
            p_bbox[[0, 2]] *= sx
            p_bbox[[1, 3]] *= sy
            boxes.append(p_bbox)
            
            # Label
            label = np.zeros(self.num_classes, dtype=np.float32)
            # 1. Object: person (Index 0 in ag_objects)
            label[0] = 1.0
            
            # 2. Actions: Charades
            fps = self.video_fps.get(video_id_full, 24.0)
            time_sec = (frame_idx - 1) / fps
            for cls_idx, start, end in self.video_actions.get(video_id, []):
                if start <= time_sec <= end:
                    # Indices 36-192 for actions
                    label[self.num_objects + cls_idx] = 1.0
            
            # 3. Relationships: Union of all interactions
            for obj in obj_info_list:
                for r_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
                    rel_list = obj.get(r_type, [])
                    if rel_list is None:
                        rel_list = []
                    for r in rel_list:
                        r_norm = self._normalize_rel(r)
                        if r_norm in self.ag_relations:
                            r_idx = self.ag_relations.index(r_norm)
                            # Indices 193-218 for relations
                            label[self.num_objects + self.num_actions + r_idx] = 1.0
            labels.append(label)

        # B. Add Object Boxes
        for obj in obj_info_list:
            if not obj.get('visible'): continue
            
            # Format is [x, y, w, h] -> convert to [x1, y1, x2, y2]
            ox, oy, ow_obj, oh_obj = obj['bbox']
            o_bbox = np.array([ox * sx, oy * sy, (ox + ow_obj) * sx, (oy + oh_obj) * sy])
            boxes.append(o_bbox)
            
            # Label
            label = np.zeros(self.num_classes, dtype=np.float32)
            # 1. Object class
            obj_name = obj['class'].lower()
            if obj_name in self.ag_objects:
                label[self.ag_objects.index(obj_name)] = 1.0
            
            # 2. Relationships specific to this object
            for r_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
                rel_list = obj.get(r_type, [])
                if rel_list is None:
                    rel_list = []
                for r in rel_list:
                    r_norm = self._normalize_rel(r)
                    if r_norm in self.ag_relations:
                        r_idx = self.ag_relations.index(r_norm)
                        label[self.num_objects + self.num_actions + r_idx] = 1.0
            labels.append(label)

        # 3. Finalize - Concatenate boxes and labels for transform
        if len(boxes) > 0:
            boxes = np.array(boxes).reshape(-1, 4)
            labels = np.array(labels).reshape(-1, self.num_classes)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0, self.num_classes), dtype=np.float32)
        
        # target: [N, 4 + C] for transform
        target = np.concatenate([boxes, labels], axis=-1)
        
        # transform
        if self.transform is not None:
            video_clip, target = self.transform(video_clip, target)
        
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)
        
        # reformat target to dict after transform
        if target.shape[0] > 0:
            target = {
                'boxes': target[:, :4].float(),
                'labels': target[:, 4:].float(),  # multi-hot labels are float for BCE loss
                'orig_size': [oh, ow]
            }
        else:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0, self.num_classes), dtype=torch.float32),
                'orig_size': [oh, ow]
            }
            
        return [video_id, frame_idx], video_clip, target
