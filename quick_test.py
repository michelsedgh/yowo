#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.getcwd())
from dataset.charades_ag import CharadesAGDataset
from dataset.transforms import BaseTransform

cfg = {'train_size': 224, 'test_size': 224, 'sampling_rate': 1, 'multi_hot': True, 'valid_num_classes': 219}
transform = BaseTransform(img_size=224)
dataset = CharadesAGDataset(cfg=cfg, data_root='/home/michel/yowo/data/ActionGenome', is_train=True, img_size=224, transform=transform, len_clip=16, sampling_rate=1)
print(f"Dataset: {len(dataset)} keyframes, {dataset.num_classes} classes")
[vid, frame_idx], video_clip, target = dataset[0]
print(f"Sample: vid={vid} frame={frame_idx} clip={video_clip.shape} boxes={target['boxes'].shape} labels={target['labels'].shape}")
print("✓ Works!")
