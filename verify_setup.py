import torch
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from config import build_dataset_config, build_model_config
from models import build_model
from utils.misc import CollateFunc
from dataset.charades_ag import CharadesAGDataset
from dataset.transforms import Augmentation

def verify_dataset():
    print("--- Verifying Dataset ---")
    d_cfg = {
        'data_root': 'data/ActionGenome/',
        'train_size': 224,
        'test_size': 224,
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'multi_hot': True,
        'valid_num_classes': 219,  # FIXED: 36 objects + 157 actions + 26 relations = 219
    }
    
    # Check if paths exist
    if not os.path.exists('data/ActionGenome/annotations/person_bbox.pkl'):
        print("Error: person_bbox.pkl not found")
        return
    
    transform = Augmentation(img_size=224)
    
    try:
        dataset = CharadesAGDataset(
            cfg=d_cfg,
            data_root='data/ActionGenome/',
            is_train=True,
            img_size=224,
            transform=transform,
            len_clip=16,
            sampling_rate=1
        )
        print(f"Dataset length: {len(dataset)}")
        
        # Pull one item
        id_info, video_clip, target = dataset[0]
        print(f"Video ID/Frame: {id_info}")
        print(f"Video clip shape: {video_clip.shape}") # Expect [3, 16, 224, 224]
        print(f"Target boxes shape: {target['boxes'].shape}")
        print(f"Target labels shape: {target['labels'].shape}")
        
        # Check if human bit is set
        human_idx = 38
        labels = target['labels']
        for i in range(labels.shape[0]):
            if labels[i, human_idx] > 0.5:
                print(f"Box {i} is Human")
                # Check for actions
                actions = torch.where(labels[i, 39:39+157] > 0.5)[0]
                if len(actions) > 0:
                    print(f"  Actions: {actions.tolist()}")
                # Check for relations
                rels = torch.where(labels[i, 196:] > 0.5)[0]
                if len(rels) > 0:
                    print(f"  Relations: {rels.tolist()}")
            else:
                # Check which object it is
                obj_idx = torch.where(labels[i, :38] > 0.5)[0]
                if len(obj_idx) > 0:
                    print(f"Box {i} is Object {obj_idx.tolist()}")
                    
    except Exception as e:
        print(f"Dataset verification failed: {e}")
        import traceback
        traceback.print_exc()

def verify_model():
    print("\n--- Verifying Model ---")
    class Args:
        version = 'yowo_v2_medium_yolo11m'
        conf_thresh = 0.1
        nms_thresh = 0.5
        topk = 40
        freeze_backbone_2d = False
        freeze_backbone_3d = False
        # Loss weights
        loss_conf_weight = 1.0
        loss_cls_weight = 1.0
        loss_reg_weight = 5.0
        focal_loss = False
        # Matcher
        center_sampling_radius = 2.5
        topk_candicate = 10
        
    args = Args()
    d_cfg = {'multi_hot': True, 'train_size': 224}
    m_cfg = build_model_config(args)
    
    device = torch.device('cpu')
    num_classes = 219  # FIXED: 36 objects + 157 actions + 26 relations
    
    try:
        model, criterion = build_model(
            args, d_cfg, m_cfg, device, num_classes, trainable=True
        )
        print("Model built successfully")
        
        # Dummy forward
        video_clips = torch.randn(2, 3, 16, 224, 224)
        outputs = model(video_clips)
        
        print("Forward pass successful")
        print("Output keys:", outputs.keys())
        # outputs['pred_cls'] is a list of [B, M, C]
        for i, (conf, cls, box) in enumerate(zip(outputs['pred_conf'], outputs['pred_cls'], outputs['pred_box'])):
            print(f"Scale {i}: conf {conf.shape}, cls {cls.shape}, box {box.shape}")
            
        # Verify classes
        if outputs['pred_cls'][0].shape[-1] == 219:
            print("Output class count matches: 219")
        else:
            print(f"Error: Output class count is {outputs['pred_cls'][0].shape[-1]}, expected 219")
            
    except Exception as e:
        print(f"Model verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset()
    verify_model()
