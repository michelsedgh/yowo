"""
Builder for YOWOMultiTask model.
"""

import torch
from .yowo_multitask import YOWOMultiTask
from .loss_multitask import build_multitask_criterion


def build_yowo_multitask(args, d_cfg, m_cfg, device, num_classes=219,
                          num_objects=36, num_actions=157, num_relations=26,
                          trainable=False, resume=None, end2end=False):
    """
    Build YOWOMultiTask model and criterion.
    
    Args:
        end2end: Enable dual-head NMS-free mode (default: False)
    """
    print("="*30)
    print("Building YOWOMultiTask")
    print(f"  Objects: {num_objects}")
    print(f"  Actions: {num_actions}")
    print(f"  Relations: {num_relations}")
    print(f"  Total classes: {num_classes}")
    print(f"  End-to-End NMS-Free: {end2end}")
    print("="*30)
    
    model = YOWOMultiTask(
        cfg=m_cfg,
        device=device,
        num_objects=num_objects,
        num_actions=num_actions,
        num_relations=num_relations,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.topk,
        trainable=trainable,
        end2end=end2end  # NEW: Enable dual-head NMS-free mode
    )
    
    # Load checkpoint if provided
    if resume is not None:
        print(f"Loading checkpoint from {resume}")
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    
    # Build criterion
    if trainable:
        # Check for smart_home class weights
        action_class_weights = None
        if hasattr(args, 'smart_home_config') and args.smart_home_config is not None:
            action_class_weights = args.smart_home_config.get('action_class_weights', None)
            if action_class_weights is not None:
                print(f"  Using class weights: [{min(action_class_weights):.2f}, {max(action_class_weights):.2f}]")
        
        criterion = build_multitask_criterion(
            args=args,
            img_size=d_cfg['train_size'],
            num_classes=num_classes,
            num_objects=num_objects,
            num_actions=num_actions,
            num_relations=num_relations,
            action_class_weights=action_class_weights
        )
    else:
        criterion = None
    
    return model, criterion

