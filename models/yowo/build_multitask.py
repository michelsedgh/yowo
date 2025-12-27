"""
Build Multi-Task YOWO for Action Genome + Charades

This module provides the factory function to build the multi-task YOWO model
with three heads (Object, Action, Relation).
"""

import torch
from .yowo_multitask import YOWOMultiTask
from .loss_multitask import build_multitask_criterion


def build_yowo_multitask(args,
                          d_cfg,
                          m_cfg, 
                          device, 
                          num_objects=36,
                          num_actions=157,
                          num_relations=26,
                          trainable=False,
                          resume=None):
    """
    Build YOWO Multi-Task model for Action Genome + Charades.
    
    Args:
        args: Command line arguments
        d_cfg: Dataset config
        m_cfg: Model config
        device: torch device
        num_objects: Number of object classes (default 36)
        num_actions: Number of action classes (default 157)
        num_relations: Number of relation classes (default 26)
        trainable: Whether model is for training
        resume: Path to checkpoint for resuming
        
    Returns:
        model: YOWOMultiTask model
        criterion: MultiTaskCriterion (if trainable)
    """
    print('==============================')
    print('Build {} (Multi-Task) ...'.format(args.version.upper()))
    print(f'  Objects: {num_objects}')
    print(f'  Actions: {num_actions}')
    print(f'  Relations: {num_relations}')

    # Build YOWO Multi-Task
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
    )

    if trainable:
        # Freeze backbone if requested
        if args.freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if args.freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # Resume from checkpoint
        if resume is not None:
            print('Resuming from: ', resume)
            checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        # Build criterion
        criterion = build_multitask_criterion(
            args, 
            d_cfg['train_size'], 
            num_objects, 
            num_actions, 
            num_relations
        )
    else:
        criterion = None
                        
    return model, criterion


