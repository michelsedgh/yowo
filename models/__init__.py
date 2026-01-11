from .yowo.build import build_yowo
from .yowo.build_multitask import build_yowo_multitask


def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    """
    Build action detection model.
    
    For Action Genome (multi_task=True):
    - Uses YOWOMultiTask model with position-aware object context
    - Separate object head, then action/relation with enriched features
    
    For standard datasets (UCF24, AVA):
    - Uses original YOWO model
    """
    # Check if multi-task
    use_multitask = (
        '_multitask' in args.version or
        m_cfg.get('multi_task', False) or
        d_cfg.get('multi_task', False)
    )
    
    if use_multitask:
        # Multi-task model for Action Genome
        num_objects = d_cfg.get('num_objects', 36)
        num_actions = d_cfg.get('num_actions', 157)
        num_relations = d_cfg.get('num_relations', 26)
        total_classes = num_objects + num_actions + num_relations
        
        model, criterion = build_yowo_multitask(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=total_classes,
            num_objects=num_objects,
            num_actions=num_actions,
            num_relations=num_relations,
            trainable=trainable,
            resume=resume
        )
    else:
        # Original YOWO for standard datasets
        model, criterion = build_yowo(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            resume=resume
        )

    return model, criterion
