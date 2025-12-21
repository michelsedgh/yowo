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
    
    Automatically selects multi-task architecture when:
    1. Model version ends with '_multitask', OR
    2. Dataset config has 'multi_task': True
    """
    # Check if multi-task architecture should be used
    use_multitask = (
        '_multitask' in args.version or
        m_cfg.get('multi_task', False) or
        d_cfg.get('multi_task', False)
    )
    
    if use_multitask and 'yowo_v2_' in args.version:
        # Multi-task architecture for Action Genome + Charades
        num_objects = d_cfg.get('num_objects', 36)
        num_actions = d_cfg.get('num_actions', 157)
        num_relations = d_cfg.get('num_relations', 26)
        
        model, criterion = build_yowo_multitask(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_objects=num_objects,
            num_actions=num_actions,
            num_relations=num_relations,
            trainable=trainable,
            resume=resume
        )
    elif 'yowo_v2_' in args.version:
        # Original single-head architecture
        model, criterion = build_yowo(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            resume=resume
        )
    else:
        raise ValueError(f"Unknown model version: {args.version}")

    return model, criterion

