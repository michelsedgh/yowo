import torch
from torch import optim


def build_optimizer(cfg, model, base_lr=0.0, resume=None):
    """
    Build optimizer and optionally load state from checkpoint.
    
    Args:
        cfg: Dataset config with optimizer settings
        model: Model to optimize
        base_lr: Base learning rate
        resume: Path to checkpoint file (optional)
        
    Returns:
        optimizer: Configured optimizer
        start_epoch: Epoch to start training from (0 for fresh, N for resume)
    """
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    # Separate parameters into groups with different learning rates
    # Backbones (pretrained) get a smaller LR, Heads/Context get full base_lr
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    param_groups = [
        {'params': head_params, 'lr': base_lr},
        {'params': backbone_params, 'lr': base_lr * 0.1}
    ]
    
    print(f'--Backbone parameters: {len(backbone_params)}')
    print(f'--Head/Context parameters: {len(head_params)}')
    print(f'--Backbone LR: {base_lr * 0.1:.6f}')
    print(f'--Head/Context LR: {base_lr:.6f}')

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            param_groups, 
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            param_groups, 
            weight_decay=cfg['weight_decay'])
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            param_groups, 
            weight_decay=cfg['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
          
    start_epoch = 0
    
    if resume is not None:
        print('Loading checkpoint for resume: ', resume)
        # PyTorch 2.6+ requires weights_only=False to load optimizer state
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        
        # Load optimizer state if available
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print('✅ Loaded optimizer state (momentum buffers preserved)')
            except Exception as e:
                print(f'⚠️ Could not load optimizer state: {e}')
                print('   Starting with fresh optimizer (this is normal if model architecture changed)')
        else:
            print('⚠️ No optimizer state in checkpoint, starting fresh')
        
        # Get epoch to resume from
        # Saved epoch is the completed epoch (0-indexed), so we start from epoch+1
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f'✅ Resuming from epoch {start_epoch}')
        else:
            print('⚠️ No epoch info in checkpoint, starting from epoch 0')

    return optimizer, start_epoch
