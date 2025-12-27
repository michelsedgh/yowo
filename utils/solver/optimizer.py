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

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=base_lr,
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=base_lr,
            weight_decay=cfg['weight_decay'])  # Fixed: was 'eight_decay'
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=base_lr,
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
