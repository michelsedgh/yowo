from .dataset_config import dataset_config
from .yowo_v2_config import yowo_v2_config


def build_model_config(args):
    print('==============================')
    print('Model Config: {} '.format(args.version.upper()))
    
    if 'yowo_v2_' in args.version:
        m_cfg = yowo_v2_config[args.version]

    return m_cfg


def build_dataset_config(args):
    print('==============================')
    print('Dataset Config: {} '.format(args.dataset.upper()))
    
    d_cfg = dataset_config[args.dataset].copy()  # Make a copy to avoid modifying the original
    
    # Override image size if specified via command line
    if hasattr(args, 'img_size') and args.img_size is not None:
        d_cfg['train_size'] = args.img_size
        d_cfg['test_size'] = args.img_size
        print(f'📐 Image size override: {args.img_size}x{args.img_size}')

    return d_cfg
