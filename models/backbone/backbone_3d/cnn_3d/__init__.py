from .resnet import build_resnet_3d
from .resnext import build_resnext_3d
from .shufflnetv2 import build_shufflenetv2_3d
from .x3d import build_x3d_3d


def build_3d_cnn(cfg, pretrained=False):
    print('==============================')
    print('3D Backbone: {}'.format(cfg['backbone_3d'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in cfg['backbone_3d']:
        model, feat_dims = build_resnet_3d(
            model_name=cfg['backbone_3d'],
            pretrained=pretrained
            )
    elif 'resnext' in cfg['backbone_3d']:
        model, feat_dims = build_resnext_3d(
            model_name=cfg['backbone_3d'],
            pretrained=pretrained
            )
    elif 'shufflenetv2' in cfg['backbone_3d']:
        model, feat_dims = build_shufflenetv2_3d(
            model_size=cfg['model_size'],
            pretrained=pretrained
            )
    elif 'x3d' in cfg['backbone_3d']:
        # X3D backbone - model_name should be 'x3d_xs', 'x3d_s', 'x3d_m', or 'x3d_l'
        model, feat_dims = build_x3d_3d(
            model_name=cfg['backbone_3d'],
            pretrained=pretrained
            )
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dims
