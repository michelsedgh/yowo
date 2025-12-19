# import 2D backbone
from .yolo_free.yolo_free import build_yolo_free
from .yolo_11 import build_yolo_11


def build_2d_cnn(cfg, pretrained=False):
    print('==============================')
    print('2D Backbone: {}'.format(cfg['backbone_2d'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone_2d'] in ['yolo_free_nano', 'yolo_free_tiny', \
                              'yolo_free_large', 'yolo_free_huge']:
        model, feat_dims = build_yolo_free(cfg['backbone_2d'], pretrained)

    elif 'yolo11' in cfg['backbone_2d']:
        model, feat_dims = build_yolo_11(cfg['backbone_2d'] + '.pt', pretrained)

    else:
        print('Unknown 2D Backbone ...')
        exit()

    return model, feat_dims
