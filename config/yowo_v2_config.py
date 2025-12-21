# Model configuration


yowo_v2_config = {
    'yowo_v2_nano': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_nano',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '1.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': True,
    },

    'yowo_v2_tiny': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_tiny',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    'yowo_v2_medium': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_large',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    'yowo_v2_medium_yolo11m': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo11m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    'yowo_v2_large': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_large',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'resnext101',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    # =========================================================================
    # X3D Backbone Configurations
    # X3D is more accurate than ShuffleNet for temporal modeling while still
    # being efficient. All X3D variants output 192 channels.
    # =========================================================================
    
    'yowo_v2_x3d_s': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_large',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - X3D-S: Small, good balance of speed and accuracy
        'backbone_3d': 'x3d_s',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    'yowo_v2_x3d_m': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_large',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - X3D-M: Medium, higher accuracy
        'backbone_3d': 'x3d_m',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    # X3D with YOLO11m - the config you want for Charades + Action Genome
    'yowo_v2_x3d_s_yolo11m': {
        # backbone
        ## 2D - YOLO11m for better 2D features
        'backbone_2d': 'yolo11m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - X3D-S for efficient temporal modeling
        'backbone_3d': 'x3d_s',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    'yowo_v2_x3d_m_yolo11m': {
        # backbone
        ## 2D - YOLO11m for better 2D features
        'backbone_2d': 'yolo11m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - X3D-M for higher temporal accuracy
        'backbone_3d': 'x3d_m',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

}