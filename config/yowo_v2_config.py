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
    # X3D Backbone Configurations (4-POOL TEMPORAL ARCHITECTURE)
    # X3D with 4-pool: Pools 16 frames into 4 temporal segments (early,
    # early_mid, late_mid, late) and concatenates to 768 channels (4 × 192).
    # This preserves temporal order for transitional actions.
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

    # =========================================================================
    # MULTI-TASK Configurations for Action Genome + Charades
    # These use the three-head architecture (Object/Action/Relation)
    # =========================================================================
    
    # Multi-Task with X3D-S + YOLO11m (RECOMMENDED for Action Genome + Charades)
    'yowo_v2_x3d_s_yolo11m_multitask': {
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
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
    },

    # Multi-Task with X3D-M + YOLO11m (Higher accuracy, slower)
    'yowo_v2_x3d_m_yolo11m_multitask': {
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
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
    },

    # =========================================================================
    # CLEAN Multi-Task Configurations (Minimal, Proven Architecture)
    # Use these for reliable training with YOLO11 + ResNeXt/ShuffleNet
    # =========================================================================
    
    # YOLO11m + ResNeXt101 - High capacity, good for Action Genome
    'yowo_v2_resnext_yolo11m_multitask': {
        # backbone
        ## 2D - YOLO11m
        'backbone_2d': 'yolo11m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ResNeXt101 (resize to 224 for fine-grained action tracking)
        'backbone_3d': 'resnext101',
        'backbone_3d_size': 224,  # 224px keeps small objects visible
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head - increased to 384 for better feature capacity
        'head_dim': 384,
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
        'clean': True,  # Use clean implementation
    },

    # YOLO11m + ShuffleNetV2 - Faster, lighter
    'yowo_v2_shufflenet_yolo11m_multitask': {
        # backbone
        ## 2D - YOLO11m
        'backbone_2d': 'yolo11m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ShuffleNetV2 (faster)
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
        'clean': True,
    },

    # =========================================================================
    # YOLO26 Configurations - Latest YOLO with NMS-free native design
    # =========================================================================
    
    # YOLO26L + ResNeXt101 - High capacity, state-of-the-art 2D features
    'yowo_v2_resnext_yolo26l_multitask': {
        # backbone
        ## 2D - YOLO26L (latest, improved small object detection)
        'backbone_2d': 'yolo26l',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ResNeXt101 (resize to 224 for fine-grained action tracking)
        'backbone_3d': 'resnext101',
        'backbone_3d_size': 224,  # 224px keeps small objects visible
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head - increased to 384 for better feature capacity
        'head_dim': 384,
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
    },

    # YOLO26M + ResNeXt101 - Medium variant, faster
    'yowo_v2_resnext_yolo26m_multitask': {
        # backbone
        ## 2D - YOLO26M (balanced)
        'backbone_2d': 'yolo26m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ResNeXt101 (resize to 224 for fine-grained action tracking)
        'backbone_3d': 'resnext101',
        'backbone_3d_size': 224,  # 224px keeps small objects visible (112px too small for phones/cups)
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head - increased from 256 to 384 for better feature capacity
        'head_dim': 384,
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
        # Motion Enhanced: RGB + diff + flow_x + flow_y (6 channels)
        'use_motion_enhanced': True,
    },

    # YOLO26L + ShuffleNetV2 - Fast inference with strong 2D features
    'yowo_v2_shufflenet_yolo26l_multitask': {
        # backbone
        ## 2D - YOLO26L
        'backbone_2d': 'yolo26l',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ShuffleNetV2 (faster)
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
    },

    # =========================================================================
    # OPTIMIZED FOR EDGE DEPLOYMENT (Orin Nano)
    # ShuffleNetV2 2.0x @ 480px - Best balance of speed and accuracy
    # =========================================================================
    
    # YOLO26M + ShuffleNetV2 2.0x - RECOMMENDED for Orin Nano @ 480px+
    # Expected: 14-18 FPS on Orin Nano with TensorRT INT8
    'yowo_v2_shufflenet_yolo26m_multitask': {
        # backbone
        ## 2D - YOLO26M (medium - balanced speed/accuracy)
        'backbone_2d': 'yolo26m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ShuffleNetV2 2.0x (66.4% K400, 976ch output, strong temporal)
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head - 256 to match ResNeXt capacity (CRITICAL for high-res)
        'head_dim': 256,
        'head_norm': 'GN',  # GroupNorm - safe for O2M/O2O dual-forward
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
        # Motion Enhanced: RGB + diff + flow_x + flow_y (6 channels)
        'use_motion_enhanced': True,
    },

    # YOLO26S + ShuffleNetV2 2.0x - Even faster, smaller 2D backbone
    'yowo_v2_shufflenet_yolo26s_multitask': {
        # backbone
        ## 2D - YOLO26S (small - fastest)
        'backbone_2d': 'yolo26s',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ShuffleNetV2 2.0x
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 256,
        'head_norm': 'GN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
        # Motion Enhanced: RGB + diff + flow_x + flow_y (6 channels)
        # Best for motion-based actions like walking, running
        'use_motion_enhanced': True,
    },

    # YOLO26S + ResNeXt101 - Faster 2D, powerful 3D temporal backbone
    # Good balance: fast spatial features + strong temporal features
    'yowo_v2_resnext_yolo26s_multitask': {
        # backbone
        ## 2D - YOLO26S (small - fast spatial features)
        'backbone_2d': 'yolo26s',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ResNeXt101 (resize to 224 for fine-grained action tracking)
        'backbone_3d': 'resnext101',
        'backbone_3d_size': 224,  # 224px keeps small objects visible
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head - increased to 384 for better feature capacity
        'head_dim': 384,
        'head_norm': 'GN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
        # Motion Enhanced: RGB + diff + flow_x + flow_y (6 channels)
        'use_motion_enhanced': True,
    },

    # =========================================================================
    # SHUFFLENET 1.0x VARIANTS (Lighter 3D backbone, faster inference)
    # Use these if 2.0x is too slow at high resolutions
    # =========================================================================
    
    # YOLO26M + ShuffleNetV2 1.0x - Faster 3D, good for 640px
    'yowo_v2_shufflenet1x_yolo26m_multitask': {
        # backbone
        ## 2D - YOLO26M
        'backbone_2d': 'yolo26m',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D - ShuffleNetV2 1.0x (464ch output, lighter than 2.0x)
        'backbone_3d': 'shufflenetv2',
        'model_size': '1.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 256,
        'head_norm': 'GN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # multi-task flag
        'multi_task': True,
    },

}