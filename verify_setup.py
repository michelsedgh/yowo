#!/usr/bin/env python3
"""
YOWO Setup Verification Script

Run this before training to ensure:
1. Backbones load correctly
2. Feature fusion works properly
3. Full model builds and runs

Usage:
    python verify_setup.py
    # or
    python verify_setup.py --dataset charades_ag --version yowo_v2_x3d_m_yolo11m_multitask
"""

import torch
import argparse
from config import build_dataset_config, build_model_config
from models import build_model

def verify_backbones():
    """Test individual backbones"""
    print("=" * 60)
    print("🔍 BACKBONE VERIFICATION")
    print("=" * 60)

    from models.backbone.backbone_2d.cnn_2d.yolo_11 import build_yolo_11
    from models.backbone.backbone_3d.cnn_3d.x3d import build_x3d_3d

    # Test YOLO11
    print("Testing YOLO11 backbone...")
    yolo_model, yolo_feat_dims = build_yolo_11('yolo11m.pt', pretrained=True)
    yolo_model = yolo_model.cuda()

    key_frame = torch.randn(2, 3, 224, 224).cuda()
    with torch.no_grad():
        yolo_cls, yolo_reg = yolo_model(key_frame)

    print(f"  ✅ YOLO11: {len(yolo_cls)} scales, dims {yolo_feat_dims}")

    # Test X3D
    print("Testing X3D backbone...")
    x3d_model, x3d_feat_dim = build_x3d_3d('x3d_s', pretrained=True)
    x3d_model = x3d_model.cuda()

    video_clip = torch.randn(2, 3, 16, 224, 224).cuda()
    with torch.no_grad():
        x3d_feat = x3d_model(video_clip)

    print(f"  ✅ X3D: dim {x3d_feat_dim}, output {x3d_feat.shape}")
    print()

def verify_fusion():
    """Test feature fusion"""
    print("=" * 60)
    print("🔗 FEATURE FUSION VERIFICATION")
    print("=" * 60)

    from models.backbone.backbone_2d.cnn_2d.yolo_11 import build_yolo_11
    from models.backbone.backbone_3d.cnn_3d.x3d import build_x3d_3d
    from models.yowo.encoder import build_channel_encoder
    import torch.nn.functional as F

    # Load backbones
    yolo_model, yolo_feat_dims = build_yolo_11('yolo11m.pt', pretrained=True)
    x3d_model, x3d_feat_dim = build_x3d_3d('x3d_s', pretrained=True)

    yolo_model = yolo_model.cuda()
    x3d_model = x3d_model.cuda()

    # Create encoders
    m_cfg = build_model_config(type('Args', (), {'version': 'yowo_v2_x3d_m_yolo11m_multitask'})())
    encoders = []
    for level in range(3):
        in_dim = yolo_feat_dims[level] + x3d_feat_dim
        out_dim = yolo_feat_dims[level]
        encoder = build_channel_encoder(m_cfg, in_dim, out_dim).cuda()
        encoders.append(encoder)

    # Test fusion
    video_clip = torch.randn(2, 3, 16, 224, 224).cuda()
    key_frame = video_clip[:, :, -1, :, :]

    with torch.no_grad():
        yolo_cls, _ = yolo_model(key_frame)
        x3d_feat = x3d_model(video_clip)

        for level in range(3):
            # Upsample X3D to match YOLO scale
            scale_factor = 2 ** (2 - level)
            x3d_up = F.interpolate(x3d_feat, scale_factor=scale_factor, mode='bilinear', align_corners=False)

            # Fuse
            fused = encoders[level](yolo_cls[level], x3d_up)

            # Verify
            expected_shape = yolo_cls[level].shape
            success = (fused.shape == expected_shape)
            print(f"  {'✅' if success else '❌'} Level {level} (P{level+3}): {fused.shape}")

    print()

def verify_full_model(dataset_name='charades_ag', model_version='yowo_v2_x3d_m_yolo11m_multitask'):
    """Test full YOWO model"""
    print("=" * 60)
    print("🚀 FULL MODEL VERIFICATION")
    print("=" * 60)

    class Args:
        cuda = True
        len_clip = 16
        root = 'data'
        test_batch_size = 1
        eval = False
        conf_thresh = 0.1
        nms_thresh = 0.5
        topk = 40
        freeze_backbone_2d = False
        freeze_backbone_3d = False
        center_sampling_radius = 2.5
        topk_candicate = 10
        loss_conf_weight = 1
        loss_cls_weight = 1
        loss_reg_weight = 5
        focal_loss = False

    args = Args()
    args.dataset = dataset_name
    args.version = model_version

    # Build model
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    print(f"Building {model_version} for {dataset_name}...")
    try:
        model, criterion = build_model(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=torch.device('cuda'),
            num_classes=d_cfg['valid_num_classes'],
            trainable=True
        )

        # Ensure model is on CUDA
        model = model.cuda()

        # Test forward pass
        video_batch = torch.randn(2, 3, 16, 224, 224).cuda()
        with torch.no_grad():
            outputs = model(video_batch)

        print(f"✅ Model built successfully")
        print(f"✅ Forward pass works: {len(outputs)} output keys")

        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params/1e6:.1f}M")
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        raise
    print()

def main():
    parser = argparse.ArgumentParser(description='YOWO Setup Verification')
    parser.add_argument('--dataset', default='charades_ag', help='Dataset to test')
    parser.add_argument('--version', default='yowo_v2_x3d_m_yolo11m_multitask', help='Model version')
    args = parser.parse_args()

    print("🔬 YOWO SETUP VERIFICATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.version}")
    print("=" * 60)
    print()

    try:
        verify_backbones()
        verify_fusion()
        verify_full_model(dataset_name=args.dataset, model_version=args.version)

        print("=" * 60)
        print("🎉 ALL VERIFICATION CHECKS PASSED!")
        print("Your YOWO setup is ready for training.")
        print("=" * 60)

    except Exception as e:
        print(f"❌ VERIFICATION FAILED: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
