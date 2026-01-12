#!/usr/bin/env python3
"""
YOWO Multi-Task ONNX Export Script

Exports the YOWO multi-task model (X3D-M + YOLO11m) to ONNX format for TensorRT optimization.

Architecture:
- Input: [B=1, C=3, T=16, H=224, W=224] video clip
- Output: [N, 224] where N is the number of detections (varies)
  - Output format per detection: [conf(1), obj(36), act(157), rel(26), box(4)]
    - conf: Object confidence score (sigmoid applied)
    - obj: Object class probabilities (softmax applied)
    - act: Action probabilities (sigmoid applied)
    - rel: Relation probabilities (sigmoid applied)
    - box: Bounding box [x1, y1, x2, y2] in pixels

Key design choices:
1. Apply sigmoid/softmax in the model for cleaner post-processing
2. Flatten multi-scale outputs into single [N, 224] tensor
3. Use fixed input shape for TensorRT optimization

Usage:
    python export_onnx.py --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_14.pth
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_dataset_config, build_model_config
from models.yowo.yowo_multitask import YOWOMultiTask


class YOWOMultiTaskONNX(nn.Module):
    """
    ONNX-exportable wrapper for YOWOMultiTask.
    
    Key differences from training model:
    1. Applies sigmoid/softmax activations inline (TRT can optimize these)
    2. Flattens all scale outputs into single tensor
    3. Includes box decoding and normalization
    4. Fixed batch size = 1 for optimal TensorRT optimization
    """
    
    def __init__(self, model: YOWOMultiTask, img_size: int = 224):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.stride = model.stride
        self.num_objects = model.num_objects
        self.num_actions = model.num_actions
        self.num_relations = model.num_relations
        
        # Register ImageNet pixel mean/std as buffers for X3D normalization
        # X3D uses ImageNet normalization: mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]
        self.register_buffer('pixel_mean', torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1))
    
    def generate_anchors(self, fmp_size, stride, device):
        """Generate anchor points for a given feature map size."""
        fmp_h, fmp_w = fmp_size
        anchor_y = torch.arange(fmp_h, device=device, dtype=torch.float32)
        anchor_x = torch.arange(fmp_w, device=device, dtype=torch.float32)
        anchor_grid_y, anchor_grid_x = torch.meshgrid(anchor_y, anchor_x, indexing='ij')
        anchor_xy = torch.stack([anchor_grid_x, anchor_grid_y], dim=-1).view(-1, 2) + 0.5
        anchor_xy *= stride
        return anchor_xy
    
    def decode_boxes(self, anchors, pred_reg, stride):
        """Decode box predictions from anchor offsets."""
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        pred_box_wh = pred_reg[..., 2:].exp() * stride
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        return pred_box
    
    def forward(self, video_clips):
        """
        Forward pass for ONNX export.
        
        Args:
            video_clips: [1, 3, T, H, W] - single clip batch
            
        Returns:
            output: [N, 224] tensor with format:
                [conf(1), obj(36), act(157), rel(26), box(4)]
                All activations (sigmoid/softmax) applied.
        """
        B = video_clips.shape[0]  # Should be 1
        device = video_clips.device
        
        # Extract key frame for 2D backbone
        key_frame = video_clips[:, :, -1, :, :]
        
        # Normalize video for 3D backbone (X3D uses ImageNet norm)
        video_clips_3d = (video_clips - self.pixel_mean) / self.pixel_std
        
        # 3D backbone
        feat_3d = self.model.backbone_3d(video_clips_3d)
        
        # 2D backbone
        cls_feats, reg_feats = self.model.backbone_2d(key_frame)
        
        # Collect all predictions
        all_outputs = []
        
        for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
            # Upsample 3D features to match 2D feature map
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            # Channel encoders
            cls_feat = self.model.cls_channel_encoders[level](cls_feat, feat_3d_up)
            reg_feat = self.model.reg_channel_encoders[level](reg_feat, feat_3d_up)
            
            # Heads
            cls_feat, reg_feat = self.model.heads[level](cls_feat, reg_feat)
            
            # ============ CASCADED PREDICTIONS ============
            # Step 1: Confidence and Object prediction
            conf_pred = self.model.conf_preds[level](reg_feat)
            obj_pred = self.model.obj_preds[level](cls_feat)
            
            # Step 2: Relation prediction (with object context)
            rel_feat = self.model.obj_context[level](cls_feat, obj_pred)
            rel_pred = self.model.rel_preds[level](rel_feat)
            
            # Step 3: Action prediction (with object+relation context)
            act_feat = self.model.rel_context[level](rel_feat, obj_pred, rel_pred)
            act_pred = self.model.act_preds[level](act_feat)
            
            # Box prediction
            reg_pred = self.model.reg_preds[level](reg_feat)
            
            # Get feature map dimensions
            fmp_h, fmp_w = conf_pred.shape[-2:]
            
            # Generate anchors for this level
            anchors = self.generate_anchors((fmp_h, fmp_w), self.stride[level], device)
            
            # Reshape predictions: [B, C, H, W] -> [B, H*W, C]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_objects)
            act_pred = act_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_actions)
            rel_pred = rel_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_relations)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            
            # Decode boxes
            box_pred = self.decode_boxes(anchors.unsqueeze(0), reg_pred, self.stride[level])
            
            # Apply activations for clean output
            conf_pred = torch.sigmoid(conf_pred)          # [B, M, 1]
            obj_pred = F.softmax(obj_pred, dim=-1)        # [B, M, 36]
            act_pred = torch.sigmoid(act_pred)            # [B, M, 157]
            rel_pred = torch.sigmoid(rel_pred)            # [B, M, 26]
            
            # Concatenate: [conf, obj, act, rel, box] = [1, 36, 157, 26, 4] = 224
            level_output = torch.cat([conf_pred, obj_pred, act_pred, rel_pred, box_pred], dim=-1)
            all_outputs.append(level_output)
        
        # Concatenate all levels
        output = torch.cat(all_outputs, dim=1)  # [B, N_total, 224]
        
        # Squeeze batch dimension for single-image inference
        output = output.squeeze(0)  # [N, 224]
        
        return output


def export_onnx(args):
    """Export model to ONNX format."""
    print("=" * 60)
    print("YOWO Multi-Task ONNX Export")
    print("=" * 60)
    
    # Force CPU for ONNX export (avoid GPU memory issues on Orin Nano)
    # TensorRT will optimize for GPU later
    device = torch.device('cpu')
    print(f"Device: {device}")
    print("  Note: Using CPU for export to avoid memory issues. TensorRT will run on GPU.")
    
    # Create a minimal args namespace for config building
    class Args:
        def __init__(self):
            self.version = 'yowo_v2_x3d_m_yolo11m_multitask'
            self.dataset = 'charades_ag'
            self.img_size = 224
            self.len_clip = 16
            self.conf_thresh = 0.1
            self.nms_thresh = 0.5
            self.topk = 50
            self.freeze_backbone_2d = False
            self.freeze_backbone_3d = False
    
    model_args = Args()
    
    # Build configs
    from config import build_dataset_config, build_model_config
    d_cfg = build_dataset_config(model_args)
    m_cfg = build_model_config(model_args)
    
    # Build model
    print(f"\nBuilding model: {model_args.version}")
    print(f"  Objects: {d_cfg['num_objects']}")
    print(f"  Actions: {d_cfg['num_actions']}")
    print(f"  Relations: {d_cfg['num_relations']}")
    
    model = YOWOMultiTask(
        cfg=m_cfg,
        device=device,
        num_objects=d_cfg['num_objects'],
        num_actions=d_cfg['num_actions'],
        num_relations=d_cfg['num_relations'],
        conf_thresh=0.1,
        nms_thresh=0.5,
        topk=50,
        trainable=False
    )
    
    # Load weights
    print(f"\nLoading weights: {args.weight}")
    checkpoint = torch.load(args.weight, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Handle potential key mismatches
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(f"  Shape mismatch: {k}")
        else:
            print(f"  Skipping: {k}")
    
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    
    print(f"  Loaded {len(filtered_state)}/{len(model_state)} parameters")
    
    # Create ONNX wrapper
    onnx_model = YOWOMultiTaskONNX(model, img_size=args.img_size)
    onnx_model.to(device)
    onnx_model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, args.len_clip, args.img_size, args.img_size, device=device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = onnx_model(dummy_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [N, 224] where N = sum of grid cells across scales")
    
    # Calculate expected N
    # For 224x224 input with strides [8, 16, 32]:
    # Level 0: 28x28 = 784
    # Level 1: 14x14 = 196  
    # Level 2: 7x7 = 49
    # Total: 1029
    expected_n = (args.img_size // 8) ** 2 + (args.img_size // 16) ** 2 + (args.img_size // 32) ** 2
    print(f"  Expected N: {expected_n}")
    
    # Export to ONNX
    output_path = args.output
    print(f"\nExporting to ONNX: {output_path}")
    
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # TensorRT 10.x supports opset 17 well
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Fixed shapes for best TensorRT optimization
    )
    
    print(f"  ONNX model saved!")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model_check = onnx.load(output_path)
    onnx.checker.check_model(onnx_model_check)
    print("  ONNX model verification passed!")
    
    # Print model info
    print(f"\n{'=' * 60}")
    print("ONNX Export Complete!")
    print(f"{'=' * 60}")
    print(f"  File: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"  Input: [1, 3, {args.len_clip}, {args.img_size}, {args.img_size}]")
    print(f"  Output: [{expected_n}, 224]")
    print()
    print("Output format per row:")
    print("  [0]: conf (sigmoid)")
    print("  [1:37]: object probs (softmax, 36 classes)")
    print("  [37:194]: action probs (sigmoid, 157 classes)")
    print("  [194:220]: relation probs (sigmoid, 26 classes)")
    print("  [220:224]: box [x1, y1, x2, y2] (pixel coords)")
    print()
    print("Next step: Build TensorRT engine:")
    print(f"  trtexec --onnx={output_path} --saveEngine=yowo_multitask_fp16.engine --fp16 --workspace=4096")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='YOWO Multi-Task ONNX Export')
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to trained weight file (.pth)')
    parser.add_argument('--output', type=str, default='yowo_multitask.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--len_clip', type=int, default=16,
                        help='Clip length (number of frames)')
    
    args = parser.parse_args()
    
    export_onnx(args)


if __name__ == '__main__':
    main()
