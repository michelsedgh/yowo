#!/usr/bin/env python3
"""
Complete Unified End-to-End deployment builder for YOWO.
Combines:
1) O2M purge from checkpoint (leaving only O2O layers for NMS-free inference)
2) ONNX Generation (fully optimized, flattened structure)
3) INT8 TensorRT Engine Building (with proper precision Fallbacks for bbox logic)

This script replaces both `purge_o2m_for_inference.py`, `export_onnx.py`, and `build_int8_engine.py`.

Usage:
    python deploy_engine.py --checkpoint path/to/yowo_v2_resnext_..._epoch_X.pth
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Models and configs must be accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import build_dataset_config, build_model_config
from models.yowo.yowo_multitask import YOWOMultiTask

# =========================================================================
# ONNX WRAPPER
# =========================================================================
class YOWONMSFreeONNXWrapper(nn.Module):
    def __init__(self, model: YOWOMultiTask, img_size: int = 224, backbone_3d_type: str = 'resnext'):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.stride = model.stride
        self.num_objects = model.num_objects
        self.num_actions = model.num_actions
        self.num_relations = model.num_relations
        self.backbone_3d_type = backbone_3d_type
        
        # Norm checks
        if 'x3d' in backbone_3d_type.lower():
            self.register_buffer('pixel_mean', torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1))
            self.register_buffer('pixel_std', torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1))
            self.needs_normalization = True
        else:
            self.register_buffer('pixel_mean', torch.zeros(1, 3, 1, 1, 1))
            self.register_buffer('pixel_std', torch.ones(1, 3, 1, 1, 1))
            self.needs_normalization = False

    def generate_anchors(self, fmp_size, stride, device):
        fmp_h, fmp_w = fmp_size
        anchor_y = torch.arange(fmp_h, device=device, dtype=torch.float32)
        anchor_x = torch.arange(fmp_w, device=device, dtype=torch.float32)
        anchor_grid_y, anchor_grid_x = torch.meshgrid(anchor_y, anchor_x, indexing='ij')
        anchor_xy = torch.stack([anchor_grid_x, anchor_grid_y], dim=-1).view(-1, 2) + 0.5
        anchor_xy *= stride
        return anchor_xy
    
    def decode_boxes(self, anchors, pred_reg, stride):
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        pred_box_wh = pred_reg[..., 2:].exp() * stride
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        return torch.cat([pred_x1y1, pred_x2y2], dim=-1)
    
    def forward(self, video_clips):
        B = video_clips.shape[0]
        device = video_clips.device
        key_frame = video_clips[:, :, -1, :, :]
        
        if self.needs_normalization:
            video_clips_3d = (video_clips - self.pixel_mean) / self.pixel_std
        else:
            video_clips_3d = video_clips
        
        feat_3d = self.model.backbone_3d(video_clips_3d)
        cls_feats, reg_feats = self.model.backbone_2d(key_frame)
        all_outputs = []
        
        for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            cls_feat = self.model.cls_channel_encoders[level](cls_feat, feat_3d_up)
            reg_feat = self.model.reg_channel_encoders[level](reg_feat, feat_3d_up)
            cls_feat, reg_feat = self.model.heads[level](cls_feat, reg_feat)
            
            # Since purge_o2m ran, the .pth ONLY contains the renamed O2O conf/reg weights!
            # These are now called self.model.conf_preds, so no end2end flag is needed
            conf_pred = self.model.conf_preds[level](reg_feat)
            obj_pred = self.model.obj_preds[level](cls_feat)
            
            # TRUE CASCADE: Object context enriches features for relation prediction
            rel_feat = self.model.obj_context[level](cls_feat, obj_pred, conf_pred)
            rel_pred = self.model.rel_preds[level](rel_feat)
            
            # TRUE CASCADE: rel_feat (NOT cls_feat!) goes into rel_context for action prediction
            # This was a critical bug - using cls_feat broke the learned cascade!
            act_feat = self.model.rel_context[level](rel_feat, obj_pred, rel_pred, conf_pred)
            act_pred = self.model.act_preds[level](act_feat)
            reg_pred = self.model.reg_preds[level](reg_feat)
            
            fmp_h, fmp_w = conf_pred.shape[-2:]
            anchors = self.generate_anchors((fmp_h, fmp_w), self.stride[level], device)
            
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_objects)
            act_pred = act_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_actions)
            rel_pred = rel_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_relations)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            box_pred = self.decode_boxes(anchors.unsqueeze(0), reg_pred, self.stride[level])
            
            conf_pred = torch.sigmoid(conf_pred)
            obj_pred = F.softmax(obj_pred, dim=-1)
            act_pred = torch.sigmoid(act_pred)
            rel_pred = torch.sigmoid(rel_pred)
            
            level_output = torch.cat([conf_pred, obj_pred, act_pred, rel_pred, box_pred], dim=-1)
            all_outputs.append(level_output)
            
        output = torch.cat(all_outputs, dim=1).squeeze(0)
        return output

# =========================================================================
# STAGE 1: PURGE O2M WEIGHTS
# =========================================================================
def stage1_purge_ckpt(ckpt_path):
    print(f"\n{'='*60}\nSTAGE 1: PURGING O2M HEADS\n{'='*60}")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    is_wrapped = 'model' in checkpoint
    state_dict = checkpoint['model'] if is_wrapped else checkpoint
    
    # Remove O2M heads entirely
    keys_to_remove = [k for k in state_dict.keys() if k.startswith('conf_preds.') or k.startswith('reg_preds.')]
    for k in keys_to_remove: del state_dict[k]
        
    # Promote O2O into primary variables
    rename_map = {}
    for k in state_dict.keys():
        if k.startswith('o2o_conf_preds.'): rename_map[k] = k.replace('o2o_conf_preds.', 'conf_preds.')
        elif k.startswith('o2o_reg_preds.'): rename_map[k] = k.replace('o2o_reg_preds.', 'reg_preds.')
    
    for old_key, new_key in rename_map.items():
        state_dict[new_key] = state_dict.pop(old_key)
        
    # Delete alias references (we keep the underlying class logic shared weights untouched)
    alias_prefixes = ['o2o_obj_preds.', 'o2o_act_preds.', 'o2o_rel_preds.', 'o2o_obj_context.', 'o2o_rel_context.']
    for k in list(state_dict.keys()):
        if any(k.startswith(p) for p in alias_prefixes):
            del state_dict[k]
            
    out_path = ckpt_path.replace('.pth', '_purged.pth')
    torch.save({'model': state_dict} if is_wrapped else state_dict, out_path)
    print(f"  ✓ Purged weights saved temporarily to: {out_path}")
    return out_path, state_dict

# =========================================================================
# STAGE 2: GENERATE ONNX
# =========================================================================
def stage2_export_onnx(purged_path, state_dict, version, dataset, len_clip=32, img_size=224):
    print(f"\n{'='*60}\nSTAGE 2: ONNX GENERATION\n{'='*60}")
    
    class Args:
        def __init__(self):
            self.version = version
            self.dataset = dataset
            self.img_size = img_size
            self.len_clip = len_clip
            self.freeze_backbone_2d = False
            self.freeze_backbone_3d = False
    
    model_args = Args()
    d_cfg = build_dataset_config(model_args)
    m_cfg = build_model_config(model_args)
    
    # Model built purely as evaluation/O2O single state
    model = YOWOMultiTask(
        cfg=m_cfg, device='cpu', 
        num_objects=d_cfg['num_objects'], num_actions=d_cfg['num_actions'], num_relations=d_cfg['num_relations'],
        conf_thresh=0.1, nms_thresh=0.5, topk=50, trainable=False, end2end=False # O2M is erased. O2O is now native.
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    backbone_type = m_cfg.get('backbone_3d', 'resnext101')
    onnx_wrapper = YOWONMSFreeONNXWrapper(model, img_size, backbone_type)
    onnx_wrapper.eval()
    
    dummy_input = torch.randn(1, 3, len_clip, img_size, img_size)
    out_path = purged_path.replace('_purged.pth', '_optimized.onnx')
    
    print("  Dynamically fusing into native ONNX...")
    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,
    )
    print(f"  ✓ ONNX Exported to: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")
    return out_path


# =========================================================================
# STAGE 3: TENSORRT CALIBRATION AND ENGINE BUILDING
# =========================================================================
def stage3_build_engine(onnx_path, calib_data_dir, len_clip=32):
    print(f"\n{'='*60}\nSTAGE 3: TENSORRT INT8 COMPILATION\n{'='*60}")
    
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import cv2
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # Mute excess logs
    
    class YOWOCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, data_dir, len_clip=32):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.cache_file = onnx_path.replace('.onnx', '.cache')
            self.len_clip = len_clip
            self.video_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))[:25] # 25 videos is enough
            self.current_idx = 0
            self.device_input = cuda.mem_alloc(int(np.prod((1, 3, len_clip, 224, 224)) * 4))
            
        def get_batch_size(self): return 1
        
        def get_batch(self, names):
            while self.current_idx < len(self.video_dirs):
                frames = sorted(glob.glob(os.path.join(self.video_dirs[self.current_idx], "*.jpg")))
                if len(frames) > 0:
                    break
                self.current_idx += 1
                
            if self.current_idx >= len(self.video_dirs): return None
            
            if len(frames) < self.len_clip: frames += [frames[-1]] * (self.len_clip - len(frames))
            
            indices = np.linspace(0, len(frames) - 1, self.len_clip, dtype=int)
            clip = []
            for idx in indices:
                img = cv2.imread(frames[idx])
                if img is None:
                    img = np.zeros((224,224,3), dtype=np.uint8)
                img = cv2.resize(img, (224, 224))
                clip.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0))
            
            clip = np.ascontiguousarray(np.expand_dims(np.stack(clip, axis=0).transpose(3, 0, 1, 2), 0))
            cuda.memcpy_htod(self.device_input, clip)
            self.current_idx += 1
            print(f"    Calibration batch {self.current_idx}/{len(self.video_dirs)} loaded.")
            return [int(self.device_input)]
        
        def read_calibration_cache(self):
            return open(self.cache_file, "rb").read() if os.path.exists(self.cache_file) else None
        
        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f: f.write(cache)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print("  Parsing ONNX layout...")
    if not parser.parse(open(onnx_path, "rb").read()):
        raise RuntimeError("ONNX parsing failed.")
    
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
    
    # !!! CRITICAL PRECISION FIX for OBJECT DETECTION !!!
    # Force box regression, context modules, and activation layers to FP16.
    # INT8 quantization destroys the attention mechanisms in context modules.
    print("  Applying precision fallback directives for critical detection heads...")
    fp16_keywords = [
        "reg_preds",      # Box regression needs precision
        "obj_context",    # Attention module - sensitive to quantization
        "rel_context",    # Attention module - sensitive to quantization
        "conf_preds",     # Confidence prediction
        "act_preds",      # Action prediction head
        "rel_preds",      # Relation prediction head
        "obj_preds",      # Object prediction head
        "softmax", "sigmoid", "exp",
        "matmul", "attention", "norm"  # Attention ops need precision
    ]
    # Skip these layer types that can't have precision changed
    skip_layer_types = {trt.LayerType.SHAPE, trt.LayerType.CONSTANT, trt.LayerType.IDENTITY}
    fp16_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        # Skip layers that don't support precision changes
        if layer.type in skip_layer_types:
            continue
        if any(key in layer.name.lower() for key in fp16_keywords):
            try:
                layer.precision = trt.float16
                layer.set_output_type(0, trt.float16)
                fp16_count += 1
            except Exception:
                pass  # Skip layers that don't support precision changes
    print(f"    → {fp16_count} layers set to FP16 precision")

    if hasattr(trt.BuilderFlag, 'OBEY_PRECISION_CONSTRAINTS'):
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    else:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    
    print(f"  Beginning INT8 Quantization Process...")
    config.int8_calibrator = YOWOCalibrator(calib_data_dir, len_clip)
    
    # Engine creation
    engine_path = onnx_path.replace('_optimized.onnx', '_int8.engine')
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT Engine building failed! The Orin Nano likely ran out of unified memory. Reboot and try again, or disable INT8.")
        
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"  ✓ INT8 Engine fully compiled: {engine_path} ({os.path.getsize(engine_path)/1024/1024:.1f} MB)")
    return engine_path


# =========================================================================
# MAIN EXECUTION
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='Unified YOWO Deployment Script')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to the trained .pth checkpoint (YOWO_v2_..._epoch_X.pth)')
    parser.add_argument('--dataset', type=str, default='smart_home', choices=['smart_home', 'charades_ag'])
    parser.add_argument('--len_clip', type=int, default=32)
    parser.add_argument('--calib_data', type=str, default='/home/michel/yowo/data/ActionGenome/frames')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Cannot find {args.checkpoint}")
        return 1
        
    # Auto-infer version config based on file name contents
    ckpt_name = os.path.basename(args.checkpoint)
    idx = ckpt_name.find('_epoch')
    version = ckpt_name[:idx] if idx != -1 else 'yowo_v2_resnext_yolo26m_multitask'

    # Run the unified build chain!
    purged_path, state_dict = stage1_purge_ckpt(args.checkpoint)
    onnx_path = stage2_export_onnx(purged_path, state_dict, version, args.dataset, args.len_clip)
    engine_path = stage3_build_engine(onnx_path, args.calib_data, args.len_clip)

    # Cleanup intermediate 
    if os.path.exists(purged_path):
        os.remove(purged_path)

    print(f"\n============================================================")
    print(f"🚀 FINAL DEPLOYMENT BUILDER COMPLETE!")
    print(f"============================================================")
    print(f" You can delete the original ONNX and Engine scripts.")
    print(f" Just run this file directly when deploying future models.")
    print(f" -> {engine_path}")

if __name__ == '__main__':
    exit(main())
