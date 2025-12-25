#!/usr/bin/env python3
"""
YOWO TensorRT Export Script for Epoch 10
Memory-optimized for Jetson Orin devices.

This script exports the YOWO multi-task model to TensorRT:
1. Export PyTorch model to ONNX (CPU-based to save memory)
2. Build TensorRT engine using trtexec (for best optimization)

Usage:
    python export_tensorrt_epoch10.py --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_10.pth --precision fp16
"""

import argparse
import os
import time
import gc
import subprocess
import numpy as np
import torch
import torch.nn as nn

from config import build_dataset_config, build_model_config
from models import build_model
from utils.misc import load_weight


class YOWOWrapper(nn.Module):
    """
    Wrapper for YOWO model that outputs raw predictions for TensorRT export.
    
    TensorRT doesn't support dynamic post-processing, so we export only the 
    backbone + heads and do post-processing in Python after TensorRT inference.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.trainable = True  # Force training mode to get raw outputs
        
    def forward(self, x):
        """
        Forward pass returning raw predictions.
        
        Returns stacked outputs for easier TensorRT handling:
        - conf: [B, N, 1]
        - obj: [B, N, 36]
        - act: [B, N, 157]
        - rel: [B, N, 26]
        - box: [B, N, 4]
        - interact: [B, N, 1]
        """
        # Get raw outputs from model in training mode
        outputs = self.model(x)
        
        # Concatenate across pyramid levels
        conf = torch.cat(outputs["pred_conf"], dim=1)      # [B, N, 1]
        obj = torch.cat(outputs["pred_obj"], dim=1)        # [B, N, 36]
        act = torch.cat(outputs["pred_act"], dim=1)        # [B, N, 157]
        rel = torch.cat(outputs["pred_rel"], dim=1)        # [B, N, 26]
        box = torch.cat(outputs["pred_box"], dim=1)        # [B, N, 4]
        interact = torch.cat(outputs["pred_interact"], dim=1)  # [B, N, 1]
        
        # Stack all predictions: [B, N, 1+36+157+26+4+1] = [B, N, 225]
        stacked = torch.cat([conf, obj, act, rel, box, interact], dim=-1)
        
        return stacked


def parse_args():
    parser = argparse.ArgumentParser(description='Export YOWO Epoch 10 to TensorRT')
    
    # Model
    parser.add_argument('--weight', default='yowo_v2_x3d_m_yolo11m_multitask_epoch_10.pth', 
                        type=str, help='Path to trained weights')
    parser.add_argument('-v', '--version', default='yowo_v2_x3d_m_yolo11m_multitask', type=str,
                        help='Model version')
    parser.add_argument('-d', '--dataset', default='charades_ag', type=str,
                        help='Dataset config')
    
    # Export settings
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'],
                        help='TensorRT precision mode')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--len_clip', default=16, type=int,
                        help='Video clip length')
    parser.add_argument('--output', default=None, type=str,
                        help='Output engine path (default: auto-generate)')
    parser.add_argument('--workspace', default=8, type=int,
                        help='TensorRT workspace size in GB')
    parser.add_argument('--onnx_only', action='store_true',
                        help='Only export ONNX (skip TensorRT engine build)')
    parser.add_argument('--skip_onnx', action='store_true',
                        help='Skip ONNX export (use existing ONNX file)')
    
    # Inference settings (for args compatibility)
    parser.add_argument('--conf_thresh', default=0.1, type=float)
    parser.add_argument('--nms_thresh', default=0.5, type=float)
    parser.add_argument('--topk', default=40, type=int)
    
    return parser.parse_args()


def export_onnx_cpu(model_cfg, dataset_cfg, weight_path, input_shape, onnx_path, args):
    """Export model to ONNX format using CPU to save memory."""
    print(f"\n[1/3] Exporting to ONNX on CPU: {onnx_path}")
    print(f"      (Using CPU to avoid Orin memory issues during tracing)")
    
    # Force CPU device
    device = torch.device('cpu')
    
    # Build model on CPU
    print("      Building model on CPU...")
    model, _ = build_model(
        args=args,
        d_cfg=dataset_cfg,
        m_cfg=model_cfg,
        device=device,
        num_classes=dataset_cfg.get('valid_num_classes', 219),
        trainable=False
    )
    
    # Load weights on CPU
    print(f"      Loading weights: {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    # Wrap and prepare for export
    wrapper = YOWOWrapper(model).eval()
    
    # Create dummy input on CPU
    dummy_input = torch.randn(1, *input_shape)
    
    print("      Running trace and export (this may take a few minutes)...")
    t_start = time.time()
    
    # Export with compatible opset
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            input_names=['video_clip'],
            output_names=['predictions'],
            dynamic_axes=None,  # Fixed batch size for TensorRT
            opset_version=17,
            do_constant_folding=True,
            verbose=False
        )
    
    export_time = time.time() - t_start
    print(f"      ONNX export complete in {export_time:.1f}s")
    print(f"      File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")
    
    # Verify ONNX
    print("      Verifying ONNX model...")
    import onnx
    from onnx import checker
    onnx_model = onnx.load(onnx_path)
    checker.check_model(onnx_model)
    print("      ONNX model verified ✓")
    
    # Clean up
    del model, wrapper, dummy_input, onnx_model
    gc.collect()
    
    return onnx_path


def build_engine_trtexec(onnx_path, engine_path, precision, workspace_gb):
    """Build TensorRT engine using trtexec (official NVIDIA tool)."""
    print(f"\n[2/3] Building TensorRT engine using trtexec: {engine_path}")
    print(f"      Precision: {precision.upper()}")
    print(f"      Workspace: {workspace_gb} GB")
    
    # Construct trtexec command
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        f'--memPoolSize=workspace:{workspace_gb}G',
    ]
    
    # Add precision flags
    if precision == 'fp16':
        cmd.append('--fp16')
    elif precision == 'int8':
        cmd.extend(['--int8', '--fp16'])  # INT8 with FP16 fallback
    
    # Add optimization flags for Orin
    cmd.extend([
        '--verbose',
        '--buildOnly',
        '--noTF32',  # Disable TF32 for consistency
        '--avgTiming=16',  # More averaging for stable timing
    ])
    
    print(f"      Command: {' '.join(cmd)}")
    print("      Building (this may take 10-30 minutes on Orin)...")
    
    t_start = time.time()
    
    # Run trtexec
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour timeout
    )
    
    build_time = time.time() - t_start
    
    if result.returncode != 0:
        print(f"\n      ERROR: trtexec failed!")
        print(f"      STDERR: {result.stderr[-2000:]}")  # Last 2000 chars
        raise RuntimeError(f"trtexec failed with exit code {result.returncode}")
    
    print(f"      Engine built in {build_time/60:.1f} minutes")
    print(f"      Engine size: {os.path.getsize(engine_path) / 1e6:.1f} MB")
    
    return engine_path


def verify_engine(engine_path, input_shape):
    """Verify the TensorRT engine works correctly."""
    print(f"\n[3/3] Verifying TensorRT engine...")
    
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Get input/output info
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    input_shape_engine = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    print(f"      Input:  {input_name} {list(input_shape_engine)}")
    print(f"      Output: {output_name} {list(output_shape)}")
    
    # Allocate buffers
    input_size = int(np.prod(input_shape_engine)) * 4
    output_size = int(np.prod(output_shape)) * 4
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    h_input = np.random.randn(*input_shape_engine).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    stream = cuda.Stream()
    
    # Set tensor addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # Run inference
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    # Quick benchmark
    print("      Running quick benchmark (10 iterations)...")
    times = []
    for _ in range(10):
        t_start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        times.append((time.perf_counter() - t_start) * 1000)
    
    avg_time = np.mean(times)
    print(f"      Average inference time: {avg_time:.2f} ms ({1000/avg_time:.1f} FPS)")
    print(f"      Output stats: min={h_output.min():.4f}, max={h_output.max():.4f}, mean={h_output.mean():.4f}")
    print("      Engine verified ✓")
    
    return True


def main():
    args = parse_args()
    
    print("="*70)
    print("YOWO TensorRT Export - Epoch 10")
    print("="*70)
    print(f"Weight file: {args.weight}")
    print(f"Precision:   {args.precision.upper()}")
    print(f"Input size:  {args.img_size}x{args.img_size}, Clip length: {args.len_clip}")
    
    # Build configs
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)
    
    # Input shape: [C, T, H, W]
    input_shape = (3, args.len_clip, args.img_size, args.img_size)
    print(f"Input shape: [1, {input_shape}]")
    
    # Output paths
    weight_name = os.path.splitext(os.path.basename(args.weight))[0]
    if args.output:
        engine_path = args.output
    else:
        engine_path = f"{weight_name}_{args.precision}.engine"
    onnx_path = engine_path.replace('.engine', '.onnx')
    
    # Step 1: Export to ONNX (on CPU to save memory)
    if not args.skip_onnx:
        if os.path.exists(onnx_path):
            print(f"\n[Info] ONNX file already exists: {onnx_path}")
            print("       Use --skip_onnx to skip re-export, or delete the file to re-export.")
        export_onnx_cpu(m_cfg, d_cfg, args.weight, input_shape, onnx_path, args)
    else:
        print(f"\n[Info] Skipping ONNX export, using existing: {onnx_path}")
    
    if args.onnx_only:
        print("\n" + "="*70)
        print("ONNX Export Complete!")
        print(f"  ONNX: {onnx_path}")
        print("="*70)
        print("\nTo build TensorRT engine manually, run:")
        print(f"  trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16")
        return
    
    # Step 2: Build TensorRT engine using trtexec
    build_engine_trtexec(onnx_path, engine_path, args.precision, args.workspace)
    
    # Step 3: Verify engine
    try:
        verify_engine(engine_path, input_shape)
    except Exception as e:
        print(f"\n      Warning: Engine verification failed: {e}")
        print("      The engine was built but verification couldn't complete.")
    
    print("\n" + "="*70)
    print("TensorRT Export Complete!")
    print("="*70)
    print(f"  ONNX:   {onnx_path}")
    print(f"  Engine: {engine_path}")
    print("\nTo run demo:")
    print(f"  python demo_tensorrt.py --engine {engine_path} --video your_video.mp4")
    print("="*70)


if __name__ == '__main__':
    main()
