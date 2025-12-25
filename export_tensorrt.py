#!/usr/bin/env python3
"""
YOWO TensorRT Export Script

Exports the YOWO multi-task model to TensorRT for optimized inference on Orin Nano.
Supports FP16 and INT8 precision modes.

Usage:
    # Export to FP16 (recommended for Orin Nano)
    python export_tensorrt.py --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_5.pth --precision fp16
    
    # Export to INT8 (requires calibration data)
    python export_tensorrt.py --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_5.pth --precision int8
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

# TensorRT
import tensorrt as trt

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
    parser = argparse.ArgumentParser(description='Export YOWO to TensorRT')
    
    # Model
    parser.add_argument('--weight', required=True, type=str,
                        help='Path to trained weights')
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
    parser.add_argument('--workspace', default=4, type=int,
                        help='TensorRT workspace size in GB')
    
    # For INT8 calibration
    parser.add_argument('--calib_data', default=None, type=str,
                        help='Path to calibration data for INT8 (optional)')
    parser.add_argument('--calib_batches', default=100, type=int,
                        help='Number of calibration batches for INT8')
    
    # Inference settings (for args compatibility)
    parser.add_argument('--conf_thresh', default=0.1, type=float)
    parser.add_argument('--nms_thresh', default=0.5, type=float)
    parser.add_argument('--topk', default=40, type=int)
    
    return parser.parse_args()


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """Simple INT8 calibrator using random data if no calibration dataset provided."""
    
    def __init__(self, batch_size, input_shape, calib_batches=100, cache_file="calibration.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.calib_batches = calib_batches
        self.cache_file = cache_file
        self.current_batch = 0
        
        # Allocate device memory for input
        self.device_input = None
        
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_batch >= self.calib_batches:
            return None
        
        # Generate random calibration data
        # In production, you'd load real data here
        batch_data = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)
        
        if self.device_input is None:
            import cuda.cuda as cuda
            import cuda.cudart as cudart
            err, self.device_input = cudart.cudaMalloc(batch_data.nbytes)
        
        import cuda.cudart as cudart
        cudart.cudaMemcpy(self.device_input, batch_data.ctypes.data, 
                         batch_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        self.current_batch += 1
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def export_onnx(model, input_shape, onnx_path):
    """Export model to ONNX format."""
    print(f"\n[1/2] Exporting to ONNX: {onnx_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).cuda()
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['video_clip'],
        output_names=['predictions'],
        dynamic_axes=None,  # Fixed batch size for TensorRT
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"    ONNX export complete: {os.path.getsize(onnx_path) / 1e6:.1f} MB")
    
    # Verify ONNX
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("    ONNX model verified!")
    
    return onnx_path


def build_engine(onnx_path, engine_path, precision, workspace_gb, calibrator=None):
    """Build TensorRT engine from ONNX model."""
    print(f"\n[2/2] Building TensorRT engine: {engine_path}")
    print(f"    Precision: {precision.upper()}")
    print(f"    Workspace: {workspace_gb} GB")
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"    ONNX parse error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX")
    
    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    
    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("    FP16 enabled")
        else:
            print("    Warning: FP16 not supported, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator:
                config.int8_calibrator = calibrator
            print("    INT8 enabled")
        else:
            print("    Warning: INT8 not supported, using FP32")
    
    # Build engine
    print("    Building engine (this may take several minutes)...")
    t_start = time.time()
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    build_time = time.time() - t_start
    print(f"    Engine built in {build_time:.1f}s")
    print(f"    Engine size: {os.path.getsize(engine_path) / 1e6:.1f} MB")
    
    return engine_path


def benchmark_engine(engine_path, input_shape, warmup=10, iterations=100):
    """Benchmark TensorRT engine performance."""
    print(f"\n[Benchmark] Running inference benchmark...")
    
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    input_size = int(np.prod(input_shape)) * 4  # float32
    output_shape = (1, 1029, 225)  # Approximate output shape
    output_size = int(np.prod(output_shape)) * 4
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    h_input = np.random.randn(1, *input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    stream = cuda.Stream()
    
    # Warmup
    print(f"    Warmup: {warmup} iterations")
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    print(f"    Benchmark: {iterations} iterations")
    times = []
    for _ in range(iterations):
        t_start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        times.append(time.perf_counter() - t_start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    print(f"\n    Results:")
    print(f"    - Mean latency: {np.mean(times):.2f} ms")
    print(f"    - Std latency:  {np.std(times):.2f} ms")
    print(f"    - Min latency:  {np.min(times):.2f} ms")
    print(f"    - Max latency:  {np.max(times):.2f} ms")
    print(f"    - FPS:          {1000 / np.mean(times):.1f}")
    
    return np.mean(times)


def main():
    args = parse_args()
    
    print("="*60)
    print("YOWO TensorRT Export")
    print("="*60)
    
    # Build model
    device = torch.device('cuda')
    
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)
    
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=d_cfg.get('valid_num_classes', 219),
        trainable=False
    )
    
    # Load weights
    print(f"\nLoading weights: {args.weight}")
    model = load_weight(model, args.weight)
    model = model.to(device).eval()
    
    # Wrap model for export
    wrapper = YOWOWrapper(model).cuda().eval()
    
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
    
    # Step 1: Export to ONNX
    export_onnx(wrapper, input_shape, onnx_path)
    
    # Step 2: Build TensorRT engine
    calibrator = None
    if args.precision == 'int8':
        calibrator = Int8Calibrator(
            batch_size=1,
            input_shape=input_shape,
            calib_batches=args.calib_batches
        )
    
    build_engine(onnx_path, engine_path, args.precision, args.workspace, calibrator)
    
    # Step 3: Benchmark
    try:
        benchmark_engine(engine_path, input_shape)
    except Exception as e:
        print(f"\n    Benchmark skipped: {e}")
        print("    Install pycuda for benchmarking: pip install pycuda")
    
    print("\n" + "="*60)
    print("Export complete!")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Engine: {engine_path}")
    print("="*60)


if __name__ == '__main__':
    main()
