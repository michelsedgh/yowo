#!/usr/bin/env python3
"""
Complete INT8 TensorRT Engine Builder for YOWO
PTH -> ONNX -> INT8 TensorRT with proper calibration

Usage:
    python build_int8_engine.py --checkpoint yowo_v2_resnext_yolo26m_multitask_epoch_7-2.pth
"""

import os
import sys
import glob
import argparse
import numpy as np
import subprocess

# Step 1: Export to ONNX
def export_onnx(checkpoint_path, output_onnx, len_clip=32, dataset="smart_home"):
    """Export PyTorch checkpoint to ONNX using yowov2 conda environment"""
    print(f"\n{'='*60}")
    print("STEP 1: Exporting to ONNX")
    print(f"{'='*60}")
    
    # Use conda run to ensure correct Python environment with all dependencies
    cmd = [
        "conda", "run", "-n", "yowov2", "--no-capture-output",
        "python", "export_onnx.py",
        "--weight", checkpoint_path,
        "--version", "yowo_v2_resnext_yolo26m_multitask",
        "--dataset", dataset,
        "--len_clip", str(len_clip),
        "--output", output_onnx,
        "--no_verify"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/home/michel/yowo")
    
    if result.returncode != 0 or not os.path.exists(output_onnx):
        print(f"ONNX export failed!")
        return False
    
    size_mb = os.path.getsize(output_onnx) / 1024 / 1024
    print(f"✓ ONNX exported: {output_onnx} ({size_mb:.1f} MB)")
    return True


# Step 2: Build INT8 Engine with Calibration
def build_int8_engine(onnx_path, engine_path, calib_data_dir, len_clip=32):
    """Build INT8 TensorRT engine with calibration"""
    print(f"\n{'='*60}")
    print("STEP 2: Building INT8 TensorRT Engine")
    print(f"{'='*60}")
    
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Initialize CUDA context
    import cv2
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Calibrator class for INT8
    class YOWOCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, data_dir, batch_size=1, input_shape=(1, 3, 32, 224, 224), cache_file="calibration.cache"):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.cache_file = cache_file
            self.batch_size = batch_size
            self.input_shape = input_shape
            self.len_clip = input_shape[2]
            
            # Find video folders
            self.video_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))[:50]  # Use 50 videos
            self.current_idx = 0
            
            # Allocate device memory using pycuda
            self.buffer_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
            self.device_input = cuda.mem_alloc(self.buffer_size)
            
            print(f"  Calibrator initialized with {len(self.video_dirs)} videos")
            print(f"  Input shape: {input_shape}")
            print(f"  Cache file: {cache_file}")
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_batch(self, names):
            if self.current_idx >= len(self.video_dirs):
                return None
            
            video_dir = self.video_dirs[self.current_idx]
            self.current_idx += 1
            
            # Load frames from video directory
            frames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
            if len(frames) < self.len_clip:
                # Pad by repeating last frame
                frames = frames + [frames[-1]] * (self.len_clip - len(frames))
            
            # Sample len_clip frames evenly
            indices = np.linspace(0, len(frames) - 1, self.len_clip, dtype=int)
            
            clip = []
            for idx in indices:
                img = cv2.imread(frames[idx])
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # ResNeXt 3D backbone expects raw [0,1] input - no ImageNet normalization
                # (X3D would need normalization, but we're using ResNeXt)
                img = img.astype(np.float32) / 255.0
                clip.append(img)
            
            # Stack: [T, H, W, C] -> [C, T, H, W]
            clip = np.stack(clip, axis=0)  # [T, H, W, C]
            clip = clip.transpose(3, 0, 1, 2)  # [C, T, H, W]
            clip = np.expand_dims(clip, 0)  # [1, C, T, H, W]
            clip = np.ascontiguousarray(clip, dtype=np.float32)
            
            # Copy to device using pycuda
            cuda.memcpy_htod(self.device_input, clip)
            
            if self.current_idx % 10 == 0:
                print(f"  Calibrating... {self.current_idx}/{len(self.video_dirs)}")
            
            return [int(self.device_input)]
        
        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                print(f"  Reading calibration cache: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None
        
        def write_calibration_cache(self, cache):
            print(f"  Writing calibration cache: {self.cache_file}")
            with open(self.cache_file, "wb") as f:
                f.write(cache)
        
        def __del__(self):
            # pycuda handles memory cleanup automatically
            pass
    
    # Build engine
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Parse ONNX
    print(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return False
    
    # Configure for INT8
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    
    # Set memory pool - use available memory
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 * 1024 * 1024 * 1024)  # 5GB
    
    # Set calibrator
    input_shape = (1, 3, len_clip, 224, 224)
    cache_file = "/home/michel/yowo/calibration.cache"
    calibrator = YOWOCalibrator(calib_data_dir, input_shape=input_shape, cache_file=cache_file)
    config.int8_calibrator = calibrator
    
    # Build
    print("  Building INT8 engine (this takes 15-30 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("  Engine build failed!")
        return False
    
    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"✓ INT8 Engine saved: {engine_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build INT8 TensorRT Engine")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/michel/yowo/yowo_v2_resnext_yolo26m_multitask_epoch_7-2.pth",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--len_clip", type=int, default=32,
                        help="Number of frames in clip")
    parser.add_argument("--dataset", type=str, default="smart_home",
                        choices=["smart_home", "charades_ag"],
                        help="Dataset config (smart_home=42 actions, charades_ag=157 actions)")
    parser.add_argument("--calib_data", type=str,
                        default="/home/michel/yowo/data/ActionGenome/frames",
                        help="Path to calibration data (video frame directories)")
    args = parser.parse_args()
    
    # Output paths
    onnx_path = "/home/michel/yowo/latest.onnx"
    engine_path = "/home/michel/yowo/latest_int8.engine"
    
    print("="*60)
    print("YOWO INT8 TensorRT Engine Builder")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Clip length: {args.len_clip}")
    print(f"Dataset: {args.dataset}")
    print(f"Calibration data: {args.calib_data}")
    print(f"Output ONNX: {onnx_path}")
    print(f"Output Engine: {engine_path}")
    
    # Free memory first
    print("\nFreeing system memory...")
    os.system("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
    os.system("sudo nvpmodel -m 0 2>/dev/null; sudo jetson_clocks 2>/dev/null")
    
    # Step 1: Export ONNX
    if not export_onnx(args.checkpoint, onnx_path, args.len_clip, args.dataset):
        print("\n❌ ONNX export failed!")
        sys.exit(1)
    
    # Step 2: Build INT8 engine
    if not build_int8_engine(onnx_path, engine_path, args.calib_data, args.len_clip):
        print("\n❌ INT8 engine build failed!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ SUCCESS!")
    print(f"{'='*60}")
    print(f"ONNX:   {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"\nTest with:")
    print(f"  python demo_smart_home.py --engine {engine_path}")


if __name__ == "__main__":
    main()
