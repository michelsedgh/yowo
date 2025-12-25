#!/usr/bin/env python3
"""
Proper INT8 TensorRT Export for YOWO

Key differences from naive approach:
1. Uses REAL calibration data from the Charades-AG dataset
2. Properly preprocesses images the same way as training
3. Uses sufficient calibration batches (100+)
4. Deletes old calibration cache to force recalibration

Usage:
    python export_int8_proper.py
"""

import os
import sys
import time
import glob
import random
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.transforms import BaseTransform


class RealDataCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator using REAL frames from Charades-AG dataset.
    
    This is the correct approach - use representative data that
    matches what the model will see during inference.
    """
    
    def __init__(self, 
                 data_root='data/ActionGenome/frames',
                 input_shape=(1, 3, 16, 224, 224),
                 num_batches=100,
                 sample_rate=5,
                 cache_file="int8_calibration_real.cache"):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_batches = num_batches
        self.sample_rate = sample_rate
        self.len_clip = input_shape[2]  # 16
        self.cache_file = cache_file
        self.current_batch = 0
        
        # Delete old cache to force recalibration
        if os.path.exists(cache_file):
            print(f"Deleting old calibration cache: {cache_file}")
            os.remove(cache_file)
        
        # Get video directories
        self.video_dirs = []
        if os.path.exists(data_root):
            self.video_dirs = sorted(glob.glob(os.path.join(data_root, '*')))
            self.video_dirs = [d for d in self.video_dirs if os.path.isdir(d)]
        
        if not self.video_dirs:
            print(f"WARNING: No video directories found in {data_root}")
            print("Falling back to synthetic data - INT8 accuracy may be poor!")
            self.use_real_data = False
        else:
            print(f"Found {len(self.video_dirs)} video directories for calibration")
            self.use_real_data = True
            random.shuffle(self.video_dirs)
        
        # Transform (same as inference)
        self.transform = BaseTransform(img_size=224)
        
        # Allocate device memory
        self.input_bytes = int(np.prod(input_shape) * 4)  # float32
        self.d_input = cuda.mem_alloc(self.input_bytes)
        
        print(f"Calibrator initialized:")
        print(f"  Batches: {num_batches}")
        print(f"  Shape: {input_shape}")
        print(f"  Real data: {self.use_real_data}")
    
    def _load_clip_from_video(self, video_dir):
        """Load a video clip from a directory of frames."""
        frames = sorted(glob.glob(os.path.join(video_dir, '*.png')))
        if not frames:
            frames = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        
        if len(frames) < self.len_clip * self.sample_rate:
            return None
        
        # Random start position
        max_start = len(frames) - self.len_clip * self.sample_rate
        start_idx = random.randint(0, max(0, max_start))
        
        # Sample frames
        clip = []
        for i in range(self.len_clip):
            frame_idx = start_idx + i * self.sample_rate
            if frame_idx >= len(frames):
                frame_idx = len(frames) - 1
            
            try:
                img = Image.open(frames[frame_idx]).convert('RGB')
                clip.append(img)
            except Exception as e:
                return None
        
        return clip
    
    def _preprocess_clip(self, clip):
        """Preprocess clip using the same transform as inference."""
        try:
            x, _ = self.transform(clip)
            import torch
            x = torch.stack(x, dim=1)  # [3, T, H, W]
            x = x.unsqueeze(0).numpy()  # [1, 3, T, H, W]
            return x.astype(np.float32)
        except Exception as e:
            return None
    
    def get_batch_size(self):
        return self.input_shape[0]
    
    def get_batch(self, names):
        if self.current_batch >= self.num_batches:
            return None
        
        batch_data = None
        
        if self.use_real_data:
            # Try to load real data
            for attempt in range(10):
                video_idx = (self.current_batch * 10 + attempt) % len(self.video_dirs)
                video_dir = self.video_dirs[video_idx]
                
                clip = self._load_clip_from_video(video_dir)
                if clip is not None:
                    batch_data = self._preprocess_clip(clip)
                    if batch_data is not None:
                        break
        
        # Fallback to synthetic if real data failed
        if batch_data is None:
            # Use realistic distribution for image data (normalized)
            batch_data = np.random.randn(*self.input_shape).astype(np.float32)
            batch_data = np.clip(batch_data, -2.5, 2.5)
        
        cuda.memcpy_htod(self.d_input, batch_data.tobytes())
        
        self.current_batch += 1
        if self.current_batch % 20 == 0:
            print(f"  Calibration batch {self.current_batch}/{self.num_batches}")
        
        return [int(self.d_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        print(f"Writing calibration cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_proper_int8_engine():
    """Build INT8 engine with proper calibration."""
    
    onnx_path = 'yowo_v2_x3d_m_yolo11m_multitask_epoch_5_fp16.onnx'
    engine_path = 'yowo_v2_x3d_m_yolo11m_multitask_epoch_5_int8_proper.engine'
    
    print("="*70)
    print("YOWO INT8 TensorRT Export - PROPER Calibration")
    print("="*70)
    
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        return
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    print(f"\nParsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"Error: {parser.get_error(i)}")
            return
    
    config = builder.create_builder_config()
    
    # Use more workspace for better optimization
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4 GB
    
    # Enable INT8 + FP16 fallback for unsupported layers
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Create calibrator with REAL data
    input_shape = (1, 3, 16, 224, 224)
    calibrator = RealDataCalibrator(
        data_root='data/ActionGenome/frames',
        input_shape=input_shape,
        num_batches=100,  # More batches for better calibration
        sample_rate=5,
        cache_file="int8_calibration_real.cache"
    )
    config.int8_calibrator = calibrator
    
    print("\nBuilding INT8 engine with REAL calibration data...")
    print("This will take 15-25 minutes on Orin Nano...")
    t_start = time.time()
    
    serialized = builder.build_serialized_network(network, config)
    
    if serialized is None:
        print("ERROR: Failed to build engine!")
        return
    
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    
    build_time = time.time() - t_start
    
    print("\n" + "="*70)
    print("Export complete!")
    print("="*70)
    print(f"  Engine: {engine_path}")
    print(f"  Size: {os.path.getsize(engine_path) / 1e6:.1f} MB")
    print(f"  Build time: {build_time:.1f}s ({build_time/60:.1f} min)")
    print("\nNow run benchmark to compare with FP16:")
    print(f"  python benchmark_tensorrt.py --engine {engine_path}")


if __name__ == '__main__':
    build_proper_int8_engine()
