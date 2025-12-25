#!/usr/bin/env python3
"""
Export YOWO to TensorRT INT8

Uses entropy calibration with synthetic data for testing.
For production, use real calibration data.
"""

import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator using synthetic data."""
    
    def __init__(self, input_shape, num_batches=50, cache_file="int8_calibration.cache"):
        super().__init__()
        self.input_shape = input_shape
        self.num_batches = num_batches
        self.cache_file = cache_file
        self.current_batch = 0
        self.batch_size = input_shape[0]
        
        # Allocate device memory
        self.input_bytes = int(np.prod(input_shape) * 4)
        self.d_input = cuda.mem_alloc(self.input_bytes)
        
        print(f"Calibrator initialized: {num_batches} batches, shape {input_shape}")
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_batch >= self.num_batches:
            return None
        
        # Generate synthetic data that mimics normalized video frames
        # Using random but bounded values typical for image normalization
        batch = np.random.randn(*self.input_shape).astype(np.float32)
        batch = np.clip(batch, -2.5, 2.5)  # Typical range for normalized images
        
        cuda.memcpy_htod(self.d_input, batch.tobytes())
        
        self.current_batch += 1
        if self.current_batch % 10 == 0:
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


def build_int8_engine():
    onnx_path = 'yowo_v2_x3d_m_yolo11m_multitask_epoch_5_fp16.onnx'
    engine_path = 'yowo_v2_x3d_m_yolo11m_multitask_epoch_5_int8.engine'
    
    print("="*60)
    print("YOWO INT8 TensorRT Export")
    print("="*60)
    
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))
    
    # Enable INT8 + FP16 fallback
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Create calibrator
    input_shape = (1, 3, 16, 224, 224)
    calibrator = EntropyCalibrator(input_shape, num_batches=50)
    config.int8_calibrator = calibrator
    
    print("\nBuilding INT8 engine (this takes 15-20 minutes)...")
    t_start = time.time()
    
    serialized = builder.build_serialized_network(network, config)
    
    if serialized is None:
        print("ERROR: Failed to build engine!")
        return
    
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    
    build_time = time.time() - t_start
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print(f"  Engine: {engine_path}")
    print(f"  Size: {os.path.getsize(engine_path) / 1e6:.1f} MB")
    print(f"  Build time: {build_time:.1f}s")


if __name__ == '__main__':
    build_int8_engine()
