#!/usr/bin/env python3
"""
Quick TensorRT Benchmark for YOWO

Tests inference speed of TensorRT engine.
"""

import argparse
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def benchmark(engine_path, warmup=20, iterations=100):
    """Benchmark TensorRT engine."""
    
    print(f"Loading engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Get input/output info
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    print(f"Input: {input_name} {list(input_shape)}")
    print(f"Output: {output_name} {list(output_shape)}")
    
    # Allocate memory
    input_size = int(np.prod(input_shape)) * 4
    output_size = int(np.prod(output_shape)) * 4
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    stream = cuda.Stream()
    
    # Set tensor addresses (TensorRT 10 API)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # Warmup
    print(f"\nWarmup: {warmup} iterations...")
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    print(f"Benchmark: {iterations} iterations...")
    times = []
    
    for i in range(iterations):
        t_start = time.perf_counter()
        
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        times.append((time.perf_counter() - t_start) * 1000)
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{iterations}] {np.mean(times[-20:]):.2f} ms")
    
    times = np.array(times)
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"  Mean latency: {np.mean(times):.2f} ms")
    print(f"  Std latency:  {np.std(times):.2f} ms")
    print(f"  Min latency:  {np.min(times):.2f} ms")
    print(f"  Max latency:  {np.max(times):.2f} ms")
    print(f"  Median:       {np.median(times):.2f} ms")
    print(f"  Throughput:   {1000 / np.mean(times):.1f} FPS")
    print("="*50)
    
    return np.mean(times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True, help='TensorRT engine path')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    
    benchmark(args.engine, args.warmup, args.iterations)
