#!/usr/bin/env python3

import time
import numpy as np
from tinygrad import Tensor
import torch

# helper function to measure execution time
def benchmark_fn(fn, *args, n_iter=100, warmup=10):
    # warmup
    for _ in range(warmup):
        result = fn(*args)
        if hasattr(result, 'realize'):
            result.realize()
        elif hasattr(result, 'cpu'):
            result.cpu()
    
    # benchmark
    start_time = time.time()
    for _ in range(n_iter):
        result = fn(*args)
        if hasattr(result, 'realize'):
            result.realize()
        elif hasattr(result, 'cpu'):
            result.cpu()
    end_time = time.time()
    
    return (end_time - start_time) * 1000 / n_iter  # ms per iteration

# calculate C-contiguous strides for a given shape
def calc_strides(shape):
    if not shape:
        return tuple()
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))

# benchmark configurations
shapes = [
    (1000,),
    (100, 100),
    (10, 10, 10),
]

def run_tinygrad_as_strided(x, size, stride, offset=0):
    return x.as_strided(size, stride, offset)

def run_torch_as_strided(x, size, stride, offset=0):
    return torch.as_strided(x, size, stride, offset)

if __name__ == "__main__":
    print("Benchmarking as_strided: tinygrad vs torch (forward only)")
    print("-" * 60)
    print(f"{'Shape':<12} {'Size':<12} {'Stride':<12} {'TinyGrad':<8} {'PyTorch':<8} {'Ratio':<6}")
    print("-" * 60)
    
    # define test cases with preset sizes and strides
    test_cases = [
        # description, input_shape, size, stride
        ("1D slice", (1000,), (500,), (1,)),
        ("2D slice", (100, 100), (50, 100), (100, 1)),
        ("2D transpose", (100, 100), (100, 100), (1, 100)),
        ("3D slice", (10, 10, 10), (5, 10, 10), (100, 10, 1)),
        ("3D transpose last dims", (10, 10, 10), (10, 10, 10), (100, 1, 10)),
        ("3D permute all dims", (10, 10, 10), (10, 10, 10), (1, 10, 100)),
    ]
    
    # larger test cases
    large_test_cases = [
        # description, input_shape, size, stride
        ("Large 1D", (100000,), (50000,), (1,)),
        ("Large 2D", (1000, 1000), (500, 1000), (1000, 1)),
        ("Large 3D", (100, 100, 100), (50, 100, 100), (10000, 100, 1)),
        ("Large 4D", (20, 20, 20, 20), (10, 20, 20, 20), (8000, 400, 20, 1)),
    ]
    
    # complex use cases - more realistic operations that would use as_strided
    # corrected to ensure indices don't go out of bounds
    complex_test_cases = [
        # description, input_shape, size, stride, offset
        ("Batch extract", (32, 64, 224, 224), (16, 32, 112, 112), (64*224*224, 224*224, 2*224, 2), 0),
        ("Sliding window", (1, 3, 32, 32), (1, 3, 16, 16, 3, 3), (3*32*32, 32*32, 32, 1, 32, 1), 0),
        ("Channel interleave", (2, 3, 10, 10), (2, 10, 10, 3), (300, 3, 30, 1), 0),
    ]

    for _, shape, size, stride in test_cases:
        # create data
        data = np.random.randn(*shape).astype(np.float32)
        
        # create tensors
        x_tiny = Tensor(data)
        x_torch = torch.tensor(data)
        
        # benchmark
        tiny_time = benchmark_fn(run_tinygrad_as_strided, x_tiny, size, stride)
        torch_time = benchmark_fn(run_torch_as_strided, x_torch, size, stride)
        
        # ratio (>1 means torch is faster, <1 means tinygrad is faster)
        ratio = tiny_time / torch_time if torch_time > 0 else float('inf')
        
        # format shape/size/stride to fit in narrow columns
        shape_str = str(shape)
        if len(shape_str) > 11: shape_str = shape_str[:9] + ".."
        
        size_str = str(size)
        if len(size_str) > 11: size_str = size_str[:9] + ".."
        
        stride_str = str(stride)
        if len(stride_str) > 11: stride_str = stride_str[:9] + ".."
        
        print(f"{shape_str:<12} {size_str:<12} {stride_str:<12} {tiny_time:.2f}ms {torch_time:.2f}ms {ratio:.1f}x")
    
    print("\n--- Larger Tensors ---")
    print("-" * 60)
    print(f"{'Shape':<12} {'Size':<12} {'Stride':<12} {'TinyGrad':<8} {'PyTorch':<8} {'Ratio':<6}")
    print("-" * 60)
    
    for _, shape, size, stride in large_test_cases:
        # create data - using lower n_iter for large tensors
        data = np.random.randn(*shape).astype(np.float32)
        
        # create tensors
        x_tiny = Tensor(data)
        x_torch = torch.tensor(data)
        
        # benchmark with fewer iterations for large tensors
        tiny_time = benchmark_fn(run_tinygrad_as_strided, x_tiny, size, stride, n_iter=10, warmup=2)
        torch_time = benchmark_fn(run_torch_as_strided, x_torch, size, stride, n_iter=10, warmup=2)
        
        # ratio
        ratio = tiny_time / torch_time if torch_time > 0 else float('inf')
        
        # format shape/size/stride to fit in narrow columns
        shape_str = str(shape)
        if len(shape_str) > 11: shape_str = shape_str[:9] + ".."
        
        size_str = str(size)
        if len(size_str) > 11: size_str = size_str[:9] + ".."
        
        stride_str = str(stride)
        if len(stride_str) > 11: stride_str = stride_str[:9] + ".."
        
        print(f"{shape_str:<12} {size_str:<12} {stride_str:<12} {tiny_time:.2f}ms {torch_time:.2f}ms {ratio:.1f}x")
    
    print("\n--- Complex Use Cases ---")  
    print("-" * 60)
    print(f"{'Shape':<12} {'Size':<12} {'Stride':<12} {'TinyGrad':<8} {'PyTorch':<8} {'Ratio':<6}")
    print("-" * 60)
    
    # verify bounds for each complex test case before running
    for _, shape, size, stride, offset in complex_test_cases:
        try:
            # create data
            data = np.random.randn(*shape).astype(np.float32)
            
            # create tensors
            x_tiny = Tensor(data)
            x_torch = torch.tensor(data)
            
            # Test if the strided view is valid (will throw if invalid)
            test_tiny = x_tiny.as_strided(size, stride, offset)
            test_torch = torch.as_strided(x_torch, size, stride, offset)
            
            # benchmark
            tiny_time = benchmark_fn(lambda x, s, st, o: run_tinygrad_as_strided(x, s, st, o), 
                                    x_tiny, size, stride, offset, n_iter=20, warmup=5)
            torch_time = benchmark_fn(lambda x, s, st, o: run_torch_as_strided(x, s, st, o), 
                                    x_torch, size, stride, offset, n_iter=20, warmup=5)
            
            # ratio
            ratio = tiny_time / torch_time if torch_time > 0 else float('inf')
            
            # format shape/size/stride to fit in narrow columns
            shape_str = str(shape)
            if len(shape_str) > 11: shape_str = shape_str[:9] + ".."
            
            size_str = str(size)
            if len(size_str) > 11: size_str = size_str[:9] + ".."
            
            stride_str = str(stride)
            if len(stride_str) > 11: stride_str = stride_str[:9] + ".."
            
            print(f"{shape_str:<12} {size_str:<12} {stride_str:<12} {tiny_time:.2f}ms {torch_time:.2f}ms {ratio:.1f}x")
        except Exception as e:
            # format shape/size/stride to fit in narrow columns
            shape_str = str(shape)
            if len(shape_str) > 11: shape_str = shape_str[:9] + ".."
            
            size_str = str(size)
            if len(size_str) > 11: size_str = size_str[:9] + ".."
            
            stride_str = str(stride)
            if len(stride_str) > 11: stride_str = stride_str[:9] + ".."
            
            print(f"{shape_str:<12} {size_str:<12} {stride_str:<12} ERROR: {str(e)[:30]}")
    
    print("-" * 60)
    print("Note: Ratio > 1 means PyTorch is faster") 