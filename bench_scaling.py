#!/usr/bin/env python3
"""Find ops where tinygrad scales worse than numpy even at large sizes."""
import numpy as np, time
from tinygrad import Tensor

def bench(fn, warmup=2, runs=5):
    for _ in range(warmup):
        r = fn()
        if hasattr(r, 'numpy'): r.numpy()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = fn()
        if hasattr(r, 'numpy'): r.numpy()
        times.append(time.perf_counter() - t0)
    return sorted(times)[runs//2]

def run_scaling(name, make_tg, make_np, sizes):
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    print(f"{'Size':<25} {'tinygrad':>10} {'numpy':>10} {'ratio':>8}")
    print("-" * 58)
    for label, args in sizes:
        tg_fn = make_tg(*args)
        np_fn = make_np(*args)
        tg_t = bench(tg_fn) * 1000
        np_t = bench(np_fn) * 1000
        ratio = tg_t / np_t if np_t > 0.001 else float('inf')
        winner = "TG" if ratio < 1 else ""
        print(f"{label:<25} {tg_t:>8.2f}ms {np_t:>8.2f}ms {ratio:>7.1f}x {winner}")


# ─── 1. Matmul scaling ───
def matmul_scaling():
    def make_tg(N):
        A, B = Tensor.randn(N, N), Tensor.randn(N, N)
        def fn(): return (A @ B).realize()
        return fn
    def make_np(N):
        A, B = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32)
        def fn(): return A @ B
        return fn
    run_scaling("Matmul (NxN @ NxN)", make_tg, make_np, [
        ("32x32", (32,)), ("128x128", (128,)), ("512x512", (512,)),
        ("1024x1024", (1024,)), ("2048x2048", (2048,)), ("4096x4096", (4096,)),
    ])

# ─── 2. Reduction scaling ───
def reduction_scaling():
    def make_tg(N):
        X = Tensor.randn(N)
        def fn(): return X.sum().realize()
        return fn
    def make_np(N):
        X = np.random.randn(N).astype(np.float32)
        def fn(): return X.sum()
        return fn
    run_scaling("Sum reduction (1D)", make_tg, make_np, [
        ("1K", (1000,)), ("10K", (10000,)), ("100K", (100000,)),
        ("1M", (1000000,)), ("10M", (10000000,)), ("100M", (100000000,)),
    ])

# ─── 3. Multi-axis reduction ───
def multiaxis_reduction_scaling():
    def make_tg(M, N):
        X = Tensor.randn(M, N)
        def fn(): return X.sum(axis=1).realize()
        return fn
    def make_np(M, N):
        X = np.random.randn(M, N).astype(np.float32)
        def fn(): return X.sum(axis=1)
        return fn
    run_scaling("Row-wise sum (MxN).sum(axis=1)", make_tg, make_np, [
        ("100x100", (100, 100)), ("1000x100", (1000, 100)),
        ("100x10000", (100, 10000)), ("10000x1000", (10000, 1000)),
        ("1000x10000", (1000, 10000)), ("10000x10000", (10000, 10000)),
    ])

# ─── 4. Chained elementwise ───
def elementwise_scaling():
    def make_tg(N):
        X = Tensor.randn(N)
        def fn(): return (X.sin().cos().exp().log() * 2 + 1).realize()
        return fn
    def make_np(N):
        X = np.random.randn(N).astype(np.float32)
        def fn():
            x = np.sin(X); x = np.cos(x); x = np.exp(x); x = np.log(x)
            return x * 2 + 1
        return fn
    run_scaling("Elementwise chain sin→cos→exp→log→mul→add", make_tg, make_np, [
        ("1K", (1000,)), ("10K", (10000,)), ("100K", (100000,)),
        ("1M", (1000000,)), ("10M", (10000000,)),
    ])

# ─── 5. Broadcasting ───
def broadcast_scaling():
    def make_tg(M, N):
        A = Tensor.randn(M, 1)
        B = Tensor.randn(1, N)
        def fn(): return (A * B).sum().realize()
        return fn
    def make_np(M, N):
        A = np.random.randn(M, 1).astype(np.float32)
        B = np.random.randn(1, N).astype(np.float32)
        def fn(): return (A * B).sum()
        return fn
    run_scaling("Broadcast outer product sum", make_tg, make_np, [
        ("100x100", (100, 100)), ("1000x1000", (1000, 1000)),
        ("5000x5000", (5000, 5000)), ("10000x10000", (10000, 10000)),
    ])

# ─── 6. Conv2d scaling ───
def conv2d_scaling():
    def make_tg(H, K):
        X = Tensor.randn(1, 1, H, H)
        W = Tensor.randn(1, 1, K, K)
        def fn(): return X.conv2d(W).realize()
        return fn
    def make_np(H, K):
        X = np.random.randn(H, H).astype(np.float32)
        W = np.random.randn(K, K).astype(np.float32)
        def fn():
            oH, oW = H-K+1, H-K+1
            i0 = np.repeat(np.arange(K), K)
            j0 = np.tile(np.arange(K), K)
            i1, j1 = np.arange(oH), np.arange(oW)
            rows = i0[:, None, None] + i1[None, :, None]
            cols = j0[:, None, None] + j1[None, None, :]
            return np.tensordot(W.ravel(), X[rows, cols], axes=([0], [0]))
        return fn
    run_scaling("Conv2d (1x1xHxH, k=K)", make_tg, make_np, [
        ("32x32 k=3", (32, 3)), ("64x64 k=3", (64, 3)), ("128x128 k=3", (128, 3)),
        ("256x256 k=3", (256, 3)), ("512x512 k=3", (512, 3)),
        ("64x64 k=7", (64, 7)), ("128x128 k=7", (128, 7)),
    ])

# ─── 7. Softmax scaling ───
def softmax_scaling():
    def make_tg(N, C):
        X = Tensor.randn(N, C)
        def fn():
            mx = X.max(axis=1, keepdim=True)
            shifted = X - mx
            e = shifted.exp()
            return (e / e.sum(axis=1, keepdim=True)).realize()
        return fn
    def make_np(N, C):
        X = np.random.randn(N, C).astype(np.float32)
        def fn():
            mx = X.max(axis=1, keepdims=True)
            shifted = X - mx
            e = np.exp(shifted)
            return e / e.sum(axis=1, keepdims=True)
        return fn
    run_scaling("Softmax (NxC)", make_tg, make_np, [
        ("10x100", (10, 100)), ("100x100", (100, 100)),
        ("1000x100", (1000, 100)), ("1000x1000", (1000, 1000)),
        ("10000x1000", (10000, 1000)), ("10000x10000", (10000, 10000)),
    ])

# ─── 8. Where/select scaling ───
def where_scaling():
    def make_tg(N):
        X = Tensor.randn(N)
        Y = Tensor.randn(N)
        cond = X > 0
        def fn(): return cond.where(X, Y).realize()
        return fn
    def make_np(N):
        X = np.random.randn(N).astype(np.float32)
        Y = np.random.randn(N).astype(np.float32)
        cond = X > 0
        def fn(): return np.where(cond, X, Y)
        return fn
    run_scaling("Where/select", make_tg, make_np, [
        ("1K", (1000,)), ("10K", (10000,)), ("100K", (100000,)),
        ("1M", (1000000,)), ("10M", (10000000,)),
    ])

# ─── 9. Concatenation scaling ───
def concat_scaling():
    def make_tg(N, K):
        tensors = [Tensor.randn(N) for _ in range(K)]
        def fn(): return Tensor.cat(*tensors).realize()
        return fn
    def make_np(N, K):
        arrays = [np.random.randn(N).astype(np.float32) for _ in range(K)]
        def fn(): return np.concatenate(arrays)
        return fn
    run_scaling("Concatenation (K arrays of N)", make_tg, make_np, [
        ("1K x 2", (1000, 2)), ("1K x 10", (1000, 10)), ("1K x 100", (1000, 100)),
        ("100K x 2", (100000, 2)), ("100K x 10", (100000, 10)),
        ("1M x 2", (1000000, 2)), ("1M x 10", (1000000, 10)),
    ])

# ─── 10. Transpose + matmul ───
def transpose_matmul_scaling():
    def make_tg(N):
        A = Tensor.randn(N, N)
        def fn(): return (A.T @ A).realize()
        return fn
    def make_np(N):
        A = np.random.randn(N, N).astype(np.float32)
        def fn(): return A.T @ A
        return fn
    run_scaling("A^T @ A (gram matrix)", make_tg, make_np, [
        ("64x64", (64,)), ("256x256", (256,)), ("512x512", (512,)),
        ("1024x1024", (1024,)), ("2048x2048", (2048,)),
    ])


if __name__ == "__main__":
    matmul_scaling()
    reduction_scaling()
    multiaxis_reduction_scaling()
    elementwise_scaling()
    broadcast_scaling()
    conv2d_scaling()
    softmax_scaling()
    where_scaling()
    concat_scaling()
    transpose_matmul_scaling()
