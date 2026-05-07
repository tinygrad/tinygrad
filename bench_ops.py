#!/usr/bin/env python3
"""Benchmark numpy-algorithms ported to tinygrad vs numpy."""
import time
import numpy as np
from tinygrad import Tensor, dtypes

def bench(name, fn_tg, fn_np, warmup=2, runs=5):
    """Benchmark tinygrad vs numpy, print results."""
    # Warmup
    for _ in range(warmup):
        fn_tg()
        fn_np()

    # Tinygrad
    times_tg = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result_tg = fn_tg()
        if hasattr(result_tg, 'numpy'):
            result_tg.numpy()  # force realize
        times_tg.append(time.perf_counter() - t0)

    # Numpy
    times_np = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn_np()
        times_np.append(time.perf_counter() - t0)

    tg_med = sorted(times_tg)[len(times_tg)//2]
    np_med = sorted(times_np)[len(times_np)//2]
    ratio = tg_med / np_med if np_med > 0 else float('inf')
    print(f"{name:<30} tinygrad: {tg_med*1000:8.2f}ms  numpy: {np_med*1000:8.2f}ms  ratio: {ratio:6.1f}x")
    return tg_med, np_med


# ─── Test cases ───

def bench_batch_normalize():
    N, D = 256, 512
    X_np = np.random.randn(N, D).astype(np.float32)
    gamma_np = np.random.randn(D).astype(np.float32)
    beta_np = np.random.randn(D).astype(np.float32)

    X_tg = Tensor(X_np)
    gamma_tg = Tensor(gamma_np)
    beta_tg = Tensor(beta_np)

    def tg():
        mean = X_tg.mean(axis=0)
        var = ((X_tg - mean) ** 2).mean(axis=0)
        norm = (X_tg - mean) / (var + 1e-5).sqrt()
        return (gamma_tg * norm + beta_tg).realize()

    def np_fn():
        mean = X_np.mean(axis=0)
        var = X_np.var(axis=0)
        norm = (X_np - mean) / np.sqrt(var + 1e-5)
        return gamma_np * norm + beta_np

    bench("batch_normalize", tg, np_fn)


def bench_softmax_cross_entropy():
    N, C = 256, 100
    logits_np = np.random.randn(N, C).astype(np.float32)
    targets_np = np.random.randint(0, C, size=N)

    logits_tg = Tensor(logits_np)

    def tg():
        mx = logits_tg.max(axis=1, keepdim=True)
        shifted = logits_tg - mx
        lse = shifted.exp().sum(axis=1).log()
        return lse.mean().realize()

    def np_fn():
        mx = logits_np.max(axis=1, keepdims=True)
        shifted = logits_np - mx
        lse = np.log(np.exp(shifted).sum(axis=1))
        return lse.mean()

    bench("softmax_cross_entropy", tg, np_fn)


def bench_matmul():
    N = 512
    A_np = np.random.randn(N, N).astype(np.float32)
    B_np = np.random.randn(N, N).astype(np.float32)

    A_tg = Tensor(A_np)
    B_tg = Tensor(B_np)

    def tg():
        return (A_tg @ B_tg).realize()

    def np_fn():
        return A_np @ B_np

    bench("matmul 512x512", tg, np_fn)


def bench_conv2d():
    H, W = 64, 64
    kH, kW = 5, 5
    X_np = np.random.randn(1, 1, H, W).astype(np.float32)
    K_np = np.random.randn(1, 1, kH, kW).astype(np.float32)

    X_tg = Tensor(X_np)
    K_tg = Tensor(K_np)

    def tg():
        return X_tg.conv2d(K_tg).realize()

    def np_fn():
        # im2col conv2d
        x = X_np[0, 0]
        k = K_np[0, 0]
        out_H = H - kH + 1
        out_W = W - kW + 1
        i0 = np.repeat(np.arange(kH), kW)
        j0 = np.tile(np.arange(kW), kH)
        i1 = np.arange(out_H)
        j1 = np.arange(out_W)
        rows = i0[:, None, None] + i1[None, :, None]
        cols = j0[:, None, None] + j1[None, None, :]
        col = x[rows, cols]
        return np.tensordot(k.ravel(), col, axes=([0], [0]))

    bench("conv2d 64x64 k=5", tg, np_fn)


def bench_reduction_chain():
    """Chain of reductions — tests scheduler fusion."""
    N = 10000
    X_np = np.random.randn(N, N).astype(np.float32)
    X_tg = Tensor(X_np)

    def tg():
        # sum rows, then normalize, then sum again
        row_sums = X_tg.sum(axis=1)
        normed = row_sums / row_sums.max()
        return normed.sum().realize()

    def np_fn():
        row_sums = X_np.sum(axis=1)
        normed = row_sums / row_sums.max()
        return normed.sum()

    bench("reduction_chain 10kx10k", tg, np_fn)


def bench_elementwise_chain():
    """Long chain of elementwise ops — tests fusion."""
    N = 1_000_000
    X_np = np.random.randn(N).astype(np.float32)
    X_tg = Tensor(X_np)

    def tg():
        x = X_tg
        x = x.sin()
        x = x.cos()
        x = x.exp()
        x = x.log()
        x = x * 2.0
        x = x + 1.0
        x = x.relu()
        x = x.sigmoid()
        return x.realize()

    def np_fn():
        x = X_np
        x = np.sin(x)
        x = np.cos(x)
        x = np.exp(x)
        x = np.log(x)
        x = x * 2.0
        x = x + 1.0
        x = np.maximum(x, 0)
        x = 1 / (1 + np.exp(-x))
        return x

    bench("elementwise_chain 1M", tg, np_fn)


def bench_gather_scatter():
    """Indexing / gather pattern."""
    N, D = 10000, 256
    X_np = np.random.randn(N, D).astype(np.float32)
    idx_np = np.random.randint(0, N, size=1000)

    X_tg = Tensor(X_np)
    idx_tg = Tensor(idx_np)

    def tg():
        return X_tg[idx_tg].realize()

    def np_fn():
        return X_np[idx_np]

    bench("gather 10k x 256", tg, np_fn)


def bench_broadcasting():
    """Heavy broadcasting pattern."""
    M, N = 1000, 1000
    A_np = np.random.randn(M, 1).astype(np.float32)
    B_np = np.random.randn(1, N).astype(np.float32)

    A_tg = Tensor(A_np)
    B_tg = Tensor(B_np)

    def tg():
        return (A_tg * B_tg + A_tg - B_tg).sum().realize()

    def np_fn():
        return (A_np * B_np + A_np - B_np).sum()

    bench("broadcast 1kx1k", tg, np_fn)


if __name__ == "__main__":
    print(f"{'Benchmark':<30} {'tinygrad':>16}  {'numpy':>14}  {'slowdown':>10}")
    print("-" * 78)
    bench_batch_normalize()
    bench_softmax_cross_entropy()
    bench_matmul()
    bench_conv2d()
    bench_elementwise_chain()
    bench_reduction_chain()
    bench_gather_scatter()
    bench_broadcasting()
