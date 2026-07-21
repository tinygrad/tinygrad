#!/usr/bin/env python3
import argparse, ctypes

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.qcom_8x4_gemm import M, N, K, check_all_ones, fill_half, make_donor_src8


def make_bufs(dev):
    a = Buffer(dev.device, (K//4)*M*4, dtypes.half, preallocate=True)
    b = Buffer(dev.device, (N//4)*K*4, dtypes.half, preallocate=True)
    c = Buffer(dev.device, M*N, dtypes.half, preallocate=True)
    if hasattr(a._buf, 'va_addr'):
        ctypes.memset(int(a._buf.va_addr), 0, a.nbytes)
        ctypes.memset(int(b._buf.va_addr), 0, b.nbytes)
        ctypes.memset(int(c._buf.va_addr), 0, c.nbytes)
    return a, b, c


def run(args):
    dev = Device[Device.DEFAULT]
    src = make_donor_src8(args.ncols, args.threads)
    lib = dev.compiler.compile_cached(src)
    a_img, b_img = dtypes.imageh((M, K//4)), dtypes.imageh((K, N//4))
    a, b, c = make_bufs(dev)
    fill_half(a, 0x3c00)
    fill_half(b, 0x3c00)
    prg = dev.runtime('gemm_h', lib, [[(0, a_img)], [(1, b_img)], [(2, dtypes.half.ptr())]])
    tile_m = (args.threads // 32) * 8
    gs, ls = (N // (128 * args.ncols), M // tile_m, 1), (args.threads, 1, 1)
    prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
    if not check_all_ones(c): return
    for _ in range(args.warmup): prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
    times = []
    for _ in range(args.iters):
        t = prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)
    best = min(times)
    print('compiler8 ncols=%d scalar_tile=8x%d threads=%d %.1f GFLOPS (%.3f ms)' % (args.ncols, args.ncols * 4, args.threads, 2*M*N*K / best / 1e9, best * 1e3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncols', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--threads', type=int, choices=(128, 256), default=128)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=20)
    run(parser.parse_args())
