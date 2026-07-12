#!/usr/bin/env python3
"""FP16 GEMM benchmark for Adreno 630 with binary patching.

Achieves ~190 GFLOPS via:
1. 4 rows x 4 cols per thread IMAGE kernel (read_imageh)
2. Binary patching to strip redundant (sy) sync flags
3. Binary patching to convert scalar MADs to (rpt3)mad.f16

Usage:
  DEV=QCOM python3 extra/gemm/qcom_gemm.py
  DEV=QCOM python3 extra/gemm/qcom_gemm.py --m 512 --n 512 --k 512
"""
import struct, ctypes, math, argparse
from tinygrad import Device, dtypes
from tinygrad.device import Buffer

def ri(buf, l):
    o = l * 8
    return struct.unpack_from('<I', buf, o+4)[0], struct.unpack_from('<I', buf, o)[0]

def wi(buf, l, h, lo):
    o = l * 8
    struct.pack_into('<I', buf, o, lo)
    struct.pack_into('<I', buf, o+4, h)

def patch_kernel(lib):
    """Strip redundant (sy) and convert eligible MAD groups to (rpt3)."""
    lib = bytearray(lib)
    io = struct.unpack_from('<I', lib, 0xc0)[0]
    isz = struct.unpack_from('<I', lib, 0x100)[0]
    s = bytearray(lib[io:io+isz])
    t = isz // 8

    # Strip all (sy) except the first on mad.f16 instructions
    first_sy = False
    for i in range(t):
        h, lo = ri(s, i)
        if (h >> 24) in (0x63, 0x73) and ((h >> 24) & 0xF) == 3 and (h >> 28) == 7:
            if first_sy:
                wi(s, i, (h & 0x0FFFFFFF) | 0x60000000, lo)
            else:
                first_sy = True

    # Convert groups of 4 scalar MADs to (rpt3)
    i = 0
    while i < t - 3:
        h0, l0 = ri(s, i)
        if not ((h0 >> 24) in (0x63, 0x73) and ((h0 >> 24) & 0xF) == 3):
            i += 1; continue
        d0, r0 = h0 & 0xFF, (h0 >> 8) & 0x7F
        s1 = l0 & 0xFF; s3 = (l0 >> 16) & 0xFF
        s2 = ((h0 >> 16) & 0xFF) * 2 + (((h0 >> 8) & 0xFF) >> 7)
        if r0 > 0 or d0 != s3:
            i += 1; continue
        ok = True
        for j in range(1, 4):
            hj, lj = ri(s, i+j)
            if not ((hj >> 24) in (0x63, 0x73) and ((hj >> 24) & 0xF) == 3):
                ok = False; break
            dj = hj & 0xFF; rj = (hj >> 8) & 0x7F
            s1j = lj & 0xFF; s3j = (lj >> 16) & 0xFF
            s2j = ((hj >> 16) & 0xFF) * 2 + (((hj >> 8) & 0xFF) >> 7)
            if rj != 0 or s1j != s1 or dj != d0+j or s2j != s2+j or s3j != d0+j:
                ok = False; break
        if ok:
            rb = ((h0 >> 8) & 0x80) | 3
            wi(s, i, (h0 & 0xFFFF00FF) | (rb << 8), l0 | 0x20000000)
            for j in range(1, 4):
                wi(s, i+j, 0, 0)
            i += 4
        else:
            i += 1

    # Merge (rpt1)+(rpt1) into (rpt3)
    for i in range(t - 1):
        h0, l0 = ri(s, i); h1, l1 = ri(s, i+1)
        if h0 == 0 or h1 == 0: continue
        if not ((h0 >> 24) in (0x63, 0x73) and ((h0 >> 24) & 0xF) == 3): continue
        if not ((h1 >> 24) in (0x63, 0x73) and ((h1 >> 24) & 0xF) == 3): continue
        if (h0 >> 8) & 0x7F != 1 or (h1 >> 8) & 0x7F != 1: continue
        d0, d1 = h0 & 0xFF, h1 & 0xFF
        s10, s11 = l0 & 0xFF, l1 & 0xFF
        s20 = ((h0 >> 16) & 0xFF) * 2 + (((h0 >> 8) & 0xFF) >> 7)
        s21 = ((h1 >> 16) & 0xFF) * 2 + (((h1 >> 8) & 0xFF) >> 7)
        if s10 != s11 or d1 != d0 + 2 or s21 != s20 + 2: continue
        rb = ((h0 >> 8) & 0x80) | 3
        wi(s, i, (h0 & 0xFFFF00FF) | (rb << 8), l0)
        wi(s, i+1, 0, 0)

    lib[io:io+isz] = s
    return bytes(lib)


def make_gemm_src(M, N, K, nrows=4):
    """Generate 4-row FP16 IMAGE GEMM kernel source."""
    K4 = K // 4
    TM = (128 // 32) * nrows  # 4 * nrows
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(128,1,1)))\n'
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global half *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tn=lid&31;\n'
    src += 'int row=get_group_id(1)*%d+tm*%d;int col4=get_group_id(0)*32+tn;\n' % (TM, nrows)
    for r in range(nrows):
        src += 'half4 r%dc0=(half4)(0),r%dc1=(half4)(0),r%dc2=(half4)(0),r%dc3=(half4)(0);\n' % (r,r,r,r)
    src += 'for(int k4=0;k4<%d;k4++){\n' % K4
    for r in range(nrows):
        src += 'half4 a%d=read_imageh(A,smp,(int2)(k4,row+%d));\n' % (r, r)
    for b in range(4):
        src += 'half4 b%d=read_imageh(B,smp,(int2)(col4,k4*4+%d));\n' % (b, b)
    for r in range(nrows):
        src += 'r%dc0+=a%d.xxxx*b0;r%dc1+=a%d.yyyy*b1;r%dc2+=a%d.zzzz*b2;r%dc3+=a%d.wwww*b3;\n' % (r,r,r,r,r,r,r,r)
    src += '}\n'
    for r in range(nrows):
        src += 'vstore4(r%dc0+r%dc1+r%dc2+r%dc3,0,C+(row+%d)*%d+col4*4);\n' % (r,r,r,r,r,N)
    src += '}\n'
    return src, TM


def run_gemm(args):
    dev = Device['QCOM']
    M, N, K = args.m, args.n, args.k
    print("device=%s M=%d N=%d K=%d" % (dev.device, M, N, K))

    src, TM = make_gemm_src(M, N, K, nrows=4)
    lib = patch_kernel(dev.compiler.compile_cached(src))

    a_img = dtypes.imageh((M, K//4))
    b_img = dtypes.imageh((K, N//4))
    a_buf = Buffer(dev.device, (K//4)*M*4, dtypes.half, preallocate=True)
    b_buf = Buffer(dev.device, (N//4)*K*4, dtypes.half, preallocate=True)
    c_buf = Buffer(dev.device, M*N, dtypes.half, preallocate=True)
    ctypes.memset(int(a_buf._buf.va_addr), 0, a_buf.nbytes)
    ctypes.memset(int(b_buf._buf.va_addr), 0, b_buf.nbytes)

    prg = dev.runtime('gemm_h', lib, [[(0, a_img)], [(1, b_img)], [(2, dtypes.half.ptr())]])
    gs = (N // 128, M // TM, 1)
    ls = (128, 1, 1)

    for _ in range(5):
        prg(a_buf._buf, b_buf._buf, c_buf._buf, global_size=gs, local_size=ls, wait=True)

    times = []
    for _ in range(args.iters):
        t = prg(a_buf._buf, b_buf._buf, c_buf._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)

    if times:
        best = min(times)
        gflops = 2 * M * N * K / best / 1e9
        print("%.1f GFLOPS  (%.1f ms)  %.0f%% of 690 peak" % (gflops, best*1e3, gflops/690*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=20)
    run_gemm(parser.parse_args())
