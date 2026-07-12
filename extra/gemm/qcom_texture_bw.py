#!/usr/bin/env python3
"""Adreno 630 texture/isam bandwidth benchmark.

Counts logical half4 image bytes issued by hand-assembled `isam.1d` loads. The
kernel issues a configurable number of B-image half4 loads per thread per K step,
then a single `(sy)` MAD waits for all pending texture results.
"""
import argparse, ctypes

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import *
from extra.gemm.ir3asm import _hreg
from extra.gemm.qcom_intensity_gemm import M, N, K, K4, make_donor_src, prologue_4x2, store_output


def make_bufs(dev):
    a = Buffer(dev.device, (K//4)*M*4, dtypes.half, preallocate=True)
    b = Buffer(dev.device, (N//4)*K*4, dtypes.half, preallocate=True)
    c = Buffer(dev.device, M*N, dtypes.half, preallocate=True)
    ctypes.memset(int(a._buf.va_addr), 0, a.nbytes)
    ctypes.memset(int(b._buf.va_addr), 0, b.nbytes)
    ctypes.memset(int(c._buf.va_addr), 0, c.nbytes)
    return a, b, c


def build_texture_shader(dev, threads, loads):
    if loads % 4 != 0 or not (4 <= loads <= 32):
        raise ValueError('loads must be one of 4, 8, ..., 32')
    nblocks = loads // 4
    instrs = prologue_4x2(dev, threads)

    # Adjust prologue's 1-col gid.x*32 base to a nblocks-col gid.x*(32*nblocks) base.
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2)]
    for _ in range(nblocks - 1): instrs.append(ADD_S_REG('r7.y', 'r7.y', 'r6.w'))

    acc = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc, 0), MOV_H(acc + 1, acc, rpt=2)]
    for base in range(acc + 4, acc + loads * 4, 4): instrs.append(MOV_H(base, acc, rpt=3))

    loop_start = len(instrs)
    instrs += [
        ADD_S('r4.z', 'r6.y', -3), ADD_S('r5.x', 'r6.y', -2),
        ADD_S('r5.z', 'r6.y', -1), MOV_F32('r6.x', 'r6.y'),
    ]

    k_regs = ['r4.z', 'r5.x', 'r5.z', 'r6.x']
    for block in range(nblocks):
        instrs.append(ADD_S('r7.z', 'r7.y', block * 32))
        for kk, k_reg in enumerate(k_regs):
            instrs += [MOV_F32('r7.w', k_reg), ISAM_F16(acc + (block * 4 + kk) * 4, 'r7.z', 1)]

    # Force all texture loads to complete. This is intentionally tiny compared to the load stream.
    instrs.append(MAD_F16(acc, acc, acc, acc, rpt=3, sy=True, r=True))
    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        ADD_S('r6.y', 'r6.y', 4),
        CMPS_S_EQ('r6.z', K4 - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    store_output(instrs, 'r7.x', 'r7.y', acc)
    instrs.append(END())
    return assemble(instrs), loop_end - loop_start


def run(args):
    dev = Device['QCOM']
    nblocks = args.loads // 4
    envelope, img_off, img_sz, reg_off = get_envelope(dev, make_donor_src(max(2, nblocks), args.threads))
    shader, loop_instrs = build_texture_shader(dev, args.threads, args.loads)
    hregs = (_hreg('hr16.x') + args.loads * 4 + 3) // 4
    lib = inject(envelope, img_off, img_sz, reg_off, shader, fregs=8, hregs=hregs)

    asm = disasm(shader)
    waves = 12288 // (hregs * args.threads)
    print('loads=%d hregs=%d waves=%d loop_instrs=%d shader_instrs=%d isam=%d sy=%d' % (
        args.loads, hregs, waves, loop_instrs, len(shader)//8, asm.count('isam'), asm.count('(sy)')))
    if args.disasm: print(asm)

    a_img, b_img = dtypes.imageh((M, K//4)), dtypes.imageh((K, N//4))
    a, b, c = make_bufs(dev)
    prg = dev.runtime('gemm_h', lib, [[(0, a_img)], [(1, b_img)], [(2, dtypes.half.ptr())]])
    tile_m = (args.threads // 32) * 4
    gs, ls = (max(1, N // (128 * nblocks)), M // tile_m, 1), (args.threads, 1, 1)
    for _ in range(5): prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
    times = []
    for _ in range(args.iters):
        t = prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)
    best = min(times)
    total_threads = gs[0] * gs[1] * args.threads
    logical_bytes = total_threads * K4 * args.loads * 8
    print('%.1f GB/s  (%.3f ms, %.1f MiB logical)' % (logical_bytes / best / 1e9, best * 1e3, logical_bytes / (1024 * 1024)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loads', type=int, choices=tuple(range(4, 33, 4)), default=8)
    parser.add_argument('--threads', type=int, choices=(64, 128), default=128)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--disasm', action='store_true')
    run(parser.parse_args())
