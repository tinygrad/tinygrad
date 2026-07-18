#!/usr/bin/env python3
"""Adreno 630 FP16 MAD throughput benchmark."""
import argparse, ctypes, struct

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import *
from extra.gemm.ir3asm import _hreg
from extra.gemm.qcom_intensity_gemm import M, N, K4, make_donor_src, prologue_4x2, store_output


def make_bufs(dev):
    a = Buffer(dev.device, (K4)*M*4, dtypes.half, preallocate=True)
    b = Buffer(dev.device, (N//4)*(K4*4)*4, dtypes.half, preallocate=True)
    c = Buffer(dev.device, M*N, dtypes.half, preallocate=True)
    ctypes.memset(int(a._buf.va_addr), 0, a.nbytes)
    ctypes.memset(int(b._buf.va_addr), 0, b.nbytes)
    ctypes.memset(int(c._buf.va_addr), 0, c.nbytes)
    return a, b, c


def emit_mov_h_block(instrs, start, end, src):
    pos = start
    while pos < end:
        rpt = min(3, end - pos - 1)
        instrs.append(MOV_H(pos, src, rpt=rpt))
        pos += rpt + 1


def build_compiler_pattern_shader(dev, threads, loops, pairs, store):
    if pairs < 2: raise ValueError('compiler-pattern needs at least two x/y MAD pairs; smaller shaders have caused QCOM hangs')
    instrs = prologue_4x2(dev, threads)
    instrs += [MOV_S32('r8.x', 0, sy=True), MOV_H_IMM('hr0.x', 0x3c00)]
    emit_mov_h_block(instrs, 1, _hreg('hr8.x'), 0)

    loop_start = len(instrs)
    for _ in range(pairs):
        # This mirrors the vec16 OpenCL MAD peak lowering: one vector MAD into x,
        # then one vector MAD into y. The split scalar lane avoids clobbering hr0.y.
        instrs += [
            MAD_F16('hr0.z', 'hr0.z', 'hr4.y', 'hr4.y', rpt=3, r=True, r1=True),
            MAD_F16('hr1.z', 'hr1.z', 'hr5.y', 'hr5.y', rpt=3, r=True, r1=True),
            MAD_F16('hr2.z', 'hr2.z', 'hr6.y', 'hr6.y', rpt=3, r=True, r1=True),
            MAD_F16('hr3.z', 'hr3.z', 'hr7.y', 'hr7.y', rpt=2, r=True, r1=True),
            MAD_F16('hr0.x', 'hr0.x', 'hr0.y', 'hr0.y'),
            MAD_F16('hr4.y', 'hr0.z', 'hr4.y', 'hr0.z', rpt=3, r=True, r1=True),
            MAD_F16('hr5.y', 'hr1.z', 'hr5.y', 'hr1.z', rpt=3, r=True, r1=True),
            MAD_F16('hr6.y', 'hr2.z', 'hr6.y', 'hr2.z', rpt=3, r=True, r1=True),
            MAD_F16('hr7.y', 'hr3.z', 'hr7.y', 'hr3.z', rpt=2, r=True, r1=True),
            MAD_F16('hr0.y', 'hr0.x', 'hr0.y', 'hr0.x'),
        ]
    instrs += [
        ADD_S('r8.y', 'r8.x', 1),
        CMPS_S_EQ('r8.x', loops - 1, nop=1),
        MOV_F32('r8.x', 'r8.y'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if store: store_output(instrs, 'r7.x', 'r7.y', 0)
    instrs.append(END())
    return assemble(instrs), loop_end - loop_start, 8, 9, pairs * 32 * 2


def build_alu_shader(dev, threads, groups, rpt, loops, unroll, independent, r1):
    if rpt > 3: raise ValueError('mad.f16 repeat counts above rpt3 encode other flags on A630, not more FP16 lanes')
    if not (1 <= loops <= 256): raise ValueError('loops must be in 1..256; current immediate compare encodes only 8 bits')
    width = rpt + 1
    instrs = prologue_4x2(dev, threads)
    instrs += [
        MOV_S32('r6.z', 0, sy=True),
        MOV_H_IMM('hr0.x', 0x3c00),
        MOV_H_IMM('hr16.x', 0), MOV_H('hr16.y', 'hr16.x', rpt=2),
    ]
    emit_mov_h_block(instrs, 1, max(width, 4), 0)
    emit_mov_h_block(instrs, _hreg('hr4.x'), _hreg('hr4.x') + max(width, 4), 0)
    acc0 = _hreg('hr16.x')
    hregs = (acc0 + groups * width + 3) // 4
    emit_mov_h_block(instrs, acc0 + 4, acc0 + groups * width, acc0)

    loop_start = len(instrs)
    for _ in range(unroll):
        for g in range(groups):
            src1 = (g * width) % max(width, 4)
            src2 = _hreg('hr4.x') + ((g * width) % max(width, 4))
            src3 = src1 if independent else acc0 + g * width
            instrs.append(MAD_F16(acc0 + g * width, src1, src2, src3, rpt=rpt, r=True, r1=r1))
    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        CMPS_S_EQ('r6.z', loops - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    store_output(instrs, 'r7.x', 'r7.y', acc0)
    instrs.append(END())
    return assemble(instrs), loop_end - loop_start, hregs, None, unroll * groups * width * 2


def gemm_check_inputs(rows, ncols):
    a = [[((row * 3 + kk) % 4 + 1) / 8 for kk in range(4)] for row in range(rows)]
    b = [[[[((col * 7 + kk * 3 + lane) % 4 + 1) / 8 for lane in range(4)] for kk in range(4)] for col in range(ncols)]][0]
    return a, b


def half_raw(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def build_gemm_pattern_shader(dev, threads, loops, rows, ncols, unroll, order, bmode, r1, check_pattern=False, store_group=0):
    if loops != 1: raise ValueError('gemm-pattern is a one-shot ALU body benchmark; use --loops 1 so loop-control regs do not clobber A/B sources')
    if rows not in (4, 8): raise ValueError('rows must be 4 or 8')
    if ncols < 1: raise ValueError('ncols must be positive')
    instrs = prologue_4x2(dev, threads)
    instrs += [MOV_S32('r6.z', 0, sy=True)]

    # A lives in hr0..hr(rows-1). B either reuses one 4-texel column group or
    # allocates one 4-texel group per output col4. Accumulators start at hr16 to
    # match the working GEMM kernels and avoid low full-register aliases.
    a_base = 0
    b_base = rows * 4
    b_groups = ncols if bmode == 'percol' else 1
    b_end = b_base + b_groups * 16
    acc0 = max(_hreg('hr16.x'), ((b_end + 3) // 4) * 4)
    if check_pattern:
        check_a, check_b = gemm_check_inputs(rows, ncols)
        for row in range(rows):
            for kk in range(4): instrs.append(MOV_H_IMM(a_base + row * 4 + kk, half_raw(check_a[row][kk])))
        for col in range(b_groups):
            for kk in range(4):
                for lane in range(4): instrs.append(MOV_H_IMM(b_base + col * 16 + kk * 4 + lane, half_raw(check_b[col][kk][lane])))
    else:
        instrs.append(MOV_H_IMM('hr0.x', 0x3c00))
        emit_mov_h_block(instrs, 1, rows * 4, 0)
        emit_mov_h_block(instrs, b_base, b_end, 0)
    for lane in range(acc0, acc0 + rows * ncols * 4): instrs.append(MOV_H_IMM(lane, 0))
    hregs = (max(b_end, acc0 + rows * ncols * 4) + 3) // 4

    loop_start = len(instrs)
    def emit(row, kk, col):
        b_col = col if bmode == 'percol' else 0
        instrs.append(MAD_F16(acc0 + (row * ncols + col) * 4, a_base + row * 4 + kk, b_base + b_col * 16 + kk * 4,
                              acc0 + (row * ncols + col) * 4, rpt=3, r=True, r1=r1))
    for _ in range(unroll):
        if order == 'kk_row_col':
            for kk in range(4):
                for row in range(rows):
                    for col in range(ncols): emit(row, kk, col)
        elif order == 'kk_col_row':
            for kk in range(4):
                for col in range(ncols):
                    for row in range(rows): emit(row, kk, col)
        elif order == 'col_kk_row':
            for col in range(ncols):
                for kk in range(4):
                    for row in range(rows): emit(row, kk, col)
        elif order == 'row_kk_col':
            for row in range(rows):
                for kk in range(4):
                    for col in range(ncols): emit(row, kk, col)
        elif order == 'row_col_kk':
            for row in range(rows):
                for col in range(ncols):
                    for kk in range(4): emit(row, kk, col)
        else: raise ValueError('unknown order %s' % order)
    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        CMPS_S_EQ('r6.z', loops - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if check_pattern:
        # The per-column B register bank aliases the donor prologue's r7 output
        # coordinates. Every lane computes the same diagnostic tile, so use one
        # common output address and bit-check the selected accumulator vector.
        instrs += [MOV_S32('r7.x', 0), MOV_S32('r7.y', 0), NOP(rpt=2)]
    store_output(instrs, 'r7.x', 'r7.y', acc0 + store_group * 4)
    instrs.append(END())
    return assemble(instrs), loop_end - loop_start, hregs, None, unroll * rows * ncols * 4 * 4 * 2


def run(args):
    dev = Device['QCOM']
    env_ncols = max(4 if args.gemm_pattern else 2, args.ncols if args.gemm_pattern else 2)
    envelope, img_off, img_sz, reg_off = get_envelope(dev, make_donor_src(env_ncols, args.threads))
    if args.compiler_pattern:
        shader, loop_instrs, hregs, fregs, flops_per_thread_loop = build_compiler_pattern_shader(dev, args.threads, args.loops, args.pairs, args.store)
    elif args.gemm_pattern:
        shader, loop_instrs, hregs, fregs, flops_per_thread_loop = build_gemm_pattern_shader(
            dev, args.threads, args.loops, args.rows, args.ncols, args.unroll, args.order, args.bmode, args.r1, args.check_gemm)
    else:
        shader, loop_instrs, hregs, fregs, flops_per_thread_loop = build_alu_shader(dev, args.threads, args.groups, args.rpt, args.loops, args.unroll, args.independent, args.r1)
        width = args.rpt + 1
    if fregs is None: fregs = args.fregs
    if hregs > 48 and not args.allow_invalid_regs:
        print('skipped: groups=%d needs hregs=%d, but A630 addressable GPR half registers stop at hr47 (hregs=48).' % (args.groups, hregs))
        return
    if len(shader) > img_sz:
        print('skipped: shader is %d bytes but envelope has only %d bytes.' % (len(shader), img_sz))
        return
    lib = inject(envelope, img_off, img_sz, reg_off, shader, fregs=fregs, hregs=hregs)
    asm = disasm(shader)
    reg_count = fregs + (hregs + 1) // 2
    wave_pairs = 96 // reg_count
    mode = 'compiler-pattern' if args.compiler_pattern else ('gemm-pattern' if args.gemm_pattern else ('independent' if args.independent else 'accumulate'))
    print('mode=%s r1=%d rows=%d ncols=%d bmode=%s order=%s groups=%d rpt=%d width=%d unroll=%d pairs=%d fregs=%d hregs=%d reg_count=%d wave_pairs=%d loop_instrs=%d shader_instrs=%d mad=%d rpt3=%d' % (
        mode, args.r1, args.rows, args.ncols, args.bmode, args.order, args.groups, args.rpt, args.rpt + 1, args.unroll, args.pairs, fregs, hregs, reg_count, wave_pairs, loop_instrs, len(shader)//8, asm.count('mad.f16'), asm.count('(rpt3)mad.f16')))
    if args.disasm: print(asm)

    a, b, c = make_bufs(dev)
    # Runtime buffer metadata now carries image shape separately from the scalar dtype.
    buf_dtypes = [((0, dtypes.half, (M, K4, 4)),), ((0, dtypes.half, (K4*4, N//4, 4)),), ((0, dtypes.half, None),)]
    prg = dev.runtime('gemm_h', lib, buf_dtypes=buf_dtypes)
    tile_m = (args.threads // 32) * 4
    gs, ls = (8, M // tile_m, 1), (args.threads, 1, 1)
    for _ in range(5): prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
    times = []
    for _ in range(args.iters):
        t = prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)
    best = min(times)
    median = sorted(times)[len(times) // 2]
    total_threads = gs[0] * gs[1] * args.threads
    flops = total_threads * args.loops * flops_per_thread_loop
    print('%.1f GFLOPS best (%.3f ms), %.1f GFLOPS median (%.3f ms), flops=%d runs=%d' %
          (flops / best / 1e9, best * 1e3, flops / median / 1e9, median * 1e3, flops, len(times)))
    if args.check_gemm:
        if not args.gemm_pattern or args.bmode != 'percol' or args.loops != 1:
            raise ValueError('--check-gemm requires --gemm-pattern --bmode percol --loops 1')
        check_a, check_b = gemm_check_inputs(args.rows, args.ncols)
        checked = 0
        for group in range(args.rows * args.ncols):
            check_shader, _, check_hregs, _, _ = build_gemm_pattern_shader(
                dev, args.threads, args.loops, args.rows, args.ncols, args.unroll, args.order, args.bmode, args.r1,
                check_pattern=True, store_group=group)
            check_lib = inject(envelope, img_off, img_sz, reg_off, check_shader, fregs=fregs, hregs=check_hregs)
            check_prg = dev.runtime('gemm_h', check_lib, buf_dtypes=buf_dtypes)
            check_prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
            raw = c.copyout(memoryview(bytearray(c.nbytes))).cast('H')
            row, col = divmod(group, args.ncols)
            expected = [half_raw(args.unroll * sum(check_a[row][kk] * check_b[col][kk][lane] for kk in range(4))) for lane in range(4)]
            bad = next((i for i, value in enumerate(raw[:4]) if value != expected[i]), None)
            if bad is not None:
                got = struct.unpack('<e', struct.pack('<H', raw[bad]))[0]
                want = struct.unpack('<e', struct.pack('<H', expected[bad]))[0]
                raise RuntimeError('GEMM CHECK FAIL group=%d index=%d got=%r expected=%r' % (group, bad, got, want))
            checked += 4
        print('GEMM CHECK PASS groups=%d scalar_outputs=%d bit_exact=true shape_per_thread=%dx%dx%d' %
              (args.rows * args.ncols, checked, args.rows, args.ncols * 4, args.unroll * 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=int, default=16)
    parser.add_argument('--rpt', type=int, choices=(0, 1, 3), default=3)
    parser.add_argument('--loops', type=int, default=K4)
    parser.add_argument('--unroll', type=int, default=1)
    parser.add_argument('--pairs', type=int, default=8, help='compiler-pattern vector MAD pairs per loop')
    parser.add_argument('--independent', action='store_true', help='remove loop-carried accumulator dependency for raw FMA issue peak')
    parser.add_argument('--r1', action='store_true', help='auto-increment mad.f16 source1 across repeat lanes')
    parser.add_argument('--compiler-pattern', action='store_true', help='use the vec16 OpenCL peak MAD source/destination pattern')
    parser.add_argument('--gemm-pattern', action='store_true', help='use true GEMM-style acc=A_scalar*B_half4+acc MADs')
    parser.add_argument('--check-gemm', action='store_true', help='use nonuniform exact inputs and bit-check every GEMM accumulator')
    parser.add_argument('--rows', type=int, choices=(4, 8), default=4)
    parser.add_argument('--ncols', type=int, default=4)
    parser.add_argument('--bmode', choices=('reuse', 'percol'), default='reuse')
    parser.add_argument('--order', choices=('kk_row_col', 'kk_col_row', 'col_kk_row', 'row_kk_col', 'row_col_kk'), default='kk_row_col')
    parser.add_argument('--store', action='store_true', help='store one result after the ALU loop')
    parser.add_argument('--threads', type=int, choices=(64, 128, 256), default=128)
    parser.add_argument('--fregs', type=int, default=8)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--allow-invalid-regs', action='store_true')
    parser.add_argument('--disasm', action='store_true')
    run(parser.parse_args())
