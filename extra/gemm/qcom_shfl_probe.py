#!/usr/bin/env python3
"""Subgroup shuffle probes for Adreno 630 ir3.

This keeps the same image+buffer donor envelope as the GEMM experiments, but
only uses the C buffer. The semantic probe stores per-lane u32 values so the
exact source lane selected by each shfl mode is visible.
"""
import argparse, ctypes, struct

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import *

M = N = K = 1024
K4 = K // 4


def make_donor_src(ncols=1, threads=128):
    tn = 32 * ncols
    tm = (threads // 32) * 4
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global half *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    src += 'int row=get_group_id(1)*%d+tm*4;int col4=get_group_id(0)*%d+tid;\n' % (tm, tn)
    for r in range(4):
        for c in range(ncols):
            src += 'half4 r%dd%dc0=(half4)(0),r%dd%dc1=(half4)(0),r%dd%dc2=(half4)(0),r%dd%dc3=(half4)(0);\n' % (r,c,r,c,r,c,r,c)
    src += 'for(int k4=0;k4<%d;k4++){\n' % K4
    for r in range(4): src += 'half4 a%d=read_imageh(A,smp,(int2)(k4,row+%d));\n' % (r, r)
    for c in range(ncols):
        for b in range(4): src += 'half4 b%d_%d=read_imageh(B,smp,(int2)(col4+%d,k4*4+%d));\n' % (c, b, c*32, b)
    for r in range(4):
        for c in range(ncols):
            src += 'r%dd%dc0+=a%d.xxxx*b%d_0;r%dd%dc1+=a%d.yyyy*b%d_1;r%dd%dc2+=a%d.zzzz*b%d_2;r%dd%dc3+=a%d.wwww*b%d_3;\n' % (r,c,r,c,r,c,r,c,r,c,r,c,r,c,r,c)
    src += '}\n'
    for r in range(4):
        for c in range(ncols):
            src += 'vstore4(r%dd%dc0+r%dd%dc1+r%dd%dc2+r%dd%dc3,0,C+(row+%d)*%d+(col4+%d)*4);\n' % (r,c,r,c,r,c,r,c,r,N,c*32)
    src += '}\n'
    return src


def prologue(dev, threads):
    lib, io, isz, _ = get_envelope(dev, make_donor_src(1, threads))
    pro = bytearray(lib[io:io + 21 * 8])
    return [bytes(pro[i:i+8]) for i in range(0, len(pro), 8)]


def emit_addr(instrs, row_reg, col_reg):
    instrs += [
        SHL_B('r0.x', row_reg, 10, jp=True),
        SHL_B('r0.y', col_reg, 2),
        ADD_S_REG('r0.x', 'r0.x', 'r0.y'),
        SHL_B('r0.x', 'r0.x', 1),
        ADD_U('r2.x', 'c20.x', 'r0.x'),
        CMPS_U_LT('r6.w', 'r2.x', 'c20.x'),
        SHR_B('r6.y', 'r0.x', 31),
        SAD_S32('r2.y', 'c20.y', 'r6.y', 'r6.w', nop=3),
    ]


def store_u32(instrs, row_reg, col_reg, data_reg):
    emit_addr(instrs, row_reg, col_reg)
    instrs += [STG_U32('r2.x', data_reg), NOP()]


def emit_linear_addr(instrs, idx_reg, byte_base=0, join=False):
    instrs.append(SHL_B('r0.y', idx_reg, 2, jp=join, ss=join, nop=3 if join else 0))
    if byte_base:
        instrs += [MOV_S32('r0.z', byte_base), NOP(rpt=2), ADD_S_REG('r0.y', 'r0.y', 'r0.z'), NOP(rpt=16)]
    instrs += [
        ADD_U('r2.x', 'c20.x', 'r0.y'),
        NOP(rpt=16),
        CMPS_U_LT('r6.w', 'r2.x', 'c20.x'),
        SHR_B('r6.y', 'r0.y', 31),
        NOP(rpt=16),
        SAD_S32('r2.y', 'c20.y', 'r6.y', 'r6.w', nop=3),
    ]


def store_u32_linear(instrs, idx_reg, data_reg, byte_base=0, join=False):
    emit_linear_addr(instrs, idx_reg, byte_base, join=join)
    instrs += [NOP(rpt=16), STG_U32('r2.x', data_reg, sy=True), NOP(rpt=16)]


def build_semantics_shader(dev, threads):
    instrs = [
        AND_B('r8.x', 'r0.x', 31),
    ]
    instrs += [
        SHFL('r8.y', 'r8.x', 0, mode=7, typ=3),
        SHFL('r8.z', 'r8.x', 1, mode=7, typ=3),
        SHFL('r8.w', 'r8.x', 2, mode=7, typ=3),
        SHFL('r9.x', 'r8.x', 4, mode=7, typ=3),
        SHFL('r9.y', 'r8.x', 8, mode=7, typ=3),
        SHFL('r9.z', 'r8.x', 16, mode=7, typ=3),
        MOV_S32('r10.x', 16),
        SHFL('r9.w', 'r8.x', 'r10.x', mode=1, typ=3),
        MOV_S32('r10.x', 0),
        QUAD_BRCST('r11.x', 'r8.x', 'r10.x', typ=3),
        MOV_S32('r10.y', 1),
        QUAD_BRCST('r11.y', 'r8.x', 'r10.y', typ=3),
        MOV_S32('r10.z', 2),
        QUAD_BRCST('r11.z', 'r8.x', 'r10.z', typ=3),
        MOV_S32('r10.w', 3),
        QUAD_BRCST('r11.w', 'r8.x', 'r10.w', typ=3),
        NOP(rpt=3),
    ]
    regs = ['r8.x', 'r8.y', 'r8.z', 'r8.w', 'r9.x', 'r9.y', 'r9.z', 'r9.w', 'r11.x', 'r11.y', 'r11.z', 'r11.w']
    for case, reg in enumerate(regs):
        store_u32_linear(instrs, 'r8.x', reg, case * threads * 4)
    instrs.append(END())
    return assemble(instrs)


def build_quad_semantics_shader(dev, threads):
    instrs = prologue(dev, threads)
    instrs += [
        # Recover lid=(row-tile lane)*32+column lane from persistent coordinates.
        SHR_B('r12.x', 'r7.x', 2), AND_B('r12.x', 'r12.x', 3), SHL_B('r12.x', 'r12.x', 5),
        AND_B('r12.y', 'r7.y', 31), ADD_S_REG('r12.x', 'r12.x', 'r12.y'), NOP(rpt=2),
        MOV_F32('r8.x', 'r12.x'),
        MOV_S32('r10.x', 0),
        QUAD_BRCST('r11.x', 'r8.x', 'r10.x', typ=3),
        QUAD_BRCST('r13.x', 'r8.x', 'r10.x', typ=3, sy=True),
        MOV_S32('r10.y', 1),
        QUAD_BRCST('r11.y', 'r8.x', 'r10.y', typ=3),
        MOV_S32('r10.z', 2),
        QUAD_BRCST('r11.z', 'r8.x', 'r10.z', typ=3),
        MOV_S32('r10.w', 3),
        QUAD_BRCST('r11.w', 'r8.x', 'r10.w', typ=3),
        NOP(rpt=5),
    ]
    regs = ['r8.x', 'r11.x', 'r13.x', 'r11.y', 'r11.z', 'r11.w']
    for case, reg in enumerate(regs):
        store_u32_linear(instrs, 'r12.x', reg, case * threads * 4)
    instrs.append(END())
    return assemble(instrs)


def build_quad_map_shader(dev, threads):
    instrs = []
    instrs += [
        MOV_F32('r14.x', 'r0.x'),
        MOV_F32('r0.z', 'r0.x'),
        AND_B('r0.x', 'r0.z', 31, nop=3),
        MOV_S32('r9.x', 1),
        NOP(rpt=5),
    ]
    store_u32_linear(instrs, 'r14.x', 'r9.x', 0)
    instrs += [
        MOV_S32('r10.x', 0),
        QUAD_BRCST('r1.x', 'r0.x', 'r10.x', typ=3, sy=True),
        MOV_S32('r10.x', 1),
        QUAD_BRCST('r1.y', 'r0.x', 'r10.x', typ=3),
        MOV_S32('r10.x', 2),
        QUAD_BRCST('r1.z', 'r0.x', 'r10.x', typ=3),
        MOV_S32('r10.x', 3),
        QUAD_BRCST('r1.w', 'r0.x', 'r10.x', typ=3),
        NOP(rpt=5),
    ]
    for case, reg in enumerate(['r1.x', 'r1.y', 'r1.z', 'r1.w'], start=1):
        store_u32_linear(instrs, 'r14.x', reg, case * threads * 4)
    instrs.append(END())
    return assemble(instrs)


def build_modes_shader(dev, threads):
    instrs = prologue(dev, threads)
    instrs += [MOV_F32('r8.x', 'r0.x'), MOV_F32('r15.x', 'r8.x')]
    cases = []
    dsts = ['r8.y', 'r8.z', 'r8.w', 'r9.x', 'r9.y', 'r9.z', 'r9.w', 'r10.x', 'r10.y', 'r10.z', 'r10.w', 'r11.x', 'r11.y', 'r11.z', 'r11.w', 'r12.x', 'r12.y', 'r12.z', 'r12.w', 'r13.x']
    for mode in [1, 2, 3, 6, 7]:
        for idx in [1, 2, 3, 4]:
            if not dsts: break
            dst = dsts.pop(0)
            instrs.append(SHFL(dst, 'r8.x', idx, mode=mode, typ=3))
            cases.append((f'm{mode}i{idx}', dst))
        if not dsts: break
    instrs.append(NOP(rpt=5))
    for case, (_, reg) in enumerate(cases): store_u32_linear(instrs, 'r15.x', reg, case * threads * 4)
    instrs.append(END())
    return assemble(instrs), [name for name, _ in cases]


def build_bench_shader(dev, threads, op, kind, ops_per_iter):
    instrs = [
        MOV_S32('r6.z', 0, sy=True),
        MOV_F32('r14.x', 'r0.x'),
        AND_B('r8.x', 'r0.x', 31),
    ]
    instrs += [
        MOV_F32('r8.y', 'r8.x'), MOV_F32('r8.z', 'r8.x'), MOV_F32('r8.w', 'r8.x'),
        MOV_F32('r9.x', 'r8.x'), MOV_F32('r9.y', 'r8.x'),
        MOV_F32('r9.z', 'r8.x'), MOV_F32('r9.w', 'r8.x'),
    ]
    if op == 'quad': instrs.append(MOV_S32('r15.x', 0))
    loop_start = len(instrs)
    if kind == 'chain':
        for _ in range(ops_per_iter):
            instrs.append(SHFL('r8.x', 'r8.x', 1, mode=7, typ=3) if op == 'shfl' else QUAD_BRCST('r8.x', 'r8.x', 'r15.x', typ=3))
    elif kind == 'throughput':
        dsts = ['r10.x', 'r10.y', 'r10.z', 'r10.w', 'r11.x', 'r11.y', 'r11.z', 'r11.w',
                'r12.x', 'r12.y', 'r12.z', 'r12.w', 'r13.x', 'r13.y', 'r13.z', 'r13.w']
        srcs = ['r8.x', 'r8.y', 'r8.z', 'r8.w', 'r9.x', 'r9.y', 'r9.z', 'r9.w']
        for i in range(ops_per_iter):
            instrs.append(SHFL(dsts[i % len(dsts)], srcs[i % len(srcs)], 1, mode=7, typ=3) if op == 'shfl' else QUAD_BRCST(dsts[i % len(dsts)], srcs[i % len(srcs)], 'r15.x', typ=3))
    else:
        raise ValueError(kind)
    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        CMPS_S_EQ('r6.z', K4 - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    store_u32_linear(instrs, 'r14.x', 'r8.x' if kind == 'chain' else 'r10.x')
    instrs.append(END())
    return assemble(instrs), loop_end - loop_start


def build_branch_shader(dev, threads):
    instrs = [
        MOV_F32('r14.x', 'r0.x'),
        MOV_F32('r0.z', 'r0.x'),
        AND_B('r8.x', 'r0.z', 31),
        MOV_S32('r9.x', 0),
        AND_B('r0.x', 'r0.z', 3, nop=3),
        CMPS_S_EQ('r0.x', 0),
        NOP(rpt=5),
        BR(2),
        MOV_S32('r9.x', 1),
    ]
    store_u32_linear(instrs, 'r14.x', 'r0.x', join=True)
    store_u32_linear(instrs, 'r14.x', 'r9.x', threads * 4)
    instrs.append(END())
    return assemble(instrs)


def build_fiber_shader(dev, threads):
    instrs = [
        MOV_F32('r14.x', 'r0.x'),
        GETFIBERID('r9.x'),
        NOP(rpt=5),
    ]
    store_u32_linear(instrs, 'r14.x', 'r9.x')
    instrs.append(END())
    return assemble(instrs)


def make_runtime(dev, shader, threads, disasm_shader=False):
    envelope, io, isz, ro = get_envelope(dev, make_donor_src(2, threads))
    if disasm_shader:
        print(disasm(shader))
    print('shader_instrs=%d bytes=%d envelope_bytes=%d' % (len(shader)//8, len(shader), isz))
    lib = inject(envelope, io, isz, ro, shader, fregs=16, hregs=16, mergedregs=False)
    return dev.runtime('gemm_h', lib, buf_dtypes=[((0, dtypes.half, (M, K//4, 4)),),
                       ((1, dtypes.half, (K, N//4, 4)),), ((2, dtypes.half, None),)])


def make_bufs(dev):
    a = Buffer(dev.device, (K//4)*M*4, dtypes.half, preallocate=True)
    b = Buffer(dev.device, (N//4)*K*4, dtypes.half, preallocate=True)
    c = Buffer(dev.device, M*N, dtypes.half, preallocate=True)
    ctypes.memset(int(a._buf.va_addr), 0, a.nbytes)
    ctypes.memset(int(b._buf.va_addr), 0, b.nbytes)
    ctypes.memset(int(c._buf.va_addr), 0, c.nbytes)
    return a, b, c


def read_u32_linear(c, idx):
    off = idx * 4
    return struct.unpack_from('<I', c.as_memoryview(), off)[0]


def run_semantics(args):
    dev = Device['QCOM']
    if args.quad_map: shader, mode_names = build_quad_map_shader(dev, args.threads), ['src   ', 'qbc0  ', 'qbc1  ', 'qbc2  ', 'qbc3  ']
    elif args.shfl_modes: shader, mode_names = build_modes_shader(dev, args.threads)
    else: shader = build_quad_semantics_shader(dev, args.threads) if args.quad_only else build_semantics_shader(dev, args.threads)
    prg = make_runtime(dev, shader, args.threads, args.disasm)
    a, b, c = make_bufs(dev)
    prg(a._buf, b._buf, c._buf, global_size=(1, 1, 1), local_size=(args.threads, 1, 1), wait=True)
    copied = bytearray(c.nbytes)
    c.copyout(memoryview(copied))
    names = mode_names if args.shfl_modes or args.quad_map else ['src   ', 'qbc0  ', 'sqbc0 ', 'qbc1  ', 'qbc2  ', 'qbc3  '] if args.quad_only else ['src   ', 'rdn0  ', 'rdn1  ', 'rdn2  ', 'rdn4  ', 'rdn8  ', 'rdn16 ', 'xor16 ', 'qbc0  ', 'qbc1  ', 'qbc2  ', 'qbc3  ']
    vals = [[struct.unpack_from('<I', copied, (case * args.threads + lane)*4)[0] for lane in range(32)] for case in range(len(names))]
    for name, row_vals in zip(names, vals):
        print('%s: %s' % (name, ' '.join('%02d' % (v & 0xff) for v in row_vals)))
    if args.quad_map:
        mv = copied
        for case, name in enumerate(names):
            nz = []
            base = case * args.threads
            for i in range(args.threads * 2):
                v = struct.unpack_from('<I', mv, (base + i) * 4)[0]
                if v: nz.append((i, v & 0xff))
            print('nonzero %s: %s' % (name.strip(), nz[:64]))


def run_bench(args):
    dev = Device['QCOM']
    shader, loop_instrs = build_bench_shader(dev, args.threads, args.op, args.bench, args.ops_per_iter)
    prg = make_runtime(dev, shader, args.threads, args.disasm)
    print('loop_instrs=%d op=%s ops_per_iter=%d' % (loop_instrs, args.op, args.ops_per_iter))
    a, b, c = make_bufs(dev)
    gs = (args.groups, 64, 1)
    ls = (args.threads, 1, 1)
    for _ in range(5):
        prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
    times = []
    for _ in range(args.iters):
        t = prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)
    best = min(times)
    total_threads = args.groups * 64 * args.threads
    total_ops = total_threads * K4 * args.ops_per_iter
    print('%.1f G%s/s  (%.3f ms)' % (total_ops / best / 1e9, args.op.upper(), best * 1e3))


def run_branch(args):
    dev = Device['QCOM']
    shader = build_branch_shader(dev, args.threads)
    prg = make_runtime(dev, shader, args.threads, args.disasm)
    a, b, c = make_bufs(dev)
    prg(a._buf, b._buf, c._buf, global_size=(1, 1, 1), local_size=(args.threads, 1, 1), wait=True)
    mods = [read_u32_linear(c, lane) for lane in range(args.threads)]
    vals = [read_u32_linear(c, args.threads + lane) for lane in range(args.threads)]
    for base in range(0, args.threads, 32):
        print('mod %03d: %s' % (base, ' '.join('%d' % (v & 0xff) for v in mods[base:base+32])))
        print('br  %03d: %s' % (base, ' '.join('%d' % (v & 0xff) for v in vals[base:base+32])))
    mv = c.as_memoryview()
    nz0, nz1 = [], []
    for i in range(4096):
        v0 = struct.unpack_from('<I', mv, i * 4)[0]
        v1 = struct.unpack_from('<I', mv, args.threads * 4 + i * 4)[0]
        if v0: nz0.append((i, v0 & 0xff))
        if v1: nz1.append((i, v1 & 0xff))
    print('nonzero mod:', nz0[:64])
    print('nonzero br :', nz1[:64])


def run_fiber(args):
    dev = Device['QCOM']
    shader = build_fiber_shader(dev, args.threads)
    prg = make_runtime(dev, shader, args.threads, args.disasm)
    a, b, c = make_bufs(dev)
    prg(a._buf, b._buf, c._buf, global_size=(1, 1, 1), local_size=(args.threads, 1, 1), wait=True)
    vals = [read_u32_linear(c, lane) for lane in range(args.threads)]
    for base in range(0, args.threads, 32): print('fiber %03d: %s' % (base, ' '.join('%02d' % (v & 0xff) for v in vals[base:base+32])))


def run_compiled_branch(args):
    dev = Device['QCOM']
    src = '__attribute__((reqd_work_group_size(%d,1,1)))\n' % args.threads
    src += '__kernel void gemm_h(__global uint *C){int lid=get_local_id(0);'
    src += 'if((lid&3)==0) C[lid]=1; else C[lid]=0;}\n'
    lib = bytearray(dev.compiler.compile_cached(src))
    io = struct.unpack_from('<I', lib, 0xc0)[0]
    isz = struct.unpack_from('<I', lib, 0x100)[0]
    print('compiled_branch_bytes=%d' % isz)
    print(disasm(bytes(lib[io:io+isz])))


def run_compiled_image_branch(args):
    dev = Device['QCOM']
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % args.threads
    src += '__kernel void gemm_h(read_only image2d_t A,__global half *C){int lid=get_local_id(0);half v=(half)0;'
    src += 'if((lid&3)==0) v=read_imageh(A,smp,(int2)(0,lid)).x; C[lid]=v;}\n'
    lib = bytearray(dev.compiler.compile_cached(src))
    io = struct.unpack_from('<I', lib, 0xc0)[0]
    isz = struct.unpack_from('<I', lib, 0x100)[0]
    print('compiled_image_branch_bytes=%d' % isz)
    print(disasm(bytes(lib[io:io+isz])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, choices=(64, 128), default=128)
    parser.add_argument('--disasm', action='store_true')
    parser.add_argument('--op', choices=('shfl', 'quad'), default='shfl')
    parser.add_argument('--bench', choices=('throughput', 'chain'))
    parser.add_argument('--branch', action='store_true')
    parser.add_argument('--fiberid', action='store_true')
    parser.add_argument('--compiled-branch', action='store_true')
    parser.add_argument('--compiled-image-branch', action='store_true')
    parser.add_argument('--quad-only', action='store_true')
    parser.add_argument('--quad-map', action='store_true')
    parser.add_argument('--shfl-modes', action='store_true')
    parser.add_argument('--ops-per-iter', type=int, default=16)
    parser.add_argument('--groups', type=int, default=8)
    parser.add_argument('--iters', type=int, default=20)
    args = parser.parse_args()
    if args.compiled_image_branch: run_compiled_image_branch(args)
    elif args.compiled_branch: run_compiled_branch(args)
    elif args.fiberid: run_fiber(args)
    elif args.branch: run_branch(args)
    elif args.bench: run_bench(args)
    else: run_semantics(args)
