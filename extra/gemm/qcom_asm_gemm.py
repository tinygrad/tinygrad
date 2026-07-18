#!/usr/bin/env python3
"""Hand-assembled GEMM kernels for Adreno 630.

Tests:
1. Pure ALU kernel (MAD throughput ceiling)
2. Pure LOAD kernel (texture throughput ceiling)
3. Full GEMM with optimal isam/mad interleaving
"""
import struct, ctypes, math
from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import *

dev = Device['QCOM']

# ============================================================
# DONOR KERNEL: compile the 4-row GEMM for the binary envelope
# ============================================================
DONOR_SRC = (
    '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    '__attribute__((reqd_work_group_size(128,1,1)))\n'
    '__kernel void gemm_h(read_only image2d_t A, read_only image2d_t B, __global half *C) {\n'
    '  int lid=get_local_id(0); int tm=lid>>5; int tn=lid&31;\n'
    '  int row=get_group_id(1)*16+tm*4; int col4=get_group_id(0)*32+tn;\n'
    '  half4 r0c0=(half4)(0); for(int k4=0;k4<256;k4++){\n'
    '    half4 a=read_imageh(A,smp,(int2)(k4,row));\n'
    '    half4 b0=read_imageh(B,smp,(int2)(col4,k4*4));\n'
    '    r0c0+=a.xxxx*b0;\n'
    '  }\n'
    '  vstore4(r0c0, 0, C+row*1024+col4*4);\n'
    '}\n'
)

# Use the 4-row GEMM as donor since it has the right metadata for image textures
_DONOR4 = (
    '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    '__attribute__((reqd_work_group_size(128,1,1)))\n'
    '__kernel void gemm_h(read_only image2d_t A, read_only image2d_t B, __global half *C) {\n'
    '  int lid=get_local_id(0); int tm=lid>>5; int tn=lid&31;\n'
    '  int row=get_group_id(1)*16+tm*4; int col4=get_group_id(0)*32+tn;\n'
    '  half4 r0c0=(half4)(0),r0c1=(half4)(0),r0c2=(half4)(0),r0c3=(half4)(0);\n'
    '  half4 r1c0=(half4)(0),r1c1=(half4)(0),r1c2=(half4)(0),r1c3=(half4)(0);\n'
    '  half4 r2c0=(half4)(0),r2c1=(half4)(0),r2c2=(half4)(0),r2c3=(half4)(0);\n'
    '  half4 r3c0=(half4)(0),r3c1=(half4)(0),r3c2=(half4)(0),r3c3=(half4)(0);\n'
    '  for (int k4=0;k4<256;k4++) {\n'
    '    half4 ar0=read_imageh(A,smp,(int2)(k4,row));\n'
    '    half4 ar1=read_imageh(A,smp,(int2)(k4,row+1));\n'
    '    half4 ar2=read_imageh(A,smp,(int2)(k4,row+2));\n'
    '    half4 ar3=read_imageh(A,smp,(int2)(k4,row+3));\n'
    '    half4 b0=read_imageh(B,smp,(int2)(col4,k4*4));\n'
    '    half4 b1=read_imageh(B,smp,(int2)(col4,k4*4+1));\n'
    '    half4 b2=read_imageh(B,smp,(int2)(col4,k4*4+2));\n'
    '    half4 b3=read_imageh(B,smp,(int2)(col4,k4*4+3));\n'
    '    r0c0+=ar0.xxxx*b0; r0c1+=ar0.yyyy*b1; r0c2+=ar0.zzzz*b2; r0c3+=ar0.wwww*b3;\n'
    '    r1c0+=ar1.xxxx*b0; r1c1+=ar1.yyyy*b1; r1c2+=ar1.zzzz*b2; r1c3+=ar1.wwww*b3;\n'
    '    r2c0+=ar2.xxxx*b0; r2c1+=ar2.yyyy*b1; r2c2+=ar2.zzzz*b2; r2c3+=ar2.wwww*b3;\n'
    '    r3c0+=ar3.xxxx*b0; r3c1+=ar3.yyyy*b1; r3c2+=ar3.zzzz*b2; r3c3+=ar3.wwww*b3;\n'
    '  }\n'
    '  vstore4(r0c0+r0c1+r0c2+r0c3, 0, C+row*1024+col4*4);\n'
    '  vstore4(r1c0+r1c1+r1c2+r1c3, 0, C+(row+1)*1024+col4*4);\n'
    '  vstore4(r2c0+r2c1+r2c2+r2c3, 0, C+(row+2)*1024+col4*4);\n'
    '  vstore4(r3c0+r3c1+r3c2+r3c3, 0, C+(row+3)*1024+col4*4);\n'
    '}\n'
)
envelope, img_off, img_sz, reg_off = get_envelope(dev, _DONOR4)

M, N, K = 1024, 1024, 1024
K4 = K // 4  # 256

def make_bufs():
    a = Buffer(dev.device, (K//4)*M*4, dtypes.half, preallocate=True)
    b = Buffer(dev.device, (N//4)*K*4, dtypes.half, preallocate=True)
    c = Buffer(dev.device, M*N, dtypes.half, preallocate=True)
    ctypes.memset(int(a._buf.va_addr), 0, a.nbytes)
    ctypes.memset(int(b._buf.va_addr), 0, b.nbytes)
    return a, b, c

def bench(lib, gs, ls, label, flops=2*1024*1024*1024, iters=20):
    a, b, c = make_bufs()
    try:
        prg = dev.runtime('gemm_h', lib, buf_dtypes=[((0, dtypes.half, (M, K//4, 4)),),
                                                     ((1, dtypes.half, (K, N//4, 4)),),
                                                     ((2, dtypes.half, None),)])
        for _ in range(5):
            prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        times = []
        for _ in range(iters):
            t = prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
            if t: times.append(t)
        if times:
            best = min(times)
            gflops = flops / best / 1e9
            print("  %s: %.1f GFLOPS (%.0fus)" % (label, gflops, best*1e6))
            return gflops
    except Exception as e:
        print("  %s: ERROR %s" % (label, str(e)[:80]))
    return 0

# ============================================================
# Register plan for 4-row GEMM (matching the compiled kernel):
#
# Address/coordinate registers (full):
#   r0.x(0)   = lid (hardware input)
#   r0.y(1)   = group_id(1) + lid_row_offset
#   r0.z(2)   = tm = lid >> 5
#   r0.w(3)   = group_id(0) + lid_col_offset
#   r2.y(9)   = A coord (k4 value for isam)
#   r2.z(10)  = A row0 coord
#   r2.w(11)  = A coord duplicate
#   r3.x(12)  = A row0+1 coord
#   r3.y(13)  = A coord dup
#   r3.z(14)  = A row0+2 coord
#   r3.w(15)  = A coord dup
#   r4.x(16)  = A row0+3 coord
#   r4.y(17)  = B col coord
#   r4.z(18)  = B K offset
#   r4.w(19)  = B K offset
#   r5.y(21)  = B col coord dup
#   r5.w(23)  = B col coord dup
#   r6.x(24)  = temp
#   r6.y(25)  = k4*4 base
#   r6.z(26)  = k4 counter
#   r7.x(28)  = row base addr
#   r7.y(29)  = col4 base addr
#
# Texture result registers (half):
#   hr0(0-3)   = A row3 texel (or temp)
#   hr1(4-7)   = A row2 texel
#   hr2(8-11)  = A row1 texel
#   hr3(12-15) = A row0 texel
#   hr4(16-19) = B texel (shared across all rows)
#
# Accumulator registers (half): 64 values = 16 groups of 4
#   Row0: hr13.z(54)-hr16.y(65) = 4 groups: K0-K3
#   Row1: hr17.z(70)-hr20.y(81) = 4 groups  [WRONG, let me read the actual mapping]
#
# Actually, from the disasm the accumulator mapping is:
#   Row0 K0: hr20.z(82),hr20.w(83),hr21.x(84),hr21.y(85)
#   Row0 K1: hr21.z(86),hr21.w(87),hr22.x(88),hr22.y(89) 
#   Row0 K2: hr22.z(90),hr22.w(91),hr23.x(92),hr23.y(93)
#   Row0 K3: hr23.z(94),hr23.w(95),hr24.x(96),hr24.y(97)
#   Row1 K0: hr24.z(98),hr24.w(99),hr25.x(100),hr25.y(101)
#   Row1 K1: hr25.z(102),hr25.w(103),hr26.x(104),hr26.y(105)
#   Row1 K2: hr26.z(106),hr26.w(107),hr27.x(108),hr27.y(109)
#   Row1 K3: hr27.z(110),hr27.w(111),hr28.x(112),hr28.y(113)
#   Row2 K0: hr28.z(114),hr28.w(115),hr29.x(116),hr29.y(117)
#   Row2 K1: hr29.z(118),hr29.w(119),hr30.x(120),hr30.y(121)
#   Row2 K2: (from rpt1+rpt1, noncontiguous)
#   Row2 K3: (from rpt3)
#   Row3 K0: hr17.z(70),hr17.w(71),hr18.x(72),hr18.y(73)
#   ... etc
# This is messy. Let me use a CLEAN register plan instead.
# ============================================================

# ============================================================
# TEST 1: PURE ALU - 16 (rpt3)mad.f16 in a loop, no texture loads
# ============================================================

print("=== TEST 1: Pure ALU (MAD throughput ceiling) ===")

# Accumulator regs: hr20.x(80) through hr35.w(143) = 64 half-regs = 16 groups of 4
# Source A: hr0.x(0) - hr0.w(3)
# Source B: hr4.x(16) - hr7.w(31) (unused, just for mad operands)

alu_instrs = [
    MOV_S32('r6.z', 0, sy=True),     # counter = 0
    MOV_H_IMM('hr0.x', 0x3c00),      # hr0.x = 1.0 (fp16)
    MOV_H('hr0.y', 'hr0.x', rpt=2),  # hr0.y,z,w = 1.0
    MOV_H_IMM('hr20.x', 0),           # zero first acc
]
# Zero all 64 accumulator regs (hr20.x=80 through hr35.w=143)
for base in range(84, 144, 4):
    alu_instrs.append(MOV_H(base, 80, rpt=3))
# Set source B regs to 1.0
for base in range(16, 32, 4):
    alu_instrs.append(MOV_H(base, 0, rpt=3))

# Loop label will be here
loop_start = len(alu_instrs)

# 16x (rpt3)mad.f16 = 64 MADs per iteration
for g in range(16):
    acc = 80 + g * 4           # accumulator base: hr20.x + g*4
    src1 = g % 4               # hr0.x, hr0.y, hr0.z, hr0.w (cycling)
    src2 = 16 + (g % 4) * 4   # hr4.x, hr5.x, hr6.x, hr7.x
    alu_instrs.append(MAD_F16(acc, src1, src2, acc, rpt=3, r=True))

# Loop control
alu_instrs.append(ADD_S('r6.z', 'r6.z', 1))
alu_instrs.append(CMPS_S_EQ('r6.z', K4 - 1))

loop_end = len(alu_instrs)
alu_instrs.append(BR(loop_start - loop_end))

# Epilogue: sum and store (minimal - just write something)
alu_instrs.append(ADD_F('hr0.x', 80, 84))
alu_instrs.append(ADD_F('hr0.y', 88, 92))
alu_instrs.append(ADD_F('hr0.z', 96, 100))
alu_instrs.append(ADD_F('hr0.w', 104, 108))
alu_instrs.append(NOP(rpt=5))
alu_instrs.append(STG_F16('r0.z', 'hr0.x'))
alu_instrs.append(END())

shader_alu = assemble(alu_instrs)
lib_alu = inject(envelope, img_off, img_sz, reg_off, shader_alu, fregs=8, hregs=64)

print("  Shader: %d instrs (loop body: %d)" % (len(alu_instrs), loop_end - loop_start))
print("  Disasm loop body:")
asm = disasm(shader_alu)
lines = asm.strip().split('\n')
for line in lines[loop_start:loop_end+2]:
    print("    " + line[:120])

total_mads = 64 * K4  # 64 MADs per iter * 256 iters
total_threads = 128 * (M // 128) * (M // 16)  # same grid as GEMM
total_flops = total_mads * 2 * total_threads
bench(lib_alu, (M//128, M//16, 1), (128, 1, 1), "PURE ALU", flops=total_flops)

# ============================================================
# TEST 2: PURE LOAD - 8 isam per iteration, accumulate results
# ============================================================

print("\n=== TEST 2: Pure LOAD (texture throughput ceiling) ===")

# Same coordinate setup as the real GEMM but no MAD - just isam + add
# We reuse the donor kernel's prologue for coordinate setup.
# Actually let's just build it from scratch with minimal coord math.

load_instrs = [
    MOV_S32('r6.y', 3, sy=True),     # k4*4 base = 3 (initial)
    MOV_S32('r6.z', 0),               # k4 counter = 0
    MOV_H_IMM('hr20.x', 0),           # zero accumulator
    MOV_H('hr20.y', 'hr20.x', rpt=2), # hr20.y,z,w = 0
    # Compute row and col4 from lid
    MOV_F32('r0.y', 'r52.x'),         # gid1 (from hardware constant)
    NOP(rpt=2),
    ADD_S('r0.y', 'r0.y', 0),         # r0.y = gid1 (simplified; real kernel adds c7.y)
]
# Copy the coordinate setup from the compiled kernel (lines 0-20)
# Actually this is getting complex. Let me just build a simple version:
# Use the compiled kernel verbatim but NOP out all the MADs.

# Load the full 4-row donor kernel
donor4 = (
    '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    '__attribute__((reqd_work_group_size(128,1,1)))\n'
    '__kernel void gemm_h(read_only image2d_t A, read_only image2d_t B, __global half *C) {\n'
    '  int lid=get_local_id(0); int tm=lid>>5; int tn=lid&31;\n'
    '  int row=get_group_id(1)*16+tm*4; int col4=get_group_id(0)*32+tn;\n'
    '  half4 r0c0=(half4)(0),r0c1=(half4)(0),r0c2=(half4)(0),r0c3=(half4)(0);\n'
    '  half4 r1c0=(half4)(0),r1c1=(half4)(0),r1c2=(half4)(0),r1c3=(half4)(0);\n'
    '  half4 r2c0=(half4)(0),r2c1=(half4)(0),r2c2=(half4)(0),r2c3=(half4)(0);\n'
    '  half4 r3c0=(half4)(0),r3c1=(half4)(0),r3c2=(half4)(0),r3c3=(half4)(0);\n'
    '  for (int k4=0;k4<256;k4++) {\n'
    '    half4 ar0=read_imageh(A,smp,(int2)(k4,row));\n'
    '    half4 ar1=read_imageh(A,smp,(int2)(k4,row+1));\n'
    '    half4 ar2=read_imageh(A,smp,(int2)(k4,row+2));\n'
    '    half4 ar3=read_imageh(A,smp,(int2)(k4,row+3));\n'
    '    half4 b0=read_imageh(B,smp,(int2)(col4,k4*4));\n'
    '    half4 b1=read_imageh(B,smp,(int2)(col4,k4*4+1));\n'
    '    half4 b2=read_imageh(B,smp,(int2)(col4,k4*4+2));\n'
    '    half4 b3=read_imageh(B,smp,(int2)(col4,k4*4+3));\n'
    '    r0c0+=ar0.xxxx*b0; r0c1+=ar0.yyyy*b1; r0c2+=ar0.zzzz*b2; r0c3+=ar0.wwww*b3;\n'
    '    r1c0+=ar1.xxxx*b0; r1c1+=ar1.yyyy*b1; r1c2+=ar1.zzzz*b2; r1c3+=ar1.wwww*b3;\n'
    '    r2c0+=ar2.xxxx*b0; r2c1+=ar2.yyyy*b1; r2c2+=ar2.zzzz*b2; r2c3+=ar2.wwww*b3;\n'
    '    r3c0+=ar3.xxxx*b0; r3c1+=ar3.yyyy*b1; r3c2+=ar3.zzzz*b2; r3c3+=ar3.wwww*b3;\n'
    '  }\n'
    '  vstore4(r0c0+r0c1+r0c2+r0c3, 0, C+row*1024+col4*4);\n'
    '  vstore4(r1c0+r1c1+r1c2+r1c3, 0, C+(row+1)*1024+col4*4);\n'
    '  vstore4(r2c0+r2c1+r2c2+r2c3, 0, C+(row+2)*1024+col4*4);\n'
    '  vstore4(r3c0+r3c1+r3c2+r3c3, 0, C+(row+3)*1024+col4*4);\n'
    '}\n'
)

lib4, io4, isz4, ro4 = get_envelope(dev, donor4)
shader4 = bytearray(lib4[io4:io4+isz4])
total4 = isz4 // 8

# NOP out all MAD instructions
for i in range(total4):
    lo, hi = struct.unpack_from('<II', shader4, i*8)
    if (hi >> 24) in (0x63, 0x73) and ((hi >> 24) & 0xF) == 3:
        struct.pack_into('<Q', shader4, i*8, 0)

lib_load = inject(lib4, io4, isz4, ro4, shader4, fregs=8, hregs=31)
bench(lib_load, (M//128, M//16, 1), (128, 1, 1), "PURE LOAD")

# ============================================================
# TEST 3: FULL GEMM - patched 4-row kernel (sy-stripped + rpt3)
# ============================================================

print("\n=== TEST 3: Patched GEMM (sy-stripped + rpt3) ===")

# Take the compiled 4-row kernel, strip extra (sy), convert to rpt3
shader_gemm = bytearray(lib4[io4:io4+isz4])

# Strip extra (sy) flags - keep only the first one
first_sy = False
for i in range(total4):
    lo, hi = struct.unpack_from('<II', shader_gemm, i*8)
    if (hi >> 24) in (0x63, 0x73) and ((hi >> 24) & 0xF) == 3 and (hi >> 28) == 7:
        if first_sy:
            struct.pack_into('<I', shader_gemm, i*8+4, (hi & 0x0FFFFFFF) | 0x60000000)
        else:
            first_sy = True

# Convert eligible 4-scalar MAD groups to (rpt3)
i = 0
while i < total4 - 3:
    lo0, hi0 = struct.unpack_from('<II', shader_gemm, i*8)
    if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0xF) == 3) or (hi0 >> 8) & 0x7F > 0 or (hi0 & 0xFF) != ((lo0 >> 16) & 0xFF):
        i += 1; continue
    d0, s1_0 = hi0 & 0xFF, lo0 & 0xFF
    s2_0 = ((hi0 >> 16) & 0xFF) * 2 + (((hi0 >> 8) & 0xFF) >> 7)
    ok = True
    for j in range(1, 4):
        lj, hj = struct.unpack_from('<II', shader_gemm, (i+j)*8)
        if not ((hj >> 24) in (0x63, 0x73) and ((hj >> 24) & 0xF) == 3): ok = False; break
        dj, rpj = hj & 0xFF, (hj >> 8) & 0x7F
        s1j, s3j = lj & 0xFF, (lj >> 16) & 0xFF
        s2j = ((hj >> 16) & 0xFF) * 2 + (((hj >> 8) & 0xFF) >> 7)
        if rpj != 0 or s1j != s1_0 or dj != d0+j or s2j != s2_0+j or s3j != d0+j: ok = False; break
    if ok:
        rb = ((hi0 >> 8) & 0x80) | 3
        struct.pack_into('<I', shader_gemm, i*8+4, (hi0 & 0xFFFF00FF) | (rb << 8))
        struct.pack_into('<I', shader_gemm, i*8, lo0 | 0x20000000)
        for j in range(1, 4): struct.pack_into('<Q', shader_gemm, (i+j)*8, 0)
        i += 4
    else:
        i += 1

# Merge (rpt1)+(rpt1) -> (rpt3)
for i in range(total4 - 1):
    lo0, hi0 = struct.unpack_from('<II', shader_gemm, i*8)
    lo1, hi1 = struct.unpack_from('<II', shader_gemm, (i+1)*8)
    if hi0 == 0 or hi1 == 0: continue
    if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0xF) == 3): continue
    if not ((hi1 >> 24) in (0x63, 0x73) and ((hi1 >> 24) & 0xF) == 3): continue
    if (hi0 >> 8) & 0x7F != 1 or (hi1 >> 8) & 0x7F != 1: continue
    d0, d1 = hi0 & 0xFF, hi1 & 0xFF
    s10, s11 = lo0 & 0xFF, lo1 & 0xFF
    s20 = ((hi0 >> 16) & 0xFF) * 2 + (((hi0 >> 8) & 0xFF) >> 7)
    s21 = ((hi1 >> 16) & 0xFF) * 2 + (((hi1 >> 8) & 0xFF) >> 7)
    if s10 != s11 or d1 != d0 + 2 or s21 != s20 + 2: continue
    rb = ((hi0 >> 8) & 0x80) | 3
    struct.pack_into('<I', shader_gemm, i*8+4, (hi0 & 0xFFFF00FF) | (rb << 8))
    struct.pack_into('<Q', shader_gemm, (i+1)*8, 0)

lib_gemm = inject(lib4, io4, isz4, ro4, shader_gemm, fregs=8, hregs=31)

# Count stats
asm_gemm = disasm(shader_gemm)
print("  mad.f16: %d, (rpt3): %d, isam: %d, (sy): %d" % (
    asm_gemm.count('mad.f16'), asm_gemm.count('(rpt3)mad.f16'),
    asm_gemm.count('isam'), asm_gemm.count('(sy)')))

bench(lib_gemm, (M//128, M//16, 1), (128, 1, 1), "PATCHED GEMM")

# ============================================================
# TEST 4: FULL GEMM at different sizes
# ============================================================

print("\n=== TEST 4: Patched GEMM at various sizes ===")
for dim in [512, 768, 1024, 2048]:
    if dim % 128 != 0 or dim % 16 != 0: continue
    K4d = dim // 4
    src_d = donor4.replace('k4<256', 'k4<%d' % K4d)
    for s in ['row*1024', '(row+1)*1024', '(row+2)*1024', '(row+3)*1024']:
        src_d = src_d.replace(s, s.replace('1024', str(dim)))
    lib_d, io_d, isz_d, ro_d = get_envelope(dev, src_d)
    s_d = bytearray(lib_d[io_d:io_d+isz_d])
    t_d = isz_d // 8
    # Apply same patches
    fsy = False
    for i in range(t_d):
        lo, hi = struct.unpack_from('<II', s_d, i*8)
        if (hi >> 24) in (0x63, 0x73) and ((hi >> 24) & 0xF) == 3 and (hi >> 28) == 7:
            if fsy: struct.pack_into('<I', s_d, i*8+4, (hi & 0x0FFFFFFF) | 0x60000000)
            else: fsy = True
    i = 0
    while i < t_d - 3:
        lo0, hi0 = struct.unpack_from('<II', s_d, i*8)
        if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0xF) == 3) or (hi0 >> 8) & 0x7F > 0 or (hi0 & 0xFF) != ((lo0 >> 16) & 0xFF):
            i += 1; continue
        d0, s1_0 = hi0 & 0xFF, lo0 & 0xFF
        s2_0 = ((hi0 >> 16) & 0xFF) * 2 + (((hi0 >> 8) & 0xFF) >> 7)
        ok = True
        for j in range(1, 4):
            lj, hj = struct.unpack_from('<II', s_d, (i+j)*8)
            if not ((hj >> 24) in (0x63, 0x73) and ((hj >> 24) & 0xF) == 3): ok = False; break
            dj, rpj = hj & 0xFF, (hj >> 8) & 0x7F
            s1j, s3j = lj & 0xFF, (lj >> 16) & 0xFF
            s2j = ((hj >> 16) & 0xFF) * 2 + (((hj >> 8) & 0xFF) >> 7)
            if rpj != 0 or s1j != s1_0 or dj != d0+j or s2j != s2_0+j or s3j != d0+j: ok = False; break
        if ok:
            rb = ((hi0 >> 8) & 0x80) | 3
            struct.pack_into('<I', s_d, i*8+4, (hi0 & 0xFFFF00FF) | (rb << 8))
            struct.pack_into('<I', s_d, i*8, lo0 | 0x20000000)
            for j in range(1, 4): struct.pack_into('<Q', s_d, (i+j)*8, 0)
            i += 4
        else: i += 1
    for i in range(t_d - 1):
        lo0, hi0 = struct.unpack_from('<II', s_d, i*8)
        lo1, hi1 = struct.unpack_from('<II', s_d, (i+1)*8)
        if hi0 == 0 or hi1 == 0: continue
        if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0xF) == 3): continue
        if not ((hi1 >> 24) in (0x63, 0x73) and ((hi1 >> 24) & 0xF) == 3): continue
        if (hi0 >> 8) & 0x7F != 1 or (hi1 >> 8) & 0x7F != 1: continue
        d0v, d1v = hi0 & 0xFF, hi1 & 0xFF
        s10, s11 = lo0 & 0xFF, lo1 & 0xFF
        s20 = ((hi0 >> 16) & 0xFF) * 2 + (((hi0 >> 8) & 0xFF) >> 7)
        s21 = ((hi1 >> 16) & 0xFF) * 2 + (((hi1 >> 8) & 0xFF) >> 7)
        if s10 != s11 or d1v != d0v + 2 or s21 != s20 + 2: continue
        rb = ((hi0 >> 8) & 0x80) | 3
        struct.pack_into('<I', s_d, i*8+4, (hi0 & 0xFFFF00FF) | (rb << 8))
        struct.pack_into('<Q', s_d, (i+1)*8, 0)
    ld = inject(lib_d, io_d, isz_d, ro_d, s_d, fregs=8, hregs=31)
    M2 = N2 = K2 = dim
    a2 = Buffer(dev.device, (K2//4)*M2*4, dtypes.half, preallocate=True)
    b2 = Buffer(dev.device, (N2//4)*K2*4, dtypes.half, preallocate=True)
    c2 = Buffer(dev.device, M2*N2, dtypes.half, preallocate=True)
    ctypes.memset(int(a2._buf.va_addr), 0, a2.nbytes)
    ctypes.memset(int(b2._buf.va_addr), 0, b2.nbytes)
    try:
        prg_d = dev.runtime('gemm_h', ld, [[(0, dtypes.imageh((M2, K2//4)))], [(1, dtypes.imageh((K2, N2//4)))], [(2, dtypes.half.ptr())]])
        gs_d = (dim//128, dim//16, 1)
        for _ in range(5): prg_d(a2._buf, b2._buf, c2._buf, global_size=gs_d, local_size=(128,1,1), wait=True)
        ts = []
        for _ in range(20):
            t = prg_d(a2._buf, b2._buf, c2._buf, global_size=gs_d, local_size=(128,1,1), wait=True)
            if t: ts.append(t)
        if ts:
            best = min(ts)
            gf = 2*dim*dim*dim / best / 1e9
            print("  %dx%d: %.1f GFLOPS (%.1fms)" % (dim, dim, gf, best*1e3))
    except Exception as e:
        print("  %d: ERROR %s" % (dim, str(e)[:60]))
