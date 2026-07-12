#!/usr/bin/env python3
"""Hand-assembled higher-intensity FP16 GEMM for Adreno 630.

The baseline 4-row kernel does 128 FLOPs from 64 bytes of texture input per
thread/K step. This experiment computes two col4 outputs per thread, reusing
the same four A texels across eight B texels: 256 FLOPs from 96 bytes.
"""
import argparse, array, ctypes, struct

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import *
from extra.gemm.ir3asm import _hreg

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


def make_direct_donor_src(ncols=4, threads=128):
    tn = 32 * ncols
    tm = (threads // 32) * 4
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global half *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    src += 'int row=get_group_id(1)*%d+tm*4;int col4=get_group_id(0)*%d+tid;\n' % (tm, tn)
    for r in range(4):
        for c in range(ncols): src += 'half4 r%dd%d=(half4)(0);\n' % (r, c)
    src += 'for(int k4=0;k4<%d;k4++){\n' % K4
    for r in range(4): src += 'half4 a%d=read_imageh(A,smp,(int2)(k4,row+%d));\n' % (r, r)
    for c in range(ncols):
        for b in range(4): src += 'half4 b%d_%d=read_imageh(B,smp,(int2)(col4+%d,k4*4+%d));\n' % (c, b, c*32, b)
    for r in range(4):
        for c in range(ncols):
            src += 'r%dd%d+=a%d.xxxx*b%d_0+a%d.yyyy*b%d_1+a%d.zzzz*b%d_2+a%d.wwww*b%d_3;\n' % (r,c,r,c,r,c,r,c,r,c)
    src += '}\n'
    for r in range(4):
        for c in range(ncols): src += 'vstore4(r%dd%d,0,C+(row+%d)*%d+(col4+%d)*4);\n' % (r,c,r,N,c*32)
    src += '}\n'
    return src


def make_direct_donor_src_fp32(ncols=1, threads=128):
    tn = 32 * ncols
    tm = (threads // 32) * 4
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global float *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    src += 'int row=get_group_id(1)*%d+tm*4;int col4=get_group_id(0)*%d+tid;\n' % (tm, tn)
    for r in range(4):
        for c in range(ncols): src += 'float4 r%dd%d=(float4)(0);\n' % (r, c)
    src += 'for(int k4=0;k4<%d;k4++){\n' % K4
    for r in range(4): src += 'float4 a%d=convert_float4(read_imageh(A,smp,(int2)(k4,row+%d)));\n' % (r, r)
    for c in range(ncols):
        for b in range(4): src += 'float4 b%d_%d=convert_float4(read_imageh(B,smp,(int2)(col4+%d,k4*4+%d)));\n' % (c, b, c*32, b)
    for r in range(4):
        for c in range(ncols):
            src += 'r%dd%d+=a%d.xxxx*b%d_0+a%d.yyyy*b%d_1+a%d.zzzz*b%d_2+a%d.wwww*b%d_3;\n' % (r,c,r,c,r,c,r,c,r,c)
    src += '}\n'
    for r in range(4):
        for c in range(ncols): src += 'vstore4(r%dd%d,0,C+(row+%d)*%d+(col4+%d)*4);\n' % (r,c,r,N,c*32)
    src += '}\n'
    return src


def prologue_4x2(dev, threads):
    lib, io, isz, _ = get_envelope(dev, make_donor_src(1, threads))
    pro = bytearray(lib[io:io + 21 * 8])
    return [bytes(pro[i:i+8]) for i in range(0, len(pro), 8)]


def prologue_direct4_fp32(dev, threads, ncols=1):
    lib, io, isz, _ = get_envelope(dev, make_direct_donor_src_fp32(ncols, threads))
    pro_instrs = 40 if ncols != 1 else 27
    pro = bytearray(lib[io:io + pro_instrs * 8])
    return [bytes(pro[i:i+8]) for i in range(0, len(pro), 8)]


def emit_addr(instrs, row_reg, col_reg):
    # Compute 64-bit C address in r2.x/r2.y. Do this before reducing into hr0,
    # since the address math clobbers low half registers through r0/r1 aliases.
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


def emit_addr_float(instrs, row_reg, col_reg):
    instrs += [
        SHL_B('r0.x', row_reg, 10, jp=True),
        SHL_B('r0.y', col_reg, 2),
        ADD_S_REG('r0.x', 'r0.x', 'r0.y'),
        SHL_B('r0.x', 'r0.x', 2),
        ADD_U('r2.x', 'c20.x', 'r0.x'),
        CMPS_U_LT('r6.w', 'r2.x', 'c20.x'),
        SHR_B('r6.y', 'r0.x', 31),
        SAD_S32('r2.y', 'c20.y', 'r6.y', 'r6.w', nop=3),
    ]


def store_output(instrs, row_reg, col_reg, data_hreg):
    emit_addr(instrs, row_reg, col_reg)
    instrs += [NOP(rpt=16), STG_F16('r2.x', data_hreg), NOP()]


def emit_addr4_rows(instrs, col_reg):
    # Compute four 64-bit C addresses for rows r7.x + 0..3 at a single col4.
    rows = [('r7.x', 'r0.x', 'r2.x', 'r2.y', 'r4.x', 'r4.y'),
            (1,      'r0.z', 'r2.z', 'r2.w', 'r4.z', 'r4.w'),
            (2,      'r1.x', 'r3.x', 'r3.y', 'r5.x', 'r5.y'),
            (3,      'r1.z', 'r3.z', 'r3.w', 'r5.z', 'r5.w')]
    for row, off, alo, ahi, carry, sign in rows:
        if isinstance(row, int):
            instrs.append(OR_B('r7.w', 'r7.x', row))
            instrs.append(NOP(rpt=16))
            row_reg = 'r7.w'
        else: row_reg = row
        instrs += [
            SHL_B(off, row_reg, 10, jp=True),
            SHL_B('r0.y', col_reg, 2),
            NOP(rpt=16),
            ADD_S_REG(off, off, 'r0.y'),
            NOP(rpt=16),
            SHL_B(off, off, 1),
            NOP(rpt=16),
            ADD_U(alo, 'c20.x', off),
            NOP(rpt=16),
            CMPS_U_LT(carry, alo, 'c20.x'),
            SHR_B(sign, off, 31),
            NOP(rpt=16),
            SAD_S32(ahi, 'c20.y', sign, carry, nop=3 if row == 3 else 0),
        ]


def emit_hand4_stores(instrs, acc0, ncols):
    hand_addr = [bytes.fromhex(x) for x in [
        '1c000a200000d04e', '1d0002200100d046', '0000000000100000',
        '000061100201b843', '000060100600b043', '0000010000003042',
        '0100020002013842', '0100060001003042', '000001200000d046',
        '020001200200d046', '030001200600d046', '010001200700d046',
        '0000000003401520', '0000000000000000', '0600000001401520',
        '0700000000401520', '0000000000100000', '5010030008001042',
        '501002000a001042', '501001000c001042', '501000000e001042',
        '0000000000100000', '0800501010009042', '03001f201100f046',
        '0a00501012009042', '02001f201300f046', '0c00501014009042',
        '01001f201500f046', '0e00501016009042', '00001f201700f046',
        '0000000000100000', '5110104009808867', '511012400b808967',
        '511014400d808a67', '519016400f888b67']]
    hand_stores = [bytes.fromhex(x) for x in [
        '000080040011c0c0', '0000000000000000', '080080040015c0c0',
        '0000000000000000', '100080040019c0c0', '0000000000000000',
        '18008004001dc0c0']]
    for col in range(ncols):
        if col != 0:
            instrs.append(ADD_S('r7.y', 'r7.y', 32))
        instrs += hand_addr
        for row in range(4):
            instrs.append(MOV_H(row * 4, acc0 + (row * ncols + col) * 4, rpt=3))
        instrs += hand_stores


def emit_coord_wait(instrs, coord_delay):
    if coord_delay >= 0: instrs.append(NOP(rpt=coord_delay))


def fvec(vec, comp=0):
    return 'r%d.%s' % (vec, 'xyzw'[comp])


def emit_f32_vec_imm(instrs, vec, imm):
    for comp in range(4): instrs.append(MOV_S32(fvec(vec, comp), imm))


def emit_isam_h_to_f32_vec(instrs, dst_vec, coord, tex):
    instrs.append(ISAM_F16('hr0.x', coord, tex))
    for comp in range(4): instrs.append(COV_F16F32(fvec(dst_vec, comp), 'hr0.%s' % 'xyzw'[comp], sy=(comp == 0)))


def emit_isam_f32_vec(instrs, dst_vec, coord, tex, sampler_per_texture=False):
    instrs.append(ISAM_F32(fvec(dst_vec), coord, tex, tex if sampler_per_texture else 0))


def emit_donor4_float_store(instrs, dev, threads, acc_vec0):
    lib, io, isz, _ = get_envelope(dev, make_direct_donor_src_fp32(1, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    instrs += [MOV_F32('r4.z', 'r7.x'), MOV_F32('r5.x', 'r7.y')]

    instrs += [MOV_F32('r5.z', fvec(acc_vec0, 0)), MOV_F32('r8.y', fvec(acc_vec0, 1)), MOV_F32('r8.z', fvec(acc_vec0, 2)), MOV_F32('r8.w', fvec(acc_vec0, 3))]
    instrs += [MOV_F32('r7.y', fvec(acc_vec0 + 1, 0)), MOV_F32('r7.z', fvec(acc_vec0 + 1, 1)), MOV_F32('r7.w', fvec(acc_vec0 + 1, 2)), MOV_F32('r8.x', fvec(acc_vec0 + 1, 3))]
    instrs += [MOV_F32('r6.y', fvec(acc_vec0 + 2, 0)), MOV_F32('r6.z', fvec(acc_vec0 + 2, 1)), MOV_F32('r6.w', fvec(acc_vec0 + 2, 2)), MOV_F32('r7.x', fvec(acc_vec0 + 2, 3))]
    instrs += [MOV_F32('r4.w', fvec(acc_vec0 + 3, 0)), MOV_F32('r5.y', fvec(acc_vec0 + 3, 1)), MOV_F32('r5.w', fvec(acc_vec0 + 3, 2)), MOV_F32('r6.x', fvec(acc_vec0 + 3, 3))]
    # The donor's post-loop branch expects p0 to still hold the loop-exit
    # predicate. Address compares inside this same store clobber p0, so make it
    # explicitly true before every reusable 4-row store epilogue.
    instrs += [MOV_S32('r0.x', 0), CMPS_S_EQ('r0.x', 0, nop=3)]
    instrs += donor[146:192]


def emit_donor4_float_store2(instrs, dev, threads, acc_vec0):
    lib, io, isz, _ = get_envelope(dev, make_direct_donor_src_fp32(2, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    tmp_vec0 = 24
    for i in range(8): instrs.append(MOV_F32(fvec(tmp_vec0 + i), fvec(acc_vec0 + i), rpt=3))

    def mov_split(dsts, src_vec):
        for comp, dst in enumerate(dsts): instrs.append(MOV_F32(dst, fvec(src_vec, comp)))

    mov_split(('r8.x', 'r15.x', 'r15.y', 'r15.z'), tmp_vec0 + 0)
    mov_split(('r15.w', 'r16.x', 'r16.y', 'r16.z'), tmp_vec0 + 4)
    instrs.append(MOV_F32('r14.x', fvec(tmp_vec0 + 1), rpt=3))
    instrs.append(MOV_F32('r13.x', fvec(tmp_vec0 + 5), rpt=3))
    instrs.append(MOV_F32('r12.x', fvec(tmp_vec0 + 2), rpt=3))
    mov_split(('r10.y', 'r10.w', 'r11.y', 'r11.w'), tmp_vec0 + 6)
    mov_split(('r8.y', 'r8.w', 'r9.y', 'r9.w'), tmp_vec0 + 3)
    instrs.append(MOV_F32('r7.x', fvec(tmp_vec0 + 7), rpt=3))
    instrs += donor[261:348]


def emit_donor4_float_store2_lowcopy(instrs, dev, threads, acc_vec0):
    lib, io, isz, _ = get_envelope(dev, make_direct_donor_src_fp32(2, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    for i in range(5): instrs.append(MOV_F32(fvec(i), fvec(acc_vec0 + i), rpt=3))

    def mov_split(dsts, src_vec):
        for comp, dst in enumerate(dsts): instrs.append(MOV_F32(dst, fvec(src_vec, comp)))

    mov_split(('r8.x', 'r15.x', 'r15.y', 'r15.z'), 0)
    mov_split(('r15.w', 'r16.x', 'r16.y', 'r16.z'), 4)
    instrs.append(MOV_F32('r14.x', fvec(1), rpt=3))
    instrs.append(MOV_F32('r13.x', fvec(acc_vec0 + 5), rpt=3))
    instrs.append(MOV_F32('r12.x', fvec(2), rpt=3))
    mov_split(('r10.y', 'r10.w', 'r11.y', 'r11.w'), acc_vec0 + 6)
    mov_split(('r8.y', 'r8.w', 'r9.y', 'r9.w'), 3)
    instrs.append(MOV_F32('r7.x', fvec(acc_vec0 + 7), rpt=3))
    instrs += donor[261:348]


def emit_f32_global_stores(instrs, acc_vec0, ncols):
    for col in range(ncols):
        instrs.append(MOV_F32('r5.y', 'r7.y') if col == 0 else ADD_S('r5.y', 'r7.y', col * 32))
        for row in range(4):
            instrs.append(MOV_F32('r5.x', 'r7.x') if row == 0 else OR_B('r5.x', 'r7.x', row))
            emit_addr_float(instrs, 'r5.x', 'r5.y')
            instrs += [MOV_F32('r0.x', fvec(acc_vec0 + col * 4 + row), rpt=3), NOP(rpt=16), STG_F32('r2.x', 'r0.x', sy=True), NOP()]


def emit_f32_global_stores_gap(instrs, acc_vec0, ncols, post_gap=16):
    for col in range(ncols):
        instrs.append(MOV_F32('r5.y', 'r7.y') if col == 0 else ADD_S('r5.y', 'r7.y', col * 32))
        for row in range(4):
            instrs.append(MOV_F32('r5.x', 'r7.x') if row == 0 else OR_B('r5.x', 'r7.x', row))
            emit_addr_float(instrs, 'r5.x', 'r5.y')
            instrs += [MOV_F32('r0.x', fvec(acc_vec0 + col * 4 + row), rpt=3), NOP(rpt=16), STG_F32('r2.x', 'r0.x', sy=True), NOP(rpt=post_gap)]


def build_4x4_fp32_shader(dev, threads, coord_delay=4, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, alu_order='kk_row_col', direct_f32_loads=False, sampler_per_texture=False, scalar_f32_mads=False, ncols=1):
    instrs = prologue_direct4_fp32(dev, threads, ncols)
    if ncols == 1:
        row_base, col_base, ky_reg, kz_reg = 'r7.x', 'r7.y', 'r6.y', 'r6.z'
        instrs += [MOV_F32(row_base, 'r4.z'), MOV_F32(col_base, 'r5.x')]
    else:
        row_base, col_base, ky_reg, kz_reg = 'r7.x', 'r7.y', 'r6.y', 'r6.z'
        instrs += [MOV_F32(row_base, 'r6.y'), MOV_F32(col_base, 'r6.z')]
    instrs += [MOV_S32(ky_reg, 3), MOV_S32(kz_reg, 0)]
    acc_vec0, b_vec0 = 12, 8
    a_vec0 = acc_vec0 + 4 * ncols
    for vec in range(acc_vec0, acc_vec0 + 4 * ncols): emit_f32_vec_imm(instrs, vec, 0)
    if skip_b_loads:
        for vec in range(b_vec0, b_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)
    if skip_a_loads:
        for vec in range(a_vec0, a_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)

    first = True
    def acc(row, col):
        return acc_vec0 + col * 4 + row

    def emit_mad(row, col, kk):
        nonlocal first
        avec, bvec, accvec = a_vec0 + row, b_vec0 + kk, acc(row, col)
        if scalar_f32_mads:
            for comp in range(4):
                instrs.append(MAD_F32(fvec(accvec, comp), fvec(avec, kk), fvec(bvec, comp), fvec(accvec, comp), sy=first))
                first = False
        else:
            instrs.append(MAD_F32(fvec(accvec), fvec(avec, kk), fvec(bvec), fvec(accvec), rpt=3, sy=first, r=True))
            first = False

    def emit_b_loads(col=0):
        if skip_b_loads: return
        instrs.append(MOV_F32('r4.x', col_base) if col == 0 else ADD_S('r4.x', col_base, col * 32))
        for kk, yoff in enumerate((-3, -2, -1, 0)):
            instrs.append(MOV_F32('r4.y', ky_reg) if yoff == 0 else ADD_S('r4.y', ky_reg, yoff))
            emit_coord_wait(instrs, coord_delay)
            if direct_f32_loads: emit_isam_f32_vec(instrs, b_vec0 + kk, 'r4.x', 1, sampler_per_texture)
            else: emit_isam_h_to_f32_vec(instrs, b_vec0 + kk, 'r4.x', 1)

    def emit_a_loads():
        if skip_a_loads: return
        instrs.append(MOV_F32('r4.x', kz_reg))
        for row in range(4):
            instrs.append(MOV_F32('r4.y', row_base) if row == 0 else OR_B('r4.y', row_base, row))
            emit_coord_wait(instrs, coord_delay)
            if direct_f32_loads: emit_isam_f32_vec(instrs, a_vec0 + row, 'r4.x', 0, sampler_per_texture)
            else: emit_isam_h_to_f32_vec(instrs, a_vec0 + row, 'r4.x', 0)

    def emit_mads_col(col):
        if alu_order == 'kk_row_col':
            for kk in range(4):
                for row in range(4): emit_mad(row, col, kk)
        elif alu_order == 'row_col_kk':
            for row in range(4):
                for kk in range(4): emit_mad(row, col, kk)
        else: raise ValueError('unsupported fp32 alu order %s' % alu_order)

    loop_start = len(instrs)
    emit_b_loads(0)
    emit_a_loads()
    emit_mads_col(0)
    for col in range(1, ncols):
        emit_b_loads(col)
        emit_mads_col(col)
    instrs += [
        ADD_S('r0.x', kz_reg, 1),
        ADD_S(ky_reg, ky_reg, 4),
        CMPS_S_EQ(kz_reg, K4 - 1, nop=1),
        MOV_F32(kz_reg, 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        for vec in range(acc_vec0, acc_vec0 + 4 * ncols): emit_f32_vec_imm(instrs, vec, 0x44800000)
    if not no_store:
        if ncols == 1: emit_donor4_float_store(instrs, dev, threads, acc_vec0)
        elif ncols == 2:
            instrs += [MOV_F32('r6.y', row_base), MOV_F32('r6.z', col_base)]
            emit_donor4_float_store2(instrs, dev, threads, acc_vec0)
        else:
            instrs += [MOV_F32('r7.x', row_base), MOV_F32('r7.y', col_base)]
            emit_f32_global_stores(instrs, acc_vec0, ncols)
    instrs.append(END())
    return assemble(instrs), 1, max(a_vec0 + 4, 32 if ncols == 2 and not no_store else 20), loop_end - loop_start


def build_4x4_fp32_compact_shader(dev, threads, coord_delay=4, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, sampler_per_texture=True):
    instrs = prologue_direct4_fp32(dev, threads, 1)
    row_base, col_base, ky_reg, kz_reg = 'r7.x', 'r7.y', 'r6.y', 'r6.z'
    instrs += [MOV_F32(row_base, 'r4.z'), MOV_F32(col_base, 'r5.x'), MOV_S32(ky_reg, 3), MOV_S32(kz_reg, 0)]
    acc_vec0, b_vec0, a_vec = 8, 0, 4
    for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0)
    if skip_b_loads:
        for vec in range(b_vec0, b_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)

    first = True
    def emit_b_loads():
        if skip_b_loads: return
        instrs.append(MOV_F32('r5.x', col_base))
        for kk, yoff in enumerate((-3, -2, -1, 0)):
            instrs.append(MOV_F32('r5.y', ky_reg) if yoff == 0 else ADD_S('r5.y', ky_reg, yoff))
            emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, b_vec0 + kk, 'r5.x', 1, sampler_per_texture)

    def emit_a_load(row):
        if skip_a_loads:
            emit_f32_vec_imm(instrs, a_vec, 0x3f800000)
            return
        instrs.append(MOV_F32('r5.x', kz_reg))
        instrs.append(MOV_F32('r5.y', row_base) if row == 0 else OR_B('r5.y', row_base, row))
        emit_coord_wait(instrs, coord_delay)
        emit_isam_f32_vec(instrs, a_vec, 'r5.x', 0, sampler_per_texture)

    def emit_row_mads(row):
        nonlocal first
        acc_vec = acc_vec0 + row
        for kk in range(4):
            instrs.append(MAD_F32(fvec(acc_vec), fvec(a_vec, kk), fvec(b_vec0 + kk), fvec(acc_vec), rpt=3, sy=(first or kk == 0), r=True))
            first = False

    loop_start = len(instrs)
    emit_b_loads()
    for row in range(4):
        emit_a_load(row)
        emit_row_mads(row)
    instrs += [
        ADD_S('r0.x', kz_reg, 1),
        ADD_S(ky_reg, ky_reg, 4),
        CMPS_S_EQ(kz_reg, K4 - 1, nop=1),
        MOV_F32(kz_reg, 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x44800000)
    if not no_store:
        emit_donor4_float_store(instrs, dev, threads, acc_vec0)
    instrs.append(END())
    return assemble(instrs), 1, 12, loop_end - loop_start


def build_4x4_fp32_compact_preload_shader(dev, threads, coord_delay=4, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, sampler_per_texture=True):
    instrs = prologue_direct4_fp32(dev, threads, 1)
    row_base, col_base, ky_reg, kz_reg = 'r12.x', 'r12.y', 'r12.z', 'r12.w'
    instrs += [MOV_F32(row_base, 'r4.z'), MOV_F32(col_base, 'r5.x'), MOV_S32(ky_reg, 3), MOV_S32(kz_reg, 0)]
    acc_vec0, b_vec0, a_vec0 = 8, 0, 4
    for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0)
    if skip_b_loads:
        for vec in range(b_vec0, b_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)
    if skip_a_loads:
        for vec in range(a_vec0, a_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)

    first = True
    def emit_b_loads():
        if skip_b_loads: return
        for kk, yoff in enumerate((-3, -2, -1, 0)):
            dst = b_vec0 + kk
            instrs.append(MOV_F32(fvec(dst, 0), col_base))
            instrs.append(MOV_F32(fvec(dst, 1), ky_reg) if yoff == 0 else ADD_S(fvec(dst, 1), ky_reg, yoff))
            emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, sampler_per_texture)

    def emit_a_loads():
        if skip_a_loads: return
        for row in range(4):
            dst = a_vec0 + row
            instrs.append(MOV_F32(fvec(dst, 0), kz_reg))
            instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))
            emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, sampler_per_texture)

    def emit_mads():
        nonlocal first
        for kk in range(4):
            for row in range(4):
                acc_vec = acc_vec0 + row
                instrs.append(MAD_F32(fvec(acc_vec), fvec(a_vec0 + row, kk), fvec(b_vec0 + kk), fvec(acc_vec), rpt=3, sy=first, r=True))
                first = False

    loop_start = len(instrs)
    emit_b_loads()
    emit_a_loads()
    emit_mads()
    instrs += [
        ADD_S('r0.x', kz_reg, 1),
        ADD_S(ky_reg, ky_reg, 4),
        CMPS_S_EQ(kz_reg, K4 - 1, nop=1),
        MOV_F32(kz_reg, 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x44800000)
    if not no_store:
        instrs += [MOV_F32('r7.x', row_base), MOV_F32('r7.y', col_base), NOP(rpt=3)]
        emit_donor4_float_store(instrs, dev, threads, acc_vec0)
    instrs.append(END())
    return assemble(instrs), 1, 13, loop_end - loop_start


def build_4x4_fp32_compact_hybrid_shader(dev, threads, coord_delay=4, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, sampler_per_texture=True):
    instrs = prologue_direct4_fp32(dev, threads, 1)
    row_base, col_base, ky_reg, kz_reg = 'r7.x', 'r7.y', 'r7.z', 'r7.w'
    instrs += [MOV_F32(row_base, 'r4.z'), MOV_F32(col_base, 'r5.x'), MOV_S32(ky_reg, 3), MOV_S32(kz_reg, 0)]
    acc_vec0, b_vec0, a_vecs = 8, 0, (4, 5, 6, 12)
    for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0)
    if skip_b_loads:
        for vec in range(b_vec0, b_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)
    if skip_a_loads:
        for vec in a_vecs: emit_f32_vec_imm(instrs, vec, 0x3f800000)

    first = True
    def emit_b_loads():
        if skip_b_loads: return
        instrs.append(MOV_F32('r4.x', col_base))
        for kk, yoff in enumerate((-3, -2, -1, 0)):
            instrs.append(MOV_F32('r4.y', ky_reg) if yoff == 0 else ADD_S('r4.y', ky_reg, yoff))
            emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, b_vec0 + kk, 'r4.x', 1, sampler_per_texture)

    def emit_a_loads():
        if skip_a_loads: return
        for row, dst in enumerate(a_vecs):
            instrs.append(MOV_F32(fvec(dst, 0), kz_reg))
            instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))
            emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, sampler_per_texture)

    def emit_mads():
        nonlocal first
        for kk in range(4):
            for row, avec in enumerate(a_vecs):
                acc_vec = acc_vec0 + row
                instrs.append(MAD_F32(fvec(acc_vec), fvec(avec, kk), fvec(b_vec0 + kk), fvec(acc_vec), rpt=3, sy=first, r=True))
                first = False

    loop_start = len(instrs)
    emit_b_loads()
    emit_a_loads()
    emit_mads()
    instrs += [
        ADD_S('r0.x', kz_reg, 1),
        ADD_S(ky_reg, ky_reg, 4),
        CMPS_S_EQ(kz_reg, K4 - 1, nop=1),
        MOV_F32(kz_reg, 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x44800000)
    if not no_store:
        emit_donor4_float_store(instrs, dev, threads, acc_vec0)
    instrs.append(END())
    return assemble(instrs), 1, 13, loop_end - loop_start


def build_4x8_fp32_low_shader(dev, threads, coord_delay=4, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, sampler_per_texture=True, alu_order='kk_row_col', preload_b=False, batch_coords=False, hand_store=False, stream_b=False, convert_loads=False, alu_reps=1, stream_b_sync=True, stream_b_wait=-1):
    instrs = prologue_direct4_fp32(dev, threads, 2)
    row_base, col_base, ky_reg, kz_reg = ('r20.x', 'r20.y', 'r20.z', 'r20.w') if preload_b else ('r16.x', 'r16.y', 'r16.z', 'r16.w')
    instrs += [MOV_F32(row_base, 'r6.y'), MOV_F32(col_base, 'r6.z'), MOV_S32(ky_reg, 3), MOV_S32(kz_reg, 0)]
    b_vecs0, b_vecs1, a_vecs, acc_vec0 = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), 12) if preload_b else ((0, 1, 2, 3), (0, 1, 2, 3), (4, 5, 6, 7), 8)
    for vec in range(acc_vec0, acc_vec0 + 8): emit_f32_vec_imm(instrs, vec, 0)
    if skip_b_loads:
        for vec in sorted(set(b_vecs0 + b_vecs1)): emit_f32_vec_imm(instrs, vec, 0x3f800000)
    if skip_a_loads:
        for vec in a_vecs: emit_f32_vec_imm(instrs, vec, 0x3f800000)

    first = True
    def acc(row, col): return acc_vec0 + col * 4 + row

    def emit_b_coords(col, b_vecs):
        if skip_b_loads: return
        for kk, dst in enumerate(b_vecs):
            if col == 0: instrs.append(MOV_F32(fvec(dst, 0), col_base))
            else: instrs.append(ADD_S(fvec(dst, 0), col_base, 32))
            yoff = kk - 3
            instrs.append(MOV_F32(fvec(dst, 1), ky_reg) if yoff == 0 else ADD_S(fvec(dst, 1), ky_reg, yoff))

    def emit_b_isams(b_vecs):
        if skip_b_loads: return
        for dst in b_vecs:
            emit_coord_wait(instrs, coord_delay)
            if convert_loads: emit_isam_h_to_f32_vec(instrs, dst, fvec(dst, 0), 1)
            else: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, sampler_per_texture)

    def emit_b_loads(col, b_vecs):
        emit_b_coords(col, b_vecs)
        emit_b_isams(b_vecs)

    def emit_a_coords():
        if skip_a_loads: return
        for row, dst in enumerate(a_vecs):
            instrs.append(MOV_F32(fvec(dst, 0), kz_reg))
            instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))

    def emit_a_isams():
        if skip_a_loads: return
        for dst in a_vecs:
            emit_coord_wait(instrs, coord_delay)
            if convert_loads: emit_isam_h_to_f32_vec(instrs, dst, fvec(dst, 0), 0)
            else: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, sampler_per_texture)

    def emit_a_loads():
        emit_a_coords()
        emit_a_isams()

    def emit_mad(row, col, kk):
        nonlocal first
        b_vecs = b_vecs1 if col else b_vecs0
        instrs.append(MAD_F32(fvec(acc(row, col)), fvec(a_vecs[row], kk), fvec(b_vecs[kk]), fvec(acc(row, col)), rpt=3, sy=first, r=True))
        first = False

    def emit_mads_col(col):
        if alu_order == 'kk_row_col':
            for kk in range(4):
                for row in range(4): emit_mad(row, col, kk)
        elif alu_order == 'row_col_kk':
            for row in range(4):
                for kk in range(4): emit_mad(row, col, kk)
        elif alu_order == 'row_kk_col':
            for row in range(4):
                for kk in range(4): emit_mad(row, col, kk)
        else: raise ValueError('unsupported low 4x8 fp32 alu order %s' % alu_order)

    def emit_mads_preload_b():
        if alu_order == 'kk_row_col':
            for kk in range(4):
                for row in range(4):
                    for col in range(2): emit_mad(row, col, kk)
        elif alu_order == 'row_col_kk':
            for row in range(4):
                for col in range(2):
                    for kk in range(4): emit_mad(row, col, kk)
        elif alu_order == 'row_kk_col':
            for row in range(4):
                for kk in range(4):
                    for col in range(2): emit_mad(row, col, kk)
        elif alu_order == 'kk_col_row':
            for kk in range(4):
                for col in range(2):
                    for row in range(4): emit_mad(row, col, kk)
        elif alu_order == 'col_kk_row':
            for col in range(2):
                for kk in range(4):
                    for row in range(4): emit_mad(row, col, kk)
        else: raise ValueError('unsupported low 4x8 fp32 alu order %s' % alu_order)

    def emit_mads_stream_b():
        nonlocal first
        if alu_order != 'row_col_kk': raise ValueError('--stream-b low 4x8 FP32 currently requires --alu-order row_col_kk')
        # Start useful ALU on the first B column, then issue the second B column
        # loads while the remaining col0 MADs keep the texture scoreboard busy.
        for kk in range(4): emit_mad(0, 0, kk)
        emit_b_loads(1, b_vecs1)
        for row in range(1, 4):
            for kk in range(4): emit_mad(row, 0, kk)
        if stream_b_sync: first = True
        elif stream_b_wait >= 0: instrs.append(NOP(rpt=stream_b_wait))
        for row in range(4):
            for kk in range(4): emit_mad(row, 1, kk)

    loop_start = len(instrs)
    reps = alu_reps if no_store else 1
    if preload_b:
        if stream_b:
            emit_b_loads(0, b_vecs0)
            emit_a_loads()
            for _ in range(reps): emit_mads_stream_b()
        elif batch_coords:
            emit_b_coords(0, b_vecs0)
            emit_b_coords(1, b_vecs1)
            emit_a_coords()
            emit_b_isams(b_vecs0)
            emit_b_isams(b_vecs1)
            emit_a_isams()
            for _ in range(reps): emit_mads_preload_b()
        else:
            emit_b_loads(0, b_vecs0)
            emit_b_loads(1, b_vecs1)
            emit_a_loads()
            for _ in range(reps): emit_mads_preload_b()
    else:
        emit_b_loads(0, b_vecs0)
        emit_a_loads()
        for _ in range(reps): emit_mads_col(0)
        emit_b_loads(1, b_vecs0)
        for _ in range(reps): emit_mads_col(1)
    instrs += [
        ADD_S('r0.x', kz_reg, 1),
        ADD_S(ky_reg, ky_reg, 4),
        CMPS_S_EQ(kz_reg, K4 - 1, nop=1),
        MOV_F32(kz_reg, 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        for vec in range(acc_vec0, acc_vec0 + 8): emit_f32_vec_imm(instrs, vec, 0x44800000)
    if not no_store:
        if hand_store:
            instrs += [MOV_F32('r7.x', row_base), MOV_F32('r7.y', col_base)]
            emit_f32_global_stores_gap(instrs, acc_vec0, 2)
        else:
            instrs += [MOV_F32('r6.y', row_base), MOV_F32('r6.z', col_base)]
            if preload_b: emit_donor4_float_store2_lowcopy(instrs, dev, threads, acc_vec0)
            else: emit_donor4_float_store2(instrs, dev, threads, acc_vec0)
    instrs.append(END())
    return assemble(instrs), 1, (21 if (hand_store or preload_b) else 32) if not no_store else (21 if preload_b else 17), loop_end - loop_start


def build_4x8_fp32_pipeline_shader(dev, threads, coord_delay=2, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, sampler_per_texture=True):
    instrs = prologue_direct4_fp32(dev, threads, 2)
    row_base, col_base, ky_reg, kz_reg = 'r29.x', 'r29.y', 'r29.z', 'r29.w'
    instrs += [MOV_F32(row_base, 'r6.y'), MOV_F32(col_base, 'r6.z'), MOV_S32(ky_reg, 3), MOV_S32(kz_reg, 0)]
    acc_vec0 = 12
    b0 = ((0, 1, 2, 3), (4, 5, 6, 7))
    b1 = ((21, 22, 23, 24), (25, 26, 27, 28))
    avecs = (8, 9, 10, 11)
    for vec in range(acc_vec0, acc_vec0 + 8): emit_f32_vec_imm(instrs, vec, 0)
    if skip_b_loads:
        for vec in sorted(set(b0[0] + b0[1] + b1[0] + b1[1])): emit_f32_vec_imm(instrs, vec, 0x3f800000)
    if skip_a_loads:
        for vec in avecs: emit_f32_vec_imm(instrs, vec, 0x3f800000)

    first = True
    def acc(row, col): return acc_vec0 + col * 4 + row

    def emit_b_pair(buf, ky_src):
        if not skip_b_loads:
            for col, bvecs in enumerate(buf):
                for kk, dst in enumerate(bvecs):
                    instrs.append(MOV_F32(fvec(dst, 0), col_base) if col == 0 else ADD_S(fvec(dst, 0), col_base, 32))
                    yoff = kk - 3
                    instrs.append(MOV_F32(fvec(dst, 1), ky_src) if yoff == 0 else ADD_S(fvec(dst, 1), ky_src, yoff))
                    emit_coord_wait(instrs, coord_delay)
                    emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, sampler_per_texture)

    def emit_a(kz_src):
        if not skip_a_loads:
            for row, dst in enumerate(avecs):
                instrs.append(MOV_F32(fvec(dst, 0), kz_src))
                instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))
                emit_coord_wait(instrs, coord_delay)
                emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, sampler_per_texture)

    def emit_mad(buf, row, col, kk):
        nonlocal first
        bvecs = buf[1] if col else buf[0]
        instrs.append(MAD_F32(fvec(acc(row, col)), fvec(avecs[row], kk), fvec(bvecs[kk]), fvec(acc(row, col)), rpt=3, sy=first, r=True))
        first = False

    def emit_row(buf, row):
        for col in range(2):
            for kk in range(4): emit_mad(buf, row, col, kk)

    emit_b_pair(b0, ky_reg)
    loop_start = len(instrs)
    emit_a(kz_reg)
    emit_row(b0, 0)
    instrs += [ADD_S('r20.z', ky_reg, 4), ADD_S('r20.w', kz_reg, 1)]
    emit_b_pair(b1, 'r20.z')
    for row in range(1, 4): emit_row(b0, row)
    emit_a('r20.w')
    emit_row(b1, 0)
    instrs += [ADD_S('r20.z', ky_reg, 8), ADD_S('r20.w', kz_reg, 2)]
    emit_b_pair(b0, 'r20.z')
    for row in range(1, 4): emit_row(b1, row)
    instrs += [
        ADD_S('r0.x', kz_reg, 2),
        ADD_S(ky_reg, ky_reg, 8),
        CMPS_S_EQ(kz_reg, K4 - 2, nop=1),
        MOV_F32(kz_reg, 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        for vec in range(acc_vec0, acc_vec0 + 8): emit_f32_vec_imm(instrs, vec, 0x44800000)
    if not no_store:
        instrs += [MOV_F32('r6.y', row_base), MOV_F32('r6.z', col_base)]
        emit_donor4_float_store2_lowcopy(instrs, dev, threads, acc_vec0)
    instrs.append(END())
    return assemble(instrs), 1, 30, (loop_end - loop_start) // 2


def reduce_store(instrs, row_reg, col_reg, acc_base):
    emit_addr(instrs, row_reg, col_reg)
    for lane in range(4): instrs.append(ADD_F(lane, acc_base + lane, acc_base + 4 + lane))
    instrs.append(NOP(rpt=3))
    for lane in range(4): instrs.append(ADD_F(lane, lane, acc_base + 8 + lane))
    instrs.append(NOP(rpt=3))
    for lane in range(4): instrs.append(ADD_F(lane, lane, acc_base + 12 + lane))
    instrs += [NOP(rpt=3), STG_F16('r2.x', 'hr0.x', sy=True), NOP()]


def emit_donor4_stores(instrs, dev, threads, acc0, ncols, start_col=0, count=None, store_gap=-1, store_or_cols=False):
    lib, io, isz, _ = get_envelope(dev, make_donor_src(1, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    addr = (donor[109:111] + [NOP(rpt=16)] + donor[115:128] + [NOP(rpt=16)] +
            donor[140:144] + [NOP(rpt=16)] + donor[160:168] + [NOP(rpt=16)] + donor[184:188])
    stores = donor[188:195]
    if count is None: count = ncols - start_col
    if store_or_cols:
        instrs.append(MOV_F32('r6.x', 'r7.y'))
    elif start_col: instrs.append(ADD_S('r7.y', 'r7.y', start_col * 32))
    for col in range(start_col, start_col + count):
        if store_or_cols:
            if col: instrs.append(OR_B('r7.y', 'r6.x', col * 32))
        elif col != start_col: instrs.append(ADD_S('r7.y', 'r7.y', 32))
        instrs += addr
        for row in range(4): instrs.append(MOV_H(row * 4, acc0 + (row * ncols + col) * 4, rpt=3))
        instrs += stores
        if store_gap >= 0 and col != start_col + count - 1: instrs.append(NOP(rpt=store_gap))


def emit_donor4_stores_scalar_offsets(instrs, dev, threads, acc0, ncols):
    if ncols != 4: raise ValueError('--store-scalar-offsets requires --ncols 4')
    lib, io, isz, _ = get_envelope(dev, make_donor_src(1, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    addr_tail = ([NOP(rpt=16)] + donor[115:128] + [NOP(rpt=16)] + donor[140:144] +
                 [NOP(rpt=16)] + donor[160:168] + [NOP(rpt=16)] + donor[184:188])
    stores = donor[188:195]
    for col in range(ncols):
        instrs += [SHL_B('r0.x', 'r7.x', 10, jp=True), SHL_B('r0.y', 'r7.y', 2)]
        if col:
            instrs += [MOV_S32('r6.x', col * 128), NOP(rpt=2), ADD_S_REG('r0.y', 'r0.y', 'r6.x')]
        instrs += addr_tail
        for row in range(4): instrs.append(MOV_H(row * 4, acc0 + (row * ncols + col) * 4, rpt=3))
        instrs += stores


def emit_donor4_stores_shlg_offsets(instrs, dev, threads, acc0, ncols):
    if ncols != 4: raise ValueError('--store-shlg-offsets requires --ncols 4')
    lib, io, isz, _ = get_envelope(dev, make_donor_src(1, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    addr_tail = ([NOP(rpt=16)] + donor[115:128] + [NOP(rpt=16)] + donor[140:144] +
                 [NOP(rpt=16)] + donor[160:168] + [NOP(rpt=16)] + donor[184:188])
    stores = donor[188:195]
    for col in range(ncols):
        instrs.append(SHL_B('r0.x', 'r7.x', 10, jp=True))
        instrs.append(SHL_B('r0.y', 'r7.y', 2) if col == 0 else SHLG_IMM('r0.y', 2, 'r7.y', col * 128))
        instrs += addr_tail
        for row in range(4): instrs.append(MOV_H(row * 4, acc0 + (row * ncols + col) * 4, rpt=3))
        instrs += stores


def emit_hybrid4_stores(instrs, dev, threads, acc0, ncols):
    if ncols != 4: raise ValueError('--hybrid-store requires --ncols 4')
    emit_donor4_stores(instrs, dev, threads, acc0, ncols, count=2)
    for col in range(2, 4):
        instrs.append(ADD_S('r7.y', 'r7.y', 32))
        for row in range(4):
            if row == 0: row_reg = 'r7.x'
            else:
                instrs.append(OR_B('r7.w', 'r7.x', row))
                row_reg = 'r7.w'
            store_output(instrs, row_reg, 'r7.y', acc0 + (row * ncols + col) * 4)
            instrs.append(NOP(rpt=16))


def emit_donor2_store(instrs, dev, threads, acc0):
    lib, io, isz, _ = get_envelope(dev, make_donor_src(2, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    instrs += [MOV_F32('r24.x', 'r7.x'), MOV_F32('r24.y', 'r7.y')]
    instrs += donor[217:330]
    for row in range(4):
        for col in range(2):
            out = (row * 2 + col) * 4
            instrs.append(MOV_H(out, acc0 + out, rpt=3))
    instrs += donor[360:383]


def emit_native4_store(instrs, dev, threads, acc0, ncols):
    if ncols != 4: raise ValueError('--native-store is implemented for --ncols 4')
    lib, io, isz, _ = get_envelope(dev, make_direct_donor_src(ncols, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    instrs += [MOV_F32('r24.x', 'r7.x'), MOV_F32('r24.y', 'r7.y')]
    instrs += donor[394:481]
    for row in range(4):
        for col in range(ncols):
            out = row * ncols + col
            instrs.append(MOV_H(out * 4, acc0 + out * 4, rpt=3))
    instrs += donor[537:575]


def build_4x16_pipeline_shader(dev, threads, native_store=False, no_store=False, coord_delay=0):
    if native_store:
        lib, io, isz, _ = get_envelope(dev, make_direct_donor_src(4, threads))
        donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
        instrs = donor[:12] + [NOP(rpt=2), MOV_F32('r7.x', 'r24.x'), MOV_F32('r7.y', 'r24.y'), MOV_S32('r6.y', 3, sy=True), MOV_S32('r6.z', 0)]
    else:
        instrs = prologue_4x2(dev, threads)
        instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2)]
        for i in range(3):
            instrs.append(ADD_S_REG('r7.y', 'r7.y', 'r6.w'))
            instrs.append(NOP(rpt=2))
    instrs += [MOV_F32('r4.y', 'r7.y'), MOV_F32('r4.w', 'r7.y'), MOV_F32('r5.y', 'r7.y'), MOV_F32('r5.w', 'r7.y')]
    instrs.append(ADD_S('r7.z', 'r7.y', 32))

    acc0 = _hreg('hr24.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    def emit_a4(a_base, kx):
        nonlocal instrs
        instrs += [
            MOV_F32('r2.z', 'r7.x'), OR_B('r3.x', 'r7.x', 1), OR_B('r3.z', 'r7.x', 2), OR_B('r4.x', 'r7.x', 3),
            MOV_F32('r2.y', kx), MOV_F32('r2.w', kx), MOV_F32('r3.y', kx), MOV_F32('r3.w', kx),
        ]
        for i, coord in enumerate(['r2.y', 'r2.w', 'r3.y', 'r3.w']):
            emit_coord_wait(instrs, coord_delay)
            instrs.append(ISAM_F16(a_base + i * 4, coord, 0))

    def emit_bpair(b_base, col_base, ky):
        nonlocal instrs
        if col_base:
            instrs.append(ADD_S('r0.y', 'r7.y', col_base * 32))
            x0 = 'r0.y'
        else: x0 = 'r7.y'
        instrs += [MOV_F32('r4.y', x0), MOV_F32('r4.w', x0), MOV_F32('r5.y', x0), MOV_F32('r5.w', x0)]
        instrs.append(ADD_S('r7.z', 'r7.y', (col_base + 1) * 32))
        instrs += [ADD_S('r4.z', ky, -3), ADD_S('r5.x', ky, -2), ADD_S('r5.z', ky, -1), MOV_F32('r6.x', ky)]
        for i, coord in enumerate(['r4.y', 'r4.w', 'r5.y', 'r5.w']):
            emit_coord_wait(instrs, coord_delay)
            instrs.append(ISAM_F16(b_base + i * 4, coord, 1))
        for i, yreg in enumerate(['r4.z', 'r5.x', 'r5.z', 'r6.x']):
            instrs.append(MOV_F32('r7.w', yreg))
            emit_coord_wait(instrs, coord_delay)
            instrs.append(ISAM_F16(b_base + 16 + i * 4, 'r7.z', 1))

    def first_mad(a_base, b_base, col_base):
        group = col_base * 4
        instrs.append(MAD_F16(acc0 + group, a_base, b_base, acc0 + group, rpt=3, sy=True, r=True))

    def emit_mads_pair(a_base, b_base, col_base, skip_first=True):
        for row in range(4):
            for col in range(2):
                for kk in range(4):
                    if skip_first and row == 0 and col == 0 and kk == 0: continue
                    out_col = col_base + col
                    group = (row * 4 + out_col) * 4
                    instrs.append(MAD_F16(acc0 + group, a_base + row * 4 + kk, b_base + col * 16 + kk * 4, acc0 + group, rpt=3, r=True))

    def emit_step(a_cur, a_next, next_k_add):
        nonlocal instrs
        first_mad(a_cur, _hreg('hr4.x'), 0)
        emit_bpair(_hreg('hr12.x'), 2, 'r6.y')
        emit_mads_pair(a_cur, _hreg('hr4.x'), 0)
        first_mad(a_cur, _hreg('hr12.x'), 2)
        instrs += [ADD_S('r0.x', 'r6.z', next_k_add), ADD_S('r0.z', 'r6.y', next_k_add * 4)]
        emit_a4(a_next, 'r0.x')
        emit_bpair(_hreg('hr4.x'), 0, 'r0.z')
        emit_mads_pair(a_cur, _hreg('hr12.x'), 2)

    emit_a4(_hreg('hr0.x'), 'r6.z')
    emit_bpair(_hreg('hr4.x'), 0, 'r6.y')
    loop_start = len(instrs)
    emit_step(_hreg('hr0.x'), _hreg('hr20.x'), 1)
    emit_step(_hreg('hr20.x'), _hreg('hr0.x'), 2)
    instrs += [CMPS_S_EQ('r6.z', K4 - 2, nop=1), MOV_F32('r6.z', 'r0.x'), MOV_F32('r6.y', 'r0.z'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    if native_store: emit_native4_store(instrs, dev, threads, acc0, 4)
    elif not no_store: emit_donor4_stores(instrs, dev, threads, acc0, 4)
    instrs.append(END())
    return assemble(instrs), (loop_end - loop_start) // 2


def build_4xn_shader(dev, threads, ncols=2, direct=False, quad_a=False, store_constant=False, post_constant=False, donor_store=False, donor2_store=False, native_store=False, hybrid_store=False, pipeline=False, preload_b=False, stream_b=False, stream_b_no_sync=False, b_kk_pipeline=False, b_first=False, compact_acc=False, stable_bx=False, stable_ay=False, low_a_coords=False, inc_coords=False, persistent_coords=False, k_unroll=1, row_col_kk=False, alu_order='auto', first_sync_only=False, row_sync=False, no_store=False, skip_a_loads=False, skip_b_loads=False, alu_reps=1, coord_delay=4, store_gap=-1, store_start=0, store_count=-1, store_or_cols=False, store_scalar_offsets=False, store_shlg_offsets=False):
    if not direct and ncols != 2: raise ValueError('partial-accumulator mode is only implemented for ncols=2')
    if pipeline:
        if not direct or ncols != 4: raise ValueError('--pipeline requires --direct --ncols 4')
        if store_constant or donor2_store: raise ValueError('--pipeline does not support --store-constant or --donor2-store')
        return build_4x16_pipeline_shader(dev, threads, native_store=native_store, no_store=no_store, coord_delay=coord_delay)
    if native_store:
        lib, io, isz, _ = get_envelope(dev, make_direct_donor_src(ncols, threads))
        donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
        instrs = donor[:12]
        instrs.append(NOP(rpt=2))
        instrs += [MOV_F32('r7.x', 'r24.x'), MOV_F32('r7.y', 'r24.y')]
        instrs += [MOV_S32('r6.y', 3, sy=True), MOV_S32('r6.z', 0)]
    else:
        instrs = prologue_4x2(dev, threads)
        instrs += [
            MOV_F32('r6.w', 'r51.w'), NOP(rpt=2),
            SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2),
        ]
        for i in range(ncols - 1):
            instrs.append(ADD_S_REG('r7.y', 'r7.y', 'r6.w'))
            instrs.append(NOP(rpt=2))
    instrs += [MOV_F32('r4.y', 'r7.y'), MOV_F32('r4.w', 'r7.y'), MOV_F32('r5.y', 'r7.y'), MOV_F32('r5.w', 'r7.y')]
    if quad_a: instrs.append(MOV_F32('r6.w', 'r0.x'))  # keep lane id for the divergent A-load mask
    instrs.append(ADD_S('r7.z', 'r7.y', 32))   # second col4 block x coordinate

    if preload_b and (not direct or ncols != 4): raise ValueError('--preload-b requires --direct --ncols 4')
    if stream_b and (not direct or ncols != 4 or preload_b or alu_reps != 1): raise ValueError('--stream-b requires --direct --ncols 4 without --preload-b/--alu-reps')
    if stream_b_no_sync and not stream_b: raise ValueError('--stream-b-no-sync requires --stream-b')
    if b_kk_pipeline and (not direct or ncols != 4 or preload_b or stream_b or alu_reps != 1): raise ValueError('--b-kk-pipeline requires --direct --ncols 4 without other B schedules')
    if b_first and (not direct or ncols != 4 or preload_b or stream_b or b_kk_pipeline or quad_a): raise ValueError('--b-first requires plain non-quad --direct --ncols 4')
    if low_a_coords and not stable_ay: raise ValueError('--low-a-coords requires --stable-ay')
    if compact_acc and (not direct or preload_b): raise ValueError('--compact-acc requires direct mode without --preload-b')
    if stable_bx and (not direct or ncols != 4 or preload_b or stream_b or b_kk_pipeline or (alu_reps != 1 and not no_store)): raise ValueError('--stable-bx requires plain --direct --ncols 4')
    if stable_ay and (quad_a or not direct or b_kk_pipeline): raise ValueError('--stable-ay requires non-quad direct mode without --b-kk-pipeline')
    if (skip_a_loads or skip_b_loads) and (not no_store or not direct or preload_b or stream_b or b_kk_pipeline): raise ValueError('--skip-*-loads are no-store plain direct profiling probes')
    if inc_coords and not (stable_bx and stable_ay): raise ValueError('--inc-coords requires --stable-bx --stable-ay')
    if persistent_coords and not inc_coords: raise ValueError('--persistent-coords requires --inc-coords')
    if K4 % k_unroll != 0: raise ValueError('--k-unroll must divide K/4')
    if alu_order == 'auto': alu_order = 'row_col_kk' if row_col_kk else 'kk_row_col'
    acc0 = _hreg('hr20.x') if preload_b else _hreg('hr12.x') if compact_acc else _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0x6400 if store_constant else 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    acc_groups = 4 * ncols if direct else 32
    for base in range(acc0 + 4, acc0 + acc_groups * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    if skip_a_loads:
        for base in range(_hreg('hr0.x'), _hreg('hr4.x'), 4): instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
    if skip_b_loads:
        for base in range(_hreg('hr4.x'), _hreg('hr12.x'), 4): instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
    if stable_bx:
        instrs += [ADD_S('r0.x', 'r7.y', 64), NOP(rpt=2), MOV_F32('r0.z', 'r0.x'), MOV_F32('r1.x', 'r0.x'), MOV_F32('r1.z', 'r0.x')]
    a_xregs = ['r2.x', 'r2.z', 'r3.x', 'r3.z'] if low_a_coords else ['r8.x', 'r8.z', 'r9.x', 'r9.z']
    a_yregs = ['r2.y', 'r2.w', 'r3.y', 'r3.w'] if low_a_coords else ['r8.y', 'r8.w', 'r9.y', 'r9.w']
    col3_x = 'r6.w' if low_a_coords else 'r3.w'
    if stable_ay:
        instrs += [MOV_F32(a_yregs[0], 'r7.x'), OR_B(a_yregs[1], 'r7.x', 1), OR_B(a_yregs[2], 'r7.x', 2), OR_B(a_yregs[3], 'r7.x', 3)]
        if inc_coords: instrs.append(ADD_S(col3_x, 'r7.y', 96))

    if store_constant:
        loop_start = loop_end = len(instrs)
    else:

        if persistent_coords:
            instrs += [
                MOV_F32(a_xregs[0], 'r6.z'), MOV_F32(a_xregs[1], 'r6.z'), MOV_F32(a_xregs[2], 'r6.z'), MOV_F32(a_xregs[3], 'r6.z'),
                ADD_S('r4.z', 'r6.y', -3), ADD_S('r5.x', 'r6.y', -2), ADD_S('r5.z', 'r6.y', -1), MOV_F32('r6.x', 'r6.y'),
            ]

        loop_start = len(instrs)

        def emit_one_k(kx_reg, ky_reg, sync_first=True, reuse_coords=False):
          nonlocal instrs
          if stable_ay and not reuse_coords:
              instrs += [MOV_F32(a_xregs[0], kx_reg), MOV_F32(a_xregs[1], kx_reg), MOV_F32(a_xregs[2], kx_reg), MOV_F32(a_xregs[3], kx_reg)]
          elif not stable_ay:
              instrs += [
                  MOV_F32('r2.z', 'r7.x'), OR_B('r3.x', 'r7.x', 1), OR_B('r3.z', 'r7.x', 2), OR_B('r4.x', 'r7.x', 3),
              ]
          if stable_bx:
              if not stable_ay: instrs.append(ADD_S('r7.z', 'r7.y', 32))
          else: instrs += [MOV_F32('r4.y', 'r7.y'), MOV_F32('r4.w', 'r7.y'), MOV_F32('r5.y', 'r7.y'), MOV_F32('r5.w', 'r7.y')]
          if not stable_ay:
              instrs += [MOV_F32('r2.y', kx_reg), MOV_F32('r2.w', kx_reg), MOV_F32('r3.y', kx_reg), MOV_F32('r3.w', kx_reg)]
          if not reuse_coords: instrs += [ADD_S('r4.z', ky_reg, -3), ADD_S('r5.x', ky_reg, -2), ADD_S('r5.z', ky_reg, -1), MOV_F32('r6.x', ky_reg)]
          if quad_a:
              instrs += [
                  MOV_F32('r0.z', 'r6.w'),
                  AND_B('r0.x', 'r0.z', 3, nop=3),
                  CMPS_S_EQ('r0.x', 0),
                  NOP(rpt=5),
                  BR(5),
              ]
          a_coords = [('hr3.x', a_xregs[0]), ('hr2.x', a_xregs[1]), ('hr1.x', a_xregs[2]), ('hr0.x', a_xregs[3])] if stable_ay else [('hr3.x', 'r2.y'), ('hr2.x', 'r2.w'), ('hr1.x', 'r3.y'), ('hr0.x', 'r3.w')]
          a_loaded = False
          def emit_a_loads_now():
            nonlocal a_loaded
            if skip_a_loads or a_loaded: return
            a_loaded = True
            for dst, coord in a_coords:
                emit_coord_wait(instrs, coord_delay)
                instrs.append(ISAM_F16(dst, coord, 0))
          if not b_first:
            emit_a_loads_now()
          if quad_a:
              instrs += [
                  SHL_B('r6.w', 'r6.w', 0, jp=True, ss=True, nop=3),
                  MOV_S32('r7.w', 0),
              ]
              for i, reg in enumerate(['r0.x', 'r0.y', 'r0.z', 'r0.w', 'r1.x', 'r1.y', 'r1.z', 'r1.w']):
                  instrs.append(QUAD_BRCST(reg, reg, 'r7.w', typ=3, wrmask=15, sy=(i == 0)))
          a_regs = [_hreg('hr3.x'), _hreg('hr2.x'), _hreg('hr1.x'), _hreg('hr0.x')]
          b_regs = [[_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg('hr7.x')],
                    [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')]]
          first = sync_first
          if direct:
            if b_kk_pipeline:
              b0_coords = [('r4.y', 'r4.z'), ('r4.w', 'r5.x'), ('r5.y', 'r5.z'), ('r5.w', 'r6.x')]
              b1_coords = [('r8.x', 'r8.y'), ('r8.z', 'r8.w'), ('r9.x', 'r9.y'), ('r9.z', 'r9.w')]
              for coords in (b0_coords, b1_coords):
                  for col, (xreg, _) in enumerate(coords):
                      instrs.append(MOV_F32(xreg, 'r7.y') if col == 0 else ADD_S(xreg, 'r7.y', col * 32))
              b0 = [_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg('hr7.x')]
              b1 = [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')]
              def emit_b_kk(buf, coords, yoff):
                  nonlocal instrs
                  if yoff == 0: instrs.append(MOV_F32('r7.w', 'r6.y'))
                  else: instrs.append(ADD_S('r7.w', 'r6.y', yoff))
                  for dst, (coord, yreg) in zip(buf, coords):
                      instrs.append(MOV_F32(yreg, 'r7.w'))
                      emit_coord_wait(instrs, coord_delay)
                      instrs.append(ISAM_F16(dst, coord, 1))
              def emit_compute_kk(buf, kk, force_sync=False):
                  nonlocal first
                  for row in range(4):
                      for col in range(4):
                          group = (row * ncols + col) * 4
                          instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, buf[col], acc0 + group, rpt=3, sy=(first or force_sync or row_sync), r=True))
                          first = False
                          force_sync = False
              emit_b_kk(b0, b0_coords, -3)
              emit_b_kk(b1, b1_coords, -2)
              emit_compute_kk(b0, 0)
              emit_b_kk(b0, b0_coords, -1)
              emit_compute_kk(b1, 1, force_sync=row_sync)
              emit_b_kk(b1, b1_coords, 0)
              emit_compute_kk(b0, 2, force_sync=row_sync)
              emit_compute_kk(b1, 3, force_sync=row_sync)
            elif stream_b:
              for dst, coord in [('hr4.x', 'r4.y'), ('hr5.x', 'r4.w'), ('hr6.x', 'r5.y'), ('hr7.x', 'r5.w')]:
                  emit_coord_wait(instrs, coord_delay)
                  instrs.append(ISAM_F16(dst, coord, 1))
              for dst, yreg in [('hr8.x', 'r4.z'), ('hr9.x', 'r5.x'), ('hr10.x', 'r5.z'), ('hr11.x', 'r6.x')]:
                  instrs.append(MOV_F32('r7.w', yreg))
                  emit_coord_wait(instrs, coord_delay)
                  instrs.append(ISAM_F16(dst, 'r7.z', 1))
              instrs += [ADD_S('r0.y', 'r7.y', 64), ADD_S('r7.z', 'r7.y', 96)]
              k_regs = ['r4.z', 'r5.x', 'r5.z', 'r6.x']
              for kk in range(4):
                  for row in range(4):
                      for col in range(2):
                          group = (row * ncols + col) * 4
                          instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, b_regs[col][kk], acc0 + group, rpt=3, sy=first, r=True))
                          first = False
                  instrs.append(MOV_F32('r0.z', k_regs[kk]))
                  emit_coord_wait(instrs, coord_delay)
                  instrs.append(ISAM_F16(b_regs[0][kk], 'r0.y', 1))
                  instrs.append(MOV_F32('r7.w', k_regs[kk]))
                  emit_coord_wait(instrs, coord_delay)
                  instrs.append(ISAM_F16(b_regs[1][kk], 'r7.z', 1))
              if not stream_b_no_sync: first = True
              for row in range(4):
                  for col in range(2):
                      out_col = 2 + col
                      for kk in range(4):
                          group = (row * ncols + out_col) * 4
                          instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, b_regs[col][kk], acc0 + group, rpt=3, sy=(first or (row_sync and kk == 0)), r=True))
                          first = False
            elif preload_b:
              def emit_b_pair(col_base, b_base):
                  nonlocal instrs
                  if col_base != 0:
                      instrs += [
                          ADD_S('r0.y', 'r7.y', col_base * 32),
                          MOV_F32('r4.y', 'r0.y'), MOV_F32('r4.w', 'r0.y'), MOV_F32('r5.y', 'r0.y'), MOV_F32('r5.w', 'r0.y'),
                          ADD_S('r7.z', 'r7.y', (col_base + 1) * 32),
                      ]
                  for i, coord in enumerate(['r4.y', 'r4.w', 'r5.y', 'r5.w']):
                      emit_coord_wait(instrs, coord_delay)
                      instrs.append(ISAM_F16(b_base + i * 4, coord, 1))
                  for i, yreg in enumerate(['r4.z', 'r5.x', 'r5.z', 'r6.x']):
                      instrs.append(MOV_F32('r7.w', yreg))
                      emit_coord_wait(instrs, coord_delay)
                      instrs.append(ISAM_F16(b_base + 16 + i * 4, 'r7.z', 1))
              emit_b_pair(0, _hreg('hr4.x'))
              emit_b_pair(2, _hreg('hr12.x'))
              b_all = [[_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg('hr7.x')],
                       [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')],
                       [_hreg('hr12.x'), _hreg('hr13.x'), _hreg('hr14.x'), _hreg('hr15.x')],
                       [_hreg('hr16.x'), _hreg('hr17.x'), _hreg('hr18.x'), _hreg('hr19.x')]]
              def emit(col, row, kk):
                  nonlocal first
                  group = (row * ncols + col) * 4
                  instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, b_all[col][kk], acc0 + group, rpt=3, sy=(first or (row_sync and kk == 0)), r=True))
                  first = False
              for _ in range(alu_reps):
                if alu_order == 'row_col_kk':
                  for row in range(4):
                    for col in range(ncols):
                      for kk in range(4): emit(col, row, kk)
                elif alu_order == 'row_kk_col':
                  for row in range(4):
                    for kk in range(4):
                      for col in range(ncols): emit(col, row, kk)
                elif alu_order == 'col_kk_row':
                  for col in range(ncols):
                    for kk in range(4):
                      for row in range(4): emit(col, row, kk)
                elif alu_order == 'kk_col_row':
                  for kk in range(4):
                    for col in range(ncols):
                      for row in range(4): emit(col, row, kk)
                else:
                  for kk in range(4):
                    for row in range(4):
                      for col in range(ncols): emit(col, row, kk)
            else:
             for col_base in range(0, ncols, 2):
              pair_cols = min(2, ncols - col_base)
              if stable_bx and col_base == 2:
                  instrs += [MOV_F32('r0.y', 'r4.z'), MOV_F32('r0.w', 'r5.x'), MOV_F32('r1.y', 'r5.z'), MOV_F32('r1.w', 'r6.x')]
                  if not (stable_ay and inc_coords): instrs.append(ADD_S('r3.w' if stable_ay else 'r7.z', 'r7.y', 96))
              elif col_base != 0:
                  instrs += [
                      ADD_S('r0.y', 'r7.y', col_base * 32),
                      MOV_F32('r4.y', 'r0.y'), MOV_F32('r4.w', 'r0.y'), MOV_F32('r5.y', 'r0.y'), MOV_F32('r5.w', 'r0.y'),
                  ]
              if pair_cols == 2 and col_base != 0 and not stable_bx: instrs.append(ADD_S('r7.z', 'r7.y', (col_base + 1) * 32))
              first_coords = [('hr4.x', 'r0.x'), ('hr5.x', 'r0.z'), ('hr6.x', 'r1.x'), ('hr7.x', 'r1.z')] if stable_bx and col_base == 2 else [('hr4.x', 'r4.y'), ('hr5.x', 'r4.w'), ('hr6.x', 'r5.y'), ('hr7.x', 'r5.w')]
              if not skip_b_loads:
                for dst, coord in first_coords:
                    emit_coord_wait(instrs, coord_delay)
                    instrs.append(ISAM_F16(dst, coord, 1))
              if pair_cols == 2 and not skip_b_loads:
                  second_coord, second_y = (col3_x, 'r4.x') if stable_bx and stable_ay and col_base == 2 else ('r7.z', 'r7.w')
                  for dst, yreg in [('hr8.x', 'r4.z'), ('hr9.x', 'r5.x'), ('hr10.x', 'r5.z'), ('hr11.x', 'r6.x')]:
                      instrs.append(MOV_F32(second_y, yreg))
                      emit_coord_wait(instrs, coord_delay)
                      instrs.append(ISAM_F16(dst, second_coord, 1))
              if b_first and col_base == 0: emit_a_loads_now()
              for _ in range(alu_reps):
                def emit_pair(row, col, kk):
                    nonlocal first
                    out_col = col_base + col
                    group = (row * ncols + out_col) * 4
                    instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, b_regs[col][kk], acc0 + group, rpt=3, sy=(first or (row_sync and kk == 0)), r=True))
                    first = False
                if alu_order == 'row_col_kk':
                  for row in range(4):
                    for col in range(pair_cols):
                      for kk in range(4): emit_pair(row, col, kk)
                elif alu_order == 'row_kk_col':
                  for row in range(4):
                    for kk in range(4):
                      for col in range(pair_cols): emit_pair(row, col, kk)
                elif alu_order == 'col_kk_row':
                  for col in range(pair_cols):
                    for kk in range(4):
                      for row in range(4): emit_pair(row, col, kk)
                elif alu_order == 'kk_col_row':
                  for kk in range(4):
                    for col in range(pair_cols):
                      for row in range(4): emit_pair(row, col, kk)
                else:
                  for kk in range(4):
                    for row in range(4):
                      for col in range(pair_cols): emit_pair(row, col, kk)
          else:
            for dst, coord in [('hr4.x', 'r4.y'), ('hr5.x', 'r4.w'), ('hr6.x', 'r5.y'), ('hr7.x', 'r5.w')]:
                emit_coord_wait(instrs, coord_delay)
                instrs.append(ISAM_F16(dst, coord, 1))
            for dst, yreg in [('hr8.x', 'r4.z'), ('hr9.x', 'r5.x'), ('hr10.x', 'r5.z'), ('hr11.x', 'r6.x')]:
                instrs.append(MOV_F32('r7.w', yreg))
                emit_coord_wait(instrs, coord_delay)
                instrs.append(ISAM_F16(dst, 'r7.z', 1))
            for row in range(4):
              for col in range(2):
                for kk in range(4):
                  group = ((row * 2 + col) * 4 + kk) * 4
                  instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, b_regs[col][kk], acc0 + group, rpt=3, sy=first, r=True))
                  first = False

        for ku in range(k_unroll):
            if ku:
                if inc_coords:
                    instrs += [
                        ADD_S(a_xregs[0], a_xregs[0], 1), ADD_S(a_xregs[1], a_xregs[1], 1), ADD_S(a_xregs[2], a_xregs[2], 1), ADD_S(a_xregs[3], a_xregs[3], 1),
                        ADD_S('r4.z', 'r4.z', 4), ADD_S('r5.x', 'r5.x', 4), ADD_S('r5.z', 'r5.z', 4), ADD_S('r6.x', 'r6.x', 4),
                    ]
                    emit_one_k(None, None, sync_first=not first_sync_only, reuse_coords=True)
                elif stable_bx:
                    instrs += [ADD_S('r6.w', 'r6.z', ku), ADD_S('r7.w', 'r6.y', 4 * ku)]
                    emit_one_k('r6.w', 'r7.w', sync_first=not first_sync_only)
                else:
                    instrs += [ADD_S('r0.x', 'r6.z', ku), ADD_S('r0.z', 'r6.y', 4 * ku)]
                    emit_one_k('r0.x', 'r0.z', sync_first=not first_sync_only)
            else:
                emit_one_k('r6.z', 'r6.y', reuse_coords=persistent_coords)

        if persistent_coords:
            instrs += [
                CMPS_S_EQ(a_xregs[0], K4 - 1, nop=1),
                ADD_S(a_xregs[0], a_xregs[0], 1), ADD_S(a_xregs[1], a_xregs[1], 1), ADD_S(a_xregs[2], a_xregs[2], 1), ADD_S(a_xregs[3], a_xregs[3], 1),
                ADD_S('r4.z', 'r4.z', 4), ADD_S('r5.x', 'r5.x', 4), ADD_S('r5.z', 'r5.z', 4), ADD_S('r6.x', 'r6.x', 4),
            ]
        else:
            next_k_reg = 'r6.w' if stable_bx else 'r0.x'
            instrs += [
                ADD_S(next_k_reg, 'r6.z', k_unroll),
                ADD_S('r6.y', 'r6.y', 4 * k_unroll),
                CMPS_S_EQ('r6.z', K4 - k_unroll, nop=1),
                MOV_F32('r6.z', next_k_reg),
                NOP(rpt=3),
            ]
        loop_end = len(instrs)
        instrs.append(BR(loop_start - loop_end))

    if post_constant:
        if not direct: raise ValueError('--post-constant requires --direct')
        instrs += [MOV_H_IMM(acc0, 0x6400), MOV_H(acc0 + 1, acc0, rpt=2)]
        for base in range(acc0 + 4, acc0 + acc_groups * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    if no_store:
        pass
    elif donor2_store:
        if not direct or ncols != 2: raise ValueError('--donor2-store requires --direct --ncols 2')
        emit_donor2_store(instrs, dev, threads, acc0)
    elif native_store:
        if not direct: raise ValueError('--native-store requires --direct')
        emit_native4_store(instrs, dev, threads, acc0, ncols)
    elif hybrid_store:
        if not direct: raise ValueError('--hybrid-store requires --direct')
        emit_hybrid4_stores(instrs, dev, threads, acc0, ncols)
    elif store_scalar_offsets:
        if not direct: raise ValueError('--store-scalar-offsets requires --direct')
        emit_donor4_stores_scalar_offsets(instrs, dev, threads, acc0, ncols)
    elif store_shlg_offsets:
        if not direct: raise ValueError('--store-shlg-offsets requires --direct')
        emit_donor4_stores_shlg_offsets(instrs, dev, threads, acc0, ncols)
    elif donor_store:
        if not direct: raise ValueError('--donor-store requires --direct')
        emit_donor4_stores(instrs, dev, threads, acc0, ncols, start_col=store_start, count=(None if store_count < 0 else store_count), store_gap=store_gap, store_or_cols=store_or_cols)
    else:
        if direct: emit_hand4_stores(instrs, acc0, ncols)
        else:
            cols = ['r7.y', 'r7.z']
            for row in range(4):
                if row == 0: row_reg = 'r7.x'
                else:
                    instrs.append(OR_B('r7.w', 'r7.x', row))
                    row_reg = 'r7.w'
                for col in range(ncols):
                    if col == 0: col_reg = 'r7.y'
                    else:
                        instrs.append(ADD_S('r7.z', 'r7.y', col * 32))
                        col_reg = 'r7.z'
                    reduce_store(instrs, row_reg, cols[col], acc0 + ((row * 2 + col) * 4) * 4)
    instrs.append(END())
    return assemble(instrs), loop_end - loop_start


def make_bufs(dev, c_dtype=dtypes.half):
    a = Buffer(dev.device, (K//4)*M*4, dtypes.half, preallocate=True)
    b = Buffer(dev.device, (N//4)*K*4, dtypes.half, preallocate=True)
    c = Buffer(dev.device, M*N, c_dtype, preallocate=True)
    ctypes.memset(int(a._buf.va_addr), 0, a.nbytes)
    ctypes.memset(int(b._buf.va_addr), 0, b.nbytes)
    ctypes.memset(int(c._buf.va_addr), 0, c.nbytes)
    return a, b, c


def fill_half(buf, raw):
    vals = array.array('H', [raw]) * buf.size
    buf.copyin(memoryview(vals).cast('B'))


def check_all_ones(c):
    out = c.copyout(memoryview(bytearray(c.nbytes))).cast('H')
    expected = struct.unpack('<H', struct.pack('<e', float(K4 * 4)))[0]
    mismatches = []
    for i, v in enumerate(out):
        if v != expected:
            mismatches.append((i, v))
            if len(mismatches) >= 10: break
    if mismatches:
        print('CHECK FAIL expected=0x%04x mismatches=%s' % (expected, ', '.join('idx%d=0x%04x' % x for x in mismatches)))
        return False
    print('CHECK PASS all %d outputs are %.1f' % (len(out), float(K4 * 4)))
    return True


def check_all_ones_float(c):
    out = c.copyout(memoryview(bytearray(c.nbytes))).cast('f')
    expected = float(K4 * 4)
    mismatches = []
    for i, v in enumerate(out):
        if v != expected:
            mismatches.append((i, v))
            if len(mismatches) >= 10: break
    if mismatches:
        print('CHECK FAIL expected=%.1f mismatches=%s' % (expected, ', '.join('idx%d=%r' % x for x in mismatches)))
        return False
    print('CHECK PASS all %d float outputs are %.1f' % (len(out), expected))
    return True


def scan_failure_pattern(c):
    out = c.copyout(memoryview(bytearray(c.nbytes))).cast('H')
    expected = struct.unpack('<H', struct.pack('<e', float(K4 * 4)))[0]
    for row in [0, 1, 127, 128]:
        chunks = []
        for col in range(0, N, 32):
            vals = out[row * N + col:row * N + col + 32]
            if all(v == expected for v in vals): tag = 'ok'
            elif all(v == 0 for v in vals): tag = 'zero'
            elif all(v == vals[0] for v in vals): tag = '0x%04x' % vals[0]
            else: tag = 'mix'
            chunks.append('%d:%s' % (col, tag))
        print('row%d chunks32 %s' % (row, ' '.join(chunks)))
    for col in [32, 64, 128, 384, 640, 896]:
        runs, last_tag, start = [], None, 0
        for row in range(M):
            vals = out[row * N + col:row * N + col + 32]
            if all(v == expected for v in vals): tag = 'ok'
            elif all(v == 0 for v in vals): tag = 'zero'
            elif all(v == vals[0] for v in vals): tag = '0x%04x' % vals[0]
            else: tag = 'mix'
            if last_tag is None: last_tag = tag
            elif tag != last_tag:
                runs.append('%d-%d:%s' % (start, row - 1, last_tag))
                start, last_tag = row, tag
        runs.append('%d-%d:%s' % (start, M - 1, last_tag))
        print('col%d row-runs %s' % (col, ' '.join(runs[:16])))


def scan_failure_pattern_float(c):
    out = c.copyout(memoryview(bytearray(c.nbytes))).cast('f')
    expected = float(K4 * 4)
    for row in [0, 1, 127, 128]:
        chunks = []
        for col in range(0, N, 32):
            vals = out[row * N + col:row * N + col + 32]
            if all(v == expected for v in vals): tag = 'ok'
            elif all(v == 0.0 for v in vals): tag = 'zero'
            elif all(v == vals[0] for v in vals): tag = '%r' % vals[0]
            else: tag = 'mix'
            chunks.append('%d:%s' % (col, tag))
        print('row%d chunks32 %s' % (row, ' '.join(chunks)))


def run(args):
    dev = Device['QCOM']
    if args.fp32_accum:
        if args.ncols not in (1, 2, 4): raise ValueError('--fp32-accum supports --ncols 1, 2, or 4')
        envelope_src = make_direct_donor_src_fp32(max(2, args.ncols), args.threads)
    else: envelope_src = make_direct_donor_src(args.ncols, args.threads) if args.native_store else make_donor_src(args.ncols, args.threads)
    envelope, img_off, img_sz, reg_off = get_envelope(dev, envelope_src)
    if args.fp32_accum:
        if args.low_4x8_fp32:
            if args.ncols != 2: raise ValueError('--low-4x8-fp32 requires --ncols 2')
            if args.pipeline:
                shader, hregs, fregs, loop_instrs = build_4x8_fp32_pipeline_shader(dev, args.threads, args.coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.sampler_per_texture)
            else:
                shader, hregs, fregs, loop_instrs = build_4x8_fp32_low_shader(dev, args.threads, args.coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.sampler_per_texture, args.alu_order if args.alu_order != 'auto' else 'kk_row_col', args.preload_b, args.batch_coords, args.native_store, args.stream_b, args.convert_f32_loads, args.alu_reps, not args.stream_b_no_sync, args.store_gap)
        elif args.compact_fp32_hybrid:
            if args.ncols != 1: raise ValueError('--compact-fp32-hybrid currently supports --ncols 1 only')
            shader, hregs, fregs, loop_instrs = build_4x4_fp32_compact_hybrid_shader(dev, args.threads, args.coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.sampler_per_texture)
        elif args.compact_fp32_preload:
            if args.ncols != 1: raise ValueError('--compact-fp32-preload currently supports --ncols 1 only')
            shader, hregs, fregs, loop_instrs = build_4x4_fp32_compact_preload_shader(dev, args.threads, args.coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.sampler_per_texture)
        elif args.compact_fp32:
            if args.ncols != 1: raise ValueError('--compact-fp32 currently supports --ncols 1 only')
            shader, hregs, fregs, loop_instrs = build_4x4_fp32_compact_shader(dev, args.threads, args.coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.sampler_per_texture)
        else:
            shader, hregs, fregs, loop_instrs = build_4x4_fp32_shader(dev, args.threads, args.coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.alu_order if args.alu_order != 'auto' else 'kk_row_col', args.direct_f32_loads, args.sampler_per_texture, args.scalar_f32_mads, args.ncols)
    else:
        shader, loop_instrs = build_4xn_shader(dev, args.threads, ncols=args.ncols, direct=args.direct, quad_a=args.quad_a, store_constant=args.store_constant, post_constant=args.post_constant, donor_store=args.donor_store, donor2_store=args.donor2_store, native_store=args.native_store, hybrid_store=args.hybrid_store, pipeline=args.pipeline, preload_b=args.preload_b, stream_b=args.stream_b, stream_b_no_sync=args.stream_b_no_sync, b_kk_pipeline=args.b_kk_pipeline, b_first=args.b_first, compact_acc=args.compact_acc, stable_bx=args.stable_bx, stable_ay=args.stable_ay, low_a_coords=args.low_a_coords, inc_coords=args.inc_coords, persistent_coords=args.persistent_coords, k_unroll=args.k_unroll, row_col_kk=args.row_col_kk, alu_order=args.alu_order, first_sync_only=args.first_sync_only, row_sync=args.row_sync, no_store=args.no_store, skip_a_loads=args.skip_a_loads, skip_b_loads=args.skip_b_loads, alu_reps=args.alu_reps, coord_delay=args.coord_delay, store_gap=args.store_gap, store_start=args.store_start, store_count=args.store_count, store_or_cols=args.store_or_cols, store_scalar_offsets=args.store_scalar_offsets, store_shlg_offsets=args.store_shlg_offsets)
        acc_start = _hreg('hr12.x') if args.compact_acc else _hreg('hr16.x')
        hregs = 40 if args.pipeline else 36 if args.preload_b else ((acc_start + (4 * args.ncols) * 4 + 3) // 4) if args.direct else 48
        fregs = 28 if args.native_store else 10 if args.b_kk_pipeline else 8
    if args.stable_ay and not args.low_a_coords: fregs = max(fregs, 10)
    if args.fregs_override >= 0: fregs = args.fregs_override
    if args.hregs_override >= 0: hregs = args.hregs_override
    if args.strip_mad_sy:
        patched = bytearray(shader)
        for off in range(0, len(patched), 8):
            lo, hi = struct.unpack_from('<II', patched, off)
            if (hi >> 24) == 0x73: struct.pack_into('<II', patched, off, lo, (hi & 0x00ffffff) | 0x63000000)
        shader = bytes(patched)
    if len(shader) > img_sz:
        print('skipped: shader is %d bytes but envelope has only %d bytes.' % (len(shader), img_sz))
        return
    lib = inject(envelope, img_off, img_sz, reg_off, shader, fregs=fregs, hregs=hregs)

    asm = disasm(shader)
    tex_bytes = (1 + 4 * args.ncols) * 8 if args.quad_a else (4 + 4 * args.ncols) * 8
    flops_per_thread_k = 128 * args.ncols
    covered_n = (N // (128 * args.ncols)) * (128 * args.ncols)
    waves = 12288 // (hregs * args.threads)
    density = 0.0 if loop_instrs == 0 else (64 * args.ncols * args.alu_reps * args.k_unroll) / loop_instrs
    print('ncols=%d covered_N=%d fregs=%d hregs=%d waves=%d intensity=%.2f flop/B mad_density=%.2f shader_instrs=%d loop_instrs=%d bytes=%d envelope_bytes=%d' % (
        args.ncols, covered_n, fregs, hregs, waves, flops_per_thread_k / tex_bytes, density, len(shader)//8, loop_instrs, len(shader), img_sz))
    print('mad.f16=%d mad.f32=%d rpt3=%d isam=%d qbc=%d sy=%d' % (asm.count('mad.f16'), asm.count('mad.f32'), asm.count('(rpt3)mad.'), asm.count('isam'), asm.count('quad_shuffle.brcst'), asm.count('(sy)')))
    if args.disasm: print(asm)

    a_img, b_img = dtypes.imageh((M, K//4)), dtypes.imageh((K, N//4))
    c_dtype = dtypes.float if args.fp32_accum else dtypes.half
    a, b, c = make_bufs(dev, c_dtype)
    if args.check:
        fill_half(a, 0x3c00)
        fill_half(b, 0x3c00)
    prg = dev.runtime('gemm_h', lib, [[(0, a_img)], [(1, b_img)], [(2, c_dtype.ptr())]])
    tile_m = (args.threads // 32) * 4
    gs, ls = (covered_n // (128 * args.ncols), M//tile_m, 1), (args.threads, 1, 1)
    if args.check:
        prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if args.scan_output:
            if args.fp32_accum: scan_failure_pattern_float(c)
            else: scan_failure_pattern(c)
        if args.fp32_accum: check_all_ones_float(c)
        else: check_all_ones(c)
        return
    for _ in range(5): prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
    times = []
    for _ in range(args.iters):
        t = prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)
    best = min(times)
    flops = 2 * M * covered_n * K * (args.alu_reps if args.no_store else 1)
    print('%.1f GFLOPS  (%.3f ms)' % (flops / best / 1e9, best*1e3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--threads', type=int, choices=(64, 128, 256), default=64)
    parser.add_argument('--ncols', type=int, choices=(1, 2, 3, 4, 8), default=2)
    parser.add_argument('--fp32-accum', action='store_true', help='experimental 4x4: imageh inputs, FP32 MADs, float output')
    parser.add_argument('--direct-f32-loads', action='store_true', help='experimental FP32 path: load imageh inputs with isam.f32 instead of isam.f16+cov')
    parser.add_argument('--convert-f32-loads', action='store_true', help='experimental FP32 path: load imageh inputs as f16 then convert to f32')
    parser.add_argument('--sampler-per-texture', action='store_true', help='use sampler index matching texture index for direct isam.f32 loads')
    parser.add_argument('--scalar-f32-mads', action='store_true', help='emit 64 scalar mad.f32 ops instead of 16 rpt3 mad.f32 ops')
    parser.add_argument('--compact-fp32', action='store_true', help='experimental FP32 4x4: stream one A vector at a time to reduce fregs')
    parser.add_argument('--compact-fp32-preload', action='store_true', help='experimental FP32 4x4: preload A/B into r0-r7 and use r12 for state')
    parser.add_argument('--compact-fp32-hybrid', action='store_true', help='experimental FP32 4x4: keep state in r7 and place A3 in r12')
    parser.add_argument('--low-4x8-fp32', action='store_true', help='experimental FP32 4x8: f17 A-reuse path with two donor4 float stores')
    parser.add_argument('--direct', action='store_true', help='faster but currently incorrect dependency-stress variant')
    parser.add_argument('--donor-store', action='store_true', help='use repeated compiler-donor 4-row store epilogue for direct outputs')
    parser.add_argument('--donor2-store', action='store_true', help='use compiler-donor 4x8 store epilogue for --direct --ncols 2')
    parser.add_argument('--native-store', action='store_true', help='use native direct compiler store epilogue for --direct --ncols 4')
    parser.add_argument('--hybrid-store', action='store_true', help='use donor stores for first two col blocks and hand stores for 4x16 tail')
    parser.add_argument('--pipeline', action='store_true', help='experimental direct 4x16: overlap next B/A loads with current ALU')
    parser.add_argument('--preload-b', action='store_true', help='experimental direct 4x16: preload all B col blocks before ALU')
    parser.add_argument('--batch-coords', action='store_true', help='experimental low FP32 4x8: issue coordinate setup before texture loads')
    parser.add_argument('--stream-b', action='store_true', help='experimental direct 4x16: stream second B pair into freed B registers')
    parser.add_argument('--stream-b-no-sync', action='store_true', help='diagnostic: do not sync before streamed second B pair ALU')
    parser.add_argument('--b-kk-pipeline', action='store_true', help='experimental direct 4x16: double-buffer B by K component')
    parser.add_argument('--b-first', action='store_true', help='experimental direct 4x16: load first B pair before A to hide B latency under A loads')
    parser.add_argument('--compact-acc', action='store_true', help='experimental direct mode: start accumulators at hr12 instead of hr16')
    parser.add_argument('--stable-bx', action='store_true', help='experimental direct 4x16: keep B x-coordinate registers stable across K')
    parser.add_argument('--stable-ay', action='store_true', help='experimental direct mode: keep A row coordinates in full regs')
    parser.add_argument('--low-a-coords', action='store_true', help='experimental stable-ay: place persistent A coords in low full regs to reduce fregs')
    parser.add_argument('--inc-coords', action='store_true', help='stable bx/ay: increment coords across unrolled K steps')
    parser.add_argument('--persistent-coords', action='store_true', help='inc-coords: keep coords live across loop iterations')
    parser.add_argument('--k-unroll', type=int, choices=(1, 2, 4, 8), default=1, help='semantically unroll K4 loop iterations')
    parser.add_argument('--row-col-kk', action='store_true', help='direct mode: emit true GEMM ALU in row/col/kk order')
    parser.add_argument('--alu-order', choices=('auto', 'kk_row_col', 'row_col_kk', 'row_kk_col', 'col_kk_row', 'kk_col_row'), default='auto', help='direct mode: explicit MAD issue order')
    parser.add_argument('--first-sync-only', action='store_true', help='direct mode: sync only the first unrolled K MAD group')
    parser.add_argument('--row-sync', action='store_true', help='diagnostic: set (sy) on first MAD for each row/column accumulator')
    parser.add_argument('--no-store', action='store_true', help='benchmark combined isam+ALU loop without the output store path')
    parser.add_argument('--skip-a-loads', action='store_true', help='profile only: initialize A regs once and skip A texture loads; requires --no-store')
    parser.add_argument('--skip-b-loads', action='store_true', help='profile only: initialize B regs once and skip B texture loads; requires --no-store')
    parser.add_argument('--alu-reps', type=int, default=1, help='throughput probe only: repeat ALU body per loaded A/B tile')
    parser.add_argument('--coord-delay', type=int, default=4, help='NOP repeat before texture isam; -1 means no wait')
    parser.add_argument('--store-gap', type=int, default=-1, help='diagnostic: NOP repeat between repeated donor store chunks')
    parser.add_argument('--store-start', type=int, default=0, help='diagnostic: first donor-store column block to emit')
    parser.add_argument('--store-count', type=int, default=-1, help='diagnostic: number of donor-store column blocks to emit')
    parser.add_argument('--store-or-cols', action='store_true', help='diagnostic: form repeated donor-store column coordinates with OR instead of ADD')
    parser.add_argument('--store-scalar-offsets', action='store_true', help='diagnostic: keep base col fixed and add scalar column offsets inside donor address path')
    parser.add_argument('--store-shlg-offsets', action='store_true', help='diagnostic: use compiler shlg immediate column offsets inside donor address path')
    parser.add_argument('--fregs-override', type=int, default=-1, help='debug/profile only: override full register metadata')
    parser.add_argument('--hregs-override', type=int, default=-1, help='debug/profile only: override half register metadata')
    parser.add_argument('--strip-mad-sy', action='store_true', help='diagnostic: remove (sy) from mad.f16 instructions')
    parser.add_argument('--quad-a', action='store_true', help='load A in one lane per quad and broadcast with quad_shuffle.brcst')
    parser.add_argument('--store-constant', action='store_true', help='diagnostic: skip GEMM and store 1024.0 through the same output path')
    parser.add_argument('--post-constant', action='store_true', help='diagnostic: run GEMM loop, then overwrite accumulators with 1024.0 before stores')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--scan-output', action='store_true')
    parser.add_argument('--disasm', action='store_true')
    run(parser.parse_args())
