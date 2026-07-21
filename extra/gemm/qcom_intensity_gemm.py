#!/usr/bin/env python3
"""Hand-assembled higher-intensity FP16 GEMM for Adreno 630.

The baseline 4-row kernel does 128 FLOPs from 64 bytes of texture input per
thread/K step. This experiment computes two col4 outputs per thread, reusing
the same four A texels across eight B texels: 256 FLOPs from 96 bytes.
"""
import argparse, array, ctypes, struct

from tinygrad import Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.device import Buffer
from extra.gemm.ir3asm import *
from extra.gemm.ir3asm import _hreg

M, N, K = getenv("GEMM_M", 1024), getenv("GEMM_N", 1024), getenv("GEMM_K", 1024)
K4 = K // 4


def make_donor_src(ncols=1, threads=128, swap_groups=False):
    tn = 32 * ncols
    tm = (threads // 32) * 4
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global half *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    row_gid, col_gid = (0, 1) if swap_groups else (1, 0)
    src += 'int row=get_group_id(%d)*%d+tm*4;int col4=get_group_id(%d)*%d+tid;\n' % (row_gid, tm, col_gid, tn)
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


def make_direct_donor_src(ncols=4, threads=128, swap_groups=False):
    tn = 32 * ncols
    tm = (threads // 32) * 4
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global half *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    row_gid, col_gid = (0, 1) if swap_groups else (1, 0)
    src += 'int row=get_group_id(%d)*%d+tm*4;int col4=get_group_id(%d)*%d+tid;\n' % (row_gid, tm, col_gid, tn)
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


def make_direct_image_donor_src(ncols=4, threads=128, swap_groups=False):
    tn, tm = 32*ncols, (threads//32)*4
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(write_only image2d_t C,read_only image2d_t A,read_only image2d_t B){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    row_gid, col_gid = (0, 1) if swap_groups else (1, 0)
    src += 'int row=get_group_id(%d)*%d+tm*4;int col4=get_group_id(%d)*%d+tid;\n' % (row_gid, tm, col_gid, tn)
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
    # Reserve enough instruction space for hand-assembled variants. Injection
    # replaces these dependent operations; only the ELF allocation matters.
    for _ in range(256): src += 'r0d0=r0d0*(half)1.0009765625h+(half4)(0.0009765625h);\n'
    for r in range(4):
        for c in range(ncols): src += 'write_imageh(C,(int2)(col4+%d,row+%d),r%dd%d);\n' % (c*32, r, r, c)
    return src + '}\n'


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


def make_direct_donor_src_u32(ncols=1, threads=128):
    """Integer-image donor with the same 4-row/col4 launch geometry as the FP32 path."""
    tn, tm = 32 * ncols, (threads // 32) * 4
    src = 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global uint *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    src += 'int row=get_group_id(1)*%d+tm*4;int col4=get_group_id(0)*%d+tid;\n' % (tm, tn)
    for r in range(4):
        for c in range(ncols): src += 'uint4 r%dd%d=(uint4)(0);\n' % (r, c)
    src += 'for(int k4=0;k4<%d;k4++){\n' % K4
    for r in range(4): src += 'uint4 a%d=read_imageui(A,smp,(int2)(k4,row+%d));\n' % (r, r)
    for c in range(ncols):
        for b in range(4): src += 'uint4 b%d_%d=read_imageui(B,smp,(int2)(col4+%d,k4*4+%d));\n' % (c, b, c*32, b)
    for r in range(4):
        for c in range(ncols):
            src += 'r%dd%d+=a%d.xxxx*b%d_0+a%d.yyyy*b%d_1+a%d.zzzz*b%d_2+a%d.wwww*b%d_3;\n' % (r,c,r,c,r,c,r,c,r,c)
    src += '}\n'
    for _ in range(128): src += 'r0d0=r0d0*(uint4)(1664525u)+(uint4)(1013904223u);\n'
    for r in range(4):
        for c in range(ncols): src += 'vstore4(r%dd%d,0,C+(row+%d)*%d+(col4+%d)*4);\n' % (r,c,r,N,c*32)
    return src + '}\n'


def make_direct_donor_src_fp32_quad(ncols=1, threads=128):
    src = make_direct_donor_src_fp32(ncols, threads)
    tm = (threads // 32) * 4
    return src.replace('int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n',
                       'int lid=get_local_id(0);int tm=lid&3;int tid=lid>>2;\n').replace(
                         'int row=get_group_id(1)*%d+tm*4;int col4=get_group_id(0)*%d+tid;\n' % (tm, 32*ncols),
                         'int row=get_group_id(1)*%d+tm*4;int col4=get_group_id(0)*%d+tid;\n' % (tm, 32*ncols))


def prologue_4x2(dev, threads, swap_groups=False):
    lib, io, isz, _ = get_envelope(dev, make_donor_src(1, threads, swap_groups=swap_groups))
    pro = bytearray(lib[io:io + 21 * 8])
    return [bytes(pro[i:i+8]) for i in range(0, len(pro), 8)]


def prologue_direct4_fp32(dev, threads, ncols=1, quad_map=False):
    donor_src = make_direct_donor_src_fp32_quad(ncols, threads) if quad_map else make_direct_donor_src_fp32(ncols, threads)
    lib, io, isz, _ = get_envelope(dev, donor_src)
    pro_instrs = 40 if ncols != 1 else 27
    pro = bytearray(lib[io:io + pro_instrs * 8])
    return [bytes(pro[i:i+8]) for i in range(0, len(pro), 8)]


def emit_addr(instrs, row_reg, col_reg, row_shift=10):
    # Compute 64-bit C address in r2.x/r2.y. Do this before reducing into hr0,
    # since the address math clobbers low half registers through r0/r1 aliases.
    instrs += [
        SHL_B('r0.x', row_reg, row_shift, jp=True),
        SHL_B('r0.y', col_reg, 2),
        NOP(rpt=16),
        ADD_S_REG('r0.x', 'r0.x', 'r0.y'),
        NOP(rpt=16),
        SHL_B('r0.x', 'r0.x', 1),
        NOP(rpt=16),
        ADD_U('r2.x', 'c20.x', 'r0.x'),
        NOP(rpt=16),
        CMPS_U_LT('r6.w', 'r2.x', 'c20.x'),
        SHR_B('r6.y', 'r0.x', 31),
        NOP(rpt=16),
        SAD_S32('r2.y', 'c20.y', 'r6.y', 'r6.w', nop=3),
    ]


def emit_linear4_stores(instrs, acc0, ncols, row_shift=10):
    """Store a compact 4x(4*ncols) tile using one computed 64-bit base pointer."""
    emit_addr(instrs, 'r11.z', 'r11.w', row_shift)
    col_bytes, row_bytes = 32*4*2, 1 << (row_shift+1)
    instrs += [MOV_F32('r10.z', 'r2.x'), MOV_F32('r10.w', 'r2.y'), NOP(rpt=2)]
    for row in range(4):
        for col in range(ncols):
            offset = row*row_bytes + col*col_bytes
            instrs += [MOV_F32('r2.x', 'r10.z'), MOV_F32('r2.y', 'r10.w')]
            if offset:
                instrs += [MOV_S32('r11.x', offset), NOP(rpt=2), ADD_S_REG('r2.x', 'r10.z', 'r11.x'), NOP(rpt=2)]
            instrs.append(STG_F16('r2.x', acc0+(row*ncols+col)*4, sy=True))
            instrs.append(NOP(rpt=16))


def emit_image4_stores(instrs, acc0, ncols, saved_coords=False, high_store=False, rotate_acc=False):
    if saved_coords:
        instrs += [MOV_F32('r7.x', 'r2.x'), MOV_F32('r7.y', 'r2.y'), NOP(rpt=2)]
    data_reg, coord_reg = ('r28.x', 'r29.x') if high_store else ('r0.x', 'r4.x')
    for row in range(4):
        for col in range(ncols):
            instrs.append(MOV_F32(coord_reg, 'r7.y') if col == 0 else ADD_S(coord_reg, 'r7.y', col*32))
            yreg = f"r{int(coord_reg[1:coord_reg.index('.')])+0}.y"
            instrs.append(MOV_F32(yreg, 'r7.x') if row == 0 else ADD_S(yreg, 'r7.x', row))
            src_row = (row+1) % 4 if rotate_acc else row
            instrs += [COV_F16F32(data_reg, acc0+(src_row*ncols+col)*4, sy=True, rpt=3, r=True), NOP(rpt=5),
                       STIB_F32(data_reg, coord_reg), NOP(rpt=16)]


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
    instrs += [NOP(rpt=16), STG_F16('r2.x', data_hreg, sy=True), NOP(rpt=16)]


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


def emit_hvec_copy(instrs, dst, src):
    # Keep this scalar so the source/destination lane mapping is explicit.
    for lane in range(4): instrs.append(MOV_H(dst + lane, src + lane))


def emit_hvec_imm(instrs, dst, imm):
    # Repeated MOV advances its source as well as its destination.  It cannot
    # broadcast the one initialized lane safely when the remaining registers
    # contain state left by an earlier graph kernel.
    for lane in range(4): instrs.append(MOV_H_IMM(dst + lane, imm))


def emit_hand4_stores(instrs, acc0, ncols, row_shift=10, selected_cols=None, repeat_first=False, repair_row1=False, repeat_each=False):
    # Keep the original compiler-derived epilogue byte-for-byte for the production
    # path.  The diagnostic store variants below deliberately add stronger waits
    # and scalar copies, but those changes regress the checked 4x16 kernel.
    if row_shift in (10, 11) and selected_cols is None and not (repeat_first or repair_row1 or repeat_each):
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
        if row_shift != 10: hand_addr[0] = SHL_B('r0.x', 'r7.x', row_shift, jp=True)
        hand_stores = [bytes.fromhex(x) for x in [
            '000080040011c0c0', '0000000000000000', '080080040015c0c0',
            '0000000000000000', '100080040019c0c0', '0000000000000000',
            '18008004001dc0c0']]
        for col in range(ncols):
            if col != 0: instrs.append(ADD_S('r7.y', 'r7.y', 32))
            instrs += hand_addr
            # Repeated half moves have device-dependent relative-source behavior.
            # Keep every lane explicit: the all-ones oracle cannot detect a
            # broadcast source, while random matrices require all four lanes.
            for row in range(4): emit_hvec_copy(instrs, row * 4, acc0 + (row * ncols + col) * 4)
            instrs += hand_stores
        return

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
    hand_addr[0] = SHL_B('r0.x', 'r7.x', row_shift, jp=True)
    hand_stores = [bytes.fromhex(x) for x in [
        '000080040011c0d0', '0000000000000000', '080080040015c0c0',
        '0000000000000000', '100080040019c0c0', '0000000000000000',
        '18008004001dc0c0']]
    first_hand_stores = list(hand_stores)
    if repair_row1: first_hand_stores[2] = NOP()
    cols = list(range(ncols)) if selected_cols is None else list(selected_cols)
    previous_col = 0
    for col in cols:
        if col != previous_col:
            instrs.append(ADD_S('r7.y', 'r7.y', (col - previous_col) * 32))
        previous_col = col
        instrs += hand_addr
        instrs.append(NOP_SS())
        for row in range(4): emit_hvec_copy(instrs, row * 4, acc0 + (row * ncols + col) * 4)
        instrs.append(NOP(rpt=16))
        instrs += first_hand_stores
        if repeat_each:
            instrs += hand_addr
            instrs.append(NOP_SS())
            for row in range(4): emit_hvec_copy(instrs, row * 4, acc0 + (row * ncols + col) * 4)
            instrs.append(NOP(rpt=16))
            instrs += hand_stores
        if repair_row1 and not repeat_each:
            instrs.append(NOP(rpt=16))
            emit_hvec_copy(instrs, 0, acc0 + (ncols + col) * 4)
            instrs += [NOP(rpt=16), MOV_S32('r0.x', 0), CMPS_S_EQ('r0.x', 0, nop=3),
                       STG_F16('r2.z', 0), NOP()]
    if repeat_first and cols and cols[0] == 0:
        if previous_col: instrs.append(ADD_S('r7.y', 'r7.y', -previous_col * 32))
        instrs += hand_addr
        instrs.append(NOP_SS())
        for row in range(4): emit_hvec_copy(instrs, row * 4, acc0 + row * ncols * 4)
        instrs.append(NOP(rpt=16))
        instrs += hand_stores


def make_thread_store_src(ncols, threads):
    gx = N // (128 * ncols)
    return f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({threads},1,1)))
__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global half *C) {{
  int idx=(get_group_id(1)*{gx}+get_group_id(0))*{threads}+get_local_id(0);
  half4 v=(half4)(1);
#pragma unroll 1
  for(int i=0;i<{4*ncols};i++) vstore4(v,0,C+idx*{16*ncols}+i*4);
}}'''


def emit_threadmajor_stores(instrs, acc0, ncols, lid_reg='r10.x'):
    """Store one thread's complete 4x(4*ncols) tile contiguously."""
    lib, io, isz, _ = get_envelope(Device['QCOM'], make_thread_store_src(ncols, 128))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    # Restore raw local id and retain the compiler's address calculation and
    # dependency scheduling exactly.  The loop body is unrolled only so each
    # store can select a different accumulator vector.
    if lid_reg is None:
        instrs += [MOV_S32('r0.x', 0), MOV_H('hr0.x', 'hr28.x'), NOP(rpt=2)]
    else: instrs.append(MOV_F32('r0.x', lid_reg))
    instrs += donor[:10]
    for out in range(4 * ncols):
        instrs += donor[10:12]     # loop-target nop; byte offset
        instrs += donor[16:24]     # induction, 64-bit pointer, dependency waits
        instrs.append(STG_F16('r0.y', acc0 + out * 4, sy=(out == 0)))
        instrs.append(donor[25])    # advance the compiler loop induction value


def emit_coord_wait(instrs, coord_delay):
    if coord_delay >= 0: instrs.append(NOP(rpt=coord_delay))


def fvec(vec, comp=0):
    return 'r%d.%s' % (vec, 'xyzw'[comp])


def emit_f32_vec_imm(instrs, vec, imm):
    for comp in range(4): instrs.append(MOV_S32(fvec(vec, comp), imm))


def emit_f32_vec_copy(instrs, dst_vec, src_vec):
    for comp in range(4): instrs.append(MOV_F32(fvec(dst_vec, comp), fvec(src_vec, comp)))


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
    for i in range(8): emit_f32_vec_copy(instrs, tmp_vec0 + i, acc_vec0 + i)

    def mov_split(dsts, src_vec):
        for comp, dst in enumerate(dsts): instrs.append(MOV_F32(dst, fvec(src_vec, comp)))

    mov_split(('r8.x', 'r15.x', 'r15.y', 'r15.z'), tmp_vec0 + 0)
    mov_split(('r15.w', 'r16.x', 'r16.y', 'r16.z'), tmp_vec0 + 4)
    emit_f32_vec_copy(instrs, 14, tmp_vec0 + 1)
    emit_f32_vec_copy(instrs, 13, tmp_vec0 + 5)
    emit_f32_vec_copy(instrs, 12, tmp_vec0 + 2)
    mov_split(('r10.y', 'r10.w', 'r11.y', 'r11.w'), tmp_vec0 + 6)
    mov_split(('r8.y', 'r8.w', 'r9.y', 'r9.w'), tmp_vec0 + 3)
    emit_f32_vec_copy(instrs, 7, tmp_vec0 + 7)
    instrs += donor[261:348]


def emit_donor4_float_store2_lowcopy(instrs, dev, threads, acc_vec0):
    lib, io, isz, _ = get_envelope(dev, make_direct_donor_src_fp32(2, threads))
    donor = [bytes(lib[io+i:io+i+8]) for i in range(0, isz, 8)]
    for i in range(5): emit_f32_vec_copy(instrs, i, acc_vec0 + i)

    def mov_split(dsts, src_vec):
        for comp, dst in enumerate(dsts): instrs.append(MOV_F32(dst, fvec(src_vec, comp)))

    mov_split(('r8.x', 'r15.x', 'r15.y', 'r15.z'), 0)
    mov_split(('r15.w', 'r16.x', 'r16.y', 'r16.z'), 4)
    emit_f32_vec_copy(instrs, 14, 1)
    emit_f32_vec_copy(instrs, 13, acc_vec0 + 5)
    emit_f32_vec_copy(instrs, 12, 2)
    mov_split(('r10.y', 'r10.w', 'r11.y', 'r11.w'), acc_vec0 + 6)
    mov_split(('r8.y', 'r8.w', 'r9.y', 'r9.w'), 3)
    emit_f32_vec_copy(instrs, 7, acc_vec0 + 7)
    instrs += donor[261:348]


def emit_f32_global_stores(instrs, acc_vec0, ncols):
    for col in range(ncols):
        instrs.append(MOV_F32('r5.y', 'r7.y') if col == 0 else ADD_S('r5.y', 'r7.y', col * 32))
        for row in range(4):
            instrs.append(MOV_F32('r5.x', 'r7.x') if row == 0 else OR_B('r5.x', 'r7.x', row))
            emit_addr_float(instrs, 'r5.x', 'r5.y')
            emit_f32_vec_copy(instrs, 0, acc_vec0 + col * 4 + row)
            instrs += [NOP(rpt=16), STG_F32('r2.x', 'r0.x', sy=True), NOP()]


def emit_f32_global_stores_gap(instrs, acc_vec0, ncols, post_gap=16):
    for col in range(ncols):
        instrs.append(MOV_F32('r5.y', 'r7.y') if col == 0 else ADD_S('r5.y', 'r7.y', col * 32))
        for row in range(4):
            instrs.append(MOV_F32('r5.x', 'r7.x') if row == 0 else OR_B('r5.x', 'r7.x', row))
            emit_addr_float(instrs, 'r5.x', 'r5.y')
            emit_f32_vec_copy(instrs, 0, acc_vec0 + col * 4 + row)
            instrs += [NOP(rpt=16), STG_F32('r2.x', 'r0.x', sy=True), NOP(rpt=post_gap)]


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


def build_4x4_dp4_shader(dev, threads, k, constant_inputs=False, constant_output=False, constant_a=False, constant_b=False,
                         combined_b_height=0, mixed=False, initial_acc=0, coord_delay=4):
    """4x4-output packed UINT8 GEMM using A630 dp4acc instructions."""
    if k % 16: raise ValueError('dp4 K must be divisible by 16')
    instrs = prologue_direct4_fp32(dev, threads, 1)
    row_base, col_base, ki, ky = 'r12.x', 'r12.y', 'r12.z', 'r12.w'
    instrs += [MOV_F32(row_base, 'r4.z'), MOV_F32(col_base, 'r5.x'), MOV_S32(ki, 0)]
    acc_vec0 = 8
    for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 123 if constant_output else initial_acc)
    loop_start = len(instrs)
    if not constant_output:
      if constant_inputs or constant_b:
        for vec in range(4): emit_f32_vec_imm(instrs, vec, 0x01010101)
      else:
        instrs.append(SHL_B(ky, ki, 2))
        for j in range(4):
          instrs += [MOV_F32('r5.x', col_base), MOV_F32('r5.y', ky) if j == 0 else ADD_S('r5.y', ky, j)]
          emit_coord_wait(instrs, coord_delay)
          instrs.append(ISAM_U32(fvec(j), 'r5.x', 0 if combined_b_height else 1, 0))
      for row in range(4):
        if constant_inputs or constant_a: emit_f32_vec_imm(instrs, 4, 0x01010101)
        else:
          instrs += [MOV_F32('r5.x', ki),
                     ADD_S('r5.y', row_base, combined_b_height+row) if combined_b_height else
                     MOV_F32('r5.y', row_base) if row == 0 else OR_B('r5.y', row_base, row)]
          emit_coord_wait(instrs, coord_delay)
          instrs.append(ISAM_U32(fvec(4), 'r5.x', 0, 0))
        for kk in range(4):
          for col in range(4):
            instrs.append(DP4ACC(fvec(acc_vec0+row, col), fvec(4, kk), fvec(kk, col),
                                 fvec(acc_vec0+row, col), sy=(kk == 0 and col == 0), mixed=mixed))
            if getenv("DP4_DELAY", 0): instrs.append(NOP(rpt=getenv("DP4_DELAY", 0)-1))
      instrs += [ADD_S('r0.x', ki, 1), CMPS_S_EQ(ki, k//16 - 1, nop=1), MOV_F32(ki, 'r0.x'), NOP(rpt=3)]
      loop_end = len(instrs)
      instrs.append(BR(loop_start - loop_end))
    else:
      loop_end = loop_start
      instrs.append(NOP(rpt=16))
    instrs += [MOV_F32('r7.x', row_base), MOV_F32('r7.y', col_base)]
    emit_donor4_float_store(instrs, dev, threads, acc_vec0)
    instrs.append(END())
    return assemble(instrs), 1, 13, loop_end-loop_start


def build_4x8_dp4_nostore_shader(dev, threads, k, constant_inputs=False):
    """Four-row, eight-column packed-U8 DP4 probe with no output epilogue."""
    if k % 16: raise ValueError('dp4 K must be divisible by 16')
    instrs = prologue_direct4_fp32(dev, threads, 2)
    row_base, col_base, ki, ky = 'r17.x', 'r17.y', 'r17.z', 'r17.w'
    instrs += [MOV_F32(row_base, 'r4.z'), MOV_F32(col_base, 'r5.x'), MOV_S32(ki, 0)]
    acc_vec0, a_vec = 9, 8
    for vec in range(acc_vec0, acc_vec0 + 8): emit_f32_vec_imm(instrs, vec, 0)
    loop_start = len(instrs)
    if constant_inputs:
      for vec in range(9): emit_f32_vec_imm(instrs, vec, 0x01010101)
    else:
      instrs.append(SHL_B(ky, ki, 2))
      for cg in range(2):
        for j in range(4):
          instrs += [MOV_F32('r18.x', col_base) if cg == 0 else ADD_S('r18.x', col_base, 32),
                     MOV_F32('r18.y', ky) if j == 0 else ADD_S('r18.y', ky, j)]
          emit_coord_wait(instrs, 4)
          instrs.append(ISAM_U32(fvec(cg*4+j), 'r18.x', 1, 0))
    for row in range(4):
      if not constant_inputs:
        instrs += [MOV_F32('r18.x', ki), MOV_F32('r18.y', row_base) if row == 0 else OR_B('r18.y', row_base, row)]
        emit_coord_wait(instrs, 4)
        instrs.append(ISAM_U32(fvec(a_vec), 'r18.x', 0, 0))
      for kk in range(4):
        for cg in range(2):
          acc = acc_vec0 + row*2 + cg
          instrs.append(DP4ACC(fvec(acc), fvec(a_vec, kk), fvec(cg*4+kk), fvec(acc), sy=(kk == 0 and cg == 0)))
    instrs += [ADD_S('r0.x', ki, 1), CMPS_S_EQ(ki, k//16 - 1, nop=1), MOV_F32(ki, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs += [BR(loop_start-loop_end), END()]
    return assemble(instrs), 1, 19, loop_end-loop_start


def build_4x4_fp32_compact_preload_shader(dev, threads, coord_delay=4, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, sampler_per_texture=True, hand_store=False, batch_coords=False, quad_map=False, quad_b=False, quad_b_load_all=False, first_coord_wait_only=False, quad_b_shfl_mode=0):
    if quad_b and (not quad_map or coord_delay != -1): raise ValueError('quad_b requires quad_map and coord_delay=-1')
    instrs = prologue_direct4_fp32(dev, threads, 1, quad_map=quad_map)
    row_base, col_base, ky_reg, kz_reg = 'r12.x', 'r12.y', 'r12.z', 'r12.w'
    instrs += [MOV_F32(row_base, 'r4.z'), MOV_F32(col_base, 'r5.x'), MOV_S32(ky_reg, 3), MOV_S32(kz_reg, 0)]
    acc_vec0, a_vec0 = 8, 4
    # The A630 quad shuffle does not reliably support an overlapping vector
    # source/destination. Keep sampled B in r0-r3 and broadcast into r13-r16.
    b_load_vec0, b_vec0 = 0, 13 if quad_b else 0
    for vec in range(acc_vec0, acc_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0)
    if skip_b_loads:
        for vec in range(b_load_vec0, b_load_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)
    if skip_a_loads:
        for vec in range(a_vec0, a_vec0 + 4): emit_f32_vec_imm(instrs, vec, 0x3f800000)

    first = True
    def emit_b_coords():
        if skip_b_loads: return
        for kk, yoff in enumerate((-3, -2, -1, 0)):
            dst = b_load_vec0 + kk
            instrs.append(MOV_F32(fvec(dst, 0), col_base))
            instrs.append(MOV_F32(fvec(dst, 1), ky_reg) if yoff == 0 else ADD_S(fvec(dst, 1), ky_reg, yoff))

    def emit_b_isams():
        nonlocal instrs
        if skip_b_loads: return
        if quad_b and not quad_b_load_all:
            instrs += [SHR_B('r7.w', row_base, 2), AND_B('r7.w', 'r7.w', 3, nop=3),
                       CMPS_S_EQ('r7.w', 0), NOP(rpt=5), BR(5)]
        for kk in range(4):
            dst = b_load_vec0 + kk
            if not first_coord_wait_only or kk == 0: emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, sampler_per_texture)
        if quad_b:
            if not quad_b_load_all: instrs.append(SHL_B('r7.w', 'r7.w', 0, jp=True, ss=True, nop=3))
            if quad_b_shfl_mode:
                instrs += [SHR_B('r7.w', row_base, 2), AND_B('r7.w', 'r7.w', 3), NOP(rpt=2)]
                for kk in range(4):
                    for comp in range(4):
                        instrs.append(SHFL(fvec(b_vec0+kk, comp), fvec(b_load_vec0+kk, comp), 'r7.w',
                                            mode=quad_b_shfl_mode, typ=1, sy=(kk == 0 and comp == 0)))
            else:
                instrs.append(MOV_S32('r7.w', 0))
                instrs += [MOV_F32('r7.w', 'r7.w', sy=True), NOP(rpt=5)]
                for kk in range(4): instrs.append(QUAD_BRCST(fvec(b_vec0+kk), fvec(b_load_vec0+kk), 'r7.w', typ=3, wrmask=15))

    def emit_a_coords():
        if skip_a_loads: return
        for row in range(4):
            dst = a_vec0 + row
            instrs.append(MOV_F32(fvec(dst, 0), kz_reg))
            instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))

    def emit_a_isams():
        if skip_a_loads: return
        for row in range(4):
            dst = a_vec0 + row
            if not first_coord_wait_only or row == 0: emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, sampler_per_texture)

    def emit_mads():
        nonlocal first
        for kk in range(4):
            for row in range(4):
                acc_vec = acc_vec0 + row
                instrs.append(MAD_F32(fvec(acc_vec), fvec(a_vec0 + row, kk), fvec(b_vec0 + kk), fvec(acc_vec), rpt=3, sy=first, r=True))
                first = False

    loop_start = len(instrs)
    if batch_coords:
        emit_b_coords()
        emit_a_coords()
        emit_b_isams()
        emit_a_isams()
    else:
        emit_b_coords()
        emit_b_isams()
        emit_a_coords()
        emit_a_isams()
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
        if hand_store: emit_f32_global_stores(instrs, acc_vec0, 1)
        else: emit_donor4_float_store(instrs, dev, threads, acc_vec0)
    instrs.append(END())
    return assemble(instrs), 1, 17 if quad_b else 13, loop_end - loop_start


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


def build_4x8_fp32_low_shader(dev, threads, coord_delay=4, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, sampler_per_texture=True, alu_order='kk_row_col', preload_b=False, batch_coords=False, hand_store=False, stream_b=False, convert_loads=False, alu_reps=1, stream_b_sync=True, stream_b_wait=-1, interleaved_a=False):
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
        if interleaved_a:
            for kk, dst in enumerate(a_vecs):
                instrs.append(SHL_B(fvec(dst, 0), kz_reg, 2))
                if kk:
                    # The integer ALU does not forward this freshly shifted value to the
                    # immediately following add.  Without the gap, K lanes 1..3 lag a loop.
                    instrs.extend([NOP(rpt=2), ADD_S(fvec(dst, 0), fvec(dst, 0), kk)])
                instrs.append(SHR_B(fvec(dst, 1), row_base, 2))
            return
        for row, dst in enumerate(a_vecs):
            instrs.append(MOV_F32(fvec(dst, 0), kz_reg))
            instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))

    def emit_a_isams():
        if skip_a_loads: return
        if interleaved_a:
            for dst in a_vecs:
                emit_coord_wait(instrs, coord_delay)
                emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, sampler_per_texture)
            return
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
        # In the model layout each A texel is four rows at one scalar K.  Consume
        # that texture result directly; copying it immediately after ISAM races
        # the asynchronous sampler on a6xx.
        a_src = fvec(a_vecs[kk], row) if interleaved_a else fvec(a_vecs[row], kk)
        instrs.append(MAD_F32(fvec(acc(row, col)), a_src, fvec(b_vecs[kk]), fvec(acc(row, col)), rpt=3, sy=first, r=True))
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
    fregs = (21 if (hand_store or preload_b) else 32) if not no_store else (21 if preload_b else 17)
    return assemble(instrs), 1, fregs, loop_end - loop_start


def build_4xn_fp32_stream_shader(dev, threads, ncols=4, coord_delay=-1, sync_each_col=True, store_gap=16,
                                 post_constant=False, pipeline_b=False, component_stream=False, component_sync_kk=0):
    """FP16-image/FP32-accumulate 4x(4*ncols) GEMM with one streamed B bank.

    Keeping only one four-vector B column live leaves room for a wider FP32
    accumulator tile.  C is a float image, so the accumulator vectors can be
    stored directly without the unreliable buffer-store repack.
    """
    if ncols not in (2, 3, 4): raise ValueError('streamed FP32 tile supports ncols 2..4')
    instrs = prologue_4x2(dev, threads)
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2)]
    for _ in range(ncols - 1): instrs += [ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2)]

    # The pipelined form alternates two B banks so one column can sample while
    # useful MADs consume the other.  The non-pipelined form minimizes fregs.
    b_banks = ((0, 1, 2, 3), (4, 5, 6, 7)) if pipeline_b else ((0, 1, 2, 3),)*2
    a_vecs, acc0 = ((8, 9, 10, 11), 12) if pipeline_b else ((4, 5, 6, 7), 8)
    state = acc0 + 4*ncols
    row_base, col_base, ky, k4 = (fvec(state, x) for x in range(4))
    instrs += [MOV_F32(row_base, 'r7.x'), MOV_F32(col_base, 'r7.y'), MOV_S32(ky, 3), MOV_S32(k4, 0)]
    for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0)

    def acc(row, col): return acc0 + col*4 + row

    def emit_b(col, b_vecs):
        for kk, dst in enumerate(b_vecs):
            instrs.append(MOV_F32(fvec(dst, 0), col_base) if col == 0 else ADD_S(fvec(dst, 0), col_base, col*32))
            instrs.append(MOV_F32(fvec(dst, 1), ky) if kk == 3 else ADD_S(fvec(dst, 1), ky, kk-3))
        for dst in b_vecs:
            emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, True)

    def emit_a():
        for row, dst in enumerate(a_vecs):
            instrs.append(MOV_F32(fvec(dst, 0), k4))
            instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))
        for dst in a_vecs:
            emit_coord_wait(instrs, coord_delay)
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, True)

    def emit_mads(col, b_vecs, sync):
        first = sync
        for kk in range(4):
            for row in range(4):
                out = acc(row, col)
                instrs.append(MAD_F32(fvec(out), fvec(a_vecs[row], kk), fvec(b_vecs[kk]),
                                      fvec(out), rpt=3, sy=first, r=True))
                first = False

    def emit_mads_component_stream(col):
        nonlocal instrs
        b_vecs = b_banks[0]
        for kk in range(4):
            for row in range(4):
                out = acc(row, col)
                instrs.append(MAD_F32(fvec(out), fvec(a_vecs[row], kk), fvec(b_vecs[kk]),
                                      fvec(out), rpt=3, sy=row == 0 and kk == (0 if col == 0 else component_sync_kk), r=True))
            if col+1 < ncols:
                dst = b_vecs[kk]
                instrs += [ADD_S(fvec(dst, 0), col_base, (col+1)*32),
                           MOV_F32(fvec(dst, 1), ky) if kk == 3 else ADD_S(fvec(dst, 1), ky, kk-3)]
                if kk: emit_isam_f32_vec(instrs, b_vecs[kk-1], fvec(b_vecs[kk-1], 0), 1, True)
        if col+1 < ncols:
            instrs.append(NOP_SS())
            emit_isam_f32_vec(instrs, b_vecs[3], fvec(b_vecs[3], 0), 1, True)

    loop_start = len(instrs)
    emit_b(0, b_banks[0])
    emit_a()
    if component_stream:
        for col in range(ncols): emit_mads_component_stream(col)
    elif pipeline_b:
        emit_b(1, b_banks[1])
        emit_mads(0, b_banks[0], True)
        for col in range(1, ncols):
            if col+1 < ncols: emit_b(col+1, b_banks[(col+1)&1])
            emit_mads(col, b_banks[col&1], col % 2 == 0)
    else:
        emit_mads(0, b_banks[0], True)
        for col in range(1, ncols):
            emit_b(col, b_banks[0])
            emit_mads(col, b_banks[0], sync_each_col)
    instrs += [ADD_S('r0.x', k4, 1), ADD_S(ky, ky, 4), CMPS_S_EQ(k4, K4-1, nop=1),
               MOV_F32(k4, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    if post_constant:
        for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0x44800000)

    for row in range(4):
        for col in range(ncols):
            instrs.append(MOV_F32('r4.x', col_base) if col == 0 else ADD_S('r4.x', col_base, col*32))
            instrs.append(MOV_F32('r4.y', row_base) if row == 0 else OR_B('r4.y', row_base, row))
            instrs += [MOV_F32(fvec(acc(row, col)), fvec(acc(row, col)), sy=True), NOP(rpt=5),
                       STIB_F32(fvec(acc(row, col)), 'r4.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, state+1, loop_end-loop_start


def build_4x8_fp32_rotate_shader(dev, threads, store_gap=16, post_constant=False, image_store=True, k_count=None,
                                batch_stride=0, batch_from_row=False, k_start=0, k_unroll=3, swap_groups=False,
                                col_from_z=False):
    """Three-bank FP32 GEMM: refill dead input registers under useful MADs."""
    if k_count is None: k_count = K4
    if k_count < 1: raise ValueError('rotating FP32 kernel requires a positive K4')
    if k_unroll < 3 or k_unroll % 3: raise ValueError('rotating FP32 K unroll must be a positive multiple of three')
    if k_start and (batch_stride or batch_from_row): raise ValueError('k_start is not implemented for batched rotating GEMM')
    if image_store:
        instrs = prologue_4x2(dev, threads, swap_groups=swap_groups)
        instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2),
                   ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2)]
        if col_from_z:
            instrs += [SHL_B('r6.w', 'r52.y', 6), NOP(rpt=2), ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2)]
        initial_row, initial_col = 'r7.x', 'r7.y'
    else:
        instrs = prologue_direct4_fp32(dev, threads, 2)
        initial_row, initial_col = 'r6.y', 'r6.z'
    banks = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
    acc0, state = 12, 20
    row_base, col_base, ky, k4 = (fvec(state, x) for x in range(4))
    # Offset reference slices need a zero-based counter because their absolute
    # K coordinate can exceed an immediate compare. Normal/batched leaves start
    # at zero and can reuse k4, saving one full register of occupancy metadata.
    counter = 'r21.x' if k_start else k4
    if batch_from_row:
        if not batch_stride or batch_stride & (batch_stride-1): raise ValueError('batch_stride must be a power of two')
        shift = batch_stride.bit_length()-1
        instrs += [MOV_F32(row_base, initial_row), SHR_B('r6.x', initial_row, shift), NOP(rpt=2),
                   SHL_B('r6.x', 'r6.x', shift), NOP(rpt=2), ADD_S(ky, 'r6.x', 3)]
    elif batch_stride:
        # r52.y is group_id.z in the compiler donor ABI. Stack each batch item
        # vertically in A/B/C images so one dispatch can execute many GEMMs.
        bits = [bit for bit in range(batch_stride.bit_length()) if batch_stride & (1 << bit)]
        instrs.append(SHL_B('r6.x', 'r52.y', bits[0]))
        for bit in bits[1:]:
            instrs += [SHL_B('r6.w', 'r52.y', bit), NOP(rpt=2), ADD_S_REG('r6.x', 'r6.x', 'r6.w')]
        instrs += [NOP(rpt=2), ADD_S_REG(row_base, initial_row, 'r6.x'), ADD_S(ky, 'r6.x', 3)]
    else:
        instrs += [MOV_F32(row_base, initial_row), MOV_S32(ky, k_start*4+3)]
    instrs += [MOV_F32(col_base, initial_col), MOV_S32(k4, k_start)]
    if k_start: instrs.append(MOV_S32(counter, 0))
    for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0)

    def setup_b_one(dst, col, kk):
        instrs.append(MOV_F32(fvec(dst, 0), col_base) if col == 0 else ADD_S(fvec(dst, 0), col_base, 32))
        instrs.append(MOV_F32(fvec(dst, 1), ky) if kk == 3 else ADD_S(fvec(dst, 1), ky, kk-3))

    def sample_b(dst):
        emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, True)

    def issue_b(bank, col):
        for kk, dst in enumerate(banks[bank]): setup_b_one(dst, col, kk)
        for dst in banks[bank]: sample_b(dst)

    def issue_a(bank):
        for row, dst in enumerate(banks[bank]):
            instrs.append(MOV_F32(fvec(dst, 0), k4))
            instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else OR_B(fvec(dst, 1), row_base, row))
        for dst in banks[bank]:
            emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, True)

    def phase(b0bank, b1bank, abank, prefetch=True):
        nonlocal instrs
        first = True
        for kk in range(4):
            for row in range(4):
                out = acc0 + row
                instrs.append(MAD_F32(fvec(out), fvec(banks[abank][row], kk), fvec(banks[b0bank][kk]),
                                      fvec(out), rpt=3, sy=first, r=True))
                first = False
        if not prefetch:
            for row in range(4):
                out = acc0 + 4 + row
                for kk in range(4):
                    instrs.append(MAD_F32(fvec(out), fvec(banks[abank][row], kk), fvec(banks[b1bank][kk]),
                                          fvec(out), rpt=3, r=True))
            return b0bank, b1bank, abank
        tmp = banks[b0bank][0]
        instrs += [ADD_S(fvec(tmp, 0), k4, 1), ADD_S(fvec(tmp, 1), ky, 4)]
        if k_start: instrs.append(ADD_S(counter, counter, 1))
        instrs += [NOP(rpt=2),
                   MOV_F32(k4, fvec(tmp, 0)), MOV_F32(ky, fvec(tmp, 1)), NOP(rpt=2)]
        issue_a(b0bank)
        # Row-major col1 work frees one A vector at a time; reuse each for one
        # next-K B1 vector while the remaining rows continue to execute.
        for row in range(4):
            out = acc0 + 4 + row
            for kk in range(4):
                instrs.append(MAD_F32(fvec(out), fvec(banks[abank][row], kk), fvec(banks[b1bank][kk]),
                                      fvec(out), rpt=3, r=True))
            setup_b_one(banks[abank][row], 1, row)
            if row: sample_b(banks[abank][row-1])
        for kk, dst in enumerate(banks[b1bank]): setup_b_one(dst, 0, kk)
        sample_b(banks[abank][3])
        for dst in banks[b1bank]: sample_b(dst)
        return b1bank, abank, b0bank

    issue_b(0, 0)
    issue_b(1, 1)
    issue_a(2)
    roles = phase(0, 1, 2, prefetch=k_count > 1)
    # Leave the final phase outside the loop. It consumes the last prefetched
    # tile without issuing the otherwise-unused next A/B sampler requests.
    loop_repeats, remainder = divmod(max(0, k_count-2), k_unroll)
    if not loop_repeats:
        loop_start = loop_end = len(instrs)
    else:
        loop_start = len(instrs)
        for _ in range(k_unroll): roles = phase(*roles)
        instrs += [CMPS_S_EQ(counter, 1+loop_repeats*k_unroll, nop=1), NOP(rpt=2)]
        loop_end = len(instrs)
        instrs.append(BR(loop_start-loop_end))
    for _ in range(remainder): roles = phase(*roles)
    if k_count > 1: roles = phase(*roles, prefetch=False)

    instrs += [MOV_F32(row_base, row_base, sy=True), NOP(rpt=8)]

    if post_constant:
        for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0x44800000)

    if image_store:
        for row in range(4):
            for col in range(2):
                instrs.append(MOV_F32('r4.x', col_base) if col == 0 else ADD_S('r4.x', col_base, 32))
                instrs.append(MOV_F32('r4.y', row_base) if row == 0 else OR_B('r4.y', row_base, row))
                out = acc0 + col*4 + row
                instrs += [MOV_F32(fvec(out), fvec(out), sy=True), NOP(rpt=5),
                           STIB_F32(fvec(out), 'r4.x'), NOP(rpt=store_gap)]
    else:
        instrs += [MOV_F32('r6.y', row_base), MOV_F32('r6.z', col_base)]
        emit_donor4_float_store2_lowcopy(instrs, dev, threads, acc0)
    instrs.append(END())
    return assemble(instrs), 1, 22 if k_start else 21, loop_end-loop_start


def build_4x4_fp32_halfwave_batch_shader(dev, threads, batch_stride=64, store_gap=16):
    """64-column FP32 leaf: map each half-wave to four distinct rows."""
    if threads != 64 or batch_stride != 64: raise ValueError('halfwave FP32 leaf requires threads=64 and stride=64')
    instrs = [MOV_F32('r13.x', 'r0.x')] + prologue_4x2(dev, threads)
    # The donor maps two waves to 8 rows and 32 float4 columns. Double its row
    # coordinate, then use lane bit 4 for another 4-row block; compact columns
    # to 16 float4 positions so every lane contributes to a 64-wide output.
    instrs += [SHL_B('r7.x', 'r7.x', 1), AND_B('r6.w', 'r13.x', 16), NOP(rpt=2),
               SHR_B('r6.w', 'r6.w', 2), AND_B('r7.y', 'r7.y', 15), NOP(rpt=2),
               ADD_S_REG('r7.x', 'r7.x', 'r6.w'), NOP(rpt=2)]
    bvecs, avecs, acc0, state = tuple(range(4)), tuple(range(4, 8)), 8, 12
    row_base, col_base, ky, k4 = (fvec(state, x) for x in range(4))
    instrs += [MOV_F32(row_base, 'r7.x'), MOV_F32(col_base, 'r7.y'), SHR_B('r6.w', 'r7.x', 6),
               NOP(rpt=2), SHL_B('r6.w', 'r6.w', 6), NOP(rpt=2), ADD_S(ky, 'r6.w', 3), MOV_S32(k4, 0)]
    for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0)
    loop_start = len(instrs)
    for kk, dst in enumerate(bvecs):
        instrs += [MOV_F32(fvec(dst, 0), col_base),
                   MOV_F32(fvec(dst, 1), ky) if kk == 3 else ADD_S(fvec(dst, 1), ky, kk-3)]
    for row, dst in enumerate(avecs):
        instrs += [MOV_F32(fvec(dst, 0), k4),
                   MOV_F32(fvec(dst, 1), row_base) if row == 0 else ADD_S(fvec(dst, 1), row_base, row)]
    for dst in bvecs: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, True)
    for dst in avecs: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, True)
    first = True
    for kk in range(4):
        for row in range(4):
            instrs.append(MAD_F32(fvec(acc0+row), fvec(avecs[row], kk), fvec(bvecs[kk]), fvec(acc0+row),
                                  rpt=3, sy=first, r=True))
            first = False
    instrs += [ADD_S('r0.x', k4, 1), ADD_S(ky, ky, 4), CMPS_S_EQ(k4, 15, nop=1),
               MOV_F32(k4, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    for row in range(4):
        instrs += [MOV_F32('r4.x', col_base), MOV_F32('r4.y', row_base) if row == 0 else ADD_S('r4.y', row_base, row),
                   MOV_F32(fvec(acc0+row), fvec(acc0+row), sy=True), NOP(rpt=5),
                   STIB_F32(fvec(acc0+row), 'r4.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, 14, loop_end-loop_start


def build_4x8_waksman_fp32_shader(dev, threads, store_gap=16, no_q=False):
    """Exact Waksman 4x4 microsteps for a 4-row, eight-column FP32 tile."""
    instrs = prologue_4x2(dev, threads)
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2),
               ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2)]
    # r0-r3 are first the raw A rows and then one B block.  r4-r7 hold the
    # transposed A K-components, r8-r15 are the eight output-column vectors.
    acc0, state = 8, 16
    row_base, col_base, ky, k4 = (fvec(state, x) for x in range(4))
    p0, p1, q0, q1, t0, t1, ts, zero = range(17, 25)
    instrs += [MOV_F32(row_base, 'r7.x'), MOV_F32(col_base, 'r7.y'), MOV_S32(ky, 3), MOV_S32(k4, 0)]
    for vec in range(acc0, acc0+8): emit_f32_vec_imm(instrs, vec, 0)
    emit_f32_vec_imm(instrs, zero, 0)

    def vadd(dst, lhs, rhs, sy=False):
        instrs.append(ADD_F(fvec(dst), fvec(lhs), fvec(rhs), rpt=3, r1=True, r2=True, sy=sy))

    def vsub(dst, lhs, rhs):
        instrs.append(SUB_F(fvec(dst), fvec(lhs), fvec(rhs), rpt=3, r1=True, r2=True))

    def add_scalar(dst, vec, scalar, sy=False):
        instrs.append(ADD_F(fvec(dst), fvec(vec), scalar, rpt=3, r1=True, sy=sy))

    def sub_scalar(dst, vec, scalar):
        instrs.append(SUB_F(fvec(dst), fvec(vec), scalar, rpt=3, r1=True))

    def product(dst, lhs, rhs):
        nonlocal instrs
        instrs += [MAD_F32(fvec(dst), fvec(lhs), fvec(rhs), fvec(zero), rpt=3, r=True, r1=True), NOP_SS()]

    loop_start = len(instrs)
    # Load four A rows, then transpose the 4x4 register block so each vector
    # contains one K scalar across the four output rows.
    for row in range(4):
        instrs += [MOV_F32(fvec(row, 0), k4),
                   MOV_F32(fvec(row, 1), row_base) if row == 0 else OR_B(fvec(row, 1), row_base, row)]
    for row in range(4): emit_isam_f32_vec(instrs, row, fvec(row, 0), 0, True)
    first = True
    for component in range(4):
        for row in range(4):
            instrs.append(MOV_F32(fvec(4+component, row), fvec(row, component), sy=first))
            if first: instrs.append(NOP(rpt=5))
            first = False

    def block(col: int) -> None:
        nonlocal instrs
        for kk in range(4):
            instrs += [MOV_F32(fvec(kk, 0), col_base) if col == 0 else ADD_S(fvec(kk, 0), col_base, 32),
                       MOV_F32(fvec(kk, 1), ky) if kk == 3 else ADD_S(fvec(kk, 1), ky, kk-3)]
        for kk in range(4): emit_isam_f32_vec(instrs, kk, fvec(kk, 0), 1, True)

        # q0=b1*(b0.x+b0), q1=b3*(b2.x+b2)
        instrs.append(ADD_F(fvec(t0), fvec(0, 0), fvec(0), rpt=3, r2=True, sy=True))
        product(q0, 1, t0)
        instrs.append(ADD_F(fvec(t0), fvec(2, 0), fvec(2), rpt=3, r2=True))
        product(q1, 3, t0)
        # p0=a0*(b0.x+a1), p1=a2*(b2.x+a3)
        add_scalar(t0, 5, fvec(0, 0)); product(p0, 4, t0)
        add_scalar(t0, 7, fvec(2, 0)); product(p1, 6, t0)
        c0 = acc0+col*4
        vadd(c0, c0, p0); vadd(c0, c0, p1)
        # r0=a1*(b1.x-a0), r1=a3*(b3.x-a2)
        instrs.append(SUB_F(fvec(t0), fvec(1, 0), fvec(4), rpt=3, r2=True))
        instrs += [MAD_F32(fvec(c0), fvec(5), fvec(t0), fvec(c0), rpt=3, r=True, r1=True), NOP_SS()]
        instrs.append(SUB_F(fvec(t0), fvec(3, 0), fvec(6), rpt=3, r2=True))
        instrs += [MAD_F32(fvec(c0), fvec(7), fvec(t0), fvec(c0), rpt=3, r=True, r1=True), NOP_SS()]

        for j in range(1, 4):
            cj = c0+j
            # (a0+b1[j])*(a1+b0[0]+b0[j]) - p0 - q0[j]
            add_scalar(t0, 4, fvec(1, j))
            instrs += [ADD_F(fvec(ts, 0), fvec(0, 0), fvec(0, j)), NOP(rpt=2)]
            add_scalar(t1, 5, fvec(ts, 0)); instrs.append(NOP(rpt=2))
            instrs += [MAD_F32(fvec(cj), fvec(t0), fvec(t1), fvec(cj), rpt=3, r=True, r1=True), NOP_SS()]
            vsub(cj, cj, p0)
            if not no_q: sub_scalar(cj, cj, fvec(q0, j))
            # (a2+b3[j])*(a3+b2[0]+b2[j]) - p1 - q1[j]
            add_scalar(t0, 6, fvec(3, j))
            instrs += [ADD_F(fvec(ts, 0), fvec(2, 0), fvec(2, j)), NOP(rpt=2)]
            add_scalar(t1, 7, fvec(ts, 0)); instrs.append(NOP(rpt=2))
            instrs += [MAD_F32(fvec(cj), fvec(t0), fvec(t1), fvec(cj), rpt=3, r=True, r1=True), NOP_SS()]
            vsub(cj, cj, p1)
            if not no_q: sub_scalar(cj, cj, fvec(q1, j))

    block(0)
    block(1)
    instrs += [ADD_S('r0.x', k4, 1), ADD_S(ky, ky, 4), CMPS_S_EQ(k4, K4-1, nop=1),
               MOV_F32(k4, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    # Waksman accumulates one vector down four rows per output column.  Build
    # each row vector in a dead high register immediately before storing it.
    for row in range(4):
        for col in range(2):
            for component in range(4):
                instrs.append(MOV_F32(fvec(p0, component), fvec(acc0+col*4+component, row),
                                      sy=(row == 0 and col == 0 and component == 0)))
                if row == 0 and col == 0 and component == 0: instrs.append(NOP(rpt=5))
            instrs += [MOV_F32('r4.x', col_base) if col == 0 else ADD_S('r4.x', col_base, 32),
                       MOV_F32('r4.y', row_base) if row == 0 else OR_B('r4.y', row_base, row),
                       MOV_F32(fvec(p0), fvec(p0), sy=True), NOP(rpt=5),
                       STIB_F32(fvec(p0), 'r4.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, 25, loop_end-loop_start


def build_8x4_fp32_shader(dev, threads, store_gap=16):
    """Eight-row, one-float4-column FP32 accumulator tile."""
    instrs = prologue_4x2(dev, threads)
    instrs += [SHL_B('r7.x', 'r7.x', 1), NOP(rpt=2)]
    b_vecs, a_vecs, acc0, state = (0, 1, 2, 3), tuple(range(4, 12)), 12, 20
    row_base, col_base, ky, k4 = (fvec(state, x) for x in range(4))
    instrs += [MOV_F32(row_base, 'r7.x'), MOV_F32(col_base, 'r7.y'), MOV_S32(ky, 3), MOV_S32(k4, 0)]
    for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0)

    loop_start = len(instrs)
    for kk, dst in enumerate(b_vecs):
        instrs.append(MOV_F32(fvec(dst, 0), col_base))
        instrs.append(MOV_F32(fvec(dst, 1), ky) if kk == 3 else ADD_S(fvec(dst, 1), ky, kk-3))
    for row, dst in enumerate(a_vecs):
        instrs.append(MOV_F32(fvec(dst, 0), k4))
        instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else ADD_S(fvec(dst, 1), row_base, row))
    for dst in b_vecs: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, True)
    for dst in a_vecs: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, True)
    first = True
    for kk in range(4):
        for row in range(8):
            out = acc0 + row
            instrs.append(MAD_F32(fvec(out), fvec(a_vecs[row], kk), fvec(b_vecs[kk]), fvec(out),
                                  rpt=3, sy=first, r=True))
            first = False
    instrs += [ADD_S('r0.x', k4, 1), ADD_S(ky, ky, 4), CMPS_S_EQ(k4, K4-1, nop=1),
               MOV_F32(k4, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    for row in range(8):
        instrs += [MOV_F32('r4.x', col_base), MOV_F32('r4.y', row_base) if row == 0 else ADD_S('r4.y', row_base, row),
                   MOV_F32(fvec(acc0+row), fvec(acc0+row), sy=True), NOP(rpt=5),
                   STIB_F32(fvec(acc0+row), 'r4.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, 21, loop_end-loop_start


def build_8x8_fp32_shader(dev, threads, store_gap=16, batch_stride=0):
    """Compact eight-row, two-float4-column FP32 accumulator tile."""
    instrs = prologue_4x2(dev, threads)
    instrs += [SHL_B('r7.x', 'r7.x', 1), MOV_F32('r6.w', 'r51.w'), NOP(rpt=2),
               SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2), ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2)]
    b_banks, a_vecs, acc0, state = (tuple(range(4)), tuple(range(4, 8))), tuple(range(8, 16)), 16, 32
    row_base, col_base, ky, k4 = (fvec(state, x) for x in range(4))
    if batch_stride:
        if batch_stride & (batch_stride-1): raise ValueError('batch_stride must be a power of two')
        shift = batch_stride.bit_length()-1
        instrs += [MOV_F32(row_base, 'r7.x'), MOV_F32(col_base, 'r7.y'), SHR_B('r6.w', 'r7.x', shift), NOP(rpt=2),
                   SHL_B('r6.w', 'r6.w', shift), NOP(rpt=2), ADD_S(ky, 'r6.w', 3), MOV_S32(k4, 0)]
    else:
        instrs += [MOV_F32(row_base, 'r7.x'), MOV_F32(col_base, 'r7.y'), MOV_S32(ky, 3), MOV_S32(k4, 0)]
    for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0)
    loop_start = len(instrs)
    for col, bank in enumerate(b_banks):
        for kk, dst in enumerate(bank):
            instrs.append(MOV_F32(fvec(dst, 0), col_base) if col == 0 else ADD_S(fvec(dst, 0), col_base, 32))
            instrs.append(MOV_F32(fvec(dst, 1), ky) if kk == 3 else ADD_S(fvec(dst, 1), ky, kk-3))
    for row, dst in enumerate(a_vecs):
        instrs.append(MOV_F32(fvec(dst, 0), k4))
        instrs.append(MOV_F32(fvec(dst, 1), row_base) if row == 0 else ADD_S(fvec(dst, 1), row_base, row))
    for bank in b_banks:
        for dst in bank: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, True)
    for dst in a_vecs: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, True)
    first = True
    for kk in range(4):
        for col, bank in enumerate(b_banks):
            for row in range(8):
                out = acc0 + col*8 + row
                instrs.append(MAD_F32(fvec(out), fvec(a_vecs[row], kk), fvec(bank[kk]), fvec(out),
                                      rpt=3, sy=first, r=True))
                first = False
    instrs += [ADD_S('r0.x', k4, 1), ADD_S(ky, ky, 4), CMPS_S_EQ(k4, K4-1, nop=1),
               MOV_F32(k4, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    for row in range(8):
        for col in range(2):
            out = acc0 + col*8 + row
            instrs += [MOV_F32('r4.x', col_base) if col == 0 else ADD_S('r4.x', col_base, 32),
                       MOV_F32('r4.y', row_base) if row == 0 else ADD_S('r4.y', row_base, row),
                       MOV_F32(fvec(out), fvec(out), sy=True), NOP(rpt=5), STIB_F32(fvec(out), 'r4.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, 33, loop_end-loop_start


def build_rx8_fp32_shader(dev, threads, rows=6, store_gap=16):
    """Two-float4-column FP32 tile with a tunable row/register footprint."""
    if rows not in (5, 6): raise ValueError('rx8 FP32 currently supports five or six rows')
    instrs = prologue_4x2(dev, threads)
    # Convert compiler row=(gy*groups+tm)*4 to (gy*groups+tm)*rows.
    instrs += [SHR_B('r6.w', 'r7.x', 2), NOP(rpt=2)]
    if rows == 5:
        instrs += [MOV_F32('r7.x', 'r6.w'), SHL_B('r6.w', 'r6.w', 2), NOP(rpt=2),
                   ADD_S_REG('r7.x', 'r7.x', 'r6.w'), NOP(rpt=2)]
    else:
        instrs += [SHL_B('r7.x', 'r6.w', 2), SHL_B('r6.w', 'r6.w', 1), NOP(rpt=2),
                   ADD_S_REG('r7.x', 'r7.x', 'r6.w'), NOP(rpt=2)]
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2),
               ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2)]
    b_banks, a0, acc0 = (tuple(range(4)), tuple(range(4, 8))), 8, 8+rows
    a_vecs = tuple(range(a0, a0+rows))
    state = acc0 + 2*rows
    row_base, col_base, ky, k4 = (fvec(state, x) for x in range(4))
    instrs += [MOV_F32(row_base, 'r7.x'), MOV_F32(col_base, 'r7.y'), MOV_S32(ky, 3), MOV_S32(k4, 0)]
    for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0)
    loop_start = len(instrs)
    for col, bank in enumerate(b_banks):
        for kk, dst in enumerate(bank):
            instrs.append(MOV_F32(fvec(dst, 0), col_base) if col == 0 else ADD_S(fvec(dst, 0), col_base, 32))
            instrs.append(MOV_F32(fvec(dst, 1), ky) if kk == 3 else ADD_S(fvec(dst, 1), ky, kk-3))
    for row, dst in enumerate(a_vecs):
        instrs += [MOV_F32(fvec(dst, 0), k4), MOV_F32(fvec(dst, 1), row_base) if row == 0 else ADD_S(fvec(dst, 1), row_base, row)]
    for bank in b_banks:
        for dst in bank: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, True)
    for dst in a_vecs: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 0, True)
    first = True
    for kk in range(4):
        for col, bank in enumerate(b_banks):
            for row in range(rows):
                out = acc0 + col*rows + row
                instrs.append(MAD_F32(fvec(out), fvec(a_vecs[row], kk), fvec(bank[kk]), fvec(out),
                                      rpt=3, sy=first, r=True))
                first = False
    instrs += [ADD_S('r0.x', k4, 1), ADD_S(ky, ky, 4), CMPS_S_EQ(k4, K4-1, nop=1), MOV_F32(k4, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    for row in range(rows):
        for col in range(2):
            out = acc0 + col*rows + row
            instrs += [MOV_F32('r4.x', col_base) if col == 0 else ADD_S('r4.x', col_base, 32),
                       MOV_F32('r4.y', row_base) if row == 0 else ADD_S('r4.y', row_base, row),
                       MOV_F32(fvec(out), fvec(out), sy=True), NOP(rpt=5), STIB_F32(fvec(out), 'r4.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, state+1, loop_end-loop_start


def build_4x8_fp32_quad_a_shader(dev, threads, store_gap=16):
    """4x8 FP32 GEMM where a quad cooperatively loads and broadcasts A rows."""
    instrs = [MOV_F32('r21.w', 'r0.x')] + prologue_4x2(dev, threads)
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2),
               ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2), AND_B('r6.w', 'r21.w', 3), NOP(rpt=2)]
    b_banks, a_sample, a_vecs, acc0, state = (tuple(range(4)), tuple(range(4, 8))), 8, tuple(range(9, 13)), 13, 21
    row_lane, col_base, k4, scratch = (fvec(state, x) for x in range(4))
    instrs += [ADD_S_REG(row_lane, 'r7.x', 'r6.w'), MOV_F32(col_base, 'r7.y'), MOV_S32(k4, 0)]
    for vec in range(acc0, state): emit_f32_vec_imm(instrs, vec, 0)
    loop_start = len(instrs)
    instrs += [SHL_B(scratch, k4, 2), NOP(rpt=2), ADD_S(scratch, scratch, 3), NOP(rpt=2)]
    for col, bank in enumerate(b_banks):
        for kk, dst in enumerate(bank):
            instrs.append(MOV_F32(fvec(dst, 0), col_base) if col == 0 else ADD_S(fvec(dst, 0), col_base, 32))
            instrs.append(MOV_F32(fvec(dst, 1), scratch) if kk == 3 else ADD_S(fvec(dst, 1), scratch, kk-3))
    instrs += [MOV_F32(fvec(a_sample, 0), k4), MOV_F32(fvec(a_sample, 1), row_lane)]
    for bank in b_banks:
        for dst in bank: emit_isam_f32_vec(instrs, dst, fvec(dst, 0), 1, True)
    emit_isam_f32_vec(instrs, a_sample, fvec(a_sample, 0), 0, True)
    instrs += [MOV_F32(scratch, scratch, sy=True), NOP(rpt=5)]
    for row, dst in enumerate(a_vecs):
        instrs += [MOV_S32(scratch, row), NOP(rpt=2)]
        # A single xyzw broadcast avoids four back-to-back partial writes to
        # the same full register, which A630 can silently drop.
        instrs.append(QUAD_BRCST(fvec(dst), fvec(a_sample), scratch, typ=3, wrmask=15))
    first = True
    for kk in range(4):
        for col, bank in enumerate(b_banks):
            for row in range(4):
                out = acc0 + col*4 + row
                instrs.append(MAD_F32(fvec(out), fvec(a_vecs[row], kk), fvec(bank[kk]), fvec(out),
                                      rpt=3, r=True, sy=first))
                first = False
    instrs += [ADD_S('r0.x', k4, 1), CMPS_S_EQ(k4, K4-1, nop=1), MOV_F32(k4, 'r0.x'), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    instrs += [SHR_B('r7.x', row_lane, 2), NOP(rpt=2), SHL_B('r7.x', 'r7.x', 2), MOV_F32('r7.y', col_base), NOP(rpt=2)]
    for row in range(4):
        for col in range(2):
            out = acc0 + col*4 + row
            instrs += [MOV_F32('r4.x', 'r7.y') if col == 0 else ADD_S('r4.x', 'r7.y', 32),
                       MOV_F32('r4.y', 'r7.x') if row == 0 else ADD_S('r4.y', 'r7.x', row),
                       MOV_F32(fvec(out), fvec(out), sy=True), NOP(rpt=5), STIB_F32(fvec(out), 'r4.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, 22, loop_end-loop_start


def build_4x8_fp32_quad_splitk_shader(dev, threads, store_gap=16, no_reduce=False, k_count=None):
    """Four quad lanes own K mod 4, then reduce their FP32 partial sums."""
    if k_count is None: k_count = K4
    instrs = [MOV_F32('r12.w', 'r0.x')] + prologue_4x2(dev, threads)
    # Standard row mapping already groups each four consecutive lanes onto the
    # same four rows. Compact 32 column lanes to eight logical quad lanes.
    instrs += [MOV_F32('r12.x', 'r7.x'), SHR_B('r13.x', 'r7.y', 5), SHR_B('r13.y', 'r12.w', 2),
               AND_B('r13.y', 'r13.y', 7), NOP(rpt=2), SHL_B('r13.x', 'r13.x', 4), NOP(rpt=2),
               ADD_S_REG('r12.y', 'r13.x', 'r13.y'), AND_B('r12.w', 'r12.w', 3), MOV_S32('r12.z', 0), NOP(rpt=2)]
    acc0 = 4
    for vec in range(acc0, acc0+8): emit_f32_vec_imm(instrs, vec, 0)
    loop_start = len(instrs)
    instrs += [SHL_B('r13.x', 'r12.z', 2), NOP(rpt=2), ADD_S_REG('r13.x', 'r13.x', 'r12.w'),
               SHR_B('r13.y', 'r12.x', 2), NOP(rpt=2),
               MOV_F32('r0.x', 'r13.x'), MOV_F32('r0.y', 'r13.y'),
               MOV_F32('r1.x', 'r12.y'), MOV_F32('r1.y', 'r13.x'),
               ADD_S('r2.x', 'r12.y', 8), MOV_F32('r2.y', 'r13.x'), NOP(rpt=2)]
    emit_isam_f32_vec(instrs, 0, 'r0.x', 0, True)
    emit_isam_f32_vec(instrs, 1, 'r1.x', 1, True)
    emit_isam_f32_vec(instrs, 2, 'r2.x', 1, True)
    first = True
    for col, bvec in enumerate((1, 2)):
        for row in range(4):
            out = acc0 + col*4 + row
            instrs.append(MAD_F32(fvec(out), fvec(0, row), fvec(bvec), fvec(out), rpt=3, sy=first, r=True))
            first = False
    instrs += [ADD_S('r14.x', 'r12.z', 1), NOP(rpt=2), CMPS_S_EQ('r12.z', k_count-1, nop=1),
               MOV_F32('r12.z', 'r14.x'), NOP(rpt=2)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    instrs += [MOV_F32('r12.z', 'r12.z', sy=True), NOP(rpt=8)]
    if no_reduce:
        instrs.append(END())
        return assemble(instrs), 1, 15, loop_end-loop_start
    emit_f32_vec_imm(instrs, 3, 0x3f800000)
    for row in range(4):
        for col in range(2):
            src = acc0 + col*4 + row
            emit_f32_vec_imm(instrs, 0, 0)
            for lane in range(4):
                instrs += [MOV_S32('r12.w', lane), NOP(rpt=2)]
                for comp in range(4):
                    instrs += [QUAD_BRCST('r13.x', fvec(src, comp), 'r12.w', typ=3), NOP(rpt=2),
                               MAD_F32(fvec(0, comp), 'r13.x', 'r3.x', fvec(0, comp))]
            instrs += [MOV_F32('r13.x', 'r12.y') if col == 0 else ADD_S('r13.x', 'r12.y', 8),
                       MOV_F32('r13.y', 'r12.x') if row == 0 else ADD_S('r13.y', 'r12.x', row),
                       MOV_F32('r0.x', 'r0.x', sy=True), NOP(rpt=5), STIB_F32('r0.x', 'r13.x'), NOP(rpt=store_gap)]
    instrs.append(END())
    return assemble(instrs), 1, 15, loop_end-loop_start


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
        for row in range(4): emit_hvec_copy(instrs, row * 4, acc0 + (row * ncols + col) * 4)
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
        for row in range(4): emit_hvec_copy(instrs, row * 4, acc0 + (row * ncols + col) * 4)
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
        for row in range(4): emit_hvec_copy(instrs, row * 4, acc0 + (row * ncols + col) * 4)
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
            emit_hvec_copy(instrs, out, acc0 + out)
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
            emit_hvec_copy(instrs, out * 4, acc0 + out * 4)
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

    acc0 = _hreg('hr20.x')
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


def build_4x16_isolated_shader(dev, threads, k_unroll=4):
    """4x16 FP16 GEMM with disjoint state, input, and accumulator banks.

    hr0..hr15 are accumulators, hr16..hr19 are A, hr20..hr27 are B, and
    r14+ owns all persistent state/texture coordinates.  This deliberately
    avoids relying on merged full/half register liveness.
    """
    if K4 % k_unroll: raise ValueError('k_unroll must divide K/4')
    instrs = prologue_4x2(dev, threads)
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2)]
    for _ in range(3): instrs += [ADD_S_REG('r7.y', 'r7.y', 'r6.w'), NOP(rpt=2)]
    # Save row/column before the accumulator initialization overwrites r0..r7.
    instrs += [MOV_F32('r14.x', 'r7.x'), MOV_F32('r14.y', 'r7.y'), MOV_S32('r14.z', 0), MOV_S32('r14.w', 0), NOP(rpt=2)]
    for base in range(_hreg('hr0.x'), _hreg('hr16.x'), 4): emit_hvec_imm(instrs, base, 0)
    loop_start = len(instrs)

    def body():
        nonlocal instrs
        acoords = ('r15.x', 'r15.z', 'r16.x', 'r16.z')
        bcoords = ('r17.x', 'r17.z', 'r18.x', 'r18.z', 'r19.x', 'r19.z', 'r20.x', 'r20.z')
        for row, coord in enumerate(acoords):
            yreg = f"r{int(coord[1:coord.index('.')])}.{'y' if coord.endswith('.x') else 'w'}"
            instrs += [MOV_F32(coord, 'r14.z'), MOV_F32(yreg, 'r14.x') if row == 0 else ADD_S(yreg, 'r14.x', row)]
        instrs.append(NOP(rpt=2))
        for row, coord in enumerate(acoords): instrs.append(ISAM_F16(_hreg('hr16.x') + row*4, coord, 0))
        # hr20..hr27 hold two B column groups.  Consume them, then reuse the
        # same bank for the other two groups; hr28+ is persistent full state.
        for col_base in (0, 2):
            for i, coord in enumerate(bcoords):
                yreg = f"r{int(coord[1:coord.index('.')])}.{'y' if coord.endswith('.x') else 'w'}"
                col, kk = divmod(i, 4)
                xoff = (col_base+col)*32
                instrs += [MOV_F32(coord, 'r14.y') if xoff == 0 else ADD_S(coord, 'r14.y', xoff),
                           MOV_F32(yreg, 'r14.w') if kk == 0 else ADD_S(yreg, 'r14.w', kk)]
            instrs.append(NOP(rpt=2))
            for i, coord in enumerate(bcoords): instrs.append(ISAM_F16(_hreg('hr20.x') + i*4, coord, 1))
            first = True
            for row in range(4):
                for col in range(2):
                    for kk in range(4):
                        dst = (row*4+col_base+col)*4
                        instrs.append(MAD_F16(dst, _hreg('hr20.x')+(col*4+kk)*4,
                                              _hreg('hr16.x')+row*4+kk, dst,
                                              rpt=3, sy=first, r1=True, r3=True))
                        first = False

    for ku in range(k_unroll):
        body()
        if ku != k_unroll-1: instrs += [ADD_S('r14.z', 'r14.z', 1), ADD_S('r14.w', 'r14.w', 4)]
    instrs += [ADD_S('r14.z', 'r14.z', 1), ADD_S('r14.w', 'r14.w', 4),
               CMPS_S_EQ('r14.z', K4, nop=1)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    for row in range(4):
        for col in range(4):
            instrs += [MOV_F32('r21.x', 'r14.y') if col == 0 else ADD_S('r21.x', 'r14.y', col*32),
                       MOV_F32('r21.y', 'r14.x') if row == 0 else ADD_S('r21.y', 'r14.x', row),
                       COV_F16F32('r22.x', (row*4+col)*4, sy=True, rpt=3, r=True), NOP(rpt=5),
                       STIB_F32('r22.x', 'r21.x'), NOP(rpt=16)]
    instrs.append(END())
    return assemble(instrs), loop_end-loop_start


def build_4xn_shader(dev, threads, ncols=2, direct=False, quad_a=False, store_constant=False, post_constant=False, donor_store=False, donor2_store=False, native_store=False, hybrid_store=False, safe_store=False, linear_store=False, image_store=False, preserve_coords=False, high_inputs=False, high_a_only=False, high_store=False, copy_b_probe=False, thread_store=False, repeat_first_store=False, repair_row1_store=False, repeat_each_store=False, pipeline=False, preload_b=False, preload_b_safe_coords=False, stream_b=False, stream_b_no_sync=False, b_kk_pipeline=False, b_first=False, compact_acc=False, stable_bx=False, stable_ay=False, low_a_coords=False, inc_coords=False, persistent_coords=False, serial_b_cols=False, single_cols_all=False, k_unroll=1, row_col_kk=False, alu_order='auto', first_sync_only=False, row_sync=False, no_store=False, skip_a_loads=False, skip_b_loads=False, alu_reps=1, coord_delay=4, stable_settle_delay=5, store_gap=-1, store_start=0, store_count=-1, store_or_cols=False, store_scalar_offsets=False, store_shlg_offsets=False, store_row_shift=10, k_start=0, k_count=None, store_row_base=0, safe_b_y=False, sync_b_y=False, first_cols_only=False, first_cols_offset=0, separate_b_coords=False, high_b_coords=False, persistent_b_coords=False, persistent_b_x=False, reuse_separate_b_y=False, interleave_second_pair=False, acc_hr=None, save_output_coords=False, vector_init=True, dynamic_split_k=0):
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
    # r10 is not used by the compact 4xN coordinate/load schedule. Preserve the
    # raw local id there for the thread-major epilogue.
    if thread_store: instrs.insert(0, MOV_F32('r10.x', 'r0.x'))
    if k_start: instrs += [MOV_S32('r6.z', k_start), MOV_S32('r6.y', k_start*4+3, sy=True)]
    if dynamic_split_k:
        if k_start or k_count is not None or K4 % dynamic_split_k: raise ValueError('dynamic split-K owns the full K range')
        chunk4 = K4 // dynamic_split_k
        if chunk4 != 96: raise ValueError('dynamic split-K currently supports 96 K/4 steps per split')
        if M // ((threads//32)*4) != 8: raise ValueError('dynamic split-K currently requires eight row workgroups')
        # Fold split into group_id(1): gy=split*8+row_group. The ordinary 2-D
        # prologue has already formed row=gy*16+thread_row, so masking to 127
        # restores the model row while gy>>3 supplies the K/output split.
        instrs += [SHR_B('r19.x', 'r7.x', 7), AND_B('r19.x', 'r19.x', dynamic_split_k-1),
                   AND_B('r7.x', 'r7.x', 127), AND_B('r2.z', 'r2.z', 127),
                   AND_B('r3.x', 'r3.x', 127), AND_B('r3.z', 'r3.z', 127), AND_B('r4.x', 'r4.x', 127), NOP(rpt=2),
                   SHL_B('r19.z', 'r19.x', 5), SHL_B('r19.w', 'r19.x', 6),
                   NOP(rpt=2), ADD_S_REG('r6.z', 'r19.z', 'r19.w'), NOP(rpt=2),
                   SHL_B('r6.y', 'r6.z', 2), NOP(rpt=2), ADD_S('r6.y', 'r6.y', 3),
                   MOV_S32('r6.w', 0), NOP(rpt=2)]
    if first_cols_offset: instrs += [ADD_S('r7.y', 'r7.y', first_cols_offset * 32), NOP(rpt=2)]
    if preserve_coords:
        if not (image_store and direct and ncols == 4 and stable_ay and not low_a_coords):
            raise ValueError('--preserve-coords requires the 4x16 stable image-store schedule')
        # r2 is unused by this coordinate schedule.  Its physical half-register
        # alias (hr2) must not be sampled while these values are live.
        instrs += [MOV_F32('r2.x', 'r7.x'), MOV_F32('r2.y', 'r7.y')]
    if high_inputs and not (direct and ncols == 4 and compact_acc):
        raise ValueError('--high-inputs requires compact direct 4x16 mode')
    if high_a_only and not (direct and compact_acc and not high_inputs):
        raise ValueError('--high-a-only requires compact direct mode without --high-inputs')
    if linear_store: instrs += [MOV_F32('r11.z', 'r7.x'), MOV_F32('r11.w', 'r7.y')]
    output_save_reg = 'r18' if persistent_b_coords or preload_b_safe_coords else 'r10'
    if save_output_coords: instrs += [MOV_F32(f'{output_save_reg}.x', 'r7.x'), MOV_F32(f'{output_save_reg}.y', 'r7.y')]
    instrs += [MOV_F32('r4.y', 'r7.y'), MOV_F32('r4.w', 'r7.y'), MOV_F32('r5.y', 'r7.y'), MOV_F32('r5.w', 'r7.y')]
    if quad_a: instrs.append(MOV_F32('r6.w', 'r0.x'))  # keep lane id for the divergent A-load mask
    instrs.append(ADD_S('r7.z', 'r7.y', 32))   # second col4 block x coordinate

    if preload_b and (not direct or (ncols != 4 and not (preload_b_safe_coords and ncols == 3))):
        raise ValueError('--preload-b requires direct ncols4, or safe-coordinate direct ncols3')
    if preload_b_safe_coords and not preload_b: raise ValueError('--preload-b-safe-coords requires --preload-b')
    if stream_b and (not direct or ncols != 4 or preload_b or alu_reps != 1): raise ValueError('--stream-b requires --direct --ncols 4 without --preload-b/--alu-reps')
    if stream_b_no_sync and not stream_b: raise ValueError('--stream-b-no-sync requires --stream-b')
    if b_kk_pipeline and (not direct or ncols != 4 or preload_b or stream_b or alu_reps != 1): raise ValueError('--b-kk-pipeline requires --direct --ncols 4 without other B schedules')
    if b_first and (not direct or ncols != 4 or preload_b or stream_b or b_kk_pipeline or quad_a): raise ValueError('--b-first requires plain non-quad --direct --ncols 4')
    if low_a_coords and not stable_ay: raise ValueError('--low-a-coords requires --stable-ay')
    if compact_acc and (not direct or preload_b): raise ValueError('--compact-acc requires direct mode without --preload-b')
    if stable_bx and (not direct or ncols != 4 or preload_b or stream_b or b_kk_pipeline or (alu_reps != 1 and not no_store)): raise ValueError('--stable-bx requires plain --direct --ncols 4')
    if stable_ay and (quad_a or not direct or b_kk_pipeline): raise ValueError('--stable-ay requires non-quad direct mode without --b-kk-pipeline')
    if (skip_a_loads or skip_b_loads) and ((not no_store and not preserve_coords) or not direct or preload_b or stream_b or b_kk_pipeline):
        raise ValueError('--skip-*-loads require no-store or the preserved image-store probe')
    if inc_coords and not (stable_bx and stable_ay): raise ValueError('--inc-coords requires --stable-bx --stable-ay')
    if persistent_coords and not inc_coords: raise ValueError('--persistent-coords requires --inc-coords')
    if persistent_b_coords and not (separate_b_coords and persistent_coords and direct and ncols == 4):
        raise ValueError('--persistent-b-coords requires direct 4x16 separate/persistent coordinates')
    if persistent_b_x and not (separate_b_coords and direct and ncols == 2):
        raise ValueError('--persistent-b-x requires direct 4x8 separate coordinates')
    if reuse_separate_b_y and not (separate_b_coords and direct and ncols == 4):
        raise ValueError('--reuse-separate-b-y requires direct 4x16 separate coordinates')
    if interleave_second_pair and not (separate_b_coords and direct and ncols == 4 and b_first and alu_order == 'kk_col_row'):
        raise ValueError('interleaved second pair requires direct 4x16 separate coords, b-first, and kk_col_row')
    if k_count is None: k_count = (K4//dynamic_split_k) if dynamic_split_k else K4-k_start
    if k_start < 0 or k_count <= 0 or k_start+k_count > K4 or k_count % k_unroll: raise ValueError('invalid split-K range')
    k_end = k_start+k_count
    if K4 % k_unroll != 0: raise ValueError('--k-unroll must divide K/4')
    if alu_order == 'auto': alu_order = 'row_col_kk' if row_col_kk else 'kk_row_col'
    acc0 = _hreg(f'hr{acc_hr}.x') if acc_hr is not None else \
           (_hreg('hr16.x') if low_a_coords else _hreg('hr20.x')) if high_inputs or high_a_only else \
           _hreg('hr20.x') if preload_b else _hreg('hr12.x') if compact_acc else _hreg('hr16.x')
    acc_groups = 4 * ncols if direct else 32
    if high_inputs:
        for lane in range(acc_groups * 4): instrs.append(MOV_H_IMM(acc0 + lane, 0x6400 if store_constant else 0))
    elif vector_init:
        instrs += [MOV_H_IMM(acc0, 0x6400 if store_constant else 0), MOV_H(acc0 + 1, acc0, rpt=2)]
        for base in range(acc0 + 4, acc0 + acc_groups * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    else:
        for base in range(acc0, acc0 + acc_groups * 4, 4):
            emit_hvec_imm(instrs, base, 0x6400 if store_constant else 0)
    if skip_a_loads:
        for base in ([*range(_hreg('hr40.x' if low_a_coords else 'hr44.x'),
                                  _hreg('hr44.x' if low_a_coords else 'hr48.x'), 4)] if high_inputs else
                     [_hreg('hr0.x'), _hreg('hr1.x'), _hreg('hr3.x')] if preserve_coords else
                     range(_hreg('hr0.x'), _hreg('hr4.x'), 4)):
            instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
    if skip_b_loads:
        for base in (range(_hreg('hr32.x' if low_a_coords else 'hr36.x'),
                           _hreg('hr40.x' if low_a_coords else 'hr44.x'), 4) if high_inputs else
                     range(_hreg('hr4.x'), _hreg('hr12.x'), 4)):
            instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
    if stable_bx:
        instrs += [ADD_S('r0.x', 'r7.y', 64), NOP(rpt=2), MOV_F32('r0.z', 'r0.x'), MOV_F32('r1.x', 'r0.x'), MOV_F32('r1.z', 'r0.x')]
    a_xregs = ['r2.x', 'r2.z', 'r3.x', 'r3.z'] if low_a_coords else ['r8.x', 'r8.z', 'r9.x', 'r9.z']
    a_yregs = ['r2.y', 'r2.w', 'r3.y', 'r3.w'] if low_a_coords else ['r8.y', 'r8.w', 'r9.y', 'r9.w']
    col3_x = 'r6.w' if low_a_coords else 'r3.w'
    if stable_ay:
        instrs += [MOV_F32(a_yregs[0], 'r7.x'), OR_B(a_yregs[1], 'r7.x', 1), OR_B(a_yregs[2], 'r7.x', 2), OR_B(a_yregs[3], 'r7.x', 3)]
        if inc_coords: instrs.append(ADD_S(col3_x, 'r7.y', 96))

    if copy_b_probe:
        if not (image_store and direct): raise ValueError('--copy-b-probe requires direct image store')
        loop_start = loop_end = len(instrs)
        instrs += [MOV_F32('r4.z', 'r4.z', sy=True)]
        if stable_settle_delay >= 0: instrs.append(NOP(rpt=stable_settle_delay))
        for row in range(4):
          for col in range(ncols):
            instrs += [MOV_F32('r4.x', 'r7.y') if col == 0 else ADD_S('r4.x', 'r7.y', col*32),
                       MOV_S32('r4.y', 0), NOP(rpt=2),
                       ISAM_F16(acc0+(row*ncols+col)*4, 'r4.x', 1), NOP(rpt=5),
                       MOV_F32('r4.z', 'r4.z', sy=True)]
    elif store_constant:
        loop_start = loop_end = len(instrs)
    else:

        if persistent_coords:
            instrs += [
                MOV_F32(a_xregs[0], 'r6.z'), MOV_F32(a_xregs[1], 'r6.z'), MOV_F32(a_xregs[2], 'r6.z'), MOV_F32(a_xregs[3], 'r6.z'),
                ADD_S('r4.z', 'r6.y', -3), ADD_S('r5.x', 'r6.y', -2), ADD_S('r5.z', 'r6.y', -1), MOV_F32('r6.x', 'r6.y'),
            ]
        persistent_b_coord_regs = tuple((f'r{10+col*2+row//2}.{"xz"[row&1]}', f'r{10+col*2+row//2}.{"yw"[row&1]}')
                                        for col in range(4) for row in range(4))
        if persistent_b_x:
            for col in range(2):
                for row in range(4):
                    xreg, _ = persistent_b_coord_regs[col*4+row]
                    instrs.append(MOV_F32(xreg, 'r7.y') if col == 0 else ADD_S(xreg, 'r7.y', col*32))
        if persistent_b_coords:
            b_y_sources = ('r4.z', 'r5.x', 'r5.z', 'r6.x')
            for col in range(4):
                for row in range(4):
                    xreg, yreg = persistent_b_coord_regs[col*4+row]
                    instrs += [MOV_F32(xreg, 'r7.y') if col == 0 else ADD_S(xreg, 'r7.y', col*32),
                               MOV_F32(yreg, b_y_sources[row])]

        loop_start = len(instrs)

        def emit_one_k(kx_reg, ky_reg, sync_first=True, reuse_coords=False):
          nonlocal instrs
          if high_a_only:
              # A lives in hr8:11, aliasing its coordinate bank. Rebuild all
              # coordinates before every K iteration; loads consume row 1 before
              # row 0 overwrites r8, and row 3 before row 2 overwrites r9.
              instrs += [MOV_F32(a_xregs[0], a_xregs[0], ss=True), MOV_F32(a_xregs[0], kx_reg, rpt=3), MOV_F32(a_xregs[2], kx_reg, rpt=3),
                         MOV_F32(a_yregs[0], 'r7.x'), ADD_S(a_yregs[1], 'r7.x', 1),
                         ADD_S(a_yregs[2], 'r7.x', 2), ADD_S(a_yregs[3], 'r7.x', 3)]
          elif preserve_coords:
              # Rebuild all A texture coordinates from the saved output row.
              # The B samples later overwrite r8/r9 through merged registers.
              instrs += [
                  MOV_F32(a_xregs[0], kx_reg), MOV_F32(a_yregs[0], 'r2.x'),
                  MOV_F32(a_xregs[1], kx_reg), ADD_S(a_yregs[1], 'r2.x', 1),
                  MOV_F32(a_xregs[2], kx_reg), ADD_S(a_yregs[2], 'r2.x', 2),
                  MOV_F32(a_xregs[3], kx_reg), ADD_S(a_yregs[3], 'r2.x', 3),
                  MOV_F32('r2.z', kx_reg),
              ]
          elif stable_ay and not reuse_coords:
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
          if not reuse_coords and not high_a_only:
              instrs += [ADD_S('r4.z', ky_reg, -3), ADD_S('r5.x', ky_reg, -2), ADD_S('r5.z', ky_reg, -1), MOV_F32('r6.x', ky_reg)]
          if quad_a:
              instrs += [
                  MOV_F32('r0.z', 'r6.w'),
                  AND_B('r0.x', 'r0.z', 3, nop=3),
                  CMPS_S_EQ('r0.x', 0),
                  NOP(rpt=stable_settle_delay),
                  BR(5),
              ]
          high_a = ('hr43.x', 'hr42.x', 'hr41.x', 'hr40.x') if low_a_coords else ('hr47.x', 'hr46.x', 'hr45.x', 'hr44.x')
          # Compact disjoint layout: accumulators hr12:27 and B hr28:35. A is
          # reloaded into hr0:3 after each pair's coordinate setup, allowing 36
          # total registers without aliasing live sampler results.
          high_a_only_regs = ('hr16.x', 'hr17.x', 'hr18.x', 'hr19.x')
          a_coords = (list(zip(high_a, a_xregs)) if high_inputs else
                      list(zip(high_a_only_regs, a_xregs if stable_ay else ('r2.y', 'r2.w', 'r3.y', 'r3.w'))) if high_a_only else
                      [('hr3.x', a_xregs[0]), ('hr1.x', a_xregs[2]), ('hr0.x', a_xregs[3])] if preserve_coords else
                      [('hr3.x', a_xregs[0]), ('hr2.x', a_xregs[1]), ('hr1.x', a_xregs[2]), ('hr0.x', a_xregs[3])]) if stable_ay else \
                     [('hr3.x', 'r2.y'), ('hr1.x', 'r2.w'), ('hr2.x', 'r3.y'), ('hr0.x', 'r3.w')]
          a_loaded = False
          def emit_a_loads_now():
            nonlocal a_loaded
            if skip_a_loads or a_loaded: return
            a_loaded = True
            load_coords = (list(reversed(a_coords)) if high_inputs else a_coords)
            for dst, coord in load_coords:
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
          a_regs = (list(map(_hreg, high_a)) if high_inputs else list(map(_hreg, high_a_only_regs)) if high_a_only else
                    [_hreg('hr3.x'), _hreg('hr2.x'), _hreg('hr1.x'), _hreg('hr0.x')] if stable_ay else
                    [_hreg('hr3.x'), _hreg('hr1.x'), _hreg('hr2.x'), _hreg('hr0.x')])
          high_b0 = [_hreg(f'hr{i}.x') for i in range(36 if not low_a_coords else 32, 40 if not low_a_coords else 36)]
          high_b1 = [_hreg(f'hr{i}.x') for i in range(40 if not low_a_coords else 36, 44 if not low_a_coords else 40)]
          b_regs = ([high_b1, high_b0] if high_inputs else
                    [[_hreg(f'hr{i}.x') for i in range(8, 12)], [_hreg(f'hr{i}.x') for i in range(0, 4)]] if high_a_only else
                    [[_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg(f'hr{12+4*ncols}.x') if safe_store else _hreg('hr7.x')],
                    ([_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr28.x')] if linear_store else
                     [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')])])
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
              b_all = [[_hreg(f'hr{4+col*4+kk}.x') for kk in range(4)] for col in range(ncols)]
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
              if preload_b_safe_coords:
                  instrs.append(MOV_F32('r20.x', 'r20.x', sy=True))
                  coord_pairs = tuple((f'r{20+i//2}.{"xz"[i&1]}', f'r{20+i//2}.{"yw"[i&1]}') for i in range(ncols*4))
                  ysrcs = ('r4.z', 'r5.x', 'r5.z', 'r6.x')
                  for col in range(ncols):
                      for kk in range(4):
                          xreg, yreg = coord_pairs[col*4+kk]
                          instrs += [MOV_F32(xreg, 'r7.y') if col == 0 else ADD_S(xreg, 'r7.y', col*32),
                                     MOV_F32(yreg, ysrcs[kk])]
                  if stable_settle_delay >= 0: instrs.append(NOP(rpt=stable_settle_delay))
                  for col in range(ncols):
                      for kk in range(4): instrs.append(ISAM_F16(b_all[col][kk], coord_pairs[col*4+kk][0], 1))
              else:
                  emit_b_pair(0, _hreg('hr4.x'))
                  emit_b_pair(2, _hreg('hr12.x'))
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
            elif serial_b_cols:
              # Load and consume one 128-column block at a time, reusing the
              # same four B vectors while retaining the already-loaded A tile.
              if b_first: emit_a_loads_now()
              # The first instruction after the final A sample carries SY;
              # putting SY on the sample itself only waits for older samples.
              instrs.append(MOV_F32('r0.w', 'r0.w', sy=True))
              for col in range(ncols):
                for xreg in ('r4.y', 'r4.w', 'r5.y', 'r5.w'):
                  instrs.append(MOV_F32(xreg, 'r7.y') if col == 0 else ADD_S(xreg, 'r7.y', col * 32))
                if not reuse_coords:
                  instrs += [ADD_S('r4.z', ky_reg, -3), ADD_S('r5.x', ky_reg, -2),
                             ADD_S('r5.z', ky_reg, -1), MOV_F32('r6.x', ky_reg)]
                serial_coords = ('r4.y', 'r4.w', 'r5.y', 'r5.w')
                for bi, (dst, coord) in enumerate(zip(b_regs[0], serial_coords)):
                  emit_coord_wait(instrs, coord_delay)
                  instrs.append(ISAM_F16(dst, coord, 1, sy=(bi == len(serial_coords)-1)))
                if high_inputs:
                  instrs += [MOV_F32('r7.w', 'r7.w', sy=True), NOP(rpt=5)]
                  first = False
                else: first = True
                for row in range(4):
                  group = (row * ncols + col) * 4
                  for kk in range(4):
                    instrs.append(MAD_F16(acc0 + group, b_regs[0][kk], a_regs[row] + kk, acc0 + group,
                                          rpt=3, sy=first, r1=True, r3=True))
                    first = False
            else:
             col_bases = (0, 2, 1, 3) if single_cols_all and ncols == 4 else range(0, ncols, 1 if single_cols_all else 2)
             for col_base in col_bases:
              if preserve_coords: instrs += [MOV_F32('r7.y', 'r2.y'), NOP(rpt=2)]
              pair_cols = 1 if first_cols_only or single_cols_all else min(2, ncols - col_base)
              if single_cols_all and col_base != 0: instrs.append(MOV_F32('r0.w', 'r0.w', sy=True))
              if (first_cols_only or single_cols_all) and col_base != 0 and ky_reg is not None:
                  instrs += [ADD_S('r4.z', ky_reg, -3), ADD_S('r5.x', ky_reg, -2),
                             ADD_S('r5.z', ky_reg, -1), MOV_F32('r6.x', ky_reg)]
              if stable_bx and col_base == 2 and not (high_inputs or high_a_only):
                  if separate_b_coords:
                      pass
                  else:
                      instrs += [MOV_F32('r0.y', 'r4.z'), MOV_F32('r0.w', 'r5.x'), MOV_F32('r1.y', 'r5.z'), MOV_F32('r1.w', 'r6.x')]
                      if not (stable_ay and inc_coords): instrs.append(ADD_S('r3.w' if stable_ay else 'r7.z', 'r7.y', 96))
              elif col_base != 0:
                  if high_b_coords: instrs.append(MOV_F32('r0.w', 'r0.w', ss=True))
                  instrs += [
                      ADD_S('r0.y', 'r7.y', col_base * 32),
                      MOV_F32('r4.y', 'r0.y'), MOV_F32('r4.w', 'r0.y'), MOV_F32('r5.y', 'r0.y'), MOV_F32('r5.w', 'r0.y'),
                  ]
              if pair_cols == 2 and col_base != 0 and not stable_bx: instrs.append(ADD_S('r7.z', 'r7.y', (col_base + 1) * 32))
              first_coords = list(zip(b_regs[0], ('r4.y', 'r4.w', 'r5.y', 'r5.w')))
              second_coords = list(zip(b_regs[1], ('r0.x', 'r0.z', 'r1.x', 'r1.z')))
              if high_b_coords and pair_cols == 2 and not skip_b_loads:
                  high_first = (('r14.x', 'r14.y'), ('r14.z', 'r14.w'), ('r15.x', 'r15.y'), ('r15.z', 'r15.w'))
                  high_second = (('r16.x', 'r16.y'), ('r16.z', 'r16.w'), ('r17.x', 'r17.y'), ('r17.z', 'r17.w'))
                  for col, pairs in ((col_base, high_first), (col_base + 1, high_second)):
                    for kk, (xreg, yreg) in enumerate(pairs):
                      instrs += [ADD_S(xreg, 'r7.y', col * 32),
                                 MOV_F32(yreg, ('r4.z', 'r5.x', 'r5.z', 'r6.x')[kk])]
                  if stable_settle_delay >= 0: instrs.append(NOP(rpt=stable_settle_delay))
                  first_coords = list(zip(b_regs[0], (x for x, _ in high_first)))
                  second_coords = list(zip(b_regs[1], (x for x, _ in high_second)))
              if high_a_only:
                  for _, coord in first_coords:
                      instrs.append(MOV_F32(coord, 'r7.y') if col_base == 0 else ADD_S(coord, 'r7.y', col_base * 32))
                  for _, coord in second_coords: instrs.append(ADD_S(coord, 'r7.y', (col_base + 1) * 32))
                  instrs += [ADD_S('r4.z', ky_reg, -3), ADD_S('r5.x', ky_reg, -2),
                             ADD_S('r5.z', ky_reg, -1), MOV_F32('r6.x', ky_reg),
                             MOV_F32('r0.y', 'r4.z'), MOV_F32('r0.w', 'r5.x'),
                             MOV_F32('r1.y', 'r5.z'), MOV_F32('r1.w', 'r6.x')]
              elif high_inputs:
                  for _, coord in first_coords:
                      instrs.append(MOV_F32(coord, 'r7.y') if col_base == 0 else ADD_S(coord, 'r7.y', col_base * 32))
                  for _, coord in second_coords:
                      instrs.append(ADD_S(coord, 'r7.y', (col_base + 1) * 32))
                  instrs += [MOV_F32('r0.y', 'r4.z'), MOV_F32('r0.w', 'r5.x'),
                             MOV_F32('r1.y', 'r5.z'), MOV_F32('r1.w', 'r6.x')]
                  if stable_settle_delay >= 0: instrs.append(NOP(rpt=stable_settle_delay))
              if separate_b_coords and pair_cols == 2 and not skip_b_loads:
                  if persistent_b_coords or persistent_b_x:
                      first_coords = list(zip(b_regs[0], (x for x, _ in persistent_b_coord_regs[col_base*4:(col_base+1)*4])))
                      second_coords = list(zip(b_regs[1], (x for x, _ in persistent_b_coord_regs[(col_base+1)*4:(col_base+2)*4])))
                      if persistent_b_x:
                          for row, (_, yreg) in enumerate(persistent_b_coord_regs[:8]):
                              instrs.append(MOV_F32(yreg, ('r4.z', 'r5.x', 'r5.z', 'r6.x')[row & 3]))
                  else:
                      # r0-r3 were sampler sources for the preceding A/B issue
                      # group.  Drain source WAR dependencies before rebuilding
                      # all eight immutable B coordinate pairs in place.
                      instrs.append(MOV_F32('r0.x', 'r0.x', ss=True))
                      first_coords = list(zip(b_regs[0], ('r0.x', 'r0.z', 'r1.x', 'r1.z')))
                      second_coords = list(zip(b_regs[1], ('r2.x', 'r2.z', 'r3.x', 'r3.z')))
                      xregs = ('r0.x', 'r0.z', 'r1.x', 'r1.z', 'r2.x', 'r2.z', 'r3.x', 'r3.z')
                      if reuse_separate_b_y and col_base == 2:
                          for xreg in xregs: instrs.append(ADD_S(xreg, xreg, 64))
                      else:
                          for xreg in xregs[:4]: instrs.append(ADD_S(xreg, 'r7.y', col_base * 32))
                          for xreg in xregs[4:]: instrs.append(ADD_S(xreg, 'r7.y', (col_base + 1) * 32))
                          for i, yreg in enumerate(('r0.y', 'r0.w', 'r1.y', 'r1.w', 'r2.y', 'r2.w', 'r3.y', 'r3.w')):
                              instrs.append(MOV_F32(yreg, ('r4.z', 'r5.x', 'r5.z', 'r6.x')[i & 3]))
                      if stable_settle_delay >= 0: instrs.append(NOP(rpt=stable_settle_delay))
                  for dst, coord in first_coords + second_coords: instrs.append(ISAM_F16(dst, coord, 1))
              elif not skip_b_loads:
                for dst, coord in first_coords:
                    emit_coord_wait(instrs, coord_delay)
                    instrs.append(ISAM_F16(dst, coord, 1))
              if pair_cols == 2 and not skip_b_loads and high_b_coords:
                  for dst, coord in second_coords: instrs.append(ISAM_F16(dst, coord, 1))
              elif pair_cols == 2 and not skip_b_loads and not separate_b_coords:
                  original_second_coord = col3_x if stable_bx and stable_ay and col_base == 2 else 'r7.z'
                  if high_inputs or high_a_only:
                      # Keep a distinct full-register coordinate pair live for
                      # every asynchronous sample. Mutating the first block's
                      # coordinates here races the sampler on Adreno 630.
                      for dst, coord in second_coords: instrs.append(ISAM_F16(dst, coord, 1))
                  elif safe_b_y:
                      instrs += [ADD_S('r14.x', 'r7.y', (col_base + 1) * 32),
                                 MOV_F32('r14.y', 'r4.z'), MOV_F32('r14.z', 'r5.x'),
                                 MOV_F32('r14.w', 'r5.z'), MOV_F32('r15.x', 'r6.x'), NOP(rpt=2)]
                  if not (high_inputs or high_a_only) and not separate_b_coords:
                      second_coord = 'r14.x' if safe_b_y else original_second_coord
                      second_y = 'r4.x' if stable_bx and stable_ay and col_base == 2 else 'r7.w'
                      for bi, (dst, yreg) in enumerate(zip(b_regs[1], ('r4.z', 'r5.x', 'r5.z', 'r6.x'))):
                          if safe_b_y: second_y = ('r14.y', 'r14.z', 'r14.w', 'r15.x')[bi]
                          else: instrs.append(MOV_F32(second_y, yreg, sy=sync_b_y))
                          emit_coord_wait(instrs, coord_delay)
                          instrs.append(ISAM_F16(dst, second_coord, 1))
                      if safe_b_y: instrs.append(MOV_F32('r14.z', 'r14.z', sy=True))
              if b_first and col_base == 0: emit_a_loads_now()
              interleaved_coords = None
              if interleave_second_pair and col_base == 0:
                  # Drain the first pair's A/B samples before reusing their full
                  # coordinate bank. The half-register values remain live.
                  instrs.append(MOV_F32('r0.x', 'r0.x', sy=True))
                  next0 = (('r0.x', 'r0.y'), ('r0.z', 'r0.w'), ('r1.x', 'r1.y'), ('r1.z', 'r1.w'))
                  next1 = (('r2.x', 'r2.y'), ('r2.z', 'r2.w'), ('r3.x', 'r3.y'), ('r3.z', 'r3.w'))
                  for col, coords in ((2, next0), (3, next1)):
                      for row, (xreg, yreg) in enumerate(coords):
                          instrs += [ADD_S(xreg, 'r7.y', col*32),
                                     MOV_F32(yreg, ('r4.z', 'r5.x', 'r5.z', 'r6.x')[row])]
                  if stable_settle_delay >= 0: instrs.append(NOP(rpt=stable_settle_delay))
                  interleaved_coords = (next0, next1)
                  first = False
              # Each column pair issues a fresh set of asynchronous texture
              # reads.  The first MAD consuming that pair must synchronize;
              # carrying `first=False` from the previous pair consumes stale B.
              if high_inputs or high_a_only or separate_b_coords:
                  first = True
              elif col_base != 0 and not first_sync_only: first = True
              if preserve_coords and col_base and not skip_a_loads:
                  # hr3 held A row 1 at the end of the previous pair. Reload
                  # row 0 without touching hr2/r2, which owns output coords.
                  instrs += [MOV_F32('r2.w', 'r2.x')]
                  emit_coord_wait(instrs, coord_delay)
                  instrs.append(ISAM_F16('hr3.x', 'r2.z', 0))
                  first = True
              for _ in range(alu_reps):
                def emit_pair(row, col, kk):
                    nonlocal first
                    out_col = col_base + col
                    group = (row * ncols + out_col) * 4
                    instrs.append(MAD_F16(acc0 + group, b_regs[col][kk], a_regs[row] + kk, acc0 + group,
                                          rpt=3, sy=(first or (row_sync and kk == 0)), r1=True, r3=True))
                    first = False
                if preserve_coords:
                  # Compute every resident A row, then replace the already-used
                  # row-0 vector with row 1 and compute that final row.
                  for row in (0, 2, 3):
                    for col in range(pair_cols):
                      for kk in range(4): emit_pair(row, col, kk)
                  if not skip_a_loads:
                    instrs += [ADD_S('r2.w', 'r2.x', 1)]
                    emit_coord_wait(instrs, coord_delay)
                    instrs.append(ISAM_F16('hr3.x', 'r2.z', 0))
                    first = True
                  for col in range(pair_cols):
                    for kk in range(4):
                      out_col = col_base + col
                      group = (1 * ncols + out_col) * 4
                      instrs.append(MAD_F16(acc0 + group, _hreg('hr3.x') + kk, b_regs[col][kk], acc0 + group,
                                            rpt=3, sy=first, r=True))
                      first = False
                elif alu_order == 'row_col_kk':
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
                    if interleaved_coords is not None:
                      instrs.append(ISAM_F16(b_regs[0][kk], interleaved_coords[0][kk][0], 1))
                      instrs.append(ISAM_F16(b_regs[1][kk], interleaved_coords[1][kk][0], 1))
                else:
                  for kk in range(4):
                    for row in range(4):
                      for col in range(pair_cols): emit_pair(row, col, kk)
              if interleaved_coords is not None:
                  first = True
                  for kk in range(4):
                      for col in range(2):
                          for row in range(4):
                              group = (row*ncols + 2+col)*4
                              instrs.append(MAD_F16(acc0+group, b_regs[col][kk], a_regs[row]+kk, acc0+group,
                                                    rpt=3, sy=first, r1=True, r3=True))
                              first = False
                  break
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
                    # Texture coordinates are consumed asynchronously.  Before unrolled
                    # iterations rewrite them, drain sampler source reads from the prior K.
                    if high_inputs or high_a_only: instrs.append(MOV_F32(a_xregs[0], a_xregs[0], ss=True))
                    instrs += [
                        ADD_S(a_xregs[0], a_xregs[0], 1), ADD_S(a_xregs[1], a_xregs[1], 1), ADD_S(a_xregs[2], a_xregs[2], 1), ADD_S(a_xregs[3], a_xregs[3], 1),
                        ADD_S('r4.z', 'r4.z', 4), ADD_S('r5.x', 'r5.x', 4), ADD_S('r5.z', 'r5.z', 4), ADD_S('r6.x', 'r6.x', 4),
                    ]
                    if persistent_b_coords:
                        instrs += [ADD_S(yreg, yreg, 4) for _, yreg in persistent_b_coord_regs]
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
                CMPS_S_EQ(a_xregs[0], k_end - 1, nop=1),
                ADD_S(a_xregs[0], a_xregs[0], 1), ADD_S(a_xregs[1], a_xregs[1], 1), ADD_S(a_xregs[2], a_xregs[2], 1), ADD_S(a_xregs[3], a_xregs[3], 1),
                ADD_S('r4.z', 'r4.z', 4), ADD_S('r5.x', 'r5.x', 4), ADD_S('r5.z', 'r5.z', 4), ADD_S('r6.x', 'r6.x', 4),
            ]
            if persistent_b_coords:
                instrs += [ADD_S(yreg, yreg, 4) for _, yreg in persistent_b_coord_regs]
        else:
            next_k_reg = 'r6.w' if stable_bx else 'r0.x'
            instrs += [ADD_S(next_k_reg, 'r6.z', k_unroll), ADD_S('r6.y', 'r6.y', 4 * k_unroll)]
            if dynamic_split_k:
                instrs += [ADD_S('r6.w', 'r6.w', k_unroll), NOP(rpt=2), CMPS_S_EQ('r6.w', k_count, nop=1)]
            else: instrs.append(CMPS_S_EQ('r6.z', k_end - k_unroll, nop=1))
            instrs += [MOV_F32('r6.z', next_k_reg), NOP(rpt=3)]
        loop_end = len(instrs)
        instrs.append(BR(loop_start - loop_end))

    if post_constant:
        if not direct: raise ValueError('--post-constant requires --direct')
        for lane in range(acc_groups * 4): instrs.append(MOV_H_IMM(acc0 + lane, 0x6400))

    if save_output_coords: instrs += [MOV_F32('r7.x', f'{output_save_reg}.x'), MOV_F32('r7.y', f'{output_save_reg}.y'), NOP(rpt=2)]

    if dynamic_split_k:
        instrs += [SHL_B('r19.z', 'r19.x', 7), NOP(rpt=2), ADD_S_REG('r7.x', 'r7.x', 'r19.z'), NOP(rpt=2)]

    # ADD_S immediates are only eight bits. Materialize larger split-K output row bases in a register.
    if store_row_base: instrs += [MOV_S32('r6.w', store_row_base), NOP(rpt=2), ADD_S_REG('r7.x', 'r7.x', 'r6.w'), NOP(rpt=2)]
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
    elif linear_store:
        if not direct: raise ValueError('--linear-store requires direct accumulators')
        emit_linear4_stores(instrs, acc0, ncols, store_row_shift)
    elif image_store:
        if not direct: raise ValueError('--image-store requires direct accumulators')
        emit_image4_stores(instrs, acc0, ncols, preserve_coords, high_store, False)
    elif safe_store:
        if not direct or store_row_shift != 10:
            raise ValueError('--safe-store requires direct accumulators and row shift 10')
        for row in range(4):
            row_reg = 'r7.x'
            if row:
                instrs.append(OR_B('r7.w', 'r7.x', row))
                row_reg = 'r7.w'
            for col in range(ncols):
                col_reg = 'r7.y'
                if col:
                    instrs.append(ADD_S('r7.z', 'r7.y', col * 32))
                    col_reg = 'r7.z'
                store_output(instrs, row_reg, col_reg, acc0 + (row * ncols + col) * 4)
    elif thread_store:
        if not direct: raise ValueError('--thread-store requires direct accumulators')
        emit_threadmajor_stores(instrs, acc0, ncols)
    else:
        if direct: emit_hand4_stores(instrs, acc0, ncols, store_row_shift,
                                     range(0, ncols, 2) if first_cols_only else None, repeat_first_store,
                                     repair_row1_store, repeat_each_store)
        else:
            # Reduce the four K-component accumulators for each output into a compact 4xN tile,
            # then use the compiler-derived vector store path. The old scalar reduce_store path
            # formed invalid addresses for most lanes/workgroups.
            instrs.append(NOP(rpt=16))
            for out in range(4*ncols):
                src, dst = acc0+out*16, acc0+out*4
                for lane in range(4): instrs.append(ADD_F(dst+lane, src+lane, src+4+lane))
                instrs.append(NOP(rpt=3))
                for lane in range(4): instrs.append(ADD_F(dst+lane, dst+lane, src+8+lane))
                instrs.append(NOP(rpt=3))
                for lane in range(4): instrs.append(ADD_F(dst+lane, dst+lane, src+12+lane))
                instrs.append(NOP(rpt=3))
            emit_donor4_stores(instrs, dev, threads, acc0, ncols)
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
    else: envelope_src = make_direct_donor_src(args.ncols, args.threads) if args.native_store else \
      make_donor_src(8 if args.large_envelope else args.ncols, args.threads)
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

    c_dtype = dtypes.float if args.fp32_accum else dtypes.half
    a, b, c = make_bufs(dev, c_dtype)
    if args.check:
        fill_half(a, 0x3c00)
        fill_half(b, 0x3c00)
    buf_dtypes = [((0, dtypes.half, (M, K//4, 4)),), ((0, dtypes.half, (K, N//4, 4)),), ((0, c_dtype, None),)]
    prg = dev.runtime('gemm_h', lib, buf_dtypes=buf_dtypes)
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
    parser.add_argument('--large-envelope', action='store_true', help='reserve a larger donor executable for hand shaders')
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
