#!/usr/bin/env python3
"""Hand-assembled 8xN FP16 GEMM probes for Adreno 630.

Each thread computes 8 rows by N col4 output vectors. In scalar GEMM terms,
--ncols 2 is an 8x8 tile, --ncols 3 is 8x12, and --ncols 4 is 8x16. Two schedules are
available:

* serial: keep only one 4-texel B column group live at a time. This uses 48
  hregs and gives 4 waves at THREAD64, but needs a sync for each B group.
* preload: load all B texels up front. This fits for --ncols 2, but --ncols 4
  exceeds the usable A630 half-register namespace.
"""
import argparse, array, ctypes, struct

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import *
from extra.gemm.ir3asm import _hreg
from extra.gemm.qcom_intensity_gemm import M, N, K, K4, make_donor_src, store_output, emit_donor4_stores, emit_addr4_rows


def make_donor_src8(ncols=4, threads=64):
    tn = 32 * ncols
    tm = (threads // 32) * 8
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global half *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    src += 'int row=get_group_id(1)*%d+tm*8;int col4=get_group_id(0)*%d+tid;\n' % (tm, tn)
    for r in range(8):
        for c in range(ncols): src += 'half4 r%dd%d=(half4)(0);\n' % (r, c)
    src += 'for(int k4=0;k4<%d;k4++){\n' % K4
    for r in range(8): src += 'half4 a%d=read_imageh(A,smp,(int2)(k4,row+%d));\n' % (r, r)
    for c in range(ncols):
        for b in range(4): src += 'half4 b%d_%d=read_imageh(B,smp,(int2)(col4+%d,k4*4+%d));\n' % (c, b, c*32, b)
    for r in range(8):
        for c in range(ncols):
            src += 'r%dd%d+=a%d.xxxx*b%d_0+a%d.yyyy*b%d_1+a%d.zzzz*b%d_2+a%d.wwww*b%d_3;\n' % (r, c, r, c, r, c, r, c, r, c)
    src += '}\n'
    for r in range(8):
        for c in range(ncols): src += 'vstore4(r%dd%d,0,C+(row+%d)*%d+(col4+%d)*4);\n' % (r, c, r, N, c*32)
    src += '}\n'
    return src


def make_donor_src8_fp32(ncols=2, threads=128):
    tn = 32 * ncols
    tm = (threads // 32) * 8
    src = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    src += 'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;\n'
    src += '__attribute__((reqd_work_group_size(%d,1,1)))\n' % threads
    src += '__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global float *C){\n'
    src += 'int lid=get_local_id(0);int tm=lid>>5;int tid=lid&31;\n'
    src += 'int row=get_group_id(1)*%d+tm*8;int col4=get_group_id(0)*%d+tid;\n' % (tm, tn)
    for r in range(8):
        for c in range(ncols): src += 'float4 r%dd%d=(float4)(0);\n' % (r, c)
    src += 'for(int k4=0;k4<%d;k4++){\n' % K4
    for r in range(8): src += 'float4 a%d=convert_float4(read_imageh(A,smp,(int2)(k4,row+%d)));\n' % (r, r)
    for c in range(ncols):
        for b in range(4): src += 'float4 b%d_%d=convert_float4(read_imageh(B,smp,(int2)(col4+%d,k4*4+%d)));\n' % (c, b, c*32, b)
    for r in range(8):
        for c in range(ncols):
            src += 'r%dd%d+=a%d.xxxx*b%d_0+a%d.yyyy*b%d_1+a%d.zzzz*b%d_2+a%d.wwww*b%d_3;\n' % (r, c, r, c, r, c, r, c, r, c)
    src += '}\n'
    for r in range(8):
        for c in range(ncols): src += 'vstore4(r%dd%d,0,C+(row+%d)*%d+(col4+%d)*4);\n' % (r, c, r, N, c*32)
    src += '}\n'
    return src


def prologue_8x4(dev, threads, swap_grid=False):
    lib, img_off, _, _ = get_envelope(dev, make_donor_src(1, threads))
    pro = bytearray(lib[img_off:img_off + 13 * 8])
    row_log = {64: 4, 128: 5, 256: 6}[threads]
    row_merge = ((threads // 32) - 1) * 8
    pro[6*8:7*8] = SHRM('r0.z', 2, 'r0.x', row_merge)
    pro[11*8:12*8] = SHLG('r7.x', row_log, 'r0.y', 'r0.z', nop=2)
    instrs = [bytes(pro[i:i+8]) for i in range(0, len(pro), 8)]
    if swap_grid:
        instrs += [MOV_F32('r4.x', 'r0.y'), MOV_F32('r0.y', 'r0.w'), NOP(rpt=2), SHLG('r7.x', row_log, 'r0.y', 'r0.z', nop=2), MOV_F32('r0.w', 'r4.x'), NOP(rpt=2), SHLG('r7.y', 5, 'r0.w', 'r0.x')]
    return instrs


def prologue_8x4_fp32(dev, threads, ncols=1):
    lib, img_off, _, _ = get_envelope(dev, make_donor_src8_fp32(ncols, threads))
    pro = bytearray(lib[img_off:img_off + 13 * 8])
    return [bytes(pro[i:i+8]) for i in range(0, len(pro), 8)]


def emit_col_stride(instrs, ncols, swap_grid=False):
    if ncols == 1: return
    # Donor prologue computes col4 = gid.x*32 + tid. Widened-column kernels
    # need col4 = gid.x*(32*ncols) + tid to avoid overlapping output tiles.
    instrs += [MOV_F32('r6.w', 'r52.x' if swap_grid else 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2)]
    for _ in range(ncols - 1):
        instrs.append(ADD_S_REG('r7.y', 'r7.y', 'r6.w'))
        instrs.append(NOP(rpt=2))


def emit_a_loads(instrs, coord_delay=-1, first_wait_only=False):
    a_regs = ['hr0.x', 'hr1.x', 'hr2.x', 'hr3.x', 'hr4.x', 'hr5.x', 'hr6.x', 'hr7.x']
    instrs.append(MOV_F32('r4.y', 'r6.z'))
    for row, reg in enumerate(a_regs):
        if row == 0: instrs.append(MOV_F32('r4.z', 'r7.x'))
        else: instrs.append(ADD_S('r4.z', 'r7.x', row))
        if not first_wait_only or row == 0: emit_coord_wait(instrs, coord_delay)
        instrs.append(ISAM_F16(reg, 'r4.y', 0))
    return [_hreg(reg) for reg in a_regs]


def emit_b_group(instrs, col, dest_regs, coord_delay=-1):
    if col: instrs.append(ADD_S('r4.y', 'r7.y', col * 32))
    else: instrs.append(MOV_F32('r4.y', 'r7.y'))
    k_ops = [lambda: ADD_S('r4.z', 'r6.y', -3), lambda: ADD_S('r4.z', 'r6.y', -2),
             lambda: ADD_S('r4.z', 'r6.y', -1), lambda: MOV_F32('r4.z', 'r6.y')]
    for kk in range(4):
        instrs.append(k_ops[kk]())
        emit_coord_wait(instrs, coord_delay)
        instrs.append(ISAM_F16(dest_regs[kk], 'r4.y', 1))


def emit_preload_state_copy(instrs):
    # Keep loop/base state above the low B registers; preloaded B occupies hr0-hr15.
    instrs += [
        MOV_F32('r8.w', 'r7.x'), MOV_F32('r9.x', 'r7.y'),
        MOV_F32('r9.y', 'r6.y'), MOV_F32('r9.z', 'r6.z'),
    ]


def preload_state(ncols):
    if ncols == 2:
        return {'row':'r8.w', 'col':'r9.x', 'ky':'r9.y', 'kz':'r9.z', 'coord':'r8.y', 'coord_y':'r8.z', 'tmp_col':'r9.w', 'tmp_row':'r8.x'}
    if ncols == 3:
        return {'row':'r10.w', 'col':'r11.x', 'ky':'r11.y', 'kz':'r11.z', 'coord':'r10.y', 'coord_y':'r10.z', 'tmp_col':'r11.w', 'tmp_row':'r10.x'}
    return {'row':'r8.w', 'col':'r9.x', 'ky':'r9.y', 'kz':'r9.z', 'coord':'r8.y', 'coord_y':'r8.z', 'tmp_col':'r9.w', 'tmp_row':'r8.x'}


def emit_preload_state_copy_n(instrs, st):
    instrs += [MOV_F32(st['row'], 'r7.x'), MOV_F32(st['col'], 'r7.y'), MOV_F32(st['ky'], 'r6.y'), MOV_F32(st['kz'], 'r6.z')]


def emit_a_loads_preload(instrs, a_regs, st):
    instrs.append(MOV_F32(st['coord'], st['kz']))
    for row, reg in enumerate(a_regs):
        if row == 0: instrs.append(MOV_F32(st['coord_y'], st['row']))
        else: instrs.append(OR_B(st['coord_y'], st['row'], row))
        instrs.append(ISAM_F16(reg, st['coord'], 0))
    return [_hreg(reg) for reg in a_regs]


def emit_b_group_preload(instrs, col, dest_regs, st):
    col_reg = st['col']
    if col:
        instrs.append(ADD_S(st['tmp_col'], st['col'], col * 32))
        col_reg = st['tmp_col']
    k_ops = [lambda: ADD_S(st['coord_y'], st['ky'], -3), lambda: ADD_S(st['coord_y'], st['ky'], -2),
             lambda: ADD_S(st['coord_y'], st['ky'], -1), lambda: MOV_F32(st['coord_y'], st['ky'])]
    instrs.append(MOV_F32(st['coord'], col_reg))
    for kk in range(4):
        instrs.append(k_ops[kk]())
        instrs.append(ISAM_F16(dest_regs[kk], st['coord'], 1))


def emit_mads(instrs, acc0, a_regs, b_regs, col, ncols, first, force_sy=False):
    for kk in range(4):
        for row in range(8):
            group = (row * ncols + col) * 4
            instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, b_regs[kk], acc0 + group, rpt=3, sy=(first or force_sy), r=True))
            first = False
            force_sy = False
    return first


def emit_loop_control(instrs):
    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        ADD_S('r6.y', 'r6.y', 4),
        CMPS_S_EQ('r6.z', K4 - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]


def emit_loop_control_preload(instrs, st):
    instrs += [
        ADD_S('r0.x', st['kz'], 1),
        ADD_S(st['ky'], st['ky'], 4),
        CMPS_S_EQ(st['kz'], K4 - 1, nop=1),
        MOV_F32(st['kz'], 'r0.x'),
        NOP(rpt=3),
    ]


def emit_all_stores(instrs, acc0, ncols):
    for row in range(8):
        if row == 0: row_reg = 'r7.x'
        else:
            instrs.append(OR_B('r7.w', 'r7.x', row))
            row_reg = 'r7.w'
        for col in range(ncols):
            if col == 0: col_reg = 'r7.y'
            else:
                instrs.append(ADD_S('r7.z', 'r7.y', col * 32))
                col_reg = 'r7.z'
            store_output(instrs, row_reg, col_reg, acc0 + (row * ncols + col) * 4)


def emit_all_stores_preload(instrs, acc0, ncols, st):
    for row in range(8):
        if row == 0: row_reg = st['row']
        else:
            instrs.append(OR_B(st['tmp_row'], st['row'], row))
            row_reg = st['tmp_row']
        for col in range(ncols):
            if col == 0: col_reg = st['col']
            else:
                instrs.append(ADD_S(st['tmp_col'], st['col'], col * 32))
                col_reg = st['tmp_col']
            store_output(instrs, row_reg, col_reg, acc0 + (row * ncols + col) * 4)


def emit_dynamic4_stores(instrs, acc0, ncols):
    for col in range(ncols):
        if col != 0: instrs.append(ADD_S('r7.y', 'r7.y', 32))
        emit_addr4_rows(instrs, 'r7.y')
        for row in range(4): instrs.append(MOV_H(row * 4, acc0 + (row * ncols + col) * 4, rpt=3))
        instrs += [STG_F16('r2.x', 'hr0.x'), NOP(), STG_F16('r2.z', 'hr1.x'), NOP(), STG_F16('r3.x', 'hr2.x'), NOP(), STG_F16('r3.z', 'hr3.x')]


def emit_donor8_stores(instrs, dev, threads, acc0, ncols, row_reg='r7.x', col_reg='r7.y', coord_delay=4):
    if ncols != 2: raise ValueError('8-row donor store currently requires ncols=2')
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8(1, threads))
    donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]
    instrs += [MOV_F32('r11.z', row_reg), MOV_F32('r11.w', col_reg)]
    for col in range(2):
        if col:
            instrs.append(ADD_S('r11.w', 'r11.w', 32))
            emit_coord_wait(instrs, coord_delay)
        instrs += donor[212:262]
        for row in range(8): instrs.append(MOV_H(row * 4, acc0 + (row * ncols + col) * 4, rpt=3))
        instrs += donor[292:315] if col == 1 else donor[292:314]


def emit_explicit8_stores(instrs, donor, data_base, mode='tight', data_stride=4):
    instrs += donor[292:300]
    for row, addr in enumerate(['r4.x', 'r4.z', 'r5.x', 'r5.z', 'r6.x', 'r6.z', 'r7.x', 'r7.z']):
        instrs.append(STG_F16(addr, data_base + row * data_stride))
        if (mode == 'donor' and 1 <= row <= 6) or (mode == 'pairs' and row in (1, 3, 5)): instrs.append(NOP())


def emit_donor8_add256_stores(instrs, dev, threads, acc0, ncols, row_reg='r7.x', col_reg='r7.y', gap=16, offset_before_gap=False, explicit_stores=False, store_mode='donor', direct_sources=False):
    if ncols not in (2, 4): raise ValueError('8-row add256 donor store currently requires ncols=2 or ncols=4')
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8(1, threads))
    donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]
    instrs += [SHL_B('r0.y', row_reg, 10, jp=True), SHL_B('r0.z', col_reg, 2)]
    instrs += donor[214:262]
    for col in range(ncols):
        if not direct_sources:
            for row in range(8): instrs.append(MOV_H(row * 4, acc0 + (row * ncols + col) * 4, rpt=3))
        if direct_sources: emit_explicit8_stores(instrs, donor, acc0 + col * 4, 'tight' if explicit_stores else store_mode, ncols * 4)
        elif explicit_stores or store_mode != 'donor': emit_explicit8_stores(instrs, donor, 0, 'tight' if explicit_stores else store_mode)
        else: instrs += donor[292:315] if col == ncols - 1 else donor[292:314]
        if col == ncols - 1: continue
        if offset_before_gap: instrs.append(MOV_S32('r0.x', 256))
        if gap >= 0: instrs.append(NOP(rpt=gap))
        if not offset_before_gap: instrs.append(MOV_S32('r0.x', 256))
        for reg in ['r4.x', 'r4.z', 'r5.x', 'r5.z', 'r6.x', 'r6.z', 'r7.x', 'r7.z']:
            instrs.append(ADD_S_REG(reg, reg, 'r0.x'))


def full_reg_name(idx):
    return 'r%d.%s' % (idx // 4, 'xyzw'[idx & 3])


def fvec(vec, comp=0):
    return 'r%d.%s' % (vec, 'xyzw'[comp])


def build_8x4_shader(dev, threads, variant, ncols, serial_syncs='all', a_coord_delay=-1, b_coord_delay=-1, post_constant=False, donor8_store=False, first_a_wait_only=False, donor8_add256_store=False, add256_gap=16, add256_offset_before_gap=False, add256_explicit_stores=False, add256_store_mode='donor', add256_direct_sources=False):
    instrs = prologue_8x4(dev, threads)
    emit_col_stride(instrs, ncols)
    if variant == 'serial':
        acc0 = _hreg('hr16.x')
        hregs, fregs = (acc0 + 8 * ncols * 4 + 3) // 4, 8
    elif variant == 'preload':
        acc0 = _hreg('hr20.x') if ncols == 2 else _hreg('hr24.x') if ncols == 3 else _hreg('hr28.x')
        hregs, fregs = (acc0 + 8 * ncols * 4 + 3) // 4, 10 if ncols == 2 else 12 if ncols == 3 else 10
    else:
        raise ValueError('unknown variant %s' % variant)

    st = preload_state(ncols)
    if variant == 'preload': emit_preload_state_copy_n(instrs, st)

    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 8 * ncols * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    if variant == 'preload':
        a_preload_regs = ['hr8.x', 'hr9.x', 'hr10.x', 'hr11.x', 'hr12.x', 'hr13.x', 'hr14.x', 'hr15.x'] if ncols == 2 else \
                         ['hr12.x', 'hr13.x', 'hr14.x', 'hr15.x', 'hr16.x', 'hr17.x', 'hr18.x', 'hr19.x'] if ncols == 3 else \
                         ['hr20.x', 'hr21.x', 'hr22.x', 'hr23.x', 'hr24.x', 'hr25.x', 'hr26.x', 'hr27.x']
        a_regs = emit_a_loads_preload(instrs, a_preload_regs, st)
    else: a_regs = emit_a_loads(instrs, a_coord_delay, first_a_wait_only)
    first = True

    if variant == 'serial':
        b_regs = [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')]
        for col in range(ncols):
            emit_b_group(instrs, col, ['hr8.x', 'hr9.x', 'hr10.x', 'hr11.x'], b_coord_delay)
            first = emit_mads(instrs, acc0, a_regs, b_regs, col, ncols, first, force_sy=(serial_syncs == 'all' and col != 0))
    else:
        b_groups = [
            ['hr0.x', 'hr1.x', 'hr2.x', 'hr3.x'],
            ['hr4.x', 'hr5.x', 'hr6.x', 'hr7.x'],
            ['hr8.x', 'hr9.x', 'hr10.x', 'hr11.x'],
            ['hr12.x', 'hr13.x', 'hr14.x', 'hr15.x'],
        ][:ncols]
        for col in range(ncols): emit_b_group_preload(instrs, col, b_groups[col], st)
        for col in range(ncols):
            first = emit_mads(instrs, acc0, a_regs, [_hreg(reg) for reg in b_groups[col]], col, ncols, first)

    if variant == 'preload': emit_loop_control_preload(instrs, st)
    else: emit_loop_control(instrs)
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        instrs += [MOV_H_IMM(acc0, 0x6400), MOV_H(acc0 + 1, acc0, rpt=2)]
        for base in range(acc0 + 4, acc0 + 8 * ncols * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    if variant == 'preload': emit_all_stores_preload(instrs, acc0, ncols, st)
    elif donor8_add256_store: emit_donor8_add256_stores(instrs, dev, threads, acc0, ncols, gap=add256_gap, offset_before_gap=add256_offset_before_gap, explicit_stores=add256_explicit_stores, store_mode=add256_store_mode, direct_sources=add256_direct_sources)
    elif donor8_store: emit_donor8_stores(instrs, dev, threads, acc0, ncols, coord_delay=b_coord_delay)
    else: emit_all_stores(instrs, acc0, ncols)
    instrs.append(END())
    if donor8_store: fregs = max(fregs, 12)
    if donor8_add256_store: fregs = max(fregs, 10)
    return assemble(instrs), hregs, fregs, loop_end - loop_start


def build_8x8_split_a_shader(dev, threads, a_coord_delay=3, b_coord_delay=-1, post_constant=False, grouped_b=False, grouped_b_cols=False, donor8_add256_store=False, pre_mad_nops=-1, second_a_sync=True, no_store=False, skip_a_loads=False, skip_b_loads=False, add256_gap=16, add256_offset_before_gap=False, add256_explicit_stores=False, add256_store_mode='donor', add256_direct_sources=False):
    instrs = prologue_8x4(dev, threads)
    emit_col_stride(instrs, 2)
    acc0 = _hreg('hr12.x')
    hregs, fregs = 28, 12
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)

    # Preload both B column groups into hr0..hr7, then reuse them for two 4-row A groups.
    if skip_b_loads:
        for base in range(_hreg('hr0.x'), _hreg('hr8.x'), 4):
            instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
    elif grouped_b_cols:
        pairs = [('r4.y', 'r4.z'), ('r4.w', 'r5.x'), ('r5.y', 'r5.z'), ('r5.w', 'r6.x')]
        for col, base in [(0, _hreg('hr0.x')), (1, _hreg('hr4.x'))]:
            if col: instrs.append(ADD_S('r6.w', 'r7.y', 32))
            xsrc = 'r7.y' if col == 0 else 'r6.w'
            for kk, (xreg, yreg) in enumerate(pairs):
                instrs.append(MOV_F32(xreg, xsrc))
                instrs.append([lambda: ADD_S(yreg, 'r6.y', -3), lambda: ADD_S(yreg, 'r6.y', -2),
                               lambda: ADD_S(yreg, 'r6.y', -1), lambda: MOV_F32(yreg, 'r6.y')][kk]())
            emit_coord_wait(instrs, b_coord_delay)
            for kk, (xreg, _) in enumerate(pairs): instrs.append(ISAM_F16(base + kk * 4, xreg, 1))
    elif grouped_b:
        pairs = [('r4.y', 'r4.z'), ('r4.w', 'r5.x'), ('r5.y', 'r5.z'), ('r5.w', 'r6.x'),
                 ('r6.w', 'r7.w'), ('r8.x', 'r8.y'), ('r8.z', 'r8.w'), ('r9.x', 'r9.y')]
        instrs.append(ADD_S('r9.z', 'r7.y', 32))
        for i, (xreg, yreg) in enumerate(pairs):
            instrs.append(MOV_F32(xreg, 'r7.y' if i < 4 else 'r9.z'))
            koff = i & 3
            instrs.append([lambda: ADD_S(yreg, 'r6.y', -3), lambda: ADD_S(yreg, 'r6.y', -2),
                           lambda: ADD_S(yreg, 'r6.y', -1), lambda: MOV_F32(yreg, 'r6.y')][koff]())
        emit_coord_wait(instrs, b_coord_delay)
        for i, (xreg, _) in enumerate(pairs): instrs.append(ISAM_F16(_hreg('hr0.x') + i * 4, xreg, 1))
    else:
        for col, base in [(0, _hreg('hr0.x')), (1, _hreg('hr4.x'))]:
            instrs.append(MOV_F32('r4.y', 'r7.y') if col == 0 else ADD_S('r4.y', 'r7.y', 32))
            for kk, yop in enumerate([lambda: ADD_S('r4.z', 'r6.y', -3), lambda: ADD_S('r4.z', 'r6.y', -2),
                                      lambda: ADD_S('r4.z', 'r6.y', -1), lambda: MOV_F32('r4.z', 'r6.y')]):
                instrs.append(yop())
                emit_coord_wait(instrs, b_coord_delay)
                instrs.append(ISAM_F16(base + kk * 4, 'r4.y', 1))

    b_regs = [[_hreg('hr0.x'), _hreg('hr1.x'), _hreg('hr2.x'), _hreg('hr3.x')],
              [_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg('hr7.x')]]
    a_regs = [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')]
    if skip_a_loads:
        for base in range(_hreg('hr8.x'), _hreg('hr12.x'), 4):
            instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
    if pre_mad_nops >= 0: instrs.append(NOP(rpt=pre_mad_nops))
    first = True
    def emit_a_load_group(row_base):
        if skip_a_loads: return
        instrs.append(MOV_F32('r4.y', 'r6.z'))
        for local_row, areg in enumerate(a_regs):
            row = row_base + local_row
            instrs.append(MOV_F32('r4.z', 'r7.x') if row == 0 else OR_B('r4.z', 'r7.x', row))
            if local_row == 0: emit_coord_wait(instrs, a_coord_delay)
            instrs.append(ISAM_F16(areg, 'r4.y', 0))

    def emit_mad(row, col, kk, force_sync=False):
        nonlocal first
        areg = a_regs[row & 3]
        group = (row * 2 + col) * 4
        instrs.append(MAD_F16(acc0 + group, areg + kk, b_regs[col][kk], acc0 + group, rpt=3, sy=(first or force_sync), r=True))
        first = False

    def emit_a_group(row_base, force_sync=False):
        emit_a_load_group(row_base)
        for local_row, areg in enumerate(a_regs):
            row = row_base + local_row
            for col in range(2):
                for kk in range(4):
                    emit_mad(row, col, kk, force_sync=force_sync)
                    force_sync = False

    emit_a_group(0)
    emit_a_group(4, force_sync=second_a_sync)

    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        ADD_S('r6.y', 'r6.y', 4),
        CMPS_S_EQ('r6.z', K4 - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        instrs += [MOV_H_IMM(acc0, 0x6400), MOV_H(acc0 + 1, acc0, rpt=2)]
        for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    if no_store:
        if donor8_add256_store: fregs = 10
        pass
    elif donor8_add256_store:
        emit_donor8_add256_stores(instrs, dev, threads, acc0, 2, gap=add256_gap, offset_before_gap=add256_offset_before_gap, explicit_stores=add256_explicit_stores, store_mode=add256_store_mode, direct_sources=add256_direct_sources)
        fregs = 10
    else: emit_donor8_stores(instrs, dev, threads, acc0, 2, coord_delay=b_coord_delay)
    instrs.append(END())
    return assemble(instrs), hregs, fregs, loop_end - loop_start


def build_8x8_split_a_unroll_shader(dev, threads, k_unroll=2, b_coord_delay=3, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, add256_gap=16, grouped_b=False, grouped_b_cols=False, fast_coords=False, alu_order='row_col_kk', stream_col1=False, stream_col1_sync=False, prefetch_next_b=False, add_a_rows=False, add256_offset_before_gap=False, add256_explicit_stores=False, add256_store_mode='donor', add256_direct_sources=False, buffer_a=False, prefetch_next_a=False, interleave_next_b=False, hoist_b0_coord=False, inline_b_wait=False, inline_b_nop=1, quad_a=False, prefetch_loop_b=False, high_a=False, pair_b_coords=False, base_b_y=False, low_a=False, stream_next_b1=False, stream_next_b0=False, swap_grid=False, pre_mad_nops=-1, quad_map='0123'):
    if K4 % k_unroll != 0: raise ValueError('--split-k-unroll must divide K/4')
    instrs = prologue_8x4(dev, threads, swap_grid=swap_grid)
    emit_col_stride(instrs, 2, swap_grid=swap_grid)
    acc0 = _hreg('hr16.x') if buffer_a else _hreg('hr8.x') if high_a else _hreg('hr12.x')
    hregs, fregs = (32 if buffer_a else 28), 10
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    if fast_coords: instrs.append(ADD_S('r6.w', 'r7.y', 32))
    if quad_a: instrs.append(AND_B('r8.z', 'r0.x', 3, nop=3))
    if base_b_y: instrs += [ADD_S('r6.y', 'r6.y', -3), NOP(rpt=2)]

    first = True
    b_regs = [[_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg('hr7.x')],
              [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')]] if low_a else \
             [[_hreg('hr0.x'), _hreg('hr1.x'), _hreg('hr2.x'), _hreg('hr3.x')],
              [_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg('hr7.x')]]
    a_regs = [_hreg('hr0.x'), _hreg('hr1.x'), _hreg('hr2.x'), _hreg('hr3.x')] if low_a else [_hreg('hr24.x'), _hreg('hr25.x'), _hreg('hr26.x'), _hreg('hr27.x')] if high_a else [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')]
    a2_regs = [_hreg('hr12.x'), _hreg('hr13.x'), _hreg('hr14.x'), _hreg('hr15.x')] if buffer_a else [_hreg('hr28.x'), _hreg('hr29.x'), _hreg('hr30.x'), _hreg('hr31.x')]

    def emit_const_regs(start, end):
        nonlocal instrs
        for base in range(start, end, 4): instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]

    def emit_mad(row, col, kk, force_sync=False, aregs=None):
        nonlocal first
        if aregs is None: aregs = a_regs
        group = (row * 2 + col) * 4
        instrs.append(MAD_F16(acc0 + group, aregs[row & 3] + kk, b_regs[col][kk], acc0 + group, rpt=3, sy=(first or force_sync), r=True))
        first = False

    def emit_b_loads():
        nonlocal instrs
        if skip_b_loads:
            for group in b_regs:
                for base in group: instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
            return
        if grouped_b_cols:
            pairs = [('r4.y', 'r4.z'), ('r4.w', 'r5.x'), ('r5.y', 'r5.z'), ('r5.w', 'r6.x')]
            for col, base in [(0, _hreg('hr0.x')), (1, _hreg('hr4.x'))]:
                if col and not fast_coords: instrs.append(ADD_S('r6.w', 'r7.y', 32))
                xsrc = 'r7.y' if col == 0 else 'r6.w'
                for kk, (xreg, yreg) in enumerate(pairs):
                    instrs.append(MOV_F32(xreg, xsrc))
                    instrs.append([lambda: ADD_S(yreg, 'r6.y', -3), lambda: ADD_S(yreg, 'r6.y', -2),
                                   lambda: ADD_S(yreg, 'r6.y', -1), lambda: MOV_F32(yreg, 'r6.y')][kk]())
                emit_coord_wait(instrs, b_coord_delay)
                for kk, (xreg, _) in enumerate(pairs): instrs.append(ISAM_F16(base + kk * 4, xreg, 1))
            return
        if grouped_b:
            pairs = [('r4.y', 'r4.z'), ('r4.w', 'r5.x'), ('r5.y', 'r5.z'), ('r5.w', 'r6.x'),
                     ('r6.w', 'r7.w'), ('r8.x', 'r8.y'), ('r8.z', 'r8.w'), ('r9.x', 'r9.y')]
            if not fast_coords: instrs.append(ADD_S('r9.z', 'r7.y', 32))
            for i, (xreg, yreg) in enumerate(pairs):
                instrs.append(MOV_F32(xreg, 'r7.y' if i < 4 else ('r6.w' if fast_coords else 'r9.z')))
                koff = i & 3
                instrs.append([lambda: ADD_S(yreg, 'r6.y', -3), lambda: ADD_S(yreg, 'r6.y', -2),
                               lambda: ADD_S(yreg, 'r6.y', -1), lambda: MOV_F32(yreg, 'r6.y')][koff]())
            emit_coord_wait(instrs, b_coord_delay)
            for i, (xreg, _) in enumerate(pairs): instrs.append(ISAM_F16(_hreg('hr0.x') + i * 4, xreg, 1))
            return
        for col, base in [(0, b_regs[0][0]), (1, b_regs[1][0])]:
            emit_b_col(col, base)

    def emit_b_col(col, base=None, next_y=False):
        nonlocal instrs
        if skip_b_loads:
            base = b_regs[col][0] if base is None else base
            for off in range(4): instrs += [MOV_H_IMM(base + off * 4, 0x3c00), MOV_H(base + off * 4 + 1, base + off * 4, rpt=2)]
            return
        if base is None: base = b_regs[col][0]
        instrs.append(MOV_F32('r4.y', 'r7.y') if col == 0 else MOV_F32('r4.y', 'r6.w') if fast_coords else ADD_S('r4.y', 'r7.y', 32))
        if base_b_y:
            if next_y:
                instrs += [ADD_S('r5.y', 'r6.y', 4), NOP(rpt=max(b_coord_delay, 0))]
                ysrc = 'r5.y'
            else: ysrc = 'r6.y'
            for kk in range(4):
                instrs.append(MOV_F32('r4.z', ysrc) if kk == 0 else OR_B('r4.z', ysrc, kk))
                instrs.append(ISAM_F16(base + kk * 4, 'r4.y', 1))
            return
        yoffs = [1, 2, 3, 4] if next_y else [-3, -2, -1, 0]
        if pair_b_coords and b_coord_delay == 0 and not inline_b_wait:
            instrs.append(MOV_F32('r4.w', 'r7.y') if col == 0 else MOV_F32('r4.w', 'r6.w') if fast_coords else ADD_S('r4.w', 'r7.y', 32))
            for kk in (0, 2):
                instrs.append(ADD_S('r4.z', 'r6.y', yoffs[kk]))
                instrs.append(MOV_F32('r5.x', 'r6.y') if yoffs[kk + 1] == 0 else ADD_S('r5.x', 'r6.y', yoffs[kk + 1]))
                emit_coord_wait(instrs, b_coord_delay)
                instrs.append(ISAM_F16(base + kk * 4, 'r4.y', 1))
                instrs.append(ISAM_F16(base + (kk + 1) * 4, 'r4.w', 1))
            return
        for kk, yoff in enumerate(yoffs):
            if inline_b_wait and b_coord_delay == 0:
                instrs.append(ADD_S('r4.z', 'r6.y', yoff, nop=inline_b_nop))
            else:
                instrs.append(MOV_F32('r4.z', 'r6.y') if yoff == 0 else ADD_S('r4.z', 'r6.y', yoff))
                emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(base + kk * 4, 'r4.y', 1))

    def emit_b_col_range(col, kk_start, kk_end, next_y=False, setup_x=True):
        nonlocal instrs
        base = b_regs[col][0]
        if skip_b_loads:
            for kk in range(kk_start, kk_end): instrs += [MOV_H_IMM(base + kk * 4, 0x3c00), MOV_H(base + kk * 4 + 1, base + kk * 4, rpt=2)]
            return
        if setup_x: instrs.append(MOV_F32('r4.y', 'r7.y') if col == 0 else MOV_F32('r4.y', 'r6.w') if fast_coords else ADD_S('r4.y', 'r7.y', 32))
        yoffs = [1, 2, 3, 4] if next_y else [-3, -2, -1, 0]
        for kk in range(kk_start, kk_end):
            if inline_b_wait and b_coord_delay == 0:
                instrs.append(ADD_S('r4.z', 'r6.y', yoffs[kk], nop=inline_b_nop))
            else:
                instrs.append(MOV_F32('r4.z', 'r6.y') if yoffs[kk] == 0 else ADD_S('r4.z', 'r6.y', yoffs[kk]))
                emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(base + kk * 4, 'r4.y', 1))

    def emit_loop_tail_b_prefetch():
        nonlocal instrs
        prefetch_instrs, saved_instrs = [], instrs
        instrs = prefetch_instrs
        emit_b_col(0, next_y=True)
        emit_b_col(1, next_y=True)
        instrs = saved_instrs
        instrs += [CMPS_S_EQ('r6.z', K4 - 1, nop=1), BR(len(prefetch_instrs) + 1, inv=False)] + prefetch_instrs

    def emit_a_load_group(row_base, setup_x=True, aregs=None):
        nonlocal instrs
        if aregs is None: aregs = a_regs
        if skip_a_loads:
            for base in aregs:
                instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
        elif quad_a:
            if setup_x: instrs.append(MOV_F32('r4.y', 'r6.z'))
            if row_base: instrs += [ADD_S('r8.w', 'r8.z', row_base), ADD_S_REG('r4.z', 'r7.x', 'r8.w')]
            else: instrs.append(ADD_S_REG('r4.z', 'r7.x', 'r8.z'))
            instrs.append(ISAM_F16(aregs[0], 'r4.y', 0))
            src0, src1 = full_reg_name(aregs[0] // 2), full_reg_name(aregs[0] // 2 + 1)
            first_qbc = True
            for local_row in (1, 2, 3, 0):
                instrs.append(MOV_S32('r7.w', int(quad_map[local_row])))
                dst0, dst1 = full_reg_name(aregs[local_row] // 2), full_reg_name(aregs[local_row] // 2 + 1)
                instrs.append(QUAD_BRCST(dst0, src0, 'r7.w', typ=3, sy=first_qbc))
                instrs.append(QUAD_BRCST(dst1, src1, 'r7.w', typ=3))
                first_qbc = False
        else:
            if setup_x: instrs.append(MOV_F32('r4.y', 'r6.z'))
            for local_row, areg in enumerate(aregs):
                row = row_base + local_row
                instrs.append(MOV_F32('r4.z', 'r7.x') if row == 0 else ADD_S('r4.z', 'r7.x', row) if add_a_rows else OR_B('r4.z', 'r7.x', row))
                instrs.append(ISAM_F16(areg, 'r4.y', 0))

    def emit_a_load_group_next(row_base, aregs=None):
        nonlocal instrs
        if aregs is None: aregs = a_regs
        if skip_a_loads:
            for base in aregs:
                instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
        else:
            instrs.append(ADD_S('r4.y', 'r6.z', 1))
            for local_row, areg in enumerate(aregs):
                row = row_base + local_row
                instrs.append(MOV_F32('r4.z', 'r7.x') if row == 0 else ADD_S('r4.z', 'r7.x', row) if add_a_rows else OR_B('r4.z', 'r7.x', row))
                instrs.append(ISAM_F16(areg, 'r4.y', 0))

    def emit_mads_group(row_base, cols=(0, 1), skip=None, force_first=False, aregs=None):
        forced = False
        if aregs is None: aregs = a_regs
        def maybe_emit(row, col, kk):
            nonlocal forced
            if col not in cols or skip == (row, col, kk): return
            emit_mad(row, col, kk, force_sync=force_first and not forced, aregs=aregs)
            forced = True
        if alu_order == 'row_col_kk':
            for local_row in range(4):
                row = row_base + local_row
                for col in range(2):
                    for kk in range(4): maybe_emit(row, col, kk)
        elif alu_order == 'row_kk_col':
            for local_row in range(4):
                row = row_base + local_row
                for kk in range(4):
                    for col in range(2): maybe_emit(row, col, kk)
        elif alu_order == 'kk_row_col':
            for kk in range(4):
                for local_row in range(4):
                    row = row_base + local_row
                    for col in range(2): maybe_emit(row, col, kk)
        elif alu_order == 'kk_col_row':
            for kk in range(4):
                for col in range(2):
                    for local_row in range(4): maybe_emit(row_base + local_row, col, kk)
        elif alu_order == 'col_kk_row':
            for col in range(2):
                for kk in range(4):
                    for local_row in range(4): maybe_emit(row_base + local_row, col, kk)
        else: raise ValueError('unsupported split alu order %s' % alu_order)

    def emit_mads_group_rows(row_base, local_rows, cols=(0, 1), aregs=None):
        if aregs is None: aregs = a_regs
        for local_row in local_rows:
            row = row_base + local_row
            for col in range(2):
                if col in cols:
                    for kk in range(4): emit_mad(row, col, kk, aregs=aregs)

    def emit_mads_group_col_stream_next_b(row_base, col, aregs=None):
        nonlocal instrs
        if aregs is None: aregs = a_regs
        if skip_b_loads:
            emit_mads_group(row_base, cols=(col,), aregs=aregs)
            for base in b_regs[col]: instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
            return
        instrs.append(MOV_F32('r4.y', 'r7.y') if col == 0 else MOV_F32('r4.y', 'r6.w') if fast_coords else ADD_S('r4.y', 'r7.y', 32))
        for kk, yoff in enumerate([1, 2, 3, 4]):
            for local_row in range(4): emit_mad(row_base + local_row, col, kk, aregs=aregs)
            instrs.append(ADD_S('r4.z', 'r6.y', yoff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(b_regs[col][kk], 'r4.y', 1))

    def emit_a_group(row_base, setup_x=True):
        emit_a_load_group(row_base, setup_x)
        emit_mads_group(row_base)

    if prefetch_loop_b:
        emit_b_loads()
    loop_start = len(instrs)

    for ku in range(k_unroll):
        if ku:
            instrs += [ADD_S('r6.z', 'r6.z', 1), ADD_S('r6.y', 'r6.y', 4)]
        if stream_col1 and not grouped_b and not grouped_b_cols and alu_order == 'row_col_kk':
            emit_b_col(0)
            emit_a_load_group(0)
            emit_mad(0, 0, 0)
            emit_b_col(1)
            emit_mads_group(0, cols=(0,), skip=(0, 0, 0))
            emit_mads_group(0, cols=(1,), force_first=stream_col1_sync)
            emit_a_group(4)
        elif prefetch_next_b and buffer_a and not grouped_b and not grouped_b_cols and alu_order == 'row_col_kk':
            if ku == 0 and not prefetch_loop_b: emit_b_loads()
            if fast_coords and not skip_a_loads: instrs.append(MOV_F32('r4.y', 'r6.z'))
            emit_a_load_group(0, setup_x=not fast_coords, aregs=a_regs)
            emit_mads_group(0, cols=(0,), aregs=a_regs)
            emit_a_load_group(4, setup_x=not fast_coords, aregs=a2_regs)
            emit_mads_group(0, cols=(1,), aregs=a_regs)
            if ku < k_unroll - 1:
                emit_mads_group(4, cols=(0,), aregs=a2_regs)
                emit_b_col(0, next_y=True)
                emit_mads_group(4, cols=(1,), aregs=a2_regs)
                emit_b_col(1, next_y=True)
            else:
                emit_mads_group(4, aregs=a2_regs)
                if prefetch_loop_b: emit_loop_tail_b_prefetch()
        elif prefetch_next_b and not grouped_b and not grouped_b_cols and alu_order == 'row_col_kk':
            if ku == 0 and not prefetch_loop_b: emit_b_loads()
            if prefetch_next_a and ku:
                emit_mads_group(0)
            else:
                if fast_coords and not skip_a_loads: instrs.append(MOV_F32('r4.y', 'r6.z'))
                emit_a_load_group(0, setup_x=not fast_coords)
                if pre_mad_nops >= 0: instrs.append(NOP(rpt=pre_mad_nops))
                emit_mads_group(0)
            emit_a_load_group(4, setup_x=not fast_coords)
            if ku < k_unroll - 1:
                if hoist_b0_coord:
                    instrs.append(MOV_F32('r8.x', 'r7.y'))
                    instrs.append(ADD_S('r8.y', 'r6.y', 1))
                if stream_next_b0: emit_mads_group_col_stream_next_b(4, 0)
                else: emit_mads_group(4, cols=(0,))
                if interleave_next_b:
                    emit_b_col_range(0, 0, 2, next_y=True)
                    emit_mads_group_rows(4, (0, 1), cols=(1,))
                    emit_b_col_range(0, 2, 4, next_y=True, setup_x=False)
                    emit_mads_group_rows(4, (2, 3), cols=(1,))
                elif hoist_b0_coord:
                    emit_coord_wait(instrs, b_coord_delay)
                    instrs.append(ISAM_F16(_hreg('hr0.x'), 'r8.x', 1))
                    emit_b_col_range(0, 1, 4, next_y=True, setup_x=False)
                    emit_mads_group(4, cols=(1,))
                else:
                    if not stream_next_b0: emit_b_col(0, next_y=True)
                    if stream_next_b1: emit_mads_group_col_stream_next_b(4, 1)
                    else: emit_mads_group(4, cols=(1,))
                if prefetch_next_a: emit_a_load_group_next(0)
                if not stream_next_b1: emit_b_col(1, next_y=True)
            else:
                emit_mads_group(4)
                if prefetch_loop_b: emit_loop_tail_b_prefetch()
        else:
            emit_b_loads()
            if fast_coords and not skip_a_loads: instrs.append(MOV_F32('r4.y', 'r6.z'))
            emit_a_group(0, setup_x=not fast_coords)
            emit_a_group(4, setup_x=not fast_coords)

    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        ADD_S('r6.y', 'r6.y', 4),
        CMPS_S_EQ('r6.z', K4 - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        instrs += [MOV_H_IMM(acc0, 0x6400), MOV_H(acc0 + 1, acc0, rpt=2)]
        for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    if not no_store: emit_donor8_add256_stores(instrs, dev, threads, acc0, 2, gap=add256_gap, offset_before_gap=add256_offset_before_gap, explicit_stores=add256_explicit_stores, store_mode=add256_store_mode, direct_sources=add256_direct_sources)
    instrs.append(END())
    return assemble(instrs), hregs, fregs, (loop_end - loop_start) // k_unroll


def emit_zero_f32_vecs(instrs, start_vec, count):
    for vec in range(start_vec, start_vec + count): emit_f32_vec_imm(instrs, vec, 0)


def emit_f32_vec_imm(instrs, vec, imm):
    for comp in range(4): instrs.append(MOV_S32(fvec(vec, comp), imm))


def emit_isam_h_to_f32_vec(instrs, dst_vec, coord, tex):
    instrs.append(ISAM_F16('hr0.x', coord, tex))
    for comp in range(4): instrs.append(COV_F16F32(fvec(dst_vec, comp), 'hr0.%s' % 'xyzw'[comp], sy=(comp == 0)))


def emit_isam_f32_vec(instrs, dst_vec, coord, tex, sampler_per_texture=False):
    instrs.append(ISAM_F32(fvec(dst_vec), coord, tex, tex if sampler_per_texture else 0))


def emit_store8_float(instrs, dev, threads, acc_vec0, ncols, row_reg='r7.x', col_reg='r7.y'):
    if ncols == 2:
        lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8_fp32(2, threads))
        donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]

        def acc(row, col): return fvec(acc_vec0 + row * 2 + col)
        def mov_vec(dst, src): instrs.append(MOV_F32(dst, src, rpt=3))
        def mov_split(dsts, src):
            for comp, dst in enumerate(dsts): instrs.append(MOV_F32(dst, fvec(src, comp)))

        # The compiler ncols=2 store epilogue packs data from this register layout
        # before clobbering the same high registers for 64-bit store addresses.
        mov_vec('r9.x', acc(0, 0))
        mov_split(('r16.y', 'r16.w', 'r17.y', 'r17.w'), acc_vec0 + 1)
        mov_split(('r18.y', 'r18.w', 'r19.y', 'r19.w'), acc_vec0 + 2)
        mov_split(('r20.y', 'r20.w', 'r21.y', 'r21.w'), acc_vec0 + 3)
        mov_split(('r22.y', 'r22.w', 'r23.y', 'r23.w'), acc_vec0 + 4)
        for store_idx, target in enumerate(['r24.x', 'r25.x', 'r26.x', 'r27.x', 'r28.x', 'r10.x', 'r11.x', 'r12.x', 'r13.x', 'r14.x', 'r15.x'], start=5):
            mov_vec(target, acc(store_idx // 2, store_idx & 1))

        instrs += [MOV_F32('r8.y', row_reg), MOV_F32('r8.z', col_reg)]
        instrs += donor[462:625]
        return
    if ncols != 1: raise ValueError('FP32 float store currently supports ncols=1 or 2')
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8_fp32(1, threads))
    donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]
    addr = donor[268:318] + donor[331:339]
    stores = donor[339:353]
    instrs += [MOV_F32('r18.y', row_reg), MOV_F32('r18.z', col_reg)]
    for col in range(ncols):
        if col == 0: instrs.append(MOV_F32('r12.x', fvec(acc_vec0), rpt=3))
        instrs.append(MOV_F32('r6.z', 'r18.y'))
        instrs.append(MOV_F32('r6.w', 'r18.z') if col == 0 else ADD_S('r6.w', 'r18.z', 32))
        instrs += addr
        for row in range(8): instrs.append(MOV_F32(fvec(row), 'r12.x' if col == 0 and row == 0 else fvec(acc_vec0 + row * ncols + col), rpt=3))
        instrs.append(NOP(rpt=16))
        instrs += stores


def build_8x8_fp32_shader(dev, threads, ncols=1, b_coord_delay=0, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, alu_order='row_col_kk', direct_f32_loads=False, sampler_per_texture=False):
    if ncols not in (1, 2): raise ValueError('FP32 accumulator path currently supports ncols=1 or 2')
    acc_vec0 = 20
    tid_save_reg = 'r36.w' if ncols == 2 else 'r7.z'
    instrs = [MOV_F32(tid_save_reg, 'r0.x')] + prologue_8x4_fp32(dev, threads)
    instrs += [MOV_F32('r7.x', 'r6.z'), MOV_F32('r7.y', 'r6.w')]
    emit_col_stride(instrs, ncols)
    instrs += [MOV_S32('r6.y', 3), MOV_S32('r6.z', 0)]
    if ncols > 1: instrs.append(ADD_S('r6.w', 'r7.y', 32))

    b_regs = [[8, 9, 10, 11], [12, 13, 14, 15]][:ncols]
    a_vec0 = 16
    emit_zero_f32_vecs(instrs, acc_vec0, 8 * ncols)

    first = True
    def emit_b_col(col):
        nonlocal instrs
        xsrc = 'r6.w' if ncols == 1 else 'r7.y' if col == 0 else 'r6.w'
        instrs.append(MOV_F32('r4.x', xsrc))
        for kk, yoff in enumerate((-3, -2, -1, 0)):
            instrs.append(MOV_F32('r4.y', 'r6.y') if yoff == 0 else ADD_S('r4.y', 'r6.y', yoff))
            emit_coord_wait(instrs, b_coord_delay)
            if skip_b_loads:
                emit_f32_vec_imm(instrs, b_regs[col][kk], 0x3f800000)
            elif direct_f32_loads:
                emit_isam_f32_vec(instrs, b_regs[col][kk], 'r4.x', 1, sampler_per_texture)
            else:
                emit_isam_h_to_f32_vec(instrs, b_regs[col][kk], 'r4.x', 1)

    def emit_a_group(row_base):
        nonlocal instrs
        if not skip_a_loads: instrs.append(MOV_F32('r4.x', 'r6.z'))
        for local_row in range(4):
            row = row_base + local_row
            if skip_a_loads:
                emit_f32_vec_imm(instrs, a_vec0 + local_row, 0x3f800000)
            else:
                instrs.append(MOV_F32('r4.y', 'r7.x') if row == 0 else OR_B('r4.y', 'r7.x', row))
                emit_coord_wait(instrs, b_coord_delay)
                if direct_f32_loads: emit_isam_f32_vec(instrs, a_vec0 + local_row, 'r4.x', 0, sampler_per_texture)
                else: emit_isam_h_to_f32_vec(instrs, a_vec0 + local_row, 'r4.x', 0)

    def emit_mad(row, col, kk):
        nonlocal first
        acc = acc_vec0 + row * ncols + col
        instrs.append(MAD_F32(fvec(acc), fvec(a_vec0 + (row & 3), kk), fvec(b_regs[col][kk]), fvec(acc), rpt=3, sy=first, r=True))
        first = False

    def emit_mads_group(row_base):
        if alu_order == 'row_col_kk':
            for local_row in range(4):
                row = row_base + local_row
                for col in range(ncols):
                    for kk in range(4): emit_mad(row, col, kk)
        elif alu_order == 'kk_row_col':
            for kk in range(4):
                for local_row in range(4):
                    row = row_base + local_row
                    for col in range(ncols): emit_mad(row, col, kk)
        elif alu_order == 'kk_col_row':
            for kk in range(4):
                for col in range(ncols):
                    for local_row in range(4): emit_mad(row_base + local_row, col, kk)
        elif alu_order == 'row_kk_col':
            for local_row in range(4):
                row = row_base + local_row
                for kk in range(4):
                    for col in range(ncols): emit_mad(row, col, kk)
        elif alu_order == 'col_kk_row':
            for col in range(ncols):
                for kk in range(4):
                    for local_row in range(4): emit_mad(row_base + local_row, col, kk)
        else: raise ValueError('unsupported fp32 alu order %s' % alu_order)

    loop_start = len(instrs)
    for col in range(ncols): emit_b_col(col)
    emit_a_group(0)
    emit_mads_group(0)
    emit_a_group(4)
    emit_mads_group(4)
    instrs += [
        ADD_S('r5.x', 'r6.z', 1),
        ADD_S('r6.y', 'r6.y', 4),
        CMPS_S_EQ('r6.z', K4 - 1, nop=1),
        MOV_F32('r6.z', 'r5.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        for vec in range(acc_vec0, acc_vec0 + 8 * ncols): emit_f32_vec_imm(instrs, vec, 0x44800000)
    if not no_store:
        if ncols == 1:
            instrs.append(MOV_F32('r0.x', tid_save_reg))
            instrs += prologue_8x4_fp32(dev, threads)[3:13]
            emit_store8_float(instrs, dev, threads, acc_vec0, ncols, row_reg='r6.z', col_reg='r6.w')
        else:
            instrs.append(MOV_F32('r0.x', tid_save_reg))
            instrs += prologue_8x4_fp32(dev, threads, ncols=2)[2:12]
            emit_store8_float(instrs, dev, threads, acc_vec0, ncols, row_reg='r8.y', col_reg='r8.z')
    instrs.append(END())
    return assemble(instrs), 0 if direct_f32_loads else 1, max(acc_vec0 + 8 * ncols, 37 if ncols == 2 else 0), loop_end - loop_start


def build_8x16_split_a_unroll_shader(dev, threads, k_unroll=2, b_coord_delay=0, post_constant=False, no_store=False, skip_a_loads=False, skip_b_loads=False, add256_gap=16, fast_coords=False, alu_order='row_col_kk', add256_offset_before_gap=False, add256_explicit_stores=False, add256_store_mode='donor', add256_direct_sources=False, swap_grid=False):
    if K4 % k_unroll != 0: raise ValueError('--split-k-unroll must divide K/4')
    instrs = prologue_8x4(dev, threads, swap_grid=swap_grid)
    emit_col_stride(instrs, 4, swap_grid=swap_grid)
    acc0 = _hreg('hr16.x')
    hregs, fregs = 48, 8
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 32 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    if fast_coords: instrs.append(ADD_S('r6.w', 'r7.y', 32))

    first = True
    bbufs = [[_hreg('hr0.x'), _hreg('hr1.x'), _hreg('hr2.x'), _hreg('hr3.x')],
             [_hreg('hr4.x'), _hreg('hr5.x'), _hreg('hr6.x'), _hreg('hr7.x')]]
    a0_regs = [_hreg('hr8.x'), _hreg('hr9.x'), _hreg('hr10.x'), _hreg('hr11.x')]
    a1_regs = [_hreg('hr12.x'), _hreg('hr13.x'), _hreg('hr14.x'), _hreg('hr15.x')]

    def emit_b_col(col, bregs):
        nonlocal instrs
        if skip_b_loads:
            for base in bregs: instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
            return
        if col == 0: instrs.append(MOV_F32('r4.y', 'r7.y'))
        elif col == 1 and fast_coords: instrs.append(MOV_F32('r4.y', 'r6.w'))
        else: instrs.append(ADD_S('r4.y', 'r7.y', col * 32))
        for kk, yoff in enumerate([-3, -2, -1, 0]):
            instrs.append(MOV_F32('r4.z', 'r6.y') if yoff == 0 else ADD_S('r4.z', 'r6.y', yoff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(bregs[kk], 'r4.y', 1))

    def emit_a_load_group(row_base, aregs, setup_x=True):
        nonlocal instrs
        if skip_a_loads:
            for base in aregs: instrs += [MOV_H_IMM(base, 0x3c00), MOV_H(base + 1, base, rpt=2)]
            return
        if setup_x: instrs.append(MOV_F32('r4.y', 'r6.z'))
        for local_row, areg in enumerate(aregs):
            row = row_base + local_row
            instrs.append(MOV_F32('r4.z', 'r7.x') if row == 0 else OR_B('r4.z', 'r7.x', row))
            instrs.append(ISAM_F16(areg, 'r4.y', 0))

    def emit_mad(row, col, kk, bregs):
        nonlocal first
        aregs = a0_regs if row < 4 else a1_regs
        group = (row * 4 + col) * 4
        instrs.append(MAD_F16(acc0 + group, aregs[row & 3] + kk, bregs[kk], acc0 + group, rpt=3, sy=first, r=True))
        first = False

    def emit_col_mads(col, bregs, prefetch_col=None, prefetch_bregs=None):
        emitted = 0
        if alu_order == 'row_col_kk':
            for row in range(8):
                for kk in range(4):
                    emit_mad(row, col, kk, bregs)
                    emitted += 1
                    if prefetch_col is not None and emitted == 1: emit_b_col(prefetch_col, prefetch_bregs)
        elif alu_order == 'kk_row_col':
            for kk in range(4):
                for row in range(8):
                    emit_mad(row, col, kk, bregs)
                    emitted += 1
                    if prefetch_col is not None and emitted == 1: emit_b_col(prefetch_col, prefetch_bregs)
        else: raise ValueError('unsupported 8x16 split alu order %s' % alu_order)

    loop_start = len(instrs)
    for ku in range(k_unroll):
        if ku: instrs += [ADD_S('r6.z', 'r6.z', 1), ADD_S('r6.y', 'r6.y', 4)]
        emit_b_col(0, bbufs[0])
        emit_b_col(1, bbufs[1])
        if fast_coords and not skip_a_loads: instrs.append(MOV_F32('r4.y', 'r6.z'))
        emit_a_load_group(0, a0_regs, setup_x=not fast_coords)
        emit_a_load_group(4, a1_regs, setup_x=not fast_coords)
        emit_col_mads(0, bbufs[0])
        emit_b_col(2, bbufs[0])
        emit_col_mads(1, bbufs[1])
        emit_b_col(3, bbufs[1])
        emit_col_mads(2, bbufs[0])
        emit_col_mads(3, bbufs[1])

    instrs += [
        ADD_S('r0.x', 'r6.z', 1),
        ADD_S('r6.y', 'r6.y', 4),
        CMPS_S_EQ('r6.z', K4 - 1, nop=1),
        MOV_F32('r6.z', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))
    if post_constant:
        instrs += [MOV_H_IMM(acc0, 0x6400), MOV_H(acc0 + 1, acc0, rpt=2)]
        for base in range(acc0 + 4, acc0 + 32 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))
    if not no_store: emit_donor8_add256_stores(instrs, dev, threads, acc0, 4, gap=add256_gap, offset_before_gap=add256_offset_before_gap, explicit_stores=add256_explicit_stores, store_mode=add256_store_mode, direct_sources=add256_direct_sources)
    instrs.append(END())
    return assemble(instrs), hregs, fregs, (loop_end - loop_start) // k_unroll


def build_8xn_serial_alu_profile_shader(dev, threads, ncols, profile_r1=False, alu_order='kk_row_col'):
    instrs = prologue_8x4(dev, threads)
    emit_col_stride(instrs, ncols)
    acc0 = _hreg('hr16.x')
    hregs, fregs = (acc0 + 8 * ncols * 4 + 3) // 4, 8

    instrs += [MOV_H_IMM(0, 0x3c00), MOV_H(1, 0, rpt=2)]
    for base in range(4, 12 * 4, 4): instrs.append(MOV_H(base, 0, rpt=3))
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 8 * ncols * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    a_regs = [_hreg(f'hr{i}.x') for i in range(8)]
    b_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
    first = True
    def emit(row, kk, col):
        nonlocal first
        group = (row * ncols + col) * 4
        instrs.append(MAD_F16(acc0 + group, a_regs[row] + kk, b_regs[kk], acc0 + group, rpt=3, sy=first, r=True, r1=profile_r1))
        first = False
    if alu_order == 'kk_row_col':
        for kk in range(4):
            for row in range(8):
                for col in range(ncols): emit(row, kk, col)
    elif alu_order == 'kk_col_row':
        for kk in range(4):
            for col in range(ncols):
                for row in range(8): emit(row, kk, col)
    elif alu_order == 'col_kk_row':
        for col in range(ncols):
            for kk in range(4):
                for row in range(8): emit(row, kk, col)
    elif alu_order == 'row_kk_col':
        for row in range(8):
            for kk in range(4):
                for col in range(ncols): emit(row, kk, col)
    elif alu_order == 'row_col_kk':
        for row in range(8):
            for col in range(ncols):
                for kk in range(4): emit(row, kk, col)
    else: raise ValueError('unknown ALU order %s' % alu_order)
    emit_loop_control(instrs)
    loop_end = len(instrs)
    instrs += [BR(loop_start - loop_end), END()]
    return assemble(instrs), hregs, fregs, loop_end - loop_start


def emit_coord_wait(instrs, coord_delay):
    if coord_delay >= 0: instrs.append(NOP(rpt=coord_delay))


def build_8x4_donor_store_shader(dev, threads, a_coord_delay=4, b_coord_delay=4):
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8(1, threads))
    donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]
    instrs = donor[:25]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 8 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    a_pairs = [('hr0.x', 'r4.w'), ('hr1.x', 'r5.y'), ('hr2.x', 'r5.w'), ('hr3.x', 'r6.y'),
               ('hr4.x', 'r6.w'), ('hr5.x', 'r7.y'), ('hr6.x', 'r7.w'), ('hr7.x', 'r8.y')]
    for dst, coord in a_pairs:
        instrs.append(MOV_F32(coord, 'r11.x'))
        emit_coord_wait(instrs, a_coord_delay)
        instrs.append(ISAM_F16(dst, coord, 0))

    b_pairs = [('hr8.x', 'r8.w', lambda: ADD_S('r9.x', 'r10.w', -3)),
               ('hr9.x', 'r9.y', lambda: ADD_S('r9.z', 'r10.w', -2)),
               ('hr10.x', 'r9.w', lambda: ADD_S('r10.x', 'r10.w', -1)),
               ('hr11.x', 'r10.y', lambda: MOV_F32('r10.z', 'r10.w'))]
    for dst, coord, set_y in b_pairs:
        instrs.append(set_y())
        emit_coord_wait(instrs, b_coord_delay)
        instrs.append(ISAM_F16(dst, coord, 1))

    a_regs = [_hreg(f'hr{i}.x') for i in range(8)]
    b_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
    first = True
    for kk in range(4):
        for row in range(8):
            instrs.append(MAD_F16(acc0 + row * 4, a_regs[row] + kk, b_regs[kk], acc0 + row * 4, rpt=3, sy=first, r=True))
            first = False

    instrs += [
        ADD_S('r0.x', 'r11.x', 1),
        ADD_S('r10.w', 'r10.w', 4),
        CMPS_S_EQ('r11.x', K4 - 1, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    # Keep the compiler donor's address calculation/stores. Replace only its
    # accumulator packing with direct moves from our contiguous accumulators.
    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 4, rpt=3))
    instrs += donor[292:315]
    shader = assemble(instrs)
    return shader, 24, 12, loop_end - loop_start, img_sz


def build_8x4_pipelined_shader(dev, threads, a_coord_delay=4, b_coord_delay=-1, grouped_a=True, grouped_b=True):
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8(1, threads))
    donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]
    instrs = donor[:25]

    acc0 = _hreg('hr24.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 8 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    a_coord_regs = ['r4.w', 'r5.y', 'r5.w', 'r6.y', 'r6.w', 'r7.y', 'r7.w', 'r8.y']
    b_coord_pairs = [('r8.w', 'r9.x', -3), ('r9.y', 'r9.z', -2), ('r9.w', 'r10.x', -1), ('r10.y', 'r10.z', 0)]
    def emit_loads(a_base, b_base, kx, by):
        if grouped_a:
            for coord in a_coord_regs: instrs.append(MOV_F32(coord, kx))
            emit_coord_wait(instrs, a_coord_delay)
            for i, coord in enumerate(a_coord_regs): instrs.append(ISAM_F16(a_base + i * 4, coord, 0))
        else:
            for i, coord in enumerate(a_coord_regs):
                instrs.append(MOV_F32(coord, kx))
                emit_coord_wait(instrs, a_coord_delay)
                instrs.append(ISAM_F16(a_base + i * 4, coord, 0))
        if grouped_b:
            for _, yreg, koff in b_coord_pairs: instrs.append(MOV_F32(yreg, by) if koff == 0 else ADD_S(yreg, by, koff))
            emit_coord_wait(instrs, b_coord_delay)
            for i, (coord, _, _) in enumerate(b_coord_pairs): instrs.append(ISAM_F16(b_base + i * 4, coord, 1))
        else:
            for i, (coord, yreg, koff) in enumerate(b_coord_pairs):
                instrs.append(MOV_F32(yreg, by) if koff == 0 else ADD_S(yreg, by, koff))
                emit_coord_wait(instrs, b_coord_delay)
                instrs.append(ISAM_F16(b_base + i * 4, coord, 1))

    def emit_mads(a_base, b_base, first_sy=False, skip_first=False):
        first = first_sy
        for kk in range(4):
            for row in range(8):
                if skip_first and kk == 0 and row == 0: continue
                base = acc0 + row * 4
                instrs.append(MAD_F16(base, a_base + row * 4 + kk, b_base + kk * 4, base, rpt=3, sy=first, r=True))
                first = False

    emit_loads(_hreg('hr0.x'), _hreg('hr8.x'), 'r11.x', 'r10.w')
    loop_start = len(instrs)
    instrs.append(MAD_F16(acc0, _hreg('hr0.x'), _hreg('hr8.x'), acc0, rpt=3, sy=True, r=True))
    instrs += [ADD_S('r0.x', 'r11.x', 1), ADD_S('r0.z', 'r10.w', 4)]
    emit_loads(_hreg('hr12.x'), _hreg('hr20.x'), 'r0.x', 'r0.z')
    emit_mads(_hreg('hr0.x'), _hreg('hr8.x'), skip_first=True)
    instrs.append(MAD_F16(acc0, _hreg('hr12.x'), _hreg('hr20.x'), acc0, rpt=3, sy=True, r=True))
    instrs += [ADD_S('r0.x', 'r11.x', 2), ADD_S('r0.z', 'r10.w', 8)]
    emit_loads(_hreg('hr0.x'), _hreg('hr8.x'), 'r0.x', 'r0.z')
    emit_mads(_hreg('hr12.x'), _hreg('hr20.x'), skip_first=True)

    instrs += [
        ADD_S('r0.x', 'r11.x', 2),
        ADD_S('r10.w', 'r10.w', 8),
        CMPS_S_EQ('r11.x', K4 - 2, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 4, rpt=3))
    instrs += donor[292:315]
    shader = assemble(instrs)
    return shader, 32, 12, (loop_end - loop_start) // 2, img_sz


def build_8x8_donor_store_shader(dev, threads, a_coord_delay=4, b_coord_delay=4, pre_mad_nops=0, alu_order='kk_row_col'):
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8(2, threads))
    donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]
    instrs = donor[:29]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    a_pairs = [('hr0.x', 'r7.w'), ('hr1.x', 'r8.y'), ('hr2.x', 'r8.w'), ('hr3.x', 'r9.y'),
               ('hr4.x', 'r9.w'), ('hr5.x', 'r10.y'), ('hr6.x', 'r10.w'), ('hr7.x', 'r11.y')]
    for dst, coord in a_pairs:
        instrs.append(MOV_F32(coord, 'r16.w'))
        emit_coord_wait(instrs, a_coord_delay)
        instrs.append(ISAM_F16(dst, coord, 0))

    b_pairs = [
        ('hr8.x',  'r24.z', 'r24.w', 0), ('hr9.x',  'r12.y', 'r12.z', 1),
        ('hr10.x', 'r12.w', 'r13.x', 2), ('hr11.x', 'r13.y', 'r13.z', 3),
        ('hr12.x', 'r13.w', 'r14.x', 0), ('hr13.x', 'r14.y', 'r14.z', 1),
        ('hr14.x', 'r14.w', 'r15.x', 2), ('hr15.x', 'r15.y', 'r15.z', 3),
    ]
    for dst, coord, yreg, koff in b_pairs:
        instrs.append(MOV_F32(yreg, 'r15.w') if koff == 0 else ADD_S(yreg, 'r15.w', koff))
        emit_coord_wait(instrs, b_coord_delay)
        instrs.append(ISAM_F16(dst, coord, 1))
    if pre_mad_nops >= 0: instrs.append(NOP(rpt=pre_mad_nops))

    a_regs = [_hreg(f'hr{i}.x') for i in range(8)]
    b0_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
    b1_regs = [_hreg(f'hr{i}.x') for i in range(12, 16)]
    emit_8x8_mad_order(instrs, acc0, a_regs, b0_regs, b1_regs, order=alu_order)

    instrs += [
        ADD_S('r8.y', 'r16.w', 1),
        ADD_S('r15.w', 'r15.w', 4),
        CMPS_S_EQ('r16.w', K4 - 1, nop=1),
        MOV_F32('r16.w', 'r8.y'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    # Keep the compiler donor's 16-address calculation and stores. The donor's
    # accumulator packing is harmless and overwritten by the contiguous moves.
    instrs += donor[354:457]
    for row in range(8):
        for col in range(2):
            out = (row * 2 + col) * 4
            instrs.append(MOV_H(out, acc0 + out, rpt=3))
    instrs += donor[496:535]
    shader = assemble(instrs)
    return shader, 32, 28, loop_end - loop_start, img_sz


def build_8x8_donor4_store_shader(dev, threads, a_coord_delay=4, b_coord_delay=4, alu_order='row_col_kk'):
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8(2, threads))
    instrs = prologue_8x4(dev, threads)
    emit_col_stride(instrs, 2)
    instrs += [MOV_S32('r6.y', 3, sy=True), MOV_S32('r6.z', 0)]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    a_regs = emit_a_loads(instrs, a_coord_delay)
    # A loads reuse one coordinate register. Sync before reusing those full regs
    # for B coordinates; hr15 is overwritten by B loads before being consumed.
    instrs.append(MAD_F16(_hreg('hr15.x'), a_regs[0], a_regs[0], _hreg('hr15.x'), rpt=3, sy=True, r=True))
    b_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
    for col in range(2):
        emit_b_group(instrs, col, ['hr8.x', 'hr9.x', 'hr10.x', 'hr11.x'], b_coord_delay)
        first = True
        def emit(row, kk):
            nonlocal first
            base = acc0 + (row * 2 + col) * 4
            instrs.append(MAD_F16(base, a_regs[row] + kk, b_regs[kk], base, rpt=3, sy=first, r=True))
            first = False
        if alu_order in ('row_col_kk', 'row_kk_col'):
            for row in range(8):
                for kk in range(4): emit(row, kk)
        else:
            for kk in range(4):
                for row in range(8): emit(row, kk)
    emit_loop_control(instrs)
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    emit_dynamic4_stores(instrs, acc0, 2)
    instrs += [ADD_S('r7.x', 'r7.x', 4), ADD_S('r7.y', 'r7.y', -32), NOP(rpt=16)]
    emit_dynamic4_stores(instrs, acc0 + 4 * 2 * 4, 2)
    instrs.append(END())
    return assemble(instrs), 32, 8, loop_end - loop_start, img_sz


def build_8x16_serial_donor_store_shader(dev, threads, a_coord_delay=4, b_coord_delay=4, alu_order='kk_row_col'):
    lib1, img_off1, _, _ = get_envelope(dev, make_donor_src8(1, threads))
    _, _, img_sz, _ = get_envelope(dev, make_donor_src8(4, threads))
    donor = [bytes(lib1[img_off1+i:img_off1+i+8]) for i in range(0, img_sz, 8)]

    instrs = donor[:21]
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2)]
    for _ in range(3):
        instrs.append(ADD_S_REG('r11.w', 'r11.w', 'r6.w'))
        instrs.append(NOP(rpt=2))
    instrs += [MOV_F32('r8.w', 'r11.w'), MOV_F32('r9.y', 'r11.w'), MOV_F32('r9.w', 'r11.w'), MOV_F32('r10.y', 'r11.w')]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 32 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    a_pairs = [('hr0.x', 'r4.w'), ('hr1.x', 'r5.y'), ('hr2.x', 'r5.w'), ('hr3.x', 'r6.y'),
               ('hr4.x', 'r6.w'), ('hr5.x', 'r7.y'), ('hr6.x', 'r7.w'), ('hr7.x', 'r8.y')]
    for dst, coord in a_pairs:
        instrs.append(MOV_F32(coord, 'r11.x'))
        emit_coord_wait(instrs, a_coord_delay)
        instrs.append(ISAM_F16(dst, coord, 0))

    a_regs = [_hreg(f'hr{i}.x') for i in range(8)]
    b_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
    b_coord_pairs = [('hr8.x', 'r8.w', 'r9.x', -3), ('hr9.x', 'r9.y', 'r9.z', -2),
                     ('hr10.x', 'r9.w', 'r10.x', -1), ('hr11.x', 'r10.y', 'r10.z', 0)]
    first = True
    for col in range(4):
        if col:
            instrs.append(ADD_S('r0.y', 'r11.w', col * 32))
            instrs += [MOV_F32('r8.w', 'r0.y'), MOV_F32('r9.y', 'r0.y'), MOV_F32('r9.w', 'r0.y'), MOV_F32('r10.y', 'r0.y')]
        for dst, coord, yreg, koff in b_coord_pairs:
            instrs.append(MOV_F32(yreg, 'r10.w') if koff == 0 else ADD_S(yreg, 'r10.w', koff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(dst, coord, 1))
        def emit(row, kk):
            nonlocal first
            base = acc0 + (row * 4 + col) * 4
            instrs.append(MAD_F16(base, a_regs[row] + kk, b_regs[kk], base, rpt=3, sy=first, r=True))
            first = False
        if alu_order in ('kk_row_col', 'kk_col_row', 'col_kk_row'):
            for kk in range(4):
                for row in range(8): emit(row, kk)
        elif alu_order in ('row_kk_col', 'row_col_kk'):
            for row in range(8):
                for kk in range(4): emit(row, kk)
        else: raise ValueError('unknown ALU order %s' % alu_order)

    instrs += [
        ADD_S('r0.x', 'r11.x', 1),
        ADD_S('r10.w', 'r10.w', 4),
        CMPS_S_EQ('r11.x', K4 - 1, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    for col in range(4):
        if col:
            instrs.append(ADD_S('r11.w', 'r11.w', 32))
            emit_coord_wait(instrs, b_coord_delay)
        instrs += donor[212:262]
        for row in range(8): instrs.append(MOV_H(row * 4, acc0 + (row * 4 + col) * 4, rpt=3))
        instrs += donor[292:315] if col == 3 else donor[292:314]
    shader = assemble(instrs)
    return shader, 48, 15, loop_end - loop_start, img_sz


def emit_8x8_mad_order(instrs, acc0, a_regs, b0_regs, b1_regs, profile_r1=False, order='kk_row_col'):
    first = True
    def emit(row, kk, col):
        nonlocal first
        base = acc0 + row * 8 + col * 4
        b_regs = b0_regs if col == 0 else b1_regs
        instrs.append(MAD_F16(base, a_regs[row] + kk, b_regs[kk], base, rpt=3, sy=first, r=True, r1=profile_r1))
        first = False
    if order == 'kk_row_col':
        for kk in range(4):
            for row in range(8):
                emit(row, kk, 0); emit(row, kk, 1)
    elif order == 'kk_col_row':
        for kk in range(4):
            for col in range(2):
                for row in range(8): emit(row, kk, col)
    elif order == 'col_kk_row':
        for col in range(2):
            for kk in range(4):
                for row in range(8): emit(row, kk, col)
    elif order == 'row_kk_col':
        for row in range(8):
            for kk in range(4):
                emit(row, kk, 0); emit(row, kk, 1)
    elif order == 'row_col_kk':
        for row in range(8):
            for col in range(2):
                for kk in range(4): emit(row, kk, col)
    else: raise ValueError('unknown ALU order %s' % order)


def build_8x8_profile_shader(dev, threads, mode, a_coord_delay=4, b_coord_delay=4, profile_r1=False, alu_order='kk_row_col'):
    if mode not in ('alu', 'isam'): raise ValueError('unknown profile mode %s' % mode)
    lib, img_off, img_sz, _ = get_envelope(dev, make_donor_src8(2, threads))
    donor = [bytes(lib[img_off+i:img_off+i+8]) for i in range(0, img_sz, 8)]
    instrs = donor[:29]

    acc0 = _hreg('hr16.x')
    if mode == 'alu':
        instrs += [MOV_H_IMM(0, 0x3c00), MOV_H(1, 0, rpt=2)]
        for base in range(4, 16 * 4, 4): instrs.append(MOV_H(base, 0, rpt=3))
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    a_regs = [_hreg(f'hr{i}.x') for i in range(8)]
    b0_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
    b1_regs = [_hreg(f'hr{i}.x') for i in range(12, 16)]
    if mode == 'isam':
        a_pairs = [('hr0.x', 'r7.w'), ('hr1.x', 'r8.y'), ('hr2.x', 'r8.w'), ('hr3.x', 'r9.y'),
                   ('hr4.x', 'r9.w'), ('hr5.x', 'r10.y'), ('hr6.x', 'r10.w'), ('hr7.x', 'r11.y')]
        for dst, coord in a_pairs:
            instrs.append(MOV_F32(coord, 'r16.w'))
            emit_coord_wait(instrs, a_coord_delay)
            instrs.append(ISAM_F16(dst, coord, 0))
        b_pairs = [
            ('hr8.x',  'r24.z', 'r24.w', 0), ('hr9.x',  'r12.y', 'r12.z', 1),
            ('hr10.x', 'r12.w', 'r13.x', 2), ('hr11.x', 'r13.y', 'r13.z', 3),
            ('hr12.x', 'r13.w', 'r14.x', 0), ('hr13.x', 'r14.y', 'r14.z', 1),
            ('hr14.x', 'r14.w', 'r15.x', 2), ('hr15.x', 'r15.y', 'r15.z', 3),
        ]
        for dst, coord, yreg, koff in b_pairs:
            instrs.append(MOV_F32(yreg, 'r15.w') if koff == 0 else ADD_S(yreg, 'r15.w', koff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(dst, coord, 1))
    else:
        emit_8x8_mad_order(instrs, acc0, a_regs, b0_regs, b1_regs, profile_r1, alu_order)

    instrs += [
        ADD_S('r8.y', 'r16.w', 1),
        ADD_S('r15.w', 'r15.w', 4),
        CMPS_S_EQ('r16.w', K4 - 1, nop=1),
        MOV_F32('r16.w', 'r8.y'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs += [BR(loop_start - loop_end), END()]
    shader = assemble(instrs)
    return shader, 32, 28, loop_end - loop_start, img_sz


def build_8x8_twopass_store_shader(dev, threads, a_coord_delay=4, b_coord_delay=4, pre_mad_nops=0, overlap_b1=False):
    lib1, img_off1, _, _ = get_envelope(dev, make_donor_src8(1, threads))
    _, _, img_sz, _ = get_envelope(dev, make_donor_src8(2, threads))
    donor = [bytes(lib1[img_off1+i:img_off1+i+8]) for i in range(0, img_sz, 8)]

    instrs = donor[:21]
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2), ADD_S_REG('r11.w', 'r11.w', 'r6.w')]
    instrs += [MOV_F32('r8.w', 'r11.w'), MOV_F32('r9.y', 'r11.w'), MOV_F32('r9.w', 'r11.w'), MOV_F32('r10.y', 'r11.w')]
    instrs += [ADD_S('r14.x', 'r11.w', 32), NOP(rpt=4), MOV_F32('r14.z', 'r14.x'), MOV_F32('r13.x', 'r14.x'), MOV_F32('r13.z', 'r14.x')]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    a_pairs = [('hr0.x', 'r4.w'), ('hr1.x', 'r5.y'), ('hr2.x', 'r5.w'), ('hr3.x', 'r6.y'),
               ('hr4.x', 'r6.w'), ('hr5.x', 'r7.y'), ('hr6.x', 'r7.w'), ('hr7.x', 'r8.y')]
    for dst, coord in a_pairs:
        instrs.append(MOV_F32(coord, 'r11.x'))
        emit_coord_wait(instrs, a_coord_delay)
        instrs.append(ISAM_F16(dst, coord, 0))

    b_pairs = [
        ('hr8.x',  'r8.w',  'r9.x',  -3), ('hr9.x',  'r9.y',  'r9.z',  -2),
        ('hr10.x', 'r9.w',  'r10.x', -1), ('hr11.x', 'r10.y', 'r10.z', 0),
        ('hr12.x', 'r14.x', 'r14.y', -3), ('hr13.x', 'r14.z', 'r14.w', -2),
        ('hr14.x', 'r13.x', 'r13.y', -1), ('hr15.x', 'r13.z', 'r13.w', 0),
    ]
    def emit_b_loads(pairs):
        for dst, coord, yreg, koff in pairs:
            instrs.append(MOV_F32(yreg, 'r10.w') if koff == 0 else ADD_S(yreg, 'r10.w', koff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(dst, coord, 1))
    if overlap_b1: emit_b_loads(b_pairs[:4])
    else:
        emit_b_loads(b_pairs)
        if pre_mad_nops >= 0: instrs.append(NOP(rpt=pre_mad_nops))

    a_regs = [_hreg(f'hr{i}.x') for i in range(8)]
    b0_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
    b1_regs = [_hreg(f'hr{i}.x') for i in range(12, 16)]
    if overlap_b1:
        instrs.append(MAD_F16(acc0, a_regs[0], b0_regs[0], acc0, rpt=3, sy=True, r=True))
        emit_b_loads(b_pairs[4:])
        for kk in range(4):
            for row in range(8):
                if kk == 0 and row == 0: continue
                base = acc0 + row * 8
                instrs.append(MAD_F16(base, a_regs[row] + kk, b0_regs[kk], base, rpt=3, r=True))
        first_col1 = True
        for kk in range(4):
            for row in range(8):
                base = acc0 + row * 8 + 4
                instrs.append(MAD_F16(base, a_regs[row] + kk, b1_regs[kk], base, rpt=3, sy=first_col1, r=True))
                first_col1 = False
    else:
        first = True
        for kk in range(4):
            for row in range(8):
                base = acc0 + row * 8
                instrs.append(MAD_F16(base, a_regs[row] + kk, b0_regs[kk], base, rpt=3, sy=first, r=True))
                first = False
                instrs.append(MAD_F16(base + 4, a_regs[row] + kk, b1_regs[kk], base + 4, rpt=3, r=True))

    instrs += [
        ADD_S('r0.x', 'r11.x', 1),
        ADD_S('r10.w', 'r10.w', 4),
        CMPS_S_EQ('r11.x', K4 - 1, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8, rpt=3))
    instrs += donor[292:314]
    instrs.append(ADD_S('r11.w', 'r11.w', 32))
    emit_coord_wait(instrs, b_coord_delay)
    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8 + 4, rpt=3))
    instrs += donor[292:315]
    shader = assemble(instrs)
    return shader, 32, 15, loop_end - loop_start, img_sz


def build_8x8_pipelined_shader(dev, threads, a_coord_delay=4, b_coord_delay=4, grouped_a=False, grouped_b=False, no_current_sy=False, no_next_sy=False):
    lib1, img_off1, _, _ = get_envelope(dev, make_donor_src8(1, threads))
    _, _, img_sz, _ = get_envelope(dev, make_donor_src8(2, threads))
    donor = [bytes(lib1[img_off1+i:img_off1+i+8]) for i in range(0, img_sz, 8)]

    instrs = donor[:21]
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2), ADD_S_REG('r11.w', 'r11.w', 'r6.w')]
    instrs += [MOV_F32('r8.w', 'r11.w'), MOV_F32('r9.y', 'r11.w'), MOV_F32('r9.w', 'r11.w'), MOV_F32('r10.y', 'r11.w')]
    instrs += [ADD_S('r14.x', 'r11.w', 32), NOP(rpt=4), MOV_F32('r14.z', 'r14.x'), MOV_F32('r13.x', 'r14.x'), MOV_F32('r13.z', 'r14.x')]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    a_coord_regs = ['r4.w', 'r5.y', 'r5.w', 'r6.y', 'r6.w', 'r7.y', 'r7.w', 'r8.y']
    b_coord_pairs = [('r8.w', 'r9.x', -3), ('r9.y', 'r9.z', -2), ('r9.w', 'r10.x', -1), ('r10.y', 'r10.z', 0),
                     ('r14.x', 'r14.y', -3), ('r14.z', 'r14.w', -2), ('r13.x', 'r13.y', -1), ('r13.z', 'r13.w', 0)]
    def emit_loads(a_base, b_base, kx, by):
        if grouped_a:
            for coord in a_coord_regs: instrs.append(MOV_F32(coord, kx))
            emit_coord_wait(instrs, a_coord_delay)
            for i, coord in enumerate(a_coord_regs): instrs.append(ISAM_F16(a_base + i * 4, coord, 0))
        else:
            for i, coord in enumerate(a_coord_regs):
                instrs.append(MOV_F32(coord, kx))
                emit_coord_wait(instrs, a_coord_delay)
                instrs.append(ISAM_F16(a_base + i * 4, coord, 0))
        if grouped_b:
            for _, yreg, koff in b_coord_pairs: instrs.append(MOV_F32(yreg, by) if koff == 0 else ADD_S(yreg, by, koff))
            emit_coord_wait(instrs, b_coord_delay)
            for i, (coord, _, _) in enumerate(b_coord_pairs): instrs.append(ISAM_F16(b_base + i * 4, coord, 1))
        else:
            for i, (coord, yreg, koff) in enumerate(b_coord_pairs):
                instrs.append(MOV_F32(yreg, by) if koff == 0 else ADD_S(yreg, by, koff))
                emit_coord_wait(instrs, b_coord_delay)
                instrs.append(ISAM_F16(b_base + i * 4, coord, 1))

    def emit_mads(a_base, b_base, first_sy=False, skip_first=False):
        first = first_sy
        for kk in range(4):
            for row in range(8):
                for col in range(2):
                    if skip_first and kk == 0 and row == 0 and col == 0: continue
                    base = acc0 + row * 8 + col * 4
                    instrs.append(MAD_F16(base, a_base + row * 4 + kk, b_base + col * 16 + kk * 4, base, rpt=3, sy=first, r=True))
                    first = False

    emit_loads(_hreg('hr0.x'), _hreg('hr8.x'), 'r11.x', 'r10.w')
    loop_start = len(instrs)
    instrs.append(MAD_F16(acc0, _hreg('hr0.x'), _hreg('hr8.x'), acc0, rpt=3, sy=not no_current_sy, r=True))
    instrs += [ADD_S('r0.x', 'r11.x', 1), ADD_S('r0.z', 'r10.w', 4)]
    emit_loads(_hreg('hr32.x'), _hreg('hr40.x'), 'r0.x', 'r0.z')
    emit_mads(_hreg('hr0.x'), _hreg('hr8.x'), skip_first=True)
    instrs.append(MAD_F16(acc0, _hreg('hr32.x'), _hreg('hr40.x'), acc0, rpt=3, sy=not no_next_sy, r=True))
    instrs += [ADD_S('r0.x', 'r11.x', 2), ADD_S('r0.z', 'r10.w', 8)]
    emit_loads(_hreg('hr0.x'), _hreg('hr8.x'), 'r0.x', 'r0.z')
    emit_mads(_hreg('hr32.x'), _hreg('hr40.x'), skip_first=True)

    instrs += [
        ADD_S('r0.x', 'r11.x', 2),
        ADD_S('r10.w', 'r10.w', 8),
        CMPS_S_EQ('r11.x', K4 - 2, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8, rpt=3))
    instrs += donor[292:314]
    instrs.append(ADD_S('r11.w', 'r11.w', 32))
    emit_coord_wait(instrs, b_coord_delay)
    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8 + 4, rpt=3))
    instrs += donor[292:315]
    shader = assemble(instrs)
    return shader, 48, 15, (loop_end - loop_start) // 2, img_sz


def build_8x8_pipeline4_shader(dev, threads, a_coord_delay=4, b_coord_delay=-1):
    lib1, img_off1, _, _ = get_envelope(dev, make_donor_src8(1, threads))
    _, _, img_sz, _ = get_envelope(dev, make_donor_src8(4, threads))
    donor = [bytes(lib1[img_off1+i:img_off1+i+8]) for i in range(0, img_sz, 8)]

    instrs = donor[:21]
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2), ADD_S_REG('r11.w', 'r11.w', 'r6.w')]
    instrs += [MOV_F32('r8.w', 'r11.w'), MOV_F32('r9.y', 'r11.w'), MOV_F32('r9.w', 'r11.w'), MOV_F32('r10.y', 'r11.w')]
    instrs += [ADD_S('r14.x', 'r11.w', 32), NOP(rpt=4), MOV_F32('r14.z', 'r14.x'), MOV_F32('r13.x', 'r14.x'), MOV_F32('r13.z', 'r14.x')]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    a_coord_regs = ['r4.w', 'r5.y', 'r5.w', 'r6.y', 'r6.w', 'r7.y', 'r7.w', 'r8.y']
    b_coord_pairs = [('r8.w', 'r9.x', -3), ('r9.y', 'r9.z', -2), ('r9.w', 'r10.x', -1), ('r10.y', 'r10.z', 0),
                     ('r14.x', 'r14.y', -3), ('r14.z', 'r14.w', -2), ('r13.x', 'r13.y', -1), ('r13.z', 'r13.w', 0)]
    def emit_loads(a_base, b_base, kx, by):
        for i, coord in enumerate(a_coord_regs):
            instrs.append(MOV_F32(coord, kx))
            emit_coord_wait(instrs, a_coord_delay)
            instrs.append(ISAM_F16(a_base + i * 4, coord, 0))
        for i, (coord, yreg, koff) in enumerate(b_coord_pairs):
            instrs.append(MOV_F32(yreg, by) if koff == 0 else ADD_S(yreg, by, koff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(b_base + i * 4, coord, 1))

    def emit_mads(a_base, b_base, first_sy=False, skip_first=False):
        first = first_sy
        for kk in range(4):
            for row in range(8):
                for col in range(2):
                    if skip_first and kk == 0 and row == 0 and col == 0: continue
                    base = acc0 + row * 8 + col * 4
                    instrs.append(MAD_F16(base, a_base + row * 4 + kk, b_base + col * 16 + kk * 4, base, rpt=3, sy=first, r=True))
                    first = False

    emit_loads(_hreg('hr0.x'), _hreg('hr8.x'), 'r11.x', 'r10.w')
    loop_start = len(instrs)
    buffers = [(_hreg('hr0.x'), _hreg('hr8.x')), (_hreg('hr32.x'), _hreg('hr40.x'))]
    for step in range(4):
        a_base, b_base = buffers[step & 1]
        next_a, next_b = buffers[(step + 1) & 1]
        instrs.append(MAD_F16(acc0, a_base, b_base, acc0, rpt=3, sy=(step == 0), r=True))
        instrs += [ADD_S('r0.x', 'r11.x', step + 1), ADD_S('r0.z', 'r10.w', (step + 1) * 4)]
        emit_loads(next_a, next_b, 'r0.x', 'r0.z')
        emit_mads(a_base, b_base, skip_first=True)

    instrs += [
        ADD_S('r0.x', 'r11.x', 4),
        ADD_S('r10.w', 'r10.w', 16),
        CMPS_S_EQ('r11.x', K4 - 4, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8, rpt=3))
    instrs += donor[292:314]
    instrs.append(ADD_S('r11.w', 'r11.w', 32))
    emit_coord_wait(instrs, b_coord_delay)
    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8 + 4, rpt=3))
    instrs += donor[292:315]
    shader = assemble(instrs)
    return shader, 48, 15, (loop_end - loop_start) // 4, img_sz


def build_8x8_batch2_shader(dev, threads, a_coord_delay=4, b_coord_delay=4):
    lib1, img_off1, _, _ = get_envelope(dev, make_donor_src8(1, threads))
    _, _, img_sz, _ = get_envelope(dev, make_donor_src8(2, threads))
    donor = [bytes(lib1[img_off1+i:img_off1+i+8]) for i in range(0, img_sz, 8)]

    instrs = donor[:21]
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2), ADD_S_REG('r11.w', 'r11.w', 'r6.w')]
    instrs += [MOV_F32('r8.w', 'r11.w'), MOV_F32('r9.y', 'r11.w'), MOV_F32('r9.w', 'r11.w'), MOV_F32('r10.y', 'r11.w')]
    instrs += [ADD_S('r14.x', 'r11.w', 32), NOP(rpt=4), MOV_F32('r14.z', 'r14.x'), MOV_F32('r13.x', 'r14.x'), MOV_F32('r13.z', 'r14.x')]

    acc0 = _hreg('hr16.x')
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    a_coord_regs = ['r4.w', 'r5.y', 'r5.w', 'r6.y', 'r6.w', 'r7.y', 'r7.w', 'r8.y']
    b_coord_pairs = [('r8.w', 'r9.x', -3), ('r9.y', 'r9.z', -2), ('r9.w', 'r10.x', -1), ('r10.y', 'r10.z', 0),
                     ('r14.x', 'r14.y', -3), ('r14.z', 'r14.w', -2), ('r13.x', 'r13.y', -1), ('r13.z', 'r13.w', 0)]
    def emit_loads(a_base, b_base, kx, by):
        for i, coord in enumerate(a_coord_regs):
            instrs.append(MOV_F32(coord, kx))
            emit_coord_wait(instrs, a_coord_delay)
            instrs.append(ISAM_F16(a_base + i * 4, coord, 0))
        for i, (coord, yreg, koff) in enumerate(b_coord_pairs):
            instrs.append(MOV_F32(yreg, by) if koff == 0 else ADD_S(yreg, by, koff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(b_base + i * 4, coord, 1))

    def emit_mads(a_base, b_base, first_sy=False):
        first = first_sy
        for kk in range(4):
            for row in range(8):
                for col in range(2):
                    base = acc0 + row * 8 + col * 4
                    instrs.append(MAD_F16(base, a_base + row * 4 + kk, b_base + col * 16 + kk * 4, base, rpt=3, sy=first, r=True))
                    first = False

    loop_start = len(instrs)
    emit_loads(_hreg('hr0.x'), _hreg('hr8.x'), 'r11.x', 'r10.w')
    instrs += [ADD_S('r0.x', 'r11.x', 1), ADD_S('r0.z', 'r10.w', 4)]
    emit_loads(_hreg('hr32.x'), _hreg('hr40.x'), 'r0.x', 'r0.z')
    emit_mads(_hreg('hr0.x'), _hreg('hr8.x'), first_sy=True)
    emit_mads(_hreg('hr32.x'), _hreg('hr40.x'))
    instrs += [
        ADD_S('r0.x', 'r11.x', 2),
        ADD_S('r10.w', 'r10.w', 8),
        CMPS_S_EQ('r11.x', K4 - 2, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs.append(BR(loop_start - loop_end))

    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8, rpt=3))
    instrs += donor[292:314]
    instrs.append(ADD_S('r11.w', 'r11.w', 32))
    emit_coord_wait(instrs, b_coord_delay)
    instrs += donor[212:262]
    for row in range(8): instrs.append(MOV_H(row * 4, acc0 + row * 8 + 4, rpt=3))
    instrs += donor[292:315]
    shader = assemble(instrs)
    return shader, 48, 15, (loop_end - loop_start) // 2, img_sz


def build_8x8_twopass_profile_shader(dev, threads, mode, a_coord_delay=4, b_coord_delay=4, profile_r1=False, alu_order='kk_row_col'):
    if mode not in ('alu', 'isam'): raise ValueError('unknown profile mode %s' % mode)
    lib1, img_off1, _, _ = get_envelope(dev, make_donor_src8(1, threads))
    _, _, img_sz, _ = get_envelope(dev, make_donor_src8(2, threads))
    donor = [bytes(lib1[img_off1+i:img_off1+i+8]) for i in range(0, img_sz, 8)]

    instrs = donor[:21]
    instrs += [MOV_F32('r6.w', 'r51.w'), NOP(rpt=2), SHL_B('r6.w', 'r6.w', 5), NOP(rpt=2), ADD_S_REG('r11.w', 'r11.w', 'r6.w')]
    instrs += [MOV_F32('r8.w', 'r11.w'), MOV_F32('r9.y', 'r11.w'), MOV_F32('r9.w', 'r11.w'), MOV_F32('r10.y', 'r11.w')]
    instrs += [ADD_S('r14.x', 'r11.w', 32), NOP(rpt=4), MOV_F32('r14.z', 'r14.x'), MOV_F32('r13.x', 'r14.x'), MOV_F32('r13.z', 'r14.x')]

    acc0 = _hreg('hr16.x')
    if mode == 'alu':
        instrs += [MOV_H_IMM(0, 0x3c00), MOV_H(1, 0, rpt=2)]
        for base in range(4, 16 * 4, 4): instrs.append(MOV_H(base, 0, rpt=3))
    instrs += [MOV_H_IMM(acc0, 0), MOV_H(acc0 + 1, acc0, rpt=2)]
    for base in range(acc0 + 4, acc0 + 16 * 4, 4): instrs.append(MOV_H(base, acc0, rpt=3))

    loop_start = len(instrs)
    if mode == 'isam':
        a_pairs = [('hr0.x', 'r4.w'), ('hr1.x', 'r5.y'), ('hr2.x', 'r5.w'), ('hr3.x', 'r6.y'),
                   ('hr4.x', 'r6.w'), ('hr5.x', 'r7.y'), ('hr6.x', 'r7.w'), ('hr7.x', 'r8.y')]
        for dst, coord in a_pairs:
            instrs.append(MOV_F32(coord, 'r11.x'))
            emit_coord_wait(instrs, a_coord_delay)
            instrs.append(ISAM_F16(dst, coord, 0))
        b_pairs = [
            ('hr8.x',  'r8.w',  'r9.x',  -3), ('hr9.x',  'r9.y',  'r9.z',  -2),
            ('hr10.x', 'r9.w',  'r10.x', -1), ('hr11.x', 'r10.y', 'r10.z', 0),
            ('hr12.x', 'r14.x', 'r14.y', -3), ('hr13.x', 'r14.z', 'r14.w', -2),
            ('hr14.x', 'r13.x', 'r13.y', -1), ('hr15.x', 'r13.z', 'r13.w', 0),
        ]
        for dst, coord, yreg, koff in b_pairs:
            instrs.append(MOV_F32(yreg, 'r10.w') if koff == 0 else ADD_S(yreg, 'r10.w', koff))
            emit_coord_wait(instrs, b_coord_delay)
            instrs.append(ISAM_F16(dst, coord, 1))
    else:
        a_regs = [_hreg(f'hr{i}.x') for i in range(8)]
        b0_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]
        b1_regs = [_hreg(f'hr{i}.x') for i in range(12, 16)]
        emit_8x8_mad_order(instrs, acc0, a_regs, b0_regs, b1_regs, profile_r1, alu_order)

    instrs += [
        ADD_S('r0.x', 'r11.x', 1),
        ADD_S('r10.w', 'r10.w', 4),
        CMPS_S_EQ('r11.x', K4 - 1, nop=1),
        MOV_F32('r11.x', 'r0.x'),
        NOP(rpt=3),
    ]
    loop_end = len(instrs)
    instrs += [BR(loop_start - loop_end), END()]
    shader = assemble(instrs)
    return shader, 32, 15, loop_end - loop_start, img_sz


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
    expected = struct.unpack('<H', struct.pack('<e', float(K)))[0]
    mismatches = []
    for i, v in enumerate(out):
        if v != expected:
            mismatches.append((i, v))
            if len(mismatches) >= 10: break
    if mismatches:
        print('CHECK FAIL expected=0x%04x mismatches=%s' % (expected, ', '.join('idx%d=0x%04x' % x for x in mismatches)))
        return False
    print('CHECK PASS all %d outputs are %.1f' % (len(out), float(K)))
    return True


def check_all_ones_float(c):
    out = c.copyout(memoryview(bytearray(c.nbytes))).cast('f')
    expected = float(K)
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
    expected = struct.unpack('<H', struct.pack('<e', float(K)))[0]
    rows = [0, 1, 127, 128]
    for row in rows:
        if row >= M: continue
        chunks = []
        for col in range(0, N, 32):
            vals = out[row * N + col:row * N + col + 32]
            if all(v == expected for v in vals): tag = 'ok'
            elif all(v == 0 for v in vals): tag = 'zero'
            elif all(v == vals[0] for v in vals): tag = '0x%04x' % vals[0]
            else: tag = 'mix'
            chunks.append('%d:%s' % (col, tag))
        print('row%d chunks32 %s' % (row, ' '.join(chunks)))
    for col in [128, 384, 640, 896]:
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
    expected = float(K)
    nz = [(i, v) for i, v in enumerate(out) if v != 0.0][:10]
    print('float nonzero first10 %s' % (', '.join('idx%d=%r' % x for x in nz) if nz else 'none'))
    rows = [0, 1, 127, 128]
    for row in rows:
        if row >= M: continue
        chunks = []
        for col in range(0, N, 32):
            vals = out[row * N + col:row * N + col + 32]
            if all(v == expected for v in vals): tag = 'ok'
            elif all(v == 0.0 for v in vals): tag = 'zero'
            elif all(v == vals[0] for v in vals): tag = '%r' % vals[0]
            else: tag = 'mix'
            chunks.append('%d:%s' % (col, tag))
        print('row%d chunks32 %s' % (row, ' '.join(chunks)))


def run_one(args, variant):
    dev = Device['QCOM']
    if args.fp32_accum and (variant != 'serial' or args.ncols not in (1, 2)):
        print('%s skipped: --fp32-accum currently supports only --variant serial --ncols 1 or 2.' % variant)
        return
    if not args.fp32_accum and args.ncols == 1 and variant != 'donor-store':
        print('%s skipped: scalar 8x%d uses the compiler-donor store path (--variant donor-store).' % (variant, args.ncols * 4))
        return
    if N % (128 * args.ncols) != 0:
        print('%s skipped: N=%d is not divisible by scalar tile column group %d; timing would not cover the full matrix.' % (variant, N, 128 * args.ncols))
        return
    env_ncols = args.ncols if args.fp32_accum else 8 if args.split_a and args.split_k_unroll >= 16 else 4 if ((variant == 'donor-store' and args.ncols == 2 and args.pipeline4) or (args.split_a and args.split_k_unroll >= 4)) else args.ncols
    envelope_src = make_donor_src8_fp32(env_ncols, args.threads) if args.fp32_accum else make_donor_src8(env_ncols, args.threads)
    envelope, img_off, img_sz, reg_off = get_envelope(dev, envelope_src)
    if args.fp32_accum:
        shader, hregs, fregs, loop_instrs = build_8x8_fp32_shader(dev, args.threads, args.ncols, args.b_coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.alu_order, args.direct_f32_loads, args.sampler_per_texture)
    elif variant == 'donor-store':
        if args.ncols == 1:
            if args.pipeline: shader, hregs, fregs, loop_instrs, img_sz = build_8x4_pipelined_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay, args.grouped_a, args.grouped_b)
            else: shader, hregs, fregs, loop_instrs, img_sz = build_8x4_donor_store_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay)
        elif args.ncols == 2:
            if args.profile != 'full':
                if args.experimental_twopass: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_twopass_profile_shader(dev, args.threads, args.profile, args.a_coord_delay, args.b_coord_delay, args.profile_r1, args.alu_order)
                else: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_profile_shader(dev, args.threads, args.profile, args.a_coord_delay, args.b_coord_delay, args.profile_r1, args.alu_order)
            elif args.donor4_store: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_donor4_store_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay, args.alu_order)
            elif args.pipeline4: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_pipeline4_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay)
            elif args.batch2: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_batch2_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay)
            elif args.pipeline: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_pipelined_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay, args.grouped_a, args.grouped_b, args.no_current_sy, args.no_next_sy)
            elif args.experimental_twopass: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_twopass_store_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay, args.pre_mad_nops, args.overlap_b1)
            else: shader, hregs, fregs, loop_instrs, img_sz = build_8x8_donor_store_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay, args.pre_mad_nops, args.alu_order)
        elif args.ncols == 4 and args.experimental_8x16: shader, hregs, fregs, loop_instrs, img_sz = build_8x16_serial_donor_store_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay, args.alu_order)
        else: raise ValueError('donor-store is implemented only for scalar 8x4/8x8 by default; use --experimental-8x16 for scalar 8x16')
    else:
        if args.profile == 'alu': shader, hregs, fregs, loop_instrs = build_8xn_serial_alu_profile_shader(dev, args.threads, args.ncols, args.profile_r1, args.alu_order)
        elif args.split_a and args.split_k_unroll > 1 and args.ncols == 4: shader, hregs, fregs, loop_instrs = build_8x16_split_a_unroll_shader(dev, args.threads, args.split_k_unroll, args.b_coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.add256_gap, args.split_fast_coords, args.split_alu_order, args.add256_offset_before_gap, args.add256_explicit_stores, args.add256_store_mode, args.add256_direct_sources, args.swap_grid)
        elif args.split_a and args.split_k_unroll > 1: shader, hregs, fregs, loop_instrs = build_8x8_split_a_unroll_shader(dev, args.threads, args.split_k_unroll, args.b_coord_delay, args.post_constant, args.no_store, args.skip_a_loads, args.skip_b_loads, args.add256_gap, args.grouped_b, args.grouped_b_cols, args.split_fast_coords, args.split_alu_order, args.split_stream_col1, args.split_stream_col1_sync, args.split_prefetch_next_b, args.split_add_a_rows, args.add256_offset_before_gap, args.add256_explicit_stores, args.add256_store_mode, args.add256_direct_sources, args.split_buffer_a, args.split_prefetch_next_a, args.split_interleave_next_b, args.split_hoist_b0_coord, args.split_inline_b_wait, args.split_inline_b_nop, args.split_quad_a, args.split_prefetch_loop_b, args.split_high_a, args.split_pair_b_coords, args.split_base_b_y, args.split_low_a, args.split_stream_next_b1, args.split_stream_next_b0, args.swap_grid, args.pre_mad_nops, args.split_quad_map)
        elif args.split_a: shader, hregs, fregs, loop_instrs = build_8x8_split_a_shader(dev, args.threads, args.a_coord_delay, args.b_coord_delay, args.post_constant, args.grouped_b, args.grouped_b_cols, args.donor8_add256_store, args.pre_mad_nops, not args.no_next_sy, args.no_store, args.skip_a_loads, args.skip_b_loads, args.add256_gap, args.add256_offset_before_gap, args.add256_explicit_stores, args.add256_store_mode, args.add256_direct_sources)
        else: shader, hregs, fregs, loop_instrs = build_8x4_shader(dev, args.threads, variant, args.ncols, args.serial_syncs, args.a_coord_delay, args.b_coord_delay, args.post_constant, args.donor8_store, args.first_a_wait_only, args.donor8_add256_store, args.add256_gap, args.add256_offset_before_gap, args.add256_explicit_stores, args.add256_store_mode, args.add256_direct_sources)
    if args.fregs_override >= 0: fregs = args.fregs_override
    if args.hregs_override >= 0: hregs = args.hregs_override
    if args.strip_mad_sy:
        patched = bytearray(shader)
        for off in range(0, len(patched), 8):
            lo, hi = struct.unpack_from('<II', patched, off)
            if (hi >> 24) == 0x73: struct.pack_into('<II', patched, off, lo, (hi & 0x00ffffff) | 0x63000000)
        shader = bytes(patched)
    if hregs > 48 and not args.allow_invalid_regs:
        print('%s skipped: uses %d half4 registers, but A630 addressable GPR half registers stop at hr47 (hregs=48).' % (variant, hregs))
        return
    if len(shader) > img_sz:
        print('%s skipped: shader is %d bytes but envelope has only %d bytes.' % (variant, len(shader), img_sz))
        return
    lib = inject(envelope, img_off, img_sz, reg_off, shader, fregs=fregs, hregs=hregs)
    asm = disasm(shader)
    reg_count = fregs + (hregs + 1) // 2
    wave_pairs = 96 // reg_count
    tex_bytes = (8 + 4 * args.ncols) * 8
    flops_per_thread_k = 8 * args.ncols * 4 * 4 * 2
    density = (8 * args.ncols * 4 * 4) / loop_instrs
    tag = variant + (':fp32' if args.fp32_accum else '' if args.profile == 'full' else ':' + args.profile)
    print('%s ncols=%d scalar_tile=8x%d threads=%d fregs=%d hregs=%d reg_count=%d wave_pairs=%d intensity=%.2f flop/B mad_density=%.2f shader_instrs=%d loop_instrs=%d bytes=%d envelope_bytes=%d' % (
        tag, args.ncols, args.ncols * 4, args.threads, fregs, hregs, reg_count, wave_pairs, flops_per_thread_k / tex_bytes, density, len(shader)//8, loop_instrs, len(shader), img_sz))
    print('mad.f16=%d mad.f32=%d rpt3=%d isam=%d sy=%d serial_syncs=%s' % (
        asm.count('mad.f16'), asm.count('mad.f32'), asm.count('(rpt3)mad.'), asm.count('isam'), asm.count('(sy)'), args.serial_syncs))
    if args.disasm: print(asm)

    a_img, b_img = dtypes.imageh((M, K//4)), dtypes.imageh((K, N//4))
    c_dtype = dtypes.float if args.fp32_accum else dtypes.half
    a, b, c = make_bufs(dev, c_dtype)
    if args.check:
        fill_half(a, 0x3c00)
        fill_half(b, 0x3c00)
    prg = dev.runtime('gemm_h', lib, [[(0, a_img)], [(1, b_img)], [(2, c_dtype.ptr())]])
    tile_m = (args.threads // 32) * 8
    gs = (M // tile_m, N // (128 * args.ncols), 1) if args.swap_grid else (N // (128 * args.ncols), M // tile_m, 1)
    ls = (args.threads, 1, 1)
    if args.check:
        prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if args.scan_output:
            if args.fp32_accum: scan_failure_pattern_float(c)
            else: scan_failure_pattern(c)
        if args.fp32_accum: check_all_ones_float(c)
        else: check_all_ones(c)
        return
    for _ in range(args.warmup): prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
    times = []
    for _ in range(args.iters):
        t = prg(a._buf, b._buf, c._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)
    best = min(times)
    print('%s %.1f GFLOPS  (%.3f ms)' % (tag, 2*M*N*K / best / 1e9, best * 1e3))


def run(args):
    variants = ['serial'] if args.fp32_accum else ['donor-store'] if args.variant == 'both' and args.ncols in (1, 2) else ['serial', 'preload'] if args.variant == 'both' else [args.variant]
    for variant in variants: run_one(args, variant)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('serial', 'preload', 'donor-store', 'both'), default='both')
    parser.add_argument('--ncols', type=int, choices=(1, 2, 3, 4), default=2, help='number of col4 vectors: 1 means scalar 8x4, 2 means scalar 8x8')
    parser.add_argument('--serial-syncs', choices=('all', 'first'), default='all', help='first is an unsafe throughput probe')
    parser.add_argument('--a-coord-delay', type=int, default=4, help='donor-store A coordinate wait: -1 means no nop, 0 means nop, 4 means rpt4 nop')
    parser.add_argument('--b-coord-delay', type=int, default=4, help='donor-store B coordinate wait: -1 means no nop, 0 means nop, 4 means rpt4 nop')
    parser.add_argument('--threads', type=int, choices=(64, 128, 256), default=128)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--allow-invalid-regs', action='store_true')
    parser.add_argument('--fregs-override', type=int, default=-1, help='debug/profile only: override full register footprint metadata')
    parser.add_argument('--hregs-override', type=int, default=-1, help='debug/profile only: override half register footprint metadata')
    parser.add_argument('--swap-grid', action='store_true', help='experimental: map rows from gid.x and columns from gid.y to change cache-reuse order')
    parser.add_argument('--strip-mad-sy', action='store_true', help='debug/profile only: remove (sy) from mad.f16 instructions')
    parser.add_argument('--pre-mad-nops', type=int, default=-1, help='insert NOP(rpt=N) after isam loads before the MAD block')
    parser.add_argument('--post-constant', action='store_true', help='diagnostic: overwrite accumulators with K before stores')
    parser.add_argument('--donor8-store', action='store_true', help='serial/preload diagnostic: use known-correct 8-row donor store epilogue')
    parser.add_argument('--donor8-add256-store', action='store_true', help='experimental 8x8: low-freg donor store, second column via +256 byte address offset')
    parser.add_argument('--add256-gap', type=int, default=16, help='donor8-add256-store: NOP repeat before second-column address offset')
    parser.add_argument('--add256-offset-before-gap', action='store_true', help='donor8-add256-store: preload +256 scalar before the inter-column gap')
    parser.add_argument('--add256-explicit-stores', action='store_true', help='donor8-add256-store: emit explicit stg/nop stores instead of donor store slices')
    parser.add_argument('--add256-store-mode', choices=('donor', 'pairs', 'tight'), default='donor', help='donor8-add256-store: generated store nop pattern')
    parser.add_argument('--add256-direct-sources', action='store_true', help='donor8-add256-store: store directly from accumulator hregs without packing through hr0-hr7')
    parser.add_argument('--first-a-wait-only', action='store_true', help='serial diagnostic: apply A coordinate wait only before the first A load')
    parser.add_argument('--fp32-accum', action='store_true', help='experimental: load imageh into fp32 regs, accumulate with mad.f32, and write float C')
    parser.add_argument('--direct-f32-loads', action='store_true', help='FP32 accumulator path: use direct isam.f32 loads from imageh')
    parser.add_argument('--sampler-per-texture', action='store_true', help='FP32 direct loads: use sampler index equal to texture index')
    parser.add_argument('--split-a', action='store_true', help='experimental 8x8: preload both B groups and compute two 4-row A groups')
    parser.add_argument('--no-store', action='store_true', help='split-a profile only: omit output stores')
    parser.add_argument('--skip-a-loads', action='store_true', help='split-a profile only: initialize A regs once and skip A texture loads')
    parser.add_argument('--skip-b-loads', action='store_true', help='split-a profile only: initialize B regs once and skip B texture loads')
    parser.add_argument('--grouped-b-cols', action='store_true', help='split-a experimental: group B coordinates and loads one output column at a time')
    parser.add_argument('--split-k-unroll', type=int, choices=(1, 2, 4, 8, 16), default=1, help='split-a experimental: semantically unroll K loop')
    parser.add_argument('--split-fast-coords', action='store_true', help='split-a K-unroll: precompute col1 B X and reuse A X setup across row groups')
    parser.add_argument('--split-alu-order', choices=('kk_row_col', 'kk_col_row', 'col_kk_row', 'row_kk_col', 'row_col_kk'), default='row_col_kk', help='split-a K-unroll MAD order')
    parser.add_argument('--split-stream-col1', action='store_true', help='split-a K-unroll: issue B col1 loads after starting col0 MADs')
    parser.add_argument('--split-stream-col1-sync', action='store_true', help='split-stream-col1: sync before first streamed col1 MAD')
    parser.add_argument('--split-prefetch-next-b', action='store_true', help='split-a K-unroll: refill dead B registers for the next unrolled K step')
    parser.add_argument('--split-add-a-rows', action='store_true', help='split-a K-unroll: form A row coordinates with add.s instead of or.b')
    parser.add_argument('--split-buffer-a', action='store_true', help='split-a K-unroll: use hr28-hr31 to overlap row-group-4 A loads')
    parser.add_argument('--split-prefetch-next-a', action='store_true', help='split-a K-unroll: prefetch next K step row-group-0 A at the end of current step')
    parser.add_argument('--split-interleave-next-b', action='store_true', help='split-a K-unroll: interleave next-B col0 refill with row-group-4 col1 MADs')
    parser.add_argument('--split-hoist-b0-coord', action='store_true', help='split-a K-unroll: hoist first next-B col0 coordinate setup before group-4 col0 MADs')
    parser.add_argument('--split-inline-b-wait', action='store_true', help='split-a K-unroll: encode b-coordinate wait in the coordinate add instruction for delay 0')
    parser.add_argument('--split-inline-b-nop', type=int, default=1, help='split-inline-b-wait: nop count encoded in b-coordinate add')
    parser.add_argument('--split-quad-a', action='store_true', help='split-a K-unroll: one A row load per quad lane plus quad broadcasts')
    parser.add_argument('--split-quad-map', default='0123', help='split-quad-a: qbc source index for local rows 0..3')
    parser.add_argument('--split-prefetch-loop-b', action='store_true', help='split-a K-unroll: prefetch B across outer unrolled-loop boundaries')
    parser.add_argument('--split-high-a', action='store_true', help='split-a K-unroll: place A regs at hr24-hr27 and accumulators at hr8-hr23')
    parser.add_argument('--split-pair-b-coords', action='store_true', help='split-a K-unroll: build two B coordinates per wait and issue paired isams')
    parser.add_argument('--split-base-b-y', action='store_true', help='split-a K-unroll: keep B y as base multiple of 4 and form kk offsets with or.b')
    parser.add_argument('--split-low-a', action='store_true', help='split-a K-unroll: place A regs at hr0-hr3 and B regs at hr4-hr11')
    parser.add_argument('--split-stream-next-b1', action='store_true', help='split-a K-unroll: stream next B1 components as each current B1 component becomes dead')
    parser.add_argument('--split-stream-next-b0', action='store_true', help='split-a K-unroll: stream next B0 components as each current B0 component becomes dead')
    parser.add_argument('--overlap-b1', action='store_true', help='experimental low-register 8x8: issue second B group while computing first column group')
    parser.add_argument('--pipeline', action='store_true', help='experimental 8x8: double-buffer A/B loads across K iterations')
    parser.add_argument('--pipeline4', action='store_true', help='experimental 8x8: double-buffered pipeline unrolled across four K4 iterations')
    parser.add_argument('--grouped-a', action='store_true', help='pipeline only: batch A coord moves, wait once, then issue A loads')
    parser.add_argument('--grouped-b', action='store_true', help='pipeline/split-a: batch B coord updates, wait once, then issue B loads')
    parser.add_argument('--no-current-sy', action='store_true', help='debug/profile only: omit first current-buffer pipeline MAD sync')
    parser.add_argument('--no-next-sy', action='store_true', help='debug/profile only: omit first next-buffer pipeline MAD sync')
    parser.add_argument('--batch2', action='store_true', help='experimental 8x8: load two K steps, one sync, then compute both')
    parser.add_argument('--profile', choices=('full', 'alu', 'isam'), default='full', help='profile only one part of scalar 8x8 loop')
    parser.add_argument('--profile-r1', action='store_true', help='ALU profile only: auto-increment mad.f16 source1 to measure vector-vector issue rate')
    parser.add_argument('--alu-order', choices=('kk_row_col', 'kk_col_row', 'col_kk_row', 'row_kk_col', 'row_col_kk'), default='kk_row_col')
    parser.add_argument('--experimental-twopass', action='store_true', help='use unstable low-register scalar 8x8 two-pass store path')
    parser.add_argument('--donor4-store', action='store_true', help='experimental low-register 8x8: store as two 4-row donor chunks')
    parser.add_argument('--experimental-8x16', action='store_true', help='use experimental scalar 8x16 donor-store path')
    parser.add_argument('--check', action='store_true', help='fill A/B with ones and verify every C output equals K')
    parser.add_argument('--scan-output', action='store_true', help='print a compact output chunk pattern during --check')
    parser.add_argument('--disasm', action='store_true')
    run(parser.parse_args())
