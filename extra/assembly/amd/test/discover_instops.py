#!/usr/bin/env python3
"""SQTT InstOp discovery tool - finds instruction opcodes by running different instructions.

Requires profiling enabled:
  echo 'profile_standard' | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level

Run with: DEBUG=1 python extra/assembly/amd/test/discover_instops.py
For full traces: DEBUG=2 python extra/assembly/amd/test/discover_instops.py
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_LIMIT_SE"] = "2"  # Force work to traced SE only
os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"  # Exclude WAVERDY, REG, EVENT, UTILCTR, WAVEALLOC, PERF

from tinygrad.helpers import DEBUG, colored
from tinygrad.runtime.ops_amd import SQTT_SIMD_SEL

from extra.assembly.amd.autogen.rdna3.ins import (
  # VALU - basic (these are safe, just register ops)
  v_mov_b32_e32, v_add_f32_e32, v_mul_f32_e32,
  v_and_b32_e32, v_or_b32_e32, v_xor_b32_e32,
  v_lshlrev_b32_e32, v_lshrrev_b32_e32,
  # VALU - transcendental
  v_exp_f32_e32, v_log_f32_e32, v_rcp_f32_e32, v_sqrt_f32_e32,
  v_sin_f32_e32, v_cos_f32_e32,
  # VALU - 64-bit
  v_lshlrev_b64, v_lshrrev_b64, v_ashrrev_i64,
  v_add_f64, v_mul_f64, v_max_f64, v_min_f64,
  v_fma_f64,
  # VALU - 64-bit transcendental
  v_rcp_f64_e32, v_rsq_f64_e32, v_sqrt_f64_e32,
  v_trunc_f64_e32, v_ceil_f64_e32, v_floor_f64_e32, v_fract_f64_e32,
  v_frexp_exp_i32_f64_e32, v_frexp_mant_f64_e32,
  # VALU - div helpers
  v_div_fixup_f32, v_div_fixup_f64, v_div_fmas_f32, v_div_fmas_f64, v_div_scale_f32,
  # VALU - MAD64
  v_mad_u64_u32, v_mad_i64_i32,
  # VALU - compare (writes to VCC, safe)
  v_cmp_eq_u32_e32,
  # VALU - cmpx (modifies EXEC) - various types
  v_cmpx_eq_u32_e32, v_cmpx_lt_u32_e32, v_cmpx_gt_u32_e32,
  v_cmpx_eq_f32_e32, v_cmpx_lt_f32_e32,
  v_cmpx_eq_i32_e32,
  v_cmpx_class_f32_e32,
  # VALU - readlane/writelane
  v_readlane_b32, v_writelane_b32,
  v_readfirstlane_b32_e32,
  # SALU - basic (safe, just register ops)
  s_mov_b32, s_add_u32, s_and_b32, s_or_b32,
  s_lshl_b32, s_lshr_b32,
  s_nop, s_endpgm, s_waitcnt,
  # SALU - float
  s_ceil_f32, s_floor_f32, s_trunc_f32,
  # SALU - branch (safe if offset is 0 = next instruction)
  s_branch, s_cbranch_scc0, s_cbranch_execz, s_cbranch_execnz,
  # SALU - message
  s_sendmsg,
  # SALU - bit manipulation
  s_brev_b32, s_bcnt1_i32_b32, s_ctz_i32_b32, s_clz_i32_u32,
  # SALU - saveexec (modifies EXEC)
  s_and_saveexec_b32, s_or_saveexec_b32, s_xor_saveexec_b32,
  # SMEM - scalar memory (load from kernarg pointer in s[0:1])
  s_load_b32, s_load_b64,
  # GLOBAL - global memory (load/store) - various widths
  global_load_u8, global_load_u16, global_load_b32, global_load_b64, global_load_b96, global_load_b128,
  global_store_b8, global_store_b16, global_store_b32, global_store_b64, global_store_b96, global_store_b128,
  # GLOBAL - atomics
  global_atomic_add_u32, global_atomic_add_u64,
  # FLAT - flat memory access
  flat_load_b32, flat_load_b64, flat_load_b96, flat_load_b128,
  flat_store_b8, flat_store_b16, flat_store_b32, flat_store_b64, flat_store_b96, flat_store_b128,
  # LDS - local data share - various widths
  ds_load_b32, ds_load_b64, ds_load_b128,
  ds_store_b32, ds_store_b64, ds_store_b128,
  # LDS - atomics
  ds_add_u32, ds_max_u32, ds_min_u32,
  # VOP3P - packed
  v_pk_add_f16, v_pk_mul_f16, v_pk_fma_f16, v_pk_add_i16,
  # VOP3 - misc
  v_bfe_u32, v_bfi_b32, v_alignbit_b32, v_fma_f32,
  v_add3_u32, v_xad_u32, v_lshl_or_b32, v_add_nc_u32_e32,
  # VOP3 - carry-out
  v_add_co_u32, v_add_co_ci_u32_e32,
  # VOPD - dual issue
  v_dual_add_f32, v_dual_mul_f32,
  # VOP2 - fmac
  v_fmac_f32_e32,
  # DOT
  v_dot2_f16_f16,
  # WMMA
  v_wmma_f32_16x16x16_f16, v_wmma_f16_16x16x16_f16, v_wmma_i32_16x16x16_iu8,
  # SrcEnum for NULL soffset
  SrcEnum,
)
from extra.assembly.amd.dsl import v, s
from extra.assembly.amd.sqtt import InstOp, INST, WAVESTART, WAVEEND, ALUEXEC, VMEMEXEC

from extra.assembly.amd.test.test_sqtt_hw import (
  run_asm_sqtt, decode_all_blobs, get_inst_ops, print_blobs, get_wave_packets, format_packet, PACKET_COLORS
)

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION TEST CASES - only safe instructions that don't access memory
# ═══════════════════════════════════════════════════════════════════════════════

# Helper: load buffer address from kernarg (s[0:1] -> s[2:3])
# The runtime passes kernarg pointer in s[0:1], kernarg contains buffer address
def _load_buf_addr():
  return [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),  # wait for SMEM load
  ]

INSTRUCTION_TESTS: dict[str, tuple[str, list]] = {
  # SALU (0x0) - scalar ALU, just register operations
  "SALU_mov": ("s_mov_b32", [s_mov_b32(s[4], 0), s_mov_b32(s[5], 1)]),
  "SALU_add": ("s_add_u32", [s_mov_b32(s[4], 1), s_mov_b32(s[5], 2), s_add_u32(s[6], s[4], s[5])]),
  "SALU_logic": ("s_and/or", [s_and_b32(s[6], s[4], s[5]), s_or_b32(s[7], s[4], s[5])]),
  "SALU_shift": ("s_lshl/lshr", [s_lshl_b32(s[6], s[4], 1), s_lshr_b32(s[7], s[4], 1)]),
  "SALU_nop": ("s_nop", [s_nop(0)]),

  # JUMP (0x3) - branch taken
  "JUMP_branch": ("s_branch", [s_branch(0)]),
  "JUMP_cbranch_execnz": ("s_cbranch_execnz", [s_cbranch_execnz(0)]),  # EXEC != 0, branch taken

  # JUMP_NO (0x4) - branch not taken
  "JUMP_NO_cbranch_execz": ("s_cbranch_execz", [s_cbranch_execz(0)]),  # EXEC != 0, branch not taken

  # VALU (0xb) - vector ALU, just register operations
  "VALU_mov": ("v_mov_b32", [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 1.0)]),
  "VALU_add": ("v_add_f32", [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[1], 2.0), v_add_f32_e32(v[2], v[0], v[1])]),
  "VALU_mul": ("v_mul_f32", [v_mul_f32_e32(v[2], v[0], v[1])]),
  "VALU_logic": ("v_and/or/xor", [v_and_b32_e32(v[2], v[0], v[1]), v_or_b32_e32(v[3], v[0], v[1]), v_xor_b32_e32(v[4], v[0], v[1])]),
  "VALU_shift": ("v_lshl/lshr", [v_lshlrev_b32_e32(v[2], 1, v[0]), v_lshrrev_b32_e32(v[3], 1, v[0])]),

  # VALU transcendental - still just register ops
  "VALU_exp": ("v_exp_f32", [v_mov_b32_e32(v[0], 1.0), v_exp_f32_e32(v[1], v[0])]),
  "VALU_log": ("v_log_f32", [v_mov_b32_e32(v[0], 1.0), v_log_f32_e32(v[1], v[0])]),
  "VALU_rcp": ("v_rcp_f32", [v_mov_b32_e32(v[0], 1.0), v_rcp_f32_e32(v[1], v[0])]),
  "VALU_sqrt": ("v_sqrt_f32", [v_mov_b32_e32(v[0], 1.0), v_sqrt_f32_e32(v[1], v[0])]),

  # VALU 64-bit shift (0xd)
  "VALU64_lshl": ("v_lshlrev_b64", [v_lshlrev_b64(v[0:1], 1, v[2:3])]),
  "VALU64_lshr": ("v_lshrrev_b64", [v_lshrrev_b64(v[0:1], 1, v[2:3])]),
  "VALU64_ashr": ("v_ashrrev_i64", [v_ashrrev_i64(v[0:1], 1, v[2:3])]),

  # VALU 64-bit arithmetic
  "VALU64_add": ("v_add_f64", [v_add_f64(v[0:1], v[2:3], v[4:5])]),
  "VALU64_mul": ("v_mul_f64", [v_mul_f64(v[0:1], v[2:3], v[4:5])]),
  "VALU64_max": ("v_max_f64", [v_max_f64(v[0:1], v[2:3], v[4:5])]),
  "VALU64_min": ("v_min_f64", [v_min_f64(v[0:1], v[2:3], v[4:5])]),
  "VALU64_fma": ("v_fma_f64", [v_fma_f64(v[0:1], v[2:3], v[4:5], v[6:7])]),

  # VALU 64-bit transcendental
  "VALU64_rcp": ("v_rcp_f64", [v_rcp_f64_e32(v[0:1], v[2:3])]),
  "VALU64_rsq": ("v_rsq_f64", [v_rsq_f64_e32(v[0:1], v[2:3])]),
  "VALU64_sqrt": ("v_sqrt_f64", [v_sqrt_f64_e32(v[0:1], v[2:3])]),

  # VALU 64-bit rounding
  "VALU64_trunc": ("v_trunc_f64", [v_trunc_f64_e32(v[0:1], v[2:3])]),
  "VALU64_ceil": ("v_ceil_f64", [v_ceil_f64_e32(v[0:1], v[2:3])]),
  "VALU64_floor": ("v_floor_f64", [v_floor_f64_e32(v[0:1], v[2:3])]),
  "VALU64_fract": ("v_fract_f64", [v_fract_f64_e32(v[0:1], v[2:3])]),

  # VALU 64-bit frexp
  "VALU64_frexp_exp": ("v_frexp_exp_i32_f64", [v_frexp_exp_i32_f64_e32(v[0], v[2:3])]),
  "VALU64_frexp_mant": ("v_frexp_mant_f64", [v_frexp_mant_f64_e32(v[0:1], v[2:3])]),

  # VALU 64-bit div helpers
  "VALU64_div_fixup": ("v_div_fixup_f64", [v_div_fixup_f64(v[0:1], v[2:3], v[4:5], v[6:7])]),
  "VALU64_div_fmas": ("v_div_fmas_f64", [v_div_fmas_f64(v[0:1], v[2:3], v[4:5], v[6:7])]),

  # VALU 32-bit div helpers
  "VALU_div_fixup": ("v_div_fixup_f32", [v_div_fixup_f32(v[0], v[1], v[2], v[3])]),
  "VALU_div_fmas": ("v_div_fmas_f32", [v_div_fmas_f32(v[0], v[1], v[2], v[3])]),
  "VALU_div_scale": ("v_div_scale_f32", [v_div_scale_f32(v[0], SrcEnum.VCC_LO, v[1], v[2], v[3])]),

  # VALU MAD64 (0xe)
  "VALU_mad64u": ("v_mad_u64_u32", [
    v_mov_b32_e32(v[2], 2),
    v_mov_b32_e32(v[3], 3),
    v_mov_b32_e32(v[4], 0),
    v_mov_b32_e32(v[5], 0),
    v_mad_u64_u32(v[0:1], SrcEnum.NULL, v[2], v[3], v[4:5]),
  ]),
  "VALU_mad64i": ("v_mad_i64_i32", [
    v_mov_b32_e32(v[2], 2),
    v_mov_b32_e32(v[3], 3),
    v_mov_b32_e32(v[4], 0),
    v_mov_b32_e32(v[5], 0),
    v_mad_i64_i32(v[0:1], SrcEnum.NULL, v[2], v[3], v[4:5]),
  ]),

  # VALU compare - writes to VCC
  "VALU_cmp": ("v_cmp_eq_u32", [v_cmp_eq_u32_e32(v[0], v[1])]),

  # VALU CMPX (0x73) - modifies EXEC
  "VALU_cmpx_eq_u32": ("v_cmpx_eq_u32", [v_cmpx_eq_u32_e32(v[0], v[1])]),

  # SALU saveexec (0x72) - modifies EXEC safely by ANDing with all-ones mask
  "SALU_saveexec": ("s_and_saveexec_b32", [
    s_mov_b32(s[5], 0xFFFFFFFF),  # all lanes mask
    s_and_saveexec_b32(s[4], s[5]),  # EXEC = EXEC & 0xFFFFFFFF = EXEC (unchanged)
  ]),

  # SALU float ops
  "SALU_ceil": ("s_ceil_f32", [s_ceil_f32(s[4], s[5])]),
  "SALU_floor": ("s_floor_f32", [s_floor_f32(s[4], s[5])]),
  "SALU_trunc": ("s_trunc_f32", [s_trunc_f32(s[4], s[5])]),

  # SALU bit ops
  "SALU_brev": ("s_brev_b32", [s_brev_b32(s[4], s[5])]),
  "SALU_bcnt1": ("s_bcnt1_i32_b32", [s_bcnt1_i32_b32(s[4], s[5])]),
  "SALU_ctz": ("s_ctz_i32_b32", [s_ctz_i32_b32(s[4], s[5])]),
  "SALU_clz": ("s_clz_i32_u32", [s_clz_i32_u32(s[4], s[5])]),

  # VALU sin/cos
  "VALU_sin": ("v_sin_f32", [v_sin_f32_e32(v[0], v[1])]),
  "VALU_cos": ("v_cos_f32", [v_cos_f32_e32(v[0], v[1])]),

  # VOP3P - packed operations
  "VALU_pk_add_f16": ("v_pk_add_f16", [v_pk_add_f16(v[0], v[1], v[2])]),
  "VALU_pk_mul_f16": ("v_pk_mul_f16", [v_pk_mul_f16(v[0], v[1], v[2])]),
  "VALU_pk_fma_f16": ("v_pk_fma_f16", [v_pk_fma_f16(v[0], v[1], v[2], v[3])]),
  "VALU_pk_add_i16": ("v_pk_add_i16", [v_pk_add_i16(v[0], v[1], v[2])]),

  # VOP3 - misc
  "VALU_bfe_u32": ("v_bfe_u32", [v_bfe_u32(v[0], v[1], 0, 8)]),
  "VALU_bfi_b32": ("v_bfi_b32", [v_bfi_b32(v[0], v[1], v[2], v[3])]),
  "VALU_alignbit": ("v_alignbit_b32", [v_alignbit_b32(v[0], v[1], v[2], 4)]),
  "VALU_fma_f32": ("v_fma_f32", [v_fma_f32(v[0], v[1], v[2], v[3])]),

  # VOP3 - integer add variants (used by tinygrad kernels)
  "VALU_add3": ("v_add3_u32", [v_add3_u32(v[0], v[1], v[2], v[3])]),
  "VALU_xad": ("v_xad_u32", [v_xad_u32(v[0], v[1], v[2], v[3])]),
  "VALU_lshl_or": ("v_lshl_or_b32", [v_lshl_or_b32(v[0], v[1], 4, v[2])]),
  "VALU_add_nc": ("v_add_nc_u32", [v_add_nc_u32_e32(v[0], v[1], v[2])]),

  # VOP3 - carry-out adds (used for 64-bit address calculation)
  "VALU_add_co": ("v_add_co_u32", [v_add_co_u32(v[0], SrcEnum.VCC_LO, v[1], v[2])]),
  "VALU_add_co_ci": ("v_add_co_ci_u32", [v_add_co_ci_u32_e32(v[0], v[1], v[2])]),

  # VOPD - dual issue (used by tinygrad kernels)
  "VALU_dual_add": ("v_dual_add_f32", [v_dual_add_f32(v[0], v[1], v[2], v[3], v[4], v[5])]),
  "VALU_dual_mul": ("v_dual_mul_f32", [v_dual_mul_f32(v[0], v[1], v[2], v[3], v[4], v[5])]),

  # VOP2 - fmac
  "VALU_fmac": ("v_fmac_f32", [v_fmac_f32_e32(v[0], v[1], v[0])]),

  # DOT products
  "VALU_dot2": ("v_dot2_f16_f16", [v_dot2_f16_f16(v[0], v[1], v[2], v[3])]),

  # WMMA - wave matrix multiply accumulate
  "VALU_wmma_f32_f16": ("v_wmma_f32_16x16x16_f16", [v_wmma_f32_16x16x16_f16(v[0:7], v[8:15], v[16:23], v[0:7])]),
  "VALU_wmma_f16_f16": ("v_wmma_f16_16x16x16_f16", [v_wmma_f16_16x16x16_f16(v[0:7], v[8:15], v[16:23], v[0:7])]),
  "VALU_wmma_i32_iu8": ("v_wmma_i32_16x16x16_iu8", [v_wmma_i32_16x16x16_iu8(v[0:7], v[8:11], v[12:15], v[0:7])]),

  # LDS atomics
  "LDS_atomic_add": ("ds_add_u32", [
    v_mov_b32_e32(v[0], 0),  # LDS address
    v_mov_b32_e32(v[1], 1),  # data to add
    ds_add_u32(addr=v[0], data0=v[1]),
    s_waitcnt(lgkmcnt=0),
  ]),

  # ═══════════════════════════════════════════════════════════════════════════════
  # GLOBAL ATOMICS - access real buffer passed via kernarg
  # ═══════════════════════════════════════════════════════════════════════════════

  # GLOBAL atomic add 32-bit (0x28 GLOBAL_ATOMIC)
  "GLOBAL_atomic_add": ("global_atomic_add_u32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),  # offset = 0
    v_mov_b32_e32(v[1], 1),  # data to add
    global_atomic_add_u32(addr=v[0], data=v[1], saddr=s[2]),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL atomic add 64-bit
  "GLOBAL_atomic_add64": ("global_atomic_add_u64", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[2], 1),
    v_mov_b32_e32(v[3], 0),
    global_atomic_add_u64(addr=v[0], data=v[2:3], saddr=s[2]),
    s_waitcnt(vmcnt=0),
  ]),

  # ═══════════════════════════════════════════════════════════════════════════════
  # MEMORY INSTRUCTIONS - access real buffer passed via kernarg
  # ═══════════════════════════════════════════════════════════════════════════════

  # SMEM (0x1) - scalar memory load from buffer
  "SMEM_load": ("s_load_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    s_load_b32(s[4], s[2], 0, soffset=SrcEnum.NULL),  # load from buffer
    s_waitcnt(lgkmcnt=0),
  ]),

  # GLOBAL load (0x21 GLOBAL_LOAD) - global memory load
  "GLOBAL_load": ("global_load_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),  # offset = 0
    global_load_b32(v[1], addr=v[0], saddr=s[2], offset=0),  # load from buffer
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL store (0x24 GLOBAL_STORE) - global memory store
  "GLOBAL_store": ("global_store_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),  # offset = 0
    v_mov_b32_e32(v[1], 42),  # data to store
    global_store_b32(addr=v[0], data=v[1], saddr=s[2], offset=0),  # store to buffer
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL 8-bit load/store
  "GLOBAL_load8": ("global_load_u8", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_u8(v[1], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  "GLOBAL_store8": ("global_store_b8", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[1], 42),
    global_store_b8(addr=v[0], data=v[1], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL 16-bit load/store
  "GLOBAL_load16": ("global_load_u16", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_u16(v[1], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  "GLOBAL_store16": ("global_store_b16", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[1], 42),
    global_store_b16(addr=v[0], data=v[1], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # LDS load (0x29 LDS_LOAD) - local data share read
  "LDS_load": ("ds_load_b32", [
    v_mov_b32_e32(v[0], 0),  # LDS address = 0
    ds_load_b32(v[1], v[0], offset=0),  # read from LDS
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS store (0x2b LDS_STORE) - local data share write
  "LDS_store": ("ds_store_b32", [
    v_mov_b32_e32(v[0], 0),  # LDS address = 0
    v_mov_b32_e32(v[1], 42),  # data to store
    ds_store_b32(v[0], v[1], offset=0),  # write to LDS
    s_waitcnt(lgkmcnt=0),
  ]),

  # ═══════════════════════════════════════════════════════════════════════════════
  # WIDER MEMORY OPERATIONS - to discover more InstOp variants
  # ═══════════════════════════════════════════════════════════════════════════════

  # GLOBAL 64-bit load
  "GLOBAL_load64": ("global_load_b64", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_b64(v[2:3], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL 96-bit load
  "GLOBAL_load96": ("global_load_b96", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_b96(v[4:6], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL 128-bit load
  "GLOBAL_load128": ("global_load_b128", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_b128(v[4:7], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL 64-bit store
  "GLOBAL_store64": ("global_store_b64", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[2], 42),
    v_mov_b32_e32(v[3], 43),
    global_store_b64(addr=v[0], data=v[2:3], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL 96-bit store
  "GLOBAL_store96": ("global_store_b96", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    global_store_b96(addr=v[0], data=v[4:6], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL 128-bit store
  "GLOBAL_store128": ("global_store_b128", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    v_mov_b32_e32(v[7], 45),
    global_store_b128(addr=v[0], data=v[4:7], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # ═══════════════════════════════════════════════════════════════════════════════
  # GLOBAL VADDR (vector-only addressing, saddr=NULL) - used by tinygrad kernels
  # ═══════════════════════════════════════════════════════════════════════════════

  # GLOBAL VADDR load (all sizes use same opcode 0x22)
  "GLOBAL_VADDR_load": ("global_load_b32 vaddr", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    global_load_b32(v[4], addr=v[0:1], saddr=SrcEnum.NULL, offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  "GLOBAL_VADDR_load128": ("global_load_b128 vaddr", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    global_load_b128(v[4:7], addr=v[0:1], saddr=SrcEnum.NULL, offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # GLOBAL VADDR stores (size encoded: 32->0x25, 64->0x26, 96->0x27, 128->0x28)
  "GLOBAL_VADDR_store": ("global_store_b32 vaddr", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[4], 42),
    global_store_b32(addr=v[0:1], data=v[4], saddr=SrcEnum.NULL, offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  "GLOBAL_VADDR_store64": ("global_store_b64 vaddr", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    global_store_b64(addr=v[0:1], data=v[4:5], saddr=SrcEnum.NULL, offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  "GLOBAL_VADDR_store96": ("global_store_b96 vaddr", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    global_store_b96(addr=v[0:1], data=v[4:6], saddr=SrcEnum.NULL, offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  "GLOBAL_VADDR_store128": ("global_store_b128 vaddr", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    v_mov_b32_e32(v[7], 45),
    global_store_b128(addr=v[0:1], data=v[4:7], saddr=SrcEnum.NULL, offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # LDS 64-bit load
  "LDS_load64": ("ds_load_b64", [
    v_mov_b32_e32(v[0], 0),
    ds_load_b64(v[2:3], v[0], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS 128-bit load
  "LDS_load128": ("ds_load_b128", [
    v_mov_b32_e32(v[0], 0),
    ds_load_b128(v[4:7], v[0], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS 64-bit store
  "LDS_store64": ("ds_store_b64", [
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[2], 42),
    v_mov_b32_e32(v[3], 43),
    ds_store_b64(v[0], v[2:3], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS 128-bit store
  "LDS_store128": ("ds_store_b128", [
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    v_mov_b32_e32(v[7], 45),
    ds_store_b128(v[0], v[4:7], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # MESSAGE (0x9) - s_sendmsg
  "MESSAGE": ("s_sendmsg", [
    s_sendmsg(0),  # send message 0 (NOP message)
  ]),

  # ═══════════════════════════════════════════════════════════════════════════════
  # FLAT MEMORY - uses 64-bit virtual address in VGPRs
  # ═══════════════════════════════════════════════════════════════════════════════

  # FLAT load - load using 64-bit address from buffer
  "FLAT_load": ("flat_load_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),  # addr lo
    v_mov_b32_e32(v[1], s[3]),  # addr hi
    flat_load_b32(v[2], addr=v[0:1]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  # FLAT store
  "FLAT_store": ("flat_store_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[2], 42),
    flat_store_b32(addr=v[0:1], data=v[2]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  # FLAT 64-bit
  "FLAT_load64": ("flat_load_b64", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    flat_load_b64(v[2:3], addr=v[0:1]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  "FLAT_store64": ("flat_store_b64", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    flat_store_b64(addr=v[0:1], data=v[4:5]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  # FLAT 96-bit
  "FLAT_load96": ("flat_load_b96", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    flat_load_b96(v[4:6], addr=v[0:1]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  "FLAT_store96": ("flat_store_b96", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    flat_store_b96(addr=v[0:1], data=v[4:6]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  # FLAT 128-bit
  "FLAT_load128": ("flat_load_b128", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    flat_load_b128(v[4:7], addr=v[0:1]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  "FLAT_store128": ("flat_store_b128", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    v_mov_b32_e32(v[7], 45),
    flat_store_b128(addr=v[0:1], data=v[4:7]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  # FLAT 8/16-bit stores
  "FLAT_store8": ("flat_store_b8", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[2], 42),
    flat_store_b8(addr=v[0:1], data=v[2]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

  "FLAT_store16": ("flat_store_b16", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], s[2]),
    v_mov_b32_e32(v[1], s[3]),
    v_mov_b32_e32(v[2], 42),
    flat_store_b16(addr=v[0:1], data=v[2]),
    s_waitcnt(vmcnt=0, lgkmcnt=0),
  ]),

}


def run_with_retry(instructions: list, max_attempts: int = 20) -> tuple[list[tuple[int, list[bytes]]], list[list], set]:
  """Run instructions multiple times to collect InstOp variants.

  Memory ops produce different InstOp values (0x2x vs 0x5x) depending on which SIMD executes them:
  - 0x2x range: wave ran on traced SIMD (matched)
  - 0x5x range: wave ran on other SIMD (not matched)

  Returns list of (traced_simd, blobs) tuples.
  """
  all_ops = set()
  all_runs: list[tuple[int, list[bytes]]] = []
  all_packets = []
  SQTT_SIMD_SEL.value = 0  # only trace SIMD 0
  for _ in range(max_attempts):
    blobs = run_asm_sqtt(instructions)
    packets = decode_all_blobs(blobs)
    # get ops from waves on traced SIMD 0 (gives 0x2x range)
    ops = get_inst_ops(packets, traced_simd=0)
    # also get ops from waves on other SIMDs (gives 0x5x range for memory ops)
    for simd in [1, 2, 3]:
      ops.update(get_inst_ops(packets, traced_simd=simd))
    all_runs.append((0, blobs))
    all_packets.append(packets)
    all_ops.update(ops)
  return all_runs, all_packets, all_ops

def discover_all_instops() -> tuple[dict[int, set[str]], dict[str, Exception]]:
  """Run all instruction tests and collect InstOp values."""
  discovered: dict[int, set[str]] = {}
  failures: dict[str, Exception] = {}

  for test_name, (instr_name, instructions) in INSTRUCTION_TESTS.items():
    try:
      all_runs, _, ops = run_with_retry(instructions)

      for op in ops:
        if op not in discovered:
          discovered[op] = set()
        discovered[op].add(f"{test_name}")

      if DEBUG >= 2:
        print(f"\n{'─'*60}")
        print(f"{test_name} ({instr_name}): ops={[hex(op) for op in sorted(ops)]}")

        # collect wave patterns from traced SIMD runs (group by exact timing)
        patterns: dict[tuple, list] = {}  # pattern (types + timing) -> list of (wave_packets, t0)
        for traced_simd, blobs in all_runs:
          for blob in blobs:
            packets = decode_all_blobs([blob])
            wave_packets = get_wave_packets(packets)
            # only include runs where wave ran on traced SIMD
            ws = next((p for p in wave_packets if isinstance(p, WAVESTART)), None)
            if ws and ws.simd == traced_simd and wave_packets:
              t0 = wave_packets[0]._time
              # pattern includes types AND normalized timing
              pattern = tuple((type(p).__name__, p._time - t0) for p in wave_packets)
              if pattern not in patterns:
                patterns[pattern] = []
              patterns[pattern].append((wave_packets, t0))

        if patterns:
          counts = {p: len(runs) for p, runs in patterns.items()}
          most_common = max(counts, key=counts.get)
          count = counts[most_common]
          total = sum(counts.values())
          print(f"\n=== most common pattern ({count}/{total} runs) ===")
          wave_packets, t0 = patterns[most_common][0]
          last_time = t0
          for p in wave_packets:
            print(format_packet(p, last_time, t0))
            last_time = p._time
          if len(patterns) > 1:
            print(f"\n  variations: {len(patterns)} unique timing patterns")

      if DEBUG >= 3:
        for traced_simd, blobs in all_runs:
          print(f"\n=== traced simd={traced_simd} ===")
          print_blobs(blobs, wave_only=False)
      if DEBUG >= 1:
        status = colored("✓", "green") if ops else colored("∅", "yellow")
        ops_str = ", ".join(hex(op) for op in sorted(ops)) if ops else "none"
        print(f"  {status} {test_name:25s} ops=[{ops_str}]")

    except Exception as e:
      failures[test_name] = e
      if DEBUG >= 1:
        print(f"  {colored('✗', 'red')} {test_name:25s} FAILED: {e}")

  return discovered, failures


def print_summary(discovered: dict[int, set[str]], failures: dict[str, Exception]) -> None:
  """Print discovery summary."""
  known_ops = {e.value for e in InstOp}
  discovered_ops = set(discovered.keys())

  print("\n" + "=" * 60)
  print("DISCOVERED INSTOP VALUES")
  print("=" * 60)

  for op in sorted(discovered_ops):
    try:
      name = InstOp(op).name
      status = colored("known", "green")
    except ValueError:
      name = f"UNKNOWN"
      status = colored("NEW!", "yellow")

    sources = ", ".join(sorted(discovered[op]))
    print(f"  0x{op:02x} {name:20s} ({status}) <- {sources}")

  # Missing from enum
  missing = known_ops - discovered_ops
  if missing:
    print("\n" + "=" * 60)
    print("ENUM VALUES NOT DISCOVERED")
    print("=" * 60)
    print("(need memory ops: SMEM, VMEM, LDS)")
    for op in sorted(missing):
      print(f"  0x{op:02x} {InstOp(op).name}")

  # New values to add
  new_ops = discovered_ops - known_ops
  if new_ops:
    print("\n" + "=" * 60)
    print(colored("NEW INSTOP VALUES TO ADD TO ENUM", "yellow"))
    print("=" * 60)
    for op in sorted(new_ops):
      sources = ", ".join(sorted(discovered[op]))
      print(f"  {op:#04x}: \"{sources}\",")

  # Stats
  print("\n" + "=" * 60)
  print("STATISTICS")
  print("=" * 60)
  print(f"  Tests run:      {len(INSTRUCTION_TESTS)}")
  print(f"  Tests passed:   {len(INSTRUCTION_TESTS) - len(failures)}")
  print(f"  Tests failed:   {len(failures)}")
  print(f"  Known ops:      {len(known_ops)}")
  print(f"  Discovered:     {len(discovered_ops)}")
  if known_ops:
    print(f"  Coverage:       {len(discovered_ops & known_ops)}/{len(known_ops)} ({100*len(discovered_ops & known_ops)//len(known_ops)}%)")
  print(f"  New ops found:  {len(new_ops)}")


if __name__ == "__main__":
  print("=" * 60)
  print("SQTT InstOp Discovery Tool")
  print("=" * 60)
  print(f"Testing {len(INSTRUCTION_TESTS)} instruction categories...\n")

  discovered, failures = discover_all_instops()
  print_summary(discovered, failures)
