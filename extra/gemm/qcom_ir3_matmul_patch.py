#!/usr/bin/env python3
"""Patch runner for compiler-generated QCOM IR3 matmul kernels.

This keeps tinygrad's normal scheduling/buffer setup and only replaces the IR3
binary attached to the main matmul program. The default patch is a no-op; it is
intended as the safety harness for assembly edits.
"""
from __future__ import annotations

import argparse, math, re, time
from dataclasses import replace

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import compile_linear, run_linear, time_call
from tinygrad.uop.ops import Ops, UOp
from tinygrad.runtime.support.compiler_mesa import IR3Compiler
from extra.gemm.ir3asm import ADD_S, BR, COV_F16F32, ISAM_F16, ISAM_F32, JUMP, MAD_F16, MAD_F32, MOV_F32, NOP, SHL_B, disasm

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def plain_name(name: str) -> str:
  return ANSI_RE.sub("", name)


def unpack_ir3(lib: bytes):
  v, cs, imm, image = IR3Compiler.unpack_lib(lib)
  return v, cs, imm, bytes(image)


def pack_ir3(v, cs, imm: bytes, image: bytes, fregs: int|None=None, hregs: int|None=None) -> bytes:
  if fregs is not None: v.info.max_reg = fregs - 1
  if hregs is not None: v.info.max_half_reg = hregs - 1
  v.info.size = len(image)
  v.instrlen = math.ceil(len(image) / 128)
  return bytes(v) + bytes(cs) + imm + image


def find_main_call(linear: UOp) -> int:
  candidates = []
  for i, call in enumerate(linear.src):
    if call.op is Ops.CALL and call.src and call.src[0].op is Ops.PROGRAM:
      name = plain_name(call.src[0].arg.name)
      if name.startswith("r_"): return i
      candidates.append((i, name))
  if len(candidates) == 1: return candidates[0][0]
  raise RuntimeError(f"could not identify main matmul program: {candidates}")


def patch_lib(lib: bytes, patch: str) -> tuple[bytes, str, dict[str, int|bool]]:
  v, cs, imm, image = unpack_ir3(lib)
  old_image = image
  fregs = hregs = None

  if patch == "noop":
    new_image = image
  elif patch == "rpt1_f32_pairs":
    new_image = patch_rpt1_f32_pairs(image)
  elif patch == "auto_rpt_f32":
    new_image = patch_auto_rpt_f32(image)
  elif patch == "reorder_rpt_f32_default":
    new_image = patch_reorder_rpt_f32_default(image)
  elif patch == "reorder_rpt_f32_compact":
    new_image = patch_reorder_rpt_f32_compact(image)
  elif patch == "rpt3_accum_f32_default":
    new_image = patch_rpt3_accum_f32_default(image)
    fregs = 14
  elif patch == "rpt3_accum_f32_accmajor":
    new_image = patch_rpt3_accum_f32_accmajor(image)
    fregs = 14
  elif patch == "rpt3_accum_f32_nosnop":
    new_image = patch_rpt3_accum_f32_nosnop(image)
    fregs = 14
  elif patch == "rpt3_accum_f32_nonopctl":
    new_image = patch_rpt3_accum_f32(image, clear_ctrl_nops=True)
    fregs = 14
  elif patch == "rpt3_accum_f32_a0early":
    new_image = patch_rpt3_accum_f32(image, load_order="a0early")
    fregs = 14
  elif patch == "rpt3_accum_f32_b0early":
    new_image = patch_rpt3_accum_f32(image, load_order="b0early")
    fregs = 14
  elif patch == "rpt3_accum_f32_criticalearly":
    new_image = patch_rpt3_accum_f32(image, load_order="criticalearly")
    fregs = 14
  elif patch == "rpt3_accum_f32_a1early":
    new_image = patch_rpt3_accum_f32(image, load_order="a1early")
    fregs = 14
  elif patch == "rpt3_accum_f32_loadearly":
    new_image = patch_rpt3_accum_f32(image, load_order="loadearly")
    fregs = 14
  elif patch == "rpt3_accum_f32_b0last":
    new_image = patch_rpt3_accum_f32(image, b0_last=True)
    fregs = 14
  elif patch == "rpt3_accum_f32_a0early_b0last":
    new_image = patch_rpt3_accum_f32(image, load_order="a0early", b0_last=True)
    fregs = 14
  elif patch == "rpt3_accum_f32_b0after38":
    new_image = patch_rpt3_accum_f32(image, load_order="b0after38")
    fregs = 14
  elif patch == "rpt3_accum_f32_b0after40":
    new_image = patch_rpt3_accum_f32(image, load_order="b0after40")
    fregs = 14
  elif patch == "rpt3_accum_f32_b0after41":
    new_image = patch_rpt3_accum_f32(image, load_order="b0after41")
    fregs = 14
  elif patch == "rpt3_accum_f32_k1230":
    new_image = patch_rpt3_accum_f32(image, k_order=(1, 2, 3, 0))
    fregs = 14
  elif patch == "rpt3_accum_f32_k2310":
    new_image = patch_rpt3_accum_f32(image, k_order=(2, 3, 1, 0))
    fregs = 14
  elif patch == "rpt3_accum_f32_k2310_b0last":
    new_image = patch_rpt3_accum_f32(image, k_order=(2, 3, 1, 0), b0_last=True)
    fregs = 14
  elif patch == "rpt3_accum_f32_f13":
    new_image = patch_rpt3_accum_f32(image)
    fregs = 13
  elif patch == "rpt3_tail_f13":
    new_image = patch_rpt3_tail_f13(image)
    fregs = 13
  elif patch == "rpt3_accum_f32_unroll2":
    new_image = patch_rpt3_accum_f32_unroll(image, 2)
    fregs = 14
  elif patch == "rpt3_accum_f32_unroll4":
    new_image = patch_rpt3_accum_f32_unroll(image, 4)
    fregs = 14
  elif patch == "rpt3_accum_f32_unroll8":
    new_image = patch_rpt3_accum_f32_unroll(image, 8)
    fregs = 14
  elif patch == "rpt3_accum_f32_unroll16":
    new_image = patch_rpt3_accum_f32_unroll(image, 16)
    fregs = 14
  elif patch == "rpt3_accum_f32_unroll32":
    new_image = patch_rpt3_accum_f32_unroll(image, 32)
    fregs = 14
  elif patch == "rpt3_accum_f32_unroll2_prefetcha0":
    new_image = patch_rpt3_accum_f32_unroll2_prefetcha0(image)
    fregs = 14
  elif patch == "rpt3_accum_f32_f13pack":
    new_image = patch_rpt3_accum_f32_f13pack(image)
    fregs = 13
  elif patch == "rpt3_accum_f32_f13pack_unroll8":
    new_image = patch_rpt3_accum_f32_f13pack(image, unroll=8)
    fregs = 13
  elif patch == "rpt3_accum_f32_f13pack_unroll16":
    new_image = patch_rpt3_accum_f32_f13pack(image, unroll=16)
    fregs = 13
  elif patch == "rpt3_n512_default":
    new_image = patch_rpt3_n512_default(image)
    fregs = 16
  elif patch == "rpt3_n384_unroll8":
    new_image = patch_rpt3_n384_unroll(image, 8)
    fregs = 12
  elif patch == "rpt3_n384_unroll16":
    new_image = patch_rpt3_n384_unroll(image, 16)
    fregs = 12
  elif patch == "rpt3_n384_unroll32":
    new_image = patch_rpt3_n384_unroll(image, 32)
    fregs = 12
  elif patch == "rpt3_n512_unroll8":
    new_image = patch_rpt3_n512_unroll(image, 8)
    fregs = 16
  elif patch == "rpt3_n512_unroll16":
    new_image = patch_rpt3_n512_unroll(image, 16)
    fregs = 16
  elif patch == "rpt3_n512_unroll32":
    new_image = patch_rpt3_n512_unroll(image, 32)
    fregs = 16
  elif patch == "rpt3_n512_tight":
    new_image = patch_rpt3_n512_tight(image)
    fregs = 16
  elif patch == "rpt3_n512_tight_unroll8":
    new_image = patch_rpt3_n512_tight_unroll(image, 8)
    fregs = 16
  elif patch == "rpt3_n512_tight_unroll16":
    new_image = patch_rpt3_n512_tight_unroll(image, 16)
    fregs = 16
  elif patch == "rpt3_n512_tight_unroll32":
    new_image = patch_rpt3_n512_tight_unroll(image, 32)
    fregs = 16
  elif patch == "rpt3_n512_nopdead":
    new_image = patch_rpt3_n512_variant(image, nop=(22, 24, 26, 27, 32, 33, 34))
    fregs = 16
  elif patch == "rpt3_n512_nopcopies":
    new_image = patch_rpt3_n512_variant(image, nop=(26, 27, 32, 33, 34))
    fregs = 16
  elif patch == "rpt3_n512_drop_latecopies":
    new_image = patch_rpt3_n512_variant(image, drop=(32, 33, 34))
    fregs = 16
  elif patch == "rpt3_n512_drop_copies":
    new_image = patch_rpt3_n512_variant(image, drop=(26, 27, 32, 33, 34))
    fregs = 16
  elif patch == "rpt3_n512_drop_copies_unroll16":
    new_image = patch_rpt3_n512_variant(image, drop=(26, 27, 32, 33, 34), unroll=16)
    fregs = 16
  elif patch == "rpt3_n512_b0low":
    new_image = patch_rpt3_n512_b0low(image)
    fregs = 15
  elif patch == "rpt3_n512_b0low_unroll8":
    new_image = patch_rpt3_n512_b0low(image, unroll=8)
    fregs = 15
  elif patch == "rpt3_n512_b0low_unroll16":
    new_image = patch_rpt3_n512_b0low(image, unroll=16)
    fregs = 15
  elif patch == "rpt3_n512_b0low_unroll32":
    new_image = patch_rpt3_n512_b0low(image, unroll=32)
    fregs = 15
  elif patch == "rpt3_n512_b0low_b0last":
    new_image = patch_rpt3_n512_b0low(image, b0_last=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_b0last_unroll8":
    new_image = patch_rpt3_n512_b0low(image, unroll=8, b0_last=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_b0last_unroll16":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, b0_last=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_k0123_unroll16":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(0, 1, 2, 3))
    fregs = 15
  elif patch == "rpt3_n512_b0low_k0123_unroll16_nosnop":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(0, 1, 2, 3), drop_ssnop=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_k2310_unroll16":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(2, 3, 1, 0))
    fregs = 15
  elif patch == "rpt3_n512_b0low_k2310_unroll16_nosnop":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(2, 3, 1, 0), drop_ssnop=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_k3210_unroll16":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(3, 2, 1, 0))
    fregs = 15
  elif patch == "rpt3_n512_b0low_k3210_unroll16_nosnop":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(3, 2, 1, 0), drop_ssnop=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_k3210_unroll32_nosnop":
    new_image = patch_rpt3_n512_b0low(image, unroll=32, k_order=(3, 2, 1, 0), drop_ssnop=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_k3210_unroll64_nosnop":
    new_image = patch_rpt3_n512_b0low(image, unroll=64, k_order=(3, 2, 1, 0), drop_ssnop=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_k3210_unroll16_lastcmp":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(3, 2, 1, 0), last_cmp_only=True)
    fregs = 15
  elif patch == "rpt3_n512_b0low_k3210_unroll16_tightctrl":
    new_image = patch_rpt3_n512_b0low(image, unroll=16, k_order=(3, 2, 1, 0), drop_ssnop=True, last_cmp_only=True)
    fregs = 15
  elif patch == "rpt3_n512_f14":
    new_image = patch_rpt3_n512_f14(image)
    fregs = 14
  elif patch == "rpt3_n512_f14_unroll8":
    new_image = patch_rpt3_n512_f14(image, unroll=8)
    fregs = 14
  elif patch == "rpt3_n512_f14_unroll16":
    new_image = patch_rpt3_n512_f14(image, unroll=16)
    fregs = 14
  elif patch == "rpt3_n512_f14_f15":
    new_image = patch_rpt3_n512_f14(image)
    fregs = 15
  elif patch == "rpt3_l25_unroll8":
    new_image = patch_rpt3_l25(image, unroll=8)
    fregs = 16
  elif patch == "rpt3_l25_unroll16":
    new_image = patch_rpt3_l25(image, unroll=16)
    fregs = 16
  elif patch == "rpt3_l25_unroll4":
    new_image = patch_rpt3_l25(image, unroll=4)
    fregs = 16
  elif patch == "rpt3_l25_unroll4_nosnop":
    new_image = patch_rpt3_l25(image, unroll=4, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll11":
    new_image = patch_rpt3_l25(image, unroll=11)
    fregs = 16
  elif patch == "rpt3_l25_unroll8_nosnop":
    new_image = patch_rpt3_l25(image, unroll=8, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll22":
    new_image = patch_rpt3_l25(image, unroll=22)
    fregs = 16
  elif patch == "rpt3_l25_unroll44":
    new_image = patch_rpt3_l25(image, unroll=44)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_f15":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True)
    fregs = 15
  elif patch == "rpt3_l25_unroll16_nosnop_f14":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True)
    fregs = 14
  elif patch == "rpt3_l25_unroll16_nosnop_f13":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True)
    fregs = 13
  elif patch == "rpt3_l25_unroll16_nostore":
    new_image = patch_rpt3_l25(image, unroll=16, no_store=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_nostore":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, no_store=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_nostore_skipa":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, no_store=True, skip_a_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_nostore_skipb":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, no_store=True, skip_b_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_nostore_skipab":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, no_store=True, skip_a_loads=True, skip_b_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_dropstart":
    new_image = patch_rpt3_l25(image, unroll=16, drop_startnop=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_dropcmpnop":
    new_image = patch_rpt3_l25(image, unroll=16, drop_postcmp_nop=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_tightnops":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, drop_startnop=True, drop_postcmp_nop=True)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_lastcmp3":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, last_cmp_nop=3)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_lastcmp1":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, last_cmp_nop=1)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_lastcmp0":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, last_cmp_nop=0)
    fregs = 16
  elif patch == "rpt3_l25_unroll16_nosnop_lastcmp0_dropcmpnop":
    new_image = patch_rpt3_l25(image, unroll=16, drop_ssnop=True, drop_postcmp_nop=True, last_cmp_nop=0)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll16":
    new_image = patch_rpt3_l25_postinc(image, unroll=16, drop_ssnop=True)
    fregs = 16
  elif (m:=re.fullmatch(r"rpt3_l25_postinc_unroll(\d+)", patch)) is not None:
    new_image = patch_rpt3_l25_postinc(image, unroll=int(m.group(1)), drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_nostore":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, no_store=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_nostore_skipa":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, no_store=True, skip_a_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_nostore_skipb":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, no_store=True, skip_b_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_nostore_skipab":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, no_store=True, skip_a_loads=True, skip_b_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_incoords_unroll22":
    new_image = patch_rpt3_l25_postinc_incoords(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_incoords_unroll22_nostore":
    new_image = patch_rpt3_l25_postinc_incoords(image, unroll=22, drop_ssnop=True, no_store=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_b0low_unroll22":
    new_image = patch_rpt3_l25_postinc_b0low(image, unroll=22, drop_ssnop=True)
    fregs = 15
  elif patch == "rpt3_l25_postinc_b0low_unroll22_f16":
    new_image = patch_rpt3_l25_postinc_b0low(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_b0low_unroll22_syafter_f16":
    new_image = patch_rpt3_l25_postinc_b0low(image, unroll=22, drop_ssnop=True, sy_after_b0=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_b0firstlow_unroll22_f16":
    new_image = patch_rpt3_l25_postinc_b0firstlow(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_b0low_unroll22_pre3_f16":
    new_image = patch_rpt3_l25_postinc_b0low(image, unroll=22, drop_ssnop=True, wait_before_b0=3)
    fregs = 16
  elif patch == "rpt3_l25_postinc_b0low_unroll22_pre7_f16":
    new_image = patch_rpt3_l25_postinc_b0low(image, unroll=22, drop_ssnop=True, wait_before_b0=7)
    fregs = 16
  elif patch == "rpt3_l25_postinc_b0low_unroll22_post3_f16":
    new_image = patch_rpt3_l25_postinc_b0low(image, unroll=22, drop_ssnop=True, wait_after_b0=3)
    fregs = 16
  elif patch == "rpt3_l25_postinc_b0low_unroll22_wait33_f16":
    new_image = patch_rpt3_l25_postinc_b0low(image, unroll=22, drop_ssnop=True, wait_before_b0=3, wait_after_b0=3)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_ssnop":
    new_image = patch_rpt3_l25_postinc(image, unroll=22)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_acc1230":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, acc_order=(1, 2, 3, 0))
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_acc3210":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, acc_order=(3, 2, 1, 0))
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_k3210":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, k_order=(3, 2, 1, 0))
    fregs = 16
  elif (m:=re.fullmatch(r"rpt3_l25_postinc_unroll22_k([0-3]{4})", patch)) is not None:
    k_order = tuple(int(x) for x in m.group(1))
    if sorted(k_order) != [0, 1, 2, 3]: raise ValueError(f"invalid l25 k_order {k_order}")
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, k_order=k_order)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_accmajor":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, acc_major=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_accmajor_nostore_skipab":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, acc_major=True, no_store=True, skip_a_loads=True, skip_b_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoisty":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(35,))
    fregs = 16
  elif (m:=re.fullmatch(r"rpt3_l25_postinc_hoisty_unroll(\d+)", patch)) is not None:
    new_image = patch_rpt3_l25_postinc(image, unroll=int(m.group(1)), drop_ssnop=True, hoist_y=(35,))
    fregs = 16
  elif (m:=re.fullmatch(r"rpt3_l25_postinc_hoisty_unroll22_k([0-3]{4})", patch)) is not None:
    k_order = tuple(int(x) for x in m.group(1))
    if sorted(k_order) != [0, 1, 2, 3]: raise ValueError(f"invalid l25 k_order {k_order}")
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(35,), k_order=k_order)
    fregs = 16
  elif (m:=re.fullmatch(r"rpt3_l25_postinc_hoisty_unroll22_acc([0-3]{4})", patch)) is not None:
    acc_order = tuple(int(x) for x in m.group(1))
    if sorted(acc_order) != [0, 1, 2, 3]: raise ValueError(f"invalid l25 acc_order {acc_order}")
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(35,), acc_order=acc_order)
    fregs = 16
  elif (m:=re.fullmatch(r"rpt3_l25_postinc_hoisty_unroll22_k([0-3]{4})_acc([0-3]{4})", patch)) is not None:
    k_order = tuple(int(x) for x in m.group(1))
    acc_order = tuple(int(x) for x in m.group(2))
    if sorted(k_order) != [0, 1, 2, 3]: raise ValueError(f"invalid l25 k_order {k_order}")
    if sorted(acc_order) != [0, 1, 2, 3]: raise ValueError(f"invalid l25 acc_order {acc_order}")
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(35,), k_order=k_order, acc_order=acc_order)
    fregs = 16
  elif patch == "rpt3_l25_postinc_hoisty_unroll22_accmajor":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(35,), acc_major=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoisty_nostore_skipab":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(35,), no_store=True, skip_a_loads=True, skip_b_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35))
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords_nostore_skipab":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35), no_store=True, skip_a_loads=True, skip_b_loads=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords_f17":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35), load_order="bcoords_f17")
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords_f17_nostore_skipab":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35), load_order="bcoords_f17", no_store=True, skip_a_loads=True, skip_b_loads=True)
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords_f18":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35), load_order="bcoords_f17")
    fregs = 18
  elif patch == "rpt3_l25_postinc_unroll22_hoistb2_f17":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 35), load_order="bcoords_f17")
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_hoistb3_f17":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(33, 35), load_order="bcoords_f17")
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords_lowpair_f17":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35), load_order="bcoords_lowpair_f17")
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords_lowpair_f16":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35), load_order="bcoords_lowpair_f17")
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoistb2_lowpair_f16":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 35), load_order="bcoords_lowpair_f17")
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoistb3_lowpair_f16":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(33, 35), load_order="bcoords_lowpair_f17")
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoistbcoords_lowpair_f17_nostore_skipab":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(31, 33, 35), load_order="bcoords_lowpair_f17", no_store=True, skip_a_loads=True, skip_b_loads=True)
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_hoistr7w":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(37,))
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_hoistr10z":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, hoist_y=(36,))
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_a0early":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, load_order="a0early")
    fregs = 16
  elif (m:=re.fullmatch(r"rpt3_l25_postinc_unroll22_a0early_k([0-3]{4})", patch)) is not None:
    k_order = tuple(int(x) for x in m.group(1))
    if sorted(k_order) != [0, 1, 2, 3]: raise ValueError(f"invalid l25 k_order {k_order}")
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, load_order="a0early", k_order=k_order)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_bfirst":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, load_order="bfirst")
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_a1mid":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, load_order="a1mid")
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_a1copyearly":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, load_order="a1copyearly")
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_a1copyearly_wait":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, load_order="a1copyearly_wait")
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_a1copyearly_f17":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, load_order="a1copyearly_f17")
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_af16":
    new_image = patch_rpt3_l25_postinc(image, unroll=22, drop_ssnop=True, a_f16=True)
    fregs, hregs = 16, 1
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0":
    new_image = patch_rpt3_l25_postinc_prefetcha0(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_wait3":
    new_image = patch_rpt3_l25_postinc_prefetcha0(image, unroll=22, drop_ssnop=True, prefetch_wait=3)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_wait7":
    new_image = patch_rpt3_l25_postinc_prefetcha0(image, unroll=22, drop_ssnop=True, prefetch_wait=7)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_ss":
    new_image = patch_rpt3_l25_postinc_prefetcha0(image, unroll=22, drop_ssnop=True, prefetch_ss=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_safe":
    new_image = patch_rpt3_l25_postinc_prefetcha0_safe(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_bfirst":
    new_image = patch_rpt3_l25_postinc_prefetcha0_bfirst(image, unroll=22)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_bfirst_wait7":
    new_image = patch_rpt3_l25_postinc_prefetcha0_bfirst(image, unroll=22, b_wait=7)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_bfirst_a0wait7":
    new_image = patch_rpt3_l25_postinc_prefetcha0_bfirst(image, unroll=22, a0_wait=7)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_r11coord":
    new_image = patch_rpt3_l25_postinc_prefetcha0_r11coord(image, unroll=22)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_r11coord_noprefetch":
    new_image = patch_rpt3_l25_postinc_prefetcha0_r11coord(image, unroll=22, do_prefetch=False)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_delaya1":
    new_image = patch_rpt3_l25_postinc_prefetcha0_delaya1(image, unroll=22)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_delaya1_noprefetch":
    new_image = patch_rpt3_l25_postinc_prefetcha0_delaya1(image, unroll=22, do_prefetch=False)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_low":
    new_image = patch_rpt3_l25_postinc_prefetcha0_low(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_r15w":
    new_image = patch_rpt3_l25_postinc_prefetcha0_r15w(image, unroll=22, drop_ssnop=True)
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_r15w_after16":
    new_image = patch_rpt3_l25_postinc_prefetcha0_r15w(image, unroll=22, drop_ssnop=True, prefetch_after=16)
    fregs = 17
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_pair":
    new_image = patch_rpt3_l25_postinc_prefetcha0_pair(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha0_pair_after16":
    new_image = patch_rpt3_l25_postinc_prefetcha0_pair(image, unroll=22, drop_ssnop=True, prefetch_after=16)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_pair_noprefetch":
    new_image = patch_rpt3_l25_postinc_prefetcha0_pair(image, unroll=22, drop_ssnop=True, do_prefetch=False)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha2":
    new_image = patch_rpt3_l25_postinc_prefetcha2(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha2_after16":
    new_image = patch_rpt3_l25_postinc_prefetcha2(image, unroll=22, drop_ssnop=True, prefetch_after=16)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha3":
    new_image = patch_rpt3_l25_postinc_prefetcha3(image, unroll=22, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_postinc_unroll22_prefetcha3_after16":
    new_image = patch_rpt3_l25_postinc_prefetcha3(image, unroll=22, drop_ssnop=True, prefetch_after=16)
    fregs = 16
  elif patch == "rpt3_l25_acc1230_unroll16":
    new_image = patch_rpt3_l25(image, unroll=16, acc_order=(1, 2, 3, 0))
    fregs = 16
  elif patch == "rpt3_l25_acc3210_unroll16":
    new_image = patch_rpt3_l25(image, unroll=16, acc_order=(3, 2, 1, 0))
    fregs = 16
  elif patch == "rpt3_l25_k3210_unroll8":
    new_image = patch_rpt3_l25(image, unroll=8, k_order=(3, 2, 1, 0))
    fregs = 16
  elif patch == "rpt3_l25_k3210_unroll16":
    new_image = patch_rpt3_l25(image, unroll=16, k_order=(3, 2, 1, 0))
    fregs = 16
  elif patch == "rpt3_l25_k3210_unroll8_nosnop":
    new_image = patch_rpt3_l25(image, unroll=8, k_order=(3, 2, 1, 0), drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l25_k3210_unroll16_nosnop":
    new_image = patch_rpt3_l25(image, unroll=16, k_order=(3, 2, 1, 0), drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l23_unroll8":
    new_image = patch_rpt3_l23(image, unroll=8)
    fregs = 16
  elif patch == "rpt3_l23_unroll16":
    new_image = patch_rpt3_l23(image, unroll=16)
    fregs = 16
  elif patch == "rpt3_l23_unroll32":
    new_image = patch_rpt3_l23(image, unroll=32)
    fregs = 16
  elif patch == "rpt3_l23_unroll16_nosnop":
    new_image = patch_rpt3_l23(image, unroll=16, drop_ssnop=True)
    fregs = 16
  elif patch == "rpt3_l23_unroll16_nostore":
    new_image = patch_rpt3_l23(image, unroll=16, no_store=True)
    fregs = 16
  elif patch == "half_f16_rpt3_padded":
    new_image = patch_half_f16_rpt3_padded(image)
    fregs, hregs = 8, 12
  elif patch == "half_f16_rpt3_compact":
    new_image = patch_half_f16_rpt3_compact(image)
    fregs, hregs = 8, 12
  elif patch == "half_f16_rpt3_tight":
    new_image = patch_half_f16_rpt3_tight(image)
    fregs, hregs = 8, 12
  else:
    raise ValueError(f"unknown patch {patch!r}")

  new_lib = pack_ir3(v, cs, imm, new_image, fregs=fregs, hregs=hregs)
  asm = disasm(new_image)
  meta = {
    "repacked_same": new_lib == lib,
    "image_bytes": len(new_image),
    "instrs": len(new_image) // 8,
    "fregs": v.info.max_reg + 1,
    "hregs": v.info.max_half_reg + 1,
    "changed_bytes": sum(a != b for a, b in zip(old_image, new_image)) + abs(len(old_image) - len(new_image)),
    "mad_f32": asm.count("mad.f32"),
    "mad_f16": asm.count("mad.f16"),
    "rpt_mad": asm.count(")mad.f"),
    "isam": asm.count("isam"),
    "stores": asm.count("stg.") + asm.count("stib"),
  }
  return new_lib, asm, meta


F32_RPT1_PAIRS = (
  (58, ("r5.z", "r8.z", "r5.z", "r2.x"), ("r5.w", "r8.z", "r5.w", "r2.y")),
  (74, ("r0.y", "r8.w", "r0.y", "r5.z"), ("r0.z", "r8.w", "r0.z", "r5.w")),
  (90, ("r7.z", "r9.x", "r7.z", "r0.y"), ("r7.w", "r9.x", "r7.w", "r0.z")),
  (106, ("r2.x", "r9.y", "r10.x", "r7.z"), ("r2.y", "r9.y", "r10.y", "r7.w")),
)


def patch_rpt1_f32_pairs(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  for idx, first, second in F32_RPT1_PAIRS:
    expected_first, expected_second = MAD_F32(*first), MAD_F32(*second)
    if instrs[idx] != expected_first or instrs[idx+1] != expected_second:
      raise RuntimeError(
        f"instruction mismatch at {idx}: "
        f"got {instrs[idx].hex()} {instrs[idx+1].hex()}, "
        f"expected {expected_first.hex()} {expected_second.hex()}"
      )
    instrs[idx] = MAD_F32(*first, rpt=1, r=True)
    instrs[idx+1] = NOP()
  return b"".join(instrs)


MAD_F32_RE = re.compile(r"(\(sy\))?mad\.f32 (r\d+\.[xyzw]), (r\d+\.[xyzw]), (r\d+\.[xyzw]), (r\d+\.[xyzw])")


def freg_idx(reg: str) -> int:
  r, c = reg[1:].split(".")
  return int(r) * 4 + "xyzw".index(c)


def freg_name(idx: int) -> str:
  return f"r{idx//4}.{ 'xyzw'[idx & 3] }"


def parse_mad_f32(line: str):
  if "(rpt" in line: return None
  if (m:=MAD_F32_RE.search(line)) is None: return None
  sy, dst, src1, src2, src3 = m.groups()
  return bool(sy), dst, src1, src2, src3


def patch_auto_rpt_f32(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  asm_lines = [line for line in disasm(image).splitlines() if not line.rstrip().endswith(":")]
  if len(asm_lines) != len(instrs):
    raise RuntimeError(f"disasm line count mismatch: {len(asm_lines)} lines for {len(instrs)} instrs")

  i = 0
  while i < len(instrs):
    first = parse_mad_f32(asm_lines[i])
    if first is None:
      i += 1
      continue
    sy, dst, src1, src2, src3 = first
    run = 1
    while run < 4 and i + run < len(instrs):
      cur = parse_mad_f32(asm_lines[i + run])
      if cur is None: break
      cur_sy, cur_dst, cur_src1, cur_src2, cur_src3 = cur
      if cur_sy or cur_src1 != src1: break
      if freg_idx(cur_dst) != freg_idx(dst) + run: break
      if freg_idx(cur_src2) != freg_idx(src2) + run: break
      if freg_idx(cur_src3) != freg_idx(src3) + run: break
      run += 1
    if run < 2:
      i += 1
      continue

    expected = [MAD_F32(dst, src1, src2, src3, sy=sy)]
    for off in range(1, run):
      expected.append(MAD_F32(freg_name(freg_idx(dst) + off), src1, freg_name(freg_idx(src2) + off), freg_name(freg_idx(src3) + off)))
    if instrs[i:i+run] != expected:
      raise RuntimeError(f"mad sequence mismatch at {i}: run={run}")
    instrs[i] = MAD_F32(dst, src1, src2, src3, rpt=run - 1, sy=sy, r=True)
    for off in range(1, run): instrs[i + off] = NOP()
    i += run
  return b"".join(instrs)


def clean_disasm_lines(image: bytes) -> list[str]:
  return [line for line in disasm(image).splitlines() if not line.rstrip().endswith(":")]


def mad_instr_from_parsed(parsed, rpt=0, r=False):
  sy, dst, src1, src2, src3 = parsed
  return MAD_F32(dst, src1, src2, src3, rpt=rpt, sy=sy, r=r)


def emit_mad_group(instrs: list[bytes], parsed: list, group: tuple[int, ...]) -> bytes:
  first = parsed[group[0]]
  if first is None: raise RuntimeError(f"missing mad at {group[0]}")
  sy, dst, src1, src2, src3 = first
  expected = []
  for off, idx in enumerate(group):
    cur = parsed[idx]
    if cur is None: raise RuntimeError(f"missing mad at {idx}")
    cur_sy, cur_dst, cur_src1, cur_src2, cur_src3 = cur
    if off == 0:
      if instrs[idx] != mad_instr_from_parsed(cur): raise RuntimeError(f"mad mismatch at {idx}")
    else:
      if cur_sy or cur_src1 != src1: raise RuntimeError(f"incompatible mad group at {idx}")
      if freg_idx(cur_dst) != freg_idx(dst) + off: raise RuntimeError(f"dst mismatch in mad group at {idx}")
      if freg_idx(cur_src2) != freg_idx(src2) + off: raise RuntimeError(f"src2 mismatch in mad group at {idx}")
      if freg_idx(cur_src3) != freg_idx(src3) + off: raise RuntimeError(f"src3 mismatch in mad group at {idx}")
      if instrs[idx] != mad_instr_from_parsed(cur): raise RuntimeError(f"mad mismatch at {idx}")
    expected.append(instrs[idx])
  return MAD_F32(dst, src1, src2, src3, rpt=len(group) - 1, sy=sy, r=len(group) > 1)


DEFAULT_RPT_GROUPS = [
  ((49, 52), (50, 53, 56), (51, 54, 57), (55,), (60,), (61,), (62,), (58, 59, 63), (64,)),
  ((65, 68), (66, 69, 72), (67, 70, 73), (71,), (76,), (77,), (78,), (74, 75, 79), (80,)),
  ((81, 84), (82, 85, 88), (83, 86, 89), (87,), (92,), (93,), (94,), (90, 91, 95), (96,)),
  ((97, 100), (98, 101, 104), (99, 102, 105), (103,), (108,), (109,), (110,), (106, 107, 111), (112,)),
]


def default_reordered_mad_blocks(instrs: list[bytes], parsed: list) -> list[list[bytes]]:
  out = []
  for block in DEFAULT_RPT_GROUPS:
    lo, hi = block[0][0], block[-1][-1] + 1
    used = sorted(i for group in block for i in group)
    if used != list(range(lo, hi)): raise RuntimeError(f"block does not cover {lo}:{hi}: {used}")
    out.append([emit_mad_group(instrs, parsed, group) for group in block])
  return out


def patch_reorder_rpt_f32_default(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"default reorder expects 144 instructions, got {len(instrs)}")
  lines = clean_disasm_lines(image)
  parsed = [parse_mad_f32(line) for line in lines]
  for block, new_block in zip(DEFAULT_RPT_GROUPS, default_reordered_mad_blocks(instrs, parsed)):
    lo, hi = block[0][0], block[-1][-1] + 1
    instrs[lo:hi] = new_block + [NOP()] * ((hi - lo) - len(new_block))
  return b"".join(instrs)


def patch_reorder_rpt_f32_compact(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"compact reorder expects 144 instructions, got {len(instrs)}")
  lines = clean_disasm_lines(image)
  parsed = [parse_mad_f32(line) for line in lines]
  b0, b1, b2, b3 = default_reordered_mad_blocks(instrs, parsed)
  loop_start = 21
  out = instrs[:49] + b0 + b1 + b2 + b3
  br_idx = len(out)
  out += [BR(3, inv=False), instrs[114], JUMP(loop_start - (br_idx + 2))]
  out += instrs[116:133]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def clear_hi_bits(instr: bytes, bits: int) -> bytes:
  lo = int.from_bytes(instr[:4], "little")
  hi = int.from_bytes(instr[4:], "little") & ~bits
  return lo.to_bytes(4, "little") + hi.to_bytes(4, "little")


def patch_rpt3_accum_f32(image: bytes, acc_major: bool=False, drop_ssnop: bool=False, clear_ctrl_nops: bool=False, load_order: str="default", b0_last: bool=False, k_order: tuple[int, ...]=(0, 1, 2, 3)) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"rpt3 accum expects 144 instructions, got {len(instrs)}")
  if instrs[48] != ISAM_F32("r5.z", "r6.x", 0, 0):
    raise RuntimeError(f"unexpected A0 load at 48: {instrs[48].hex()}")
  instrs[48] = ISAM_F32("r13.x", "r6.x", 0, 0)
  if clear_ctrl_nops:
    instrs[36] = clear_hi_bits(instrs[36], 0x00080800)
    instrs[37] = clear_hi_bits(instrs[37], 0x00000800)

  accs = ("r5.x", "r4.x", "r3.x", "r2.x")
  avecs = ("r13.x", "r0.y", "r7.z", "r10.x")
  bcols = (("r6.z", "r11.x", "r12.x", "r8.z"),
           ("r6.w", "r11.y", "r12.y", "r8.w"),
           ("r7.x", "r11.z", "r12.z", "r9.x"),
           ("r7.y", "r11.w", "r12.w", "r9.y"))
  mads, first = [], True
  if acc_major:
    for acc_idx, acc in enumerate(accs):
      for avec, bvec in zip(avecs, bcols):
        mads.append(MAD_F32(acc, bvec[acc_idx], avec, acc, rpt=3, sy=first, r=True))
        first = False
  else:
    acc_order = (1, 2, 3, 0) if b0_last else (0, 1, 2, 3)
    for k_idx in k_order:
      avec, bvec = avecs[k_idx], bcols[k_idx]
      for acc_idx in acc_order:
        mads.append(MAD_F32(accs[acc_idx], bvec[acc_idx], avec, accs[acc_idx], rpt=3, sy=first, r=True))
        first = False

  loop_start = 21
  if load_order == "default":
    prefix = instrs[:41] + instrs[42:46] if drop_ssnop else instrs[:46]
    out = prefix + instrs[48:49] + mads
  elif load_order == "a0early":
    out = instrs[:35] + instrs[48:49] + instrs[35:46] + mads
  elif load_order == "b0early":
    out = instrs[:35] + instrs[43:44] + instrs[35:43] + instrs[44:46] + instrs[48:49] + mads
  elif load_order == "criticalearly":
    out = instrs[:35] + instrs[43:44] + instrs[48:49] + instrs[35:43] + instrs[44:46] + mads
  elif load_order == "a1early":
    out = instrs[:39] + instrs[44:46] + instrs[39:44] + instrs[48:49] + mads
  elif load_order == "loadearly":
    out = instrs[:35] + instrs[43:44] + instrs[48:49] + instrs[35:39] + instrs[44:46] + instrs[39:43] + mads
  elif load_order == "b0after38":
    out = instrs[:39] + instrs[43:44] + instrs[39:43] + instrs[44:46] + instrs[48:49] + mads
  elif load_order == "b0after40":
    out = instrs[:41] + instrs[43:44] + instrs[41:43] + instrs[44:46] + instrs[48:49] + mads
  elif load_order == "b0after41":
    out = instrs[:42] + instrs[43:44] + instrs[42:43] + instrs[44:46] + instrs[48:49] + mads
  else:
    raise ValueError(f"unknown load_order {load_order!r}")
  br_idx = len(out)
  out += [BR(3, inv=False), instrs[114], JUMP(loop_start - (br_idx + 2))]
  out += instrs[116:133]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_accum_f32_default(image: bytes) -> bytes:
  return patch_rpt3_accum_f32(image)


def patch_rpt3_accum_f32_accmajor(image: bytes) -> bytes:
  return patch_rpt3_accum_f32(image, acc_major=True)


def patch_rpt3_accum_f32_nosnop(image: bytes) -> bytes:
  return patch_rpt3_accum_f32(image, drop_ssnop=True)


def rpt3_accum_mads(acc_major: bool=False, b0_last: bool=False) -> list[bytes]:
  accs = ("r5.x", "r4.x", "r3.x", "r2.x")
  avecs = ("r13.x", "r0.y", "r7.z", "r10.x")
  bcols = (("r6.z", "r11.x", "r12.x", "r8.z"),
           ("r6.w", "r11.y", "r12.y", "r8.w"),
           ("r7.x", "r11.z", "r12.z", "r9.x"),
           ("r7.y", "r11.w", "r12.w", "r9.y"))
  mads, first = [], True
  if acc_major:
    for acc_idx, acc in enumerate(accs):
      for avec, bvec in zip(avecs, bcols):
        mads.append(MAD_F32(acc, bvec[acc_idx], avec, acc, rpt=3, sy=first, r=True))
        first = False
  else:
    acc_order = (1, 2, 3, 0) if b0_last else (0, 1, 2, 3)
    for avec, bvec in zip(avecs, bcols):
      for acc_idx in acc_order:
        mads.append(MAD_F32(accs[acc_idx], bvec[acc_idx], avec, accs[acc_idx], rpt=3, sy=first, r=True))
        first = False
  return mads


def rpt3_accum_f13pack_mads(k_order: tuple[int, ...]=(0, 1, 2, 3), acc_order: tuple[int, ...]=(0, 1, 2, 3)) -> list[bytes]:
  accs = ("r5.x", "r4.x", "r3.x", "r2.x")
  avecs = ("r9.x", "r0.y", "r7.x", "r10.x")
  bcols = (("r6.x", "r11.x", "r12.x", "r8.x"),
           ("r6.y", "r11.y", "r12.y", "r8.y"),
           ("r6.z", "r11.z", "r12.z", "r8.z"),
           ("r6.w", "r11.w", "r12.w", "r8.w"))
  mads, first = [], True
  for k_idx in k_order:
    avec, bvec = avecs[k_idx], bcols[k_idx]
    for acc_idx in acc_order:
      mads.append(MAD_F32(accs[acc_idx], bvec[acc_idx], avec, accs[acc_idx], rpt=3, sy=first, r=True))
      first = False
  return mads


def patch_rpt3_accum_f32_unroll(image: bytes, unroll: int) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"rpt3 unroll expects 144 instructions, got {len(instrs)}")
  if 256 % unroll != 0: raise RuntimeError(f"unroll must divide 256, got {unroll}")
  if instrs[48] != ISAM_F32("r5.z", "r6.x", 0, 0):
    raise RuntimeError(f"unexpected A0 load at 48: {instrs[48].hex()}")
  instrs[48] = ISAM_F32("r13.x", "r6.x", 0, 0)
  body = instrs[21:46] + instrs[48:49] + rpt3_accum_mads()

  loop_start = 21
  out = instrs[:21]
  for i in range(unroll):
    if i: out.append(instrs[114])
    out += body
  br_idx = len(out)
  out += [BR(3, inv=False), instrs[114], JUMP(loop_start - (br_idx + 2))]
  out += instrs[116:133]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_accum_f32_unroll2_prefetcha0(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"rpt3 unroll2 prefetch expects 144 instructions, got {len(instrs)}")
  if instrs[48] != ISAM_F32("r5.z", "r6.x", 0, 0):
    raise RuntimeError(f"unexpected A0 load at 48: {instrs[48].hex()}")
  a0_load = ISAM_F32("r13.x", "r6.x", 0, 0)
  mads = rpt3_accum_mads()

  first_body = instrs[21:46] + [a0_load, instrs[21]] + mads[:4] + [a0_load] + mads[4:]
  second_body = instrs[21:46] + mads
  loop_start = 21
  out = instrs[:21] + first_body + [instrs[114]] + second_body
  br_idx = len(out)
  out += [BR(3, inv=False), instrs[114], JUMP(loop_start - (br_idx + 2))]
  out += instrs[116:133]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_accum_f32_f13pack(image: bytes, unroll: int=1) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"rpt3 f13pack expects 144 instructions, got {len(instrs)}")
  if 256 % unroll != 0: raise RuntimeError(f"unroll must divide 256, got {unroll}")
  if instrs[32] != ISAM_F32("r7.z", "r1.y", 0, 0):
    raise RuntimeError(f"unexpected A2 load at 32: {instrs[32].hex()}")
  if instrs[42] != ISAM_F32("r8.z", "r9.x", 1, 1):
    raise RuntimeError(f"unexpected B3 load at 42: {instrs[42].hex()}")
  if instrs[43] != ISAM_F32("r6.z", "r6.z", 1, 1):
    raise RuntimeError(f"unexpected B0 load at 43: {instrs[43].hex()}")
  if instrs[48] != ISAM_F32("r5.z", "r6.x", 0, 0):
    raise RuntimeError(f"unexpected A0 load at 48: {instrs[48].hex()}")

  body = instrs[21:32] + [ISAM_F32("r7.x", "r1.y", 0, 0)] + instrs[33:42] + [
    ISAM_F32("r8.x", "r9.x", 1, 1),
    instrs[44],
    instrs[45],
    ISAM_F32("r9.x", "r6.x", 0, 0),
    ISAM_F32("r6.x", "r6.z", 1, 1),
  ] + rpt3_accum_f13pack_mads(k_order=(1, 2, 3, 0), acc_order=(1, 2, 3, 0))

  loop_start = 21
  out = instrs[:21]
  for i in range(unroll):
    if i: out.append(instrs[114])
    out += body
  br_idx = len(out)
  out += [BR(3, inv=False), instrs[114], JUMP(loop_start - (br_idx + 2))]
  out += instrs[116:133]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_tail_f13(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"rpt3 tail f13 expects 144 instructions, got {len(instrs)}")
  if instrs[48] != ISAM_F32("r5.z", "r6.x", 0, 0):
    raise RuntimeError(f"unexpected A0 load at 48: {instrs[48].hex()}")

  accs = ("r5.x", "r4.x", "r3.x", "r2.x")
  avecs = ("r0.y", "r7.z", "r10.x")
  bcols = (("r6.z", "r11.x", "r12.x", "r8.z"),
           ("r6.w", "r11.y", "r12.y", "r8.w"),
           ("r7.x", "r11.z", "r12.z", "r9.x"),
           ("r7.y", "r11.w", "r12.w", "r9.y"))

  mads = [
    MAD_F32("r4.x", "r11.x", "r5.z", "r4.x", rpt=3, sy=True, r=True),
    MAD_F32("r3.x", "r12.x", "r5.z", "r3.x", rpt=3, r=True),
    MAD_F32("r2.x", "r8.z", "r5.z", "r2.x", rpt=3, r=True),
    MAD_F32("r5.x", "r6.z", "r5.z", "r5.x"),
    MAD_F32("r5.y", "r6.z", "r5.w", "r5.y"),
    MAD_F32("r5.z", "r6.z", "r6.x", "r9.z"),
    MAD_F32("r5.w", "r6.z", "r6.y", "r9.w"),
  ]
  for avec, bvec in zip(avecs, bcols[1:]):
    for acc, b in zip(accs, bvec):
      mads.append(MAD_F32(acc, b, avec, acc, rpt=3, r=True))

  loop_start = 21
  out = instrs[:49] + mads
  br_idx = len(out)
  out += [BR(3, inv=False), instrs[114], JUMP(loop_start - (br_idx + 2))]
  out += instrs[116:133]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_n512_default(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"n512 rpt3 expects 144 instructions, got {len(instrs)}")
  if instrs[47] != ISAM_F32("r8.w", "r6.y", 0, 0):
    raise RuntimeError(f"unexpected N512 A0 load at 47: {instrs[47].hex()}")

  mads = rpt3_n512_mads()

  loop_start = 21
  out = instrs[:48] + mads
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[114:132]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_n384_unroll(image: bytes, unroll: int) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 112: raise RuntimeError(f"n384 unroll expects 112 instructions, got {len(instrs)}")
  if 96 % unroll != 0: raise RuntimeError(f"unroll must divide 96, got {unroll}")
  if instrs[27] != ISAM_F32("r5.w", "r4.w", 0, 0):
    raise RuntimeError(f"unexpected N384 A0 load at 27: {instrs[27].hex()}")
  if instrs[81] != BR(2, inv=False):
    raise RuntimeError(f"unexpected N384 loop branch at 81: {instrs[81].hex()}")

  loop_start = 19
  body = instrs[19:81]
  out = instrs[:loop_start]
  for _ in range(unroll): out += body
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[83:97]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def rpt3_n512_mads(b0_low: bool=False, k_order: tuple[int, ...]=(0, 1, 2, 3), acc_order: tuple[int, ...]=(0, 1, 2, 3)) -> list[bytes]:
  accs = ("r4.z", "r3.z", "r2.z", "r1.z")
  avecs = ("r8.w", "r6.w", "r7.w", "r10.y")
  b0 = ("r5.z", "r5.w", "r6.x", "r6.y") if b0_low else ("r14.y", "r14.z", "r14.w", "r15.x")
  bcols = ((b0[0], "r11.y", "r12.y", "r13.y"),
           (b0[1], "r11.z", "r12.z", "r13.z"),
           (b0[2], "r11.w", "r12.w", "r13.w"),
           (b0[3], "r12.x", "r13.x", "r14.x"))
  mads, first = [], True
  for k_idx in k_order:
    avec, bvec = avecs[k_idx], bcols[k_idx]
    for acc_idx in acc_order:
      mads.append(MAD_F32(accs[acc_idx], bvec[acc_idx], avec, accs[acc_idx], rpt=3, sy=first, r=True))
      first = False
  return mads


def rpt3_n512_f14_mads(k_order: tuple[int, ...]=(2, 3, 0, 1)) -> list[bytes]:
  accs = ("r4.z", "r3.z", "r2.z", "r1.z")
  avecs = ("r8.w", "r6.w", "r7.w", "r9.w")
  bcols = (("r5.z", "r10.w", "r11.w", "r12.w"),
           ("r5.w", "r11.x", "r12.x", "r13.x"),
           ("r6.x", "r11.y", "r12.y", "r13.y"),
           ("r6.y", "r11.z", "r12.z", "r13.z"))
  mads, first = [], True
  for k_idx in k_order:
    avec, bvec = avecs[k_idx], bcols[k_idx]
    for acc_idx in (1, 2, 3, 0):
      mads.append(MAD_F32(accs[acc_idx], bvec[acc_idx], avec, accs[acc_idx], rpt=3, sy=first, r=True))
      first = False
  return mads


def patch_rpt3_n512_tight(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"n512 tight expects 144 instructions, got {len(instrs)}")
  if instrs[47] != ISAM_F32("r8.w", "r6.y", 0, 0):
    raise RuntimeError(f"unexpected N512 A0 load at 47: {instrs[47].hex()}")

  loop_start = 21
  out = instrs[:21] + rpt3_n512_tight_body(instrs)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[114:132]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def rpt3_n512_tight_body(instrs: list[bytes]) -> list[bytes]:
  # Drop dead scalar copies left behind after the 4-wide rpt3 accumulator rewrite.
  return [instrs[i] for i in (21, 23, 25, 28, 29, 30, 31, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)] + rpt3_n512_mads()


def patch_rpt3_n512_unroll(image: bytes, unroll: int) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"n512 unroll expects 144 instructions, got {len(instrs)}")
  if 128 % unroll != 0: raise RuntimeError(f"unroll must divide 128, got {unroll}")
  if instrs[47] != ISAM_F32("r8.w", "r6.y", 0, 0):
    raise RuntimeError(f"unexpected N512 A0 load at 47: {instrs[47].hex()}")

  body = instrs[21:48] + rpt3_n512_mads()

  loop_start = 21
  out = instrs[:21]
  for _ in range(unroll): out += body
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[114:132]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_n512_tight_unroll(image: bytes, unroll: int) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"n512 tight unroll expects 144 instructions, got {len(instrs)}")
  if 128 % unroll != 0: raise RuntimeError(f"unroll must divide 128, got {unroll}")
  if instrs[47] != ISAM_F32("r8.w", "r6.y", 0, 0):
    raise RuntimeError(f"unexpected N512 A0 load at 47: {instrs[47].hex()}")

  body = rpt3_n512_tight_body(instrs)
  loop_start = 21
  out = instrs[:21]
  for _ in range(unroll): out += body
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[114:132]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_n512_variant(image: bytes, drop: tuple[int, ...]=(), nop: tuple[int, ...]=(), unroll: int=1) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"n512 variant expects 144 instructions, got {len(instrs)}")
  if 128 % unroll != 0: raise RuntimeError(f"unroll must divide 128, got {unroll}")
  if instrs[47] != ISAM_F32("r8.w", "r6.y", 0, 0):
    raise RuntimeError(f"unexpected N512 A0 load at 47: {instrs[47].hex()}")

  drop_set, nop_set = set(drop), set(nop)
  if drop_set & nop_set: raise RuntimeError(f"overlapping drop/nop indices: {sorted(drop_set & nop_set)}")
  body = [(NOP() if i in nop_set else instrs[i]) for i in range(21, 48) if i not in drop_set] + rpt3_n512_mads()

  loop_start = 21
  out = instrs[:21]
  for _ in range(unroll): out += body
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[114:132]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_n512_b0low(image: bytes, unroll: int=1, b0_last: bool=False, k_order: tuple[int, ...]=(1, 2, 3, 0), drop_ssnop: bool=False, last_cmp_only: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"n512 b0low expects 144 instructions, got {len(instrs)}")
  if 128 % unroll != 0: raise RuntimeError(f"unroll must divide 128, got {unroll}")
  if instrs[44] != ISAM_F32("r14.y", "r6.w", 1, 1):
    raise RuntimeError(f"unexpected N512 B0 load at 44: {instrs[44].hex()}")
  if instrs[47] != ISAM_F32("r8.w", "r6.y", 0, 0):
    raise RuntimeError(f"unexpected N512 A0 load at 47: {instrs[47].hex()}")

  acc_order = (1, 2, 3, 0) if b0_last else (0, 1, 2, 3)
  body = instrs[21:44] + ([] if drop_ssnop else instrs[45:46]) + instrs[46:48] + [ISAM_F32("r5.z", "r6.w", 1, 1)] + rpt3_n512_mads(b0_low=True, k_order=k_order, acc_order=acc_order)
  body_nocmp = [NOP(rpt=1) if ins == instrs[39] else ins for ins in body] if last_cmp_only else body
  loop_start = 21
  out = instrs[:21]
  for i in range(unroll): out += body if i == unroll - 1 else body_nocmp
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[114:132]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_n512_f14(image: bytes, unroll: int=1) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"n512 f14 expects 144 instructions, got {len(instrs)}")
  if 128 % unroll != 0: raise RuntimeError(f"unroll must divide 128, got {unroll}")
  if instrs[40] != ISAM_F32("r10.y", "r5.w", 0, 0):
    raise RuntimeError(f"unexpected N512 A3 load at 40: {instrs[40].hex()}")
  if instrs[41] != ISAM_F32("r11.y", "r7.y", 1, 1):
    raise RuntimeError(f"unexpected N512 B1 load at 41: {instrs[41].hex()}")
  if instrs[42] != ISAM_F32("r12.y", "r8.w", 1, 1):
    raise RuntimeError(f"unexpected N512 B2 load at 42: {instrs[42].hex()}")
  if instrs[43] != ISAM_F32("r13.y", "r9.y", 1, 1):
    raise RuntimeError(f"unexpected N512 B3 load at 43: {instrs[43].hex()}")

  body = instrs[21:40] + [
    ISAM_F32("r9.w", "r5.w", 0, 0),
    ISAM_F32("r10.w", "r7.y", 1, 1),
    ISAM_F32("r11.w", "r8.w", 1, 1),
    ISAM_F32("r12.w", "r9.y", 1, 1),
    instrs[45],
    instrs[47],
    ISAM_F32("r5.z", "r6.w", 1, 1),
  ] + rpt3_n512_f14_mads(k_order=(2, 3)) + [instrs[46]] + rpt3_n512_f14_mads(k_order=(0, 1))
  loop_start = 21
  out = instrs[:21]
  for _ in range(unroll): out += body
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[114:132]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def l25_loop_count(image: bytes) -> int:
  lines = clean_disasm_lines(image)
  if (m:=re.search(r"cmps\.s\.ge p0\.x, r10\.x, (\d+)", lines[42])) is None:
    raise RuntimeError(f"unexpected l25 cmp at 42: {lines[42]}")
  return int(m.group(1))


def replace_freg_src(instr: bytes, src: str) -> bytes:
  lo = (int.from_bytes(instr[:4], "little") & ~0xff) | (freg_idx(src) & 0xff)
  return lo.to_bytes(4, "little") + instr[4:]


def replace_freg_dst(instr: bytes, dst: str) -> bytes:
  hi = (int.from_bytes(instr[4:], "little") & ~0xff) | (freg_idx(dst) & 0xff)
  return instr[:4] + hi.to_bytes(4, "little")


def replace_isam_coord(instr: bytes, coord: str) -> bytes:
  lo = (int.from_bytes(instr[:4], "little") & ~0x1ff) | ((freg_idx(coord) * 2 + 1) & 0x1ff)
  return lo.to_bytes(4, "little") + instr[4:]


def ADD_U_REG(dst: str, src1: str, src2: str) -> bytes:
  return (((freg_idx(src2) & 0xff) << 16) | (freg_idx(src1) & 0xff)).to_bytes(4, "little") + (0x42100000 | (freg_idx(dst) & 0xff)).to_bytes(4, "little")


def ADD_U_IMM(dst: str, src: str, imm: int, ss: bool=False, nop: int=0) -> bytes:
  hi = 0x42100000 | (freg_idx(dst) & 0xff)
  if ss: hi |= 0x1000
  if nop & 1: hi |= 0x0800
  return ((0x20 << 24) | ((imm & 0xff) << 16) | (freg_idx(src) & 0xff)).to_bytes(4, "little") + hi.to_bytes(4, "little")


def MOV_U32(dst: str, src: str) -> bytes:
  return (freg_idx(src) & 0xff).to_bytes(4, "little") + (0x200cc000 | (freg_idx(dst) & 0xff)).to_bytes(4, "little")


def isam_f16_to_f32_vec(dst: str, coord: str, tex: int) -> list[bytes]:
  base = freg_idx(dst)
  comps = "xyzw"
  return [ISAM_F16("hr0.x", coord, tex, tex)] + [COV_F16F32(freg_name(base + i), f"hr0.{comps[i]}", sy=i == 0) for i in range(4)]


def rpt3_l25_mads(k_order: tuple[int, ...]=(0, 1, 2, 3), acc_order: tuple[int, ...]=(0, 1, 2, 3), sy_first: bool=True) -> list[bytes]:
  accs = ("r5.w", "r4.w", "r3.w", "r2.w")
  avecs = ("r6.w", "r8.x", "r9.x", "r10.w")
  bcols = (("r14.w", "r11.w", "r12.w", "r13.w"),
           ("r15.x", "r12.x", "r13.x", "r14.x"),
           ("r15.y", "r12.y", "r13.y", "r14.y"),
           ("r15.z", "r12.z", "r13.z", "r14.z"))
  mads, first = [], sy_first
  for k_idx in k_order:
    avec, bvec = avecs[k_idx], bcols[k_idx]
    for acc_idx in acc_order:
      mads.append(MAD_F32(accs[acc_idx], bvec[acc_idx], avec, accs[acc_idx], rpt=3, sy=first, r=True))
      first = False
  return mads


def rpt3_l25_mads_accmajor(acc_order: tuple[int, ...]=(0, 1, 2, 3), k_order: tuple[int, ...]=(0, 1, 2, 3), sy_first: bool=True) -> list[bytes]:
  accs = ("r5.w", "r4.w", "r3.w", "r2.w")
  avecs = ("r6.w", "r8.x", "r9.x", "r10.w")
  bcols = (("r14.w", "r11.w", "r12.w", "r13.w"),
           ("r15.x", "r12.x", "r13.x", "r14.x"),
           ("r15.y", "r12.y", "r13.y", "r14.y"),
           ("r15.z", "r12.z", "r13.z", "r14.z"))
  mads, first = [], sy_first
  for acc_idx in acc_order:
    for k_idx in k_order:
      mads.append(MAD_F32(accs[acc_idx], bcols[k_idx][acc_idx], avecs[k_idx], accs[acc_idx], rpt=3, sy=first, r=True))
      first = False
  return mads


def patch_rpt3_l25(image: bytes, unroll: int=1, k_order: tuple[int, ...]=(0, 1, 2, 3), acc_order: tuple[int, ...]=(0, 1, 2, 3), drop_ssnop: bool=False, no_store: bool=False, drop_startnop: bool=False, drop_postcmp_nop: bool=False, skip_a_loads: bool=False, skip_b_loads: bool=False, last_cmp_nop: int|None=None) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 rpt3 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[40] != ISAM_F32("r9.x", "r7.x", 0, 0):
    raise RuntimeError(f"unexpected l25 A2 load at 40: {instrs[40].hex()}")
  if instrs[44] != ISAM_F32("r10.w", "r7.x", 0, 0):
    raise RuntimeError(f"unexpected l25 A3 load at 44: {instrs[44].hex()}")
  if instrs[48] != ISAM_F32("r14.w", "r2.y", 1, 1):
    raise RuntimeError(f"unexpected l25 B0 load at 48: {instrs[48].hex()}")
  if instrs[50] != ISAM_F32("r8.x", "r10.y", 0, 0):
    raise RuntimeError(f"unexpected l25 A1 load at 50: {instrs[50].hex()}")
  if instrs[51] != ISAM_F32("r6.w", "r7.z", 0, 0):
    raise RuntimeError(f"unexpected l25 A0 load at 51: {instrs[51].hex()}")

  skip_a, skip_b = {40, 44, 50, 51}, {45, 46, 47, 48}
  def make_body(use_cmp: bool=True) -> list[bytes]:
    def body_instr(i: int) -> bytes:
      if (skip_a_loads and i in skip_a) or (skip_b_loads and i in skip_b): return NOP()
      if i == 42 and not use_cmp: return NOP(rpt=last_cmp_nop or 0)
      return instrs[i]
    body_prefix = [body_instr(i) for i in range(25, 49) if not (drop_startnop and i == 25) and not (drop_postcmp_nop and i == 43)]
    return body_prefix + ([] if drop_ssnop else instrs[49:50]) + [body_instr(i) for i in range(50, 52)] + rpt3_l25_mads(k_order=k_order, acc_order=acc_order)
  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll):
    out += make_body(last_cmp_nop is None or i == unroll - 1)
    if i != unroll - 1: out.append(instrs[117])
  br_idx = len(out)
  out += [BR(3, inv=False), instrs[117], JUMP(loop_start - (br_idx + 2))]
  out += [(NOP() if no_store and i in (126, 129, 131, 132) else instrs[i]) for i in range(119, 134)]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc(image: bytes, unroll: int=1, drop_ssnop: bool=False, k_order: tuple[int, ...]=(0, 1, 2, 3), acc_order: tuple[int, ...]=(0, 1, 2, 3), no_store: bool=False, skip_a_loads: bool=False, skip_b_loads: bool=False, acc_major: bool=False, hoist_y: tuple[int, ...]=(), load_order: str="default", a_f16: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 postinc expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[39] != bytes.fromhex("0900012028081042"):
    raise RuntimeError(f"unexpected l25 next-k add at 39: {instrs[39].hex()}")

  skip_a, skip_b = {40, 44, 50, 51}, {45, 46, 47, 48}
  hoist_set = set(hoist_y)
  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  def make_body(use_cmp: bool) -> list[bytes]:
    def body_instrs(i: int) -> list[bytes]:
      if (skip_a_loads and i in skip_a) or (skip_b_loads and i in skip_b): return [NOP()]
      if a_f16 and i == 40: return isam_f16_to_f32_vec("r9.x", "r7.x", 0)
      if a_f16 and i == 44: return isam_f16_to_f32_vec("r10.w", "r7.x", 0)
      if a_f16 and i == 50: return isam_f16_to_f32_vec("r8.x", "r10.y", 0)
      if a_f16 and i == 51: return isam_f16_to_f32_vec("r6.w", "r7.z", 0)
      if load_order == "bcoords_f17" and i == 29 and 31 in hoist_set: return [replace_freg_dst(instrs[i], "r15.w")]
      if load_order == "bcoords_f17" and i == 32 and 33 in hoist_set: return [replace_freg_dst(instrs[i], "r16.y")]
      if load_order == "bcoords_f17" and i == 46 and 31 in hoist_set: return [replace_isam_coord(instrs[i], "r15.w")]
      if load_order == "bcoords_f17" and i == 47 and 33 in hoist_set: return [replace_isam_coord(instrs[i], "r16.y")]
      if load_order == "bcoords_lowpair_f17" and i == 29 and 31 in hoist_set: return [replace_freg_dst(instrs[i], "r1.x")]
      if load_order == "bcoords_lowpair_f17" and i == 32 and 33 in hoist_set: return [replace_freg_dst(instrs[i], "r0.x")]
      if load_order == "bcoords_lowpair_f17" and i == 46 and 31 in hoist_set: return [replace_isam_coord(instrs[i], "r1.x")]
      if load_order == "bcoords_lowpair_f17" and i == 47 and 33 in hoist_set: return [replace_isam_coord(instrs[i], "r0.x")]
      return [instrs[i]]
    if load_order in ("default", "bcoords_f17", "bcoords_lowpair_f17"):
      body = [op for i in range(25, 49) if i not in (25, 39, 42, 43) and i not in hoist_set for op in body_instrs(i)]
    else:
      body = [op for i in range(26, 39) if i not in hoist_set for op in body_instrs(i)]
      if load_order == "a0early": load_idxs = (40, 41, 44, 51, 45, 46, 47, 48, 50)
      elif load_order == "bfirst": load_idxs = (45, 46, 47, 48, 40, 41, 44, 50, 51)
      elif load_order == "a1mid": load_idxs = (40, 41, 44, 46, 47, 50, 51, 45, 48)
      elif load_order in ("a1copyearly", "a1copyearly_wait", "a1copyearly_f17"):
        # A1 overwrites r8.x-r8.w, which are still needed as B2/B3 coords.
        # Copy those coord pairs after A3, because A3 writes r10.w-r11.z.
        load_idxs = (40, 41, 44, 50, 51, 47, 46, 45, 48)
      else: raise ValueError(f"unknown l25 postinc load_order {load_order!r}")
      for i in load_idxs:
        if load_order == "a1copyearly" and i == 47:
          body.append(ISAM_F32("r13.w", "r11.z", 1, 1))
        elif load_order == "a1copyearly_wait" and i == 47:
          body += [ISAM_F32("r13.w", "r11.z", 1, 1), NOP(rpt=3)]
        elif load_order == "a1copyearly_f17" and i == 47:
          body.append(ISAM_F32("r13.w", "r16.y", 1, 1))
        elif load_order in ("a1copyearly", "a1copyearly_wait") and i == 46:
          body.append(ISAM_F32("r12.w", "r11.x", 1, 1))
        elif load_order == "a1copyearly_f17" and i == 46:
          body.append(ISAM_F32("r12.w", "r15.w", 1, 1))
        else:
          body += body_instrs(i)
        if load_order in ("a1copyearly", "a1copyearly_wait", "a1copyearly_f17") and i == 44:
          if load_order == "a1copyearly_f17":
            body += [MOV_U32("r15.w", "r8.x"), MOV_U32("r16.x", "r8.y"), MOV_U32("r16.y", "r8.z"), MOV_U32("r16.z", "r8.w")]
          else:
            body += [MOV_U32("r11.x", "r8.x"), MOV_U32("r11.y", "r8.y"), MOV_U32("r11.z", "r8.z"), MOV_U32("r11.w", "r8.w")]
    mads = rpt3_l25_mads_accmajor(acc_order=acc_order, k_order=k_order) if acc_major else rpt3_l25_mads(k_order=k_order, acc_order=acc_order)
    body += [] if drop_ssnop else instrs[49:50]
    if load_order in ("default", "bcoords_f17", "bcoords_lowpair_f17"): body += [op for i in range(50, 52) for op in body_instrs(i)]
    body += [ADD_S("r2.y", "r2.y", 1)] + mads
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  if load_order == "bcoords_lowpair_f17":
    out += [MOV_U32("r15.w", "r1.x"), MOV_U32("r10.x", "r0.y")]
    loop_start = len(out)
  if hoist_set & {36, 37}: out.append(instrs[27])
  if load_order == "bcoords_f17":
    if 31 in hoist_set: out.append(MOV_U32("r16.x", "r1.w"))
    if 33 in hoist_set: out.append(MOV_U32("r16.z", "r1.w"))
    out += [instrs[i] for i in (35, 36, 37) if i in hoist_set]
  elif load_order == "bcoords_lowpair_f17":
    if 31 in hoist_set: out.append(MOV_U32("r1.y", "r1.w"))
    if 33 in hoist_set: out.append(MOV_U32("r0.y", "r1.w"))
    out += [instrs[i] for i in (35, 36, 37) if i in hoist_set]
  else:
    out += [instrs[i] for i in (31, 33, 35, 36, 37) if i in hoist_set]
  for i in range(unroll): out += make_body(i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  if load_order == "bcoords_lowpair_f17": out += [MOV_U32("r1.x", "r15.w"), MOV_U32("r0.y", "r10.x")]
  out += [(NOP() if no_store and i in (126, 129, 131, 132) else instrs[i]) for i in range(119, 134)]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_incoords(image: bytes, unroll: int=22, drop_ssnop: bool=True, no_store: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 postinc incoords expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[39] != bytes.fromhex("0900012028081042"):
    raise RuntimeError(f"unexpected l25 next-k add at 39: {instrs[39].hex()}")
  cmp_r2y, mads = replace_freg_src(instrs[42], "r2.y"), rpt3_l25_mads()

  def tail(use_cmp: bool) -> list[bytes]:
    body = [ADD_S("r2.y", "r2.y", 1)] + mads
    if use_cmp: body.append(cmp_r2y)
    return body

  def full_body(use_cmp: bool) -> list[bytes]:
    body = [instrs[i] for i in range(25, 49) if i not in (25, 39, 42, 43)]
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:52]
    return body + tail(use_cmp)

  def inc_body(use_cmp: bool) -> list[bytes]:
    body = [
      ADD_U_IMM("r10.y", "r10.y", 4),
      ADD_S("r7.z", "r10.y", -1),
      MOV_U32("r7.y", "r10.z"),
      ADD_U_IMM("r7.x", "r7.z", 2),
      instrs[28], instrs[29], instrs[31], instrs[32], instrs[33],
      instrs[40],
      ADD_U_IMM("r7.x", "r7.z", 3, ss=True, nop=1),
      instrs[44], instrs[45], instrs[46], instrs[47], instrs[48],
    ]
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:52]
    return body + tail(use_cmp)

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += (full_body if i == 0 else inc_body)(i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += [(NOP() if no_store and i in (126, 129, 131, 132) else instrs[i]) for i in range(119, 134)]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0(image: bytes, unroll: int=1, drop_ssnop: bool=False, prefetch_wait: int=0, prefetch_ss: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads()
  prefetch_a0 = [SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r7.z", "r0.w", "r10.x"), ISAM_F32("r6.w", "r7.z", 0, 0)]

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = []
    for i in range(25, 49):
      if i in (25, 39, 42, 43): continue
      if i == 26: body.append(SHL_B("r10.x", "r2.y", 2)); continue
      if i == 30: body.append(ADD_U_REG("r7.z", "r0.w", "r10.x")); continue
      body.append(instrs[i])
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:51]
    if load_a0: body.append(instrs[51])
    body.append(ADD_S("r2.y", "r2.y", 1))
    body += mads[:4]
    if prefetch_next and prefetch_wait: body.append(NOP(rpt=prefetch_wait))
    if prefetch_next and prefetch_ss: body.append(instrs[49])
    if prefetch_next: body += prefetch_a0
    body += mads[4:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0, i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0_safe(image: bytes, unroll: int=1, drop_ssnop: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 safe prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads()

  def acoord_setup() -> list[bytes]:
    return [
      SHL_B("r6.x", "r2.y", 2),
      instrs[28], instrs[29],
      ADD_U_REG("r6.y", "r0.w", "r6.x"),
      MOV_F32("r6.x", "r48.x"),
      instrs[31], instrs[32], instrs[33],
      ADD_S("r6.z", "r6.y", 2),
      instrs[35],
      MOV_F32("r10.z", "r6.x"),
      MOV_F32("r7.w", "r6.x"),
      ADD_S("r10.y", "r6.y", 1),
    ]

  def prefetch_next_a0() -> list[bytes]:
    return [SHL_B("r6.x", "r2.y", 2), ADD_U_REG("r6.y", "r0.w", "r6.x"), ISAM_F32("r6.w", "r6.y", 0, 0)]

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = acoord_setup() + [
      ISAM_F32("r9.x", "r6.z", 0, 0),
      ADD_S("r10.x", "r6.y", 3),
      ISAM_F32("r10.w", "r10.x", 0, 0),
    ] + instrs[45:49]
    body += [] if drop_ssnop else instrs[49:50]
    body += [instrs[50]]
    if load_a0: body.append(ISAM_F32("r6.w", "r6.y", 0, 0))
    body.append(ADD_S("r2.y", "r2.y", 1))
    body += mads
    if prefetch_next: body += prefetch_next_a0()
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0, i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0_bfirst(image: bytes, unroll: int=1, b_wait: int=0, a0_wait: int=0) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 bfirst prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[51] != ISAM_F32("r6.w", "r7.z", 0, 0):
    raise RuntimeError(f"unexpected l25 A0 load at 51: {instrs[51].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads()

  def b_loads() -> list[bytes]:
    return [instrs[i] for i in (28, 29, 31, 32, 33, 35, 45, 46, 47, 48)]

  def load_a0_current() -> list[bytes]:
    return [SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r7.z", "r0.w", "r10.x"), MOV_U32("r7.w", "r48.x"), ISAM_F32("r6.w", "r7.z", 0, 0)]

  def load_a123_current() -> list[bytes]:
    return [
      SHL_B("r6.w", "r2.y", 2), instrs[27], ADD_U_REG("r7.z", "r0.w", "r6.w"),
      instrs[34], instrs[36], instrs[37], instrs[38], instrs[40], instrs[41], instrs[44], instrs[50],
    ]

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = b_loads()
    if load_a0: body += load_a0_current()
    if b_wait: body.append(NOP(rpt=b_wait))
    body += mads[:4]
    if a0_wait: body.append(NOP(rpt=a0_wait))
    body += load_a123_current()
    body.append(ADD_S("r2.y", "r2.y", 1))
    if prefetch_next: body += load_a0_current()
    body += mads[4:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0, i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0_r11coord(image: bytes, unroll: int=1, do_prefetch: bool=True) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 r11coord prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[51] != ISAM_F32("r6.w", "r7.z", 0, 0):
    raise RuntimeError(f"unexpected l25 A0 load at 51: {instrs[51].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads()

  def load_a0_current() -> list[bytes]:
    return [SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r7.z", "r0.w", "r10.x"), MOV_U32("r7.w", "r48.x"), ISAM_F32("r6.w", "r7.z", 0, 0)]

  def load_a123_current() -> list[bytes]:
    return [
      SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r11.z", "r0.w", "r10.x"), MOV_U32("r11.y", "r48.x"),
      ADD_U_IMM("r11.x", "r11.z", 2), ISAM_F32("r9.x", "r11.x", 0, 0),
      ADD_U_IMM("r11.x", "r11.z", 3, ss=True, nop=1), ISAM_F32("r10.w", "r11.x", 0, 0),
      ADD_U_IMM("r10.y", "r11.z", 1), MOV_U32("r10.z", "r48.x"), ISAM_F32("r8.x", "r10.y", 0, 0),
    ]

  def b_loads() -> list[bytes]:
    return [instrs[i] for i in (28, 29, 31, 32, 33, 35, 45, 46, 47, 48)]

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = load_a123_current()
    if load_a0: body += load_a0_current()
    body += b_loads() + [ADD_S("r2.y", "r2.y", 1)] + mads[:4]
    if prefetch_next: body += load_a0_current()
    body += mads[4:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0 or not do_prefetch, do_prefetch and i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0_delaya1(image: bytes, unroll: int=1, do_prefetch: bool=True) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 delaya1 prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[51] != ISAM_F32("r6.w", "r7.z", 0, 0):
    raise RuntimeError(f"unexpected l25 A0 load at 51: {instrs[51].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads(k_order=(0, 2, 3, 1))

  def b_loads() -> list[bytes]:
    return [instrs[i] for i in (28, 29, 31, 32, 33, 35, 45, 46, 47, 48)]

  def load_a0_current() -> list[bytes]:
    return [SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r7.z", "r0.w", "r10.x"), MOV_U32("r7.w", "r48.x"), ISAM_F32("r6.w", "r7.z", 0, 0)]

  def load_a23_current() -> list[bytes]:
    return [
      SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r7.w", "r0.w", "r10.x"), MOV_U32("r10.y", "r48.x"),
      ADD_U_IMM("r10.x", "r7.w", 2), ISAM_F32("r9.x", "r10.x", 0, 0),
      ADD_U_IMM("r10.x", "r7.w", 3, ss=True, nop=1), ISAM_F32("r10.w", "r10.x", 0, 0),
    ]

  def load_a1_current() -> list[bytes]:
    return [
      SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r10.y", "r0.w", "r10.x"), ADD_U_IMM("r10.y", "r10.y", 1),
      MOV_U32("r10.z", "r48.x"), ISAM_F32("r8.x", "r10.y", 0, 0),
    ]

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = b_loads()
    if load_a0: body += load_a0_current()
    body += load_a23_current()
    body += mads[:12] + load_a1_current() + [ADD_S("r2.y", "r2.y", 1)]
    if prefetch_next: body += load_a0_current()
    body += mads[12:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0 or not do_prefetch, do_prefetch and i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0_low(image: bytes, unroll: int=1, drop_ssnop: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 low prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[26] != SHL_B("r6.w", "r2.y", 2):
    raise RuntimeError(f"unexpected l25 shift at 26: {instrs[26].hex()}")
  if instrs[30] != ADD_U_REG("r7.z", "r0.w", "r6.w"):
    raise RuntimeError(f"unexpected l25 A-base add at 30: {instrs[30].hex()}")
  if instrs[51] != ISAM_F32("r6.w", "r7.z", 0, 0):
    raise RuntimeError(f"unexpected l25 A0 load at 51: {instrs[51].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = []
  first = True
  accs = ("r5.w", "r4.w", "r3.w", "r2.w")
  bcols = (("r14.w", "r11.w", "r12.w", "r13.w"),
           ("r15.x", "r12.x", "r13.x", "r14.x"),
           ("r15.y", "r12.y", "r13.y", "r14.y"),
           ("r15.z", "r12.z", "r13.z", "r14.z"))
  for avec, bvec in zip(("r6.x", "r8.x", "r9.x", "r10.w"), bcols):
    for acc, b in zip(accs, bvec):
      mads.append(MAD_F32(acc, b, avec, acc, rpt=3, sy=first, r=True))
      first = False

  def prefetch_a0() -> list[bytes]:
    return [SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r7.z", "r0.w", "r10.x"), ISAM_F32("r6.x", "r7.z", 0, 0)]

  def body_prefix(load_a0: bool) -> list[bytes]:
    body = []
    for i in range(25, 49):
      if i in (25, 39, 42, 43): continue
      if i == 26: body.append(SHL_B("r10.x", "r2.y", 2)); continue
      if i == 30: body.append(ADD_U_REG("r7.z", "r0.w", "r10.x")); continue
      body.append(instrs[i])
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:51]
    if load_a0: body.append(ISAM_F32("r6.x", "r7.z", 0, 0))
    return body

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = body_prefix(load_a0) + [ADD_S("r2.y", "r2.y", 1)] + mads[:4]
    if prefetch_next: body += prefetch_a0()
    body += mads[4:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0, i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0_r15w(image: bytes, unroll: int=1, drop_ssnop: bool=False, prefetch_after: int=4) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 r15w prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if prefetch_after < 4 or prefetch_after > 16:
    raise RuntimeError(f"prefetch_after must be in [4, 16], got {prefetch_after}")
  if instrs[51] != ISAM_F32("r6.w", "r7.z", 0, 0):
    raise RuntimeError(f"unexpected l25 A0 load at 51: {instrs[51].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = []
  first = True
  accs = ("r5.w", "r4.w", "r3.w", "r2.w")
  bcols = (("r14.w", "r11.w", "r12.w", "r13.w"),
           ("r15.x", "r12.x", "r13.x", "r14.x"),
           ("r15.y", "r12.y", "r13.y", "r14.y"),
           ("r15.z", "r12.z", "r13.z", "r14.z"))
  for avec, bvec in zip(("r15.w", "r8.x", "r9.x", "r10.w"), bcols):
    for acc, b in zip(accs, bvec):
      mads.append(MAD_F32(acc, b, avec, acc, rpt=3, sy=first, r=True))
      first = False

  def prefetch_a0() -> list[bytes]:
    return [SHL_B("r6.w", "r2.y", 2), ADD_U_REG("r7.z", "r0.w", "r6.w"), ISAM_F32("r15.w", "r7.z", 0, 0)]

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = [instrs[i] for i in range(25, 49) if i not in (25, 39, 42, 43)]
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:51]
    if load_a0: body.append(ISAM_F32("r15.w", "r7.z", 0, 0))
    body.append(ADD_S("r2.y", "r2.y", 1))
    body += mads[:prefetch_after]
    if prefetch_next: body += prefetch_a0()
    body += mads[prefetch_after:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0, i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha0_pair(image: bytes, unroll: int=1, drop_ssnop: bool=False, prefetch_after: int=4, do_prefetch: bool=True) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 pair prefetch-a0 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if prefetch_after < 4 or prefetch_after > 16:
    raise RuntimeError(f"prefetch_after must be in [4, 16], got {prefetch_after}")
  if instrs[51] != ISAM_F32("r6.w", "r7.z", 0, 0):
    raise RuntimeError(f"unexpected l25 A0 load at 51: {instrs[51].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads()

  def b_loads() -> list[bytes]:
    return [instrs[i] for i in (28, 29, 31, 32, 33, 35)] + instrs[45:49]

  def a23_loads() -> list[bytes]:
    return [
      SHL_B("r10.x", "r2.y", 2),
      ADD_U_REG("r8.z", "r0.w", "r10.x"),
      MOV_U32("r8.y", "r48.x"),
      ADD_U_IMM("r8.x", "r8.z", 2),
      ISAM_F32("r9.x", "r8.x", 0, 0),
      ADD_U_IMM("r8.x", "r8.z", 3, ss=True, nop=1),
      ISAM_F32("r10.w", "r8.x", 0, 0),
    ]

  def a01_loads(load_a0: bool) -> list[bytes]:
    body = []
    if load_a0:
      body += [MOV_U32("r7.z", "r8.z"), MOV_U32("r7.w", "r48.x"), ISAM_F32("r6.w", "r7.z", 0, 0)]
    body += [ADD_U_IMM("r10.y", "r8.z", 1), MOV_U32("r10.z", "r48.x"), ISAM_F32("r8.x", "r10.y", 0, 0)]
    return body

  def prefetch_a0() -> list[bytes]:
    return [SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r7.z", "r0.w", "r10.x"), MOV_U32("r7.w", "r48.x"), ISAM_F32("r6.w", "r7.z", 0, 0)]

  def make_body(load_a0: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = b_loads() + a23_loads()
    body += [] if drop_ssnop else instrs[49:50]
    body += a01_loads(load_a0)
    body.append(ADD_S("r2.y", "r2.y", 1))
    body += mads[:prefetch_after]
    if prefetch_next: body += prefetch_a0()
    body += mads[prefetch_after:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0 or not do_prefetch, do_prefetch and i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha2(image: bytes, unroll: int=1, drop_ssnop: bool=False, prefetch_after: int=12) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 prefetch-a2 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if prefetch_after < 12 or prefetch_after > 16:
    raise RuntimeError(f"prefetch_after must be in [12, 16], got {prefetch_after}")
  if instrs[40] != ISAM_F32("r9.x", "r7.x", 0, 0):
    raise RuntimeError(f"unexpected l25 A2 load at 40: {instrs[40].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads()

  def prefetch_a2() -> list[bytes]:
    return [
      SHL_B("r10.x", "r2.y", 2),
      ADD_U_REG("r10.x", "r0.w", "r10.x"),
      MOV_U32("r10.y", "r48.x"),
      ADD_U_IMM("r10.x", "r10.x", 2),
      ISAM_F32("r9.x", "r10.x", 0, 0),
    ]

  def make_body(load_a2: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = []
    for i in range(25, 49):
      if i in (25, 39, 42, 43): continue
      if i == 40 and not load_a2: continue
      body.append(instrs[i])
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:52] + [ADD_S("r2.y", "r2.y", 1)] + mads[:prefetch_after]
    if prefetch_next: body += prefetch_a2()
    body += mads[prefetch_after:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0, i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_prefetcha3(image: bytes, unroll: int=1, drop_ssnop: bool=False, prefetch_after: int=4) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 prefetch-a3 expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if prefetch_after < 4 or prefetch_after > 16:
    raise RuntimeError(f"prefetch_after must be in [4, 16], got {prefetch_after}")
  if instrs[44] != ISAM_F32("r10.w", "r7.x", 0, 0):
    raise RuntimeError(f"unexpected l25 A3 load at 44: {instrs[44].hex()}")

  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  mads = rpt3_l25_mads(k_order=(3, 0, 1, 2))

  def prefetch_a3() -> list[bytes]:
    return [
      SHL_B("r10.x", "r2.y", 2), ADD_U_REG("r10.x", "r0.w", "r10.x"), MOV_U32("r10.y", "r48.x"),
      ADD_U_IMM("r10.x", "r10.x", 3, ss=True, nop=1), ISAM_F32("r10.w", "r10.x", 0, 0),
    ]

  def make_body(load_a3: bool, prefetch_next: bool, use_cmp: bool) -> list[bytes]:
    body = []
    for i in range(25, 49):
      if i in (25, 39, 42, 43): continue
      if i == 44 and not load_a3: continue
      body.append(instrs[i])
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:52] + [ADD_S("r2.y", "r2.y", 1)] + mads[:prefetch_after]
    if prefetch_next: body += prefetch_a3()
    body += mads[prefetch_after:]
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == 0, i != unroll - 1, i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def rpt3_l25_b0low_split_mads(sy_after_b0: bool=False) -> tuple[list[bytes], list[bytes]]:
  accs = ("r5.w", "r4.w", "r3.w", "r2.w")
  avecs = ("r6.w", "r8.x", "r9.x", "r10.w")
  b0_or_b3 = ("r13.w", "r14.x", "r14.y", "r14.z")
  b1 = ("r11.w", "r12.x", "r12.y", "r12.z")
  b2 = ("r12.w", "r13.x", "r13.y", "r13.z")
  first, before, after = True, [], []
  for k_idx in range(4):
    before.append(MAD_F32(accs[3], b0_or_b3[k_idx], avecs[k_idx], accs[3], rpt=3, sy=first, r=True))
    first = False
  for k_idx in range(4):
    after.append(MAD_F32(accs[1], b1[k_idx], avecs[k_idx], accs[1], rpt=3, sy=sy_after_b0 and k_idx == 0, r=True))
    after.append(MAD_F32(accs[2], b2[k_idx], avecs[k_idx], accs[2], rpt=3, r=True))
  for k_idx in range(4): after.append(MAD_F32(accs[0], b0_or_b3[k_idx], avecs[k_idx], accs[0], rpt=3, r=True))
  return before, after


def rpt3_l25_b0firstlow_split_mads() -> tuple[list[bytes], list[bytes]]:
  accs = ("r5.w", "r4.w", "r3.w", "r2.w")
  avecs = ("r6.w", "r8.x", "r9.x", "r10.w")
  b0_or_b3 = ("r13.w", "r14.x", "r14.y", "r14.z")
  b1 = ("r11.w", "r12.x", "r12.y", "r12.z")
  b2 = ("r12.w", "r13.x", "r13.y", "r13.z")
  first, before, after = True, [], []
  for k_idx in range(4):
    before.append(MAD_F32(accs[0], b0_or_b3[k_idx], avecs[k_idx], accs[0], rpt=3, sy=first, r=True))
    first = False
  for k_idx in range(4):
    after.append(MAD_F32(accs[1], b1[k_idx], avecs[k_idx], accs[1], rpt=3, r=True))
    after.append(MAD_F32(accs[2], b2[k_idx], avecs[k_idx], accs[2], rpt=3, r=True))
  for k_idx in range(4): after.append(MAD_F32(accs[3], b0_or_b3[k_idx], avecs[k_idx], accs[3], rpt=3, r=True))
  return before, after


def patch_rpt3_l25_postinc_b0firstlow(image: bytes, unroll: int=1, drop_ssnop: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 postinc b0firstlow expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[47] != ISAM_F32("r13.w", "r8.z", 1, 1):
    raise RuntimeError(f"unexpected l25 B3 load at 47: {instrs[47].hex()}")
  if instrs[48] != ISAM_F32("r14.w", "r2.y", 1, 1):
    raise RuntimeError(f"unexpected l25 B0 load at 48: {instrs[48].hex()}")
  if instrs[39] != bytes.fromhex("0900012028081042"):
    raise RuntimeError(f"unexpected l25 next-k add at 39: {instrs[39].hex()}")

  before, after = rpt3_l25_b0firstlow_split_mads()
  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  def make_body(use_cmp: bool) -> list[bytes]:
    body = [instrs[i] for i in range(25, 49) if i not in (25, 39, 42, 43, 47, 48)]
    body += [ISAM_F32("r13.w", "r2.y", 1, 1)]
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:52] + before + [ISAM_F32("r13.w", "r8.z", 1, 1), ADD_S("r2.y", "r2.y", 1)] + after
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def patch_rpt3_l25_postinc_b0low(image: bytes, unroll: int=1, drop_ssnop: bool=False, wait_before_b0: int=0, wait_after_b0: int=0, sy_after_b0: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l25 postinc b0low expects 144 instructions, got {len(instrs)}")
  loop_count = l25_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[48] != ISAM_F32("r14.w", "r2.y", 1, 1):
    raise RuntimeError(f"unexpected l25 B0 load at 48: {instrs[48].hex()}")
  if instrs[39] != bytes.fromhex("0900012028081042"):
    raise RuntimeError(f"unexpected l25 next-k add at 39: {instrs[39].hex()}")

  before, after = rpt3_l25_b0low_split_mads(sy_after_b0=sy_after_b0)
  cmp_r2y = replace_freg_src(instrs[42], "r2.y")
  def make_body(use_cmp: bool) -> list[bytes]:
    body = [instrs[i] for i in range(25, 49) if i not in (25, 39, 42, 43, 48)]
    body += [] if drop_ssnop else instrs[49:50]
    body += instrs[50:52] + before
    if wait_before_b0: body.append(NOP(rpt=wait_before_b0))
    body.append(ISAM_F32("r13.w", "r2.y", 1, 1))
    if wait_after_b0: body.append(NOP(rpt=wait_after_b0))
    body += [ADD_S("r2.y", "r2.y", 1)] + after
    if use_cmp: body.append(cmp_r2y)
    return body

  loop_start = 25
  out = instrs[:loop_start]
  for i in range(unroll): out += make_body(i == unroll - 1)
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += instrs[119:134]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


def l23_loop_count(image: bytes) -> int:
  lines = clean_disasm_lines(image)
  if (m:=re.search(r"cmps\.s\.ge p0\.x, r1\.z, (\d+)", lines[41])) is None:
    raise RuntimeError(f"unexpected l23 cmp at 41: {lines[41]}")
  return int(m.group(1))


def rpt3_l23_mads(k_order: tuple[int, ...]=(0, 1, 2, 3), acc_order: tuple[int, ...]=(0, 1, 2, 3)) -> list[bytes]:
  accs = ("r4.w", "r3.w", "r2.w", "r1.w")
  avecs = ("r9.x", "r7.x", "r8.x", "r10.z")
  bcols = (("r14.z", "r11.z", "r12.z", "r13.z"),
           ("r14.w", "r11.w", "r12.w", "r13.w"),
           ("r15.x", "r12.x", "r13.x", "r14.x"),
           ("r15.y", "r12.y", "r13.y", "r14.y"))
  mads, first = [], True
  for k_idx in k_order:
    avec, bvec = avecs[k_idx], bcols[k_idx]
    for acc_idx in acc_order:
      mads.append(MAD_F32(accs[acc_idx], bvec[acc_idx], avec, accs[acc_idx], rpt=3, sy=first, r=True))
      first = False
  return mads


def patch_rpt3_l23(image: bytes, unroll: int=1, k_order: tuple[int, ...]=(0, 1, 2, 3), acc_order: tuple[int, ...]=(0, 1, 2, 3), drop_ssnop: bool=False, no_store: bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 144: raise RuntimeError(f"l23 rpt3 expects 144 instructions, got {len(instrs)}")
  loop_count = l23_loop_count(image)
  if loop_count % unroll != 0: raise RuntimeError(f"unroll {unroll} must divide loop count {loop_count}")
  if instrs[38] != ISAM_F32("r8.x", "r6.x", 0, 0):
    raise RuntimeError(f"unexpected l23 A2 load at 38: {instrs[38].hex()}")
  if instrs[42] != ISAM_F32("r10.z", "r6.x", 0, 0):
    raise RuntimeError(f"unexpected l23 A3 load at 42: {instrs[42].hex()}")
  if instrs[46] != ISAM_F32("r14.z", "r7.x", 1, 1):
    raise RuntimeError(f"unexpected l23 B0 load at 46: {instrs[46].hex()}")
  if instrs[48] != ISAM_F32("r7.x", "r10.x", 0, 0):
    raise RuntimeError(f"unexpected l23 A1 load at 48: {instrs[48].hex()}")
  if instrs[49] != ISAM_F32("r9.x", "r6.z", 0, 0):
    raise RuntimeError(f"unexpected l23 A0 load at 49: {instrs[49].hex()}")

  body = instrs[23:47] + ([] if drop_ssnop else instrs[47:48]) + instrs[48:50] + rpt3_l23_mads(k_order=k_order, acc_order=acc_order)
  loop_start = 23
  out = instrs[:loop_start]
  for _ in range(unroll): out += body
  br_idx = len(out)
  out += [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))]
  out += [(NOP() if no_store and i in (126, 129, 131, 132) else instrs[i]) for i in range(116, 134)]
  out += [NOP()] * (len(instrs) - len(out))
  return b"".join(out)


HALF_F16_ISAMS = (
  (39, "r3.z", "hr5.x", "r3.x", 0),
  (41, "r6.z", "hr6.x", "r4.z", 0),
  (43, "r7.z", "hr7.x", "r5.x", 0),
  (44, "r8.z", "hr4.x", "r2.x", 0),
  (46, "r9.z", "hr9.x", "r1.y", 1),
  (48, "r10.z", "hr10.x", "r5.z", 1),
  (50, "r4.z", "hr11.x", "r6.x", 1),
  (52, "r5.z", "hr8.x", "r2.z", 1),
)


def patch_half_f16_rpt3_padded(image: bytes) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 208:
    raise RuntimeError(f"half_f16_rpt3_padded expects 208 instructions, got {len(instrs)}")
  for idx, old_dst, new_dst, coord, tex in HALF_F16_ISAMS:
    expected = ISAM_F32(old_dst, coord, tex, tex)
    if instrs[idx] != expected:
      raise RuntimeError(f"isam mismatch at {idx}: got {instrs[idx].hex()}, expected {expected.hex()}")
    instrs[idx] = ISAM_F16(new_dst, coord, tex, tex)

  acc_rows, a_rows, b_rows, comps = ("hr3", "hr2", "hr1", "hr0"), ("hr4", "hr5", "hr6", "hr7"), ("hr8", "hr9", "hr10", "hr11"), "xyzw"
  mads = []
  first = True
  for k, comp in enumerate(comps):
    for acc, a in zip(acc_rows, a_rows):
      mads.append(MAD_F16(f"{acc}.x", f"{a}.{comp}", f"{b_rows[k]}.x", f"{acc}.x", rpt=3, sy=first, r=True))
      first = False

  start, end = 53, 181
  instrs[start:end] = mads + [NOP()] * (end - start - len(mads))
  return b"".join(instrs)


def patch_half_f16_rpt3_compact(image: bytes) -> bytes:
  padded = patch_half_f16_rpt3_padded(image)
  instrs = [padded[i:i+8] for i in range(0, len(padded), 8)]
  loop_start = 22
  prefix = instrs[:69]
  br_idx = len(prefix)
  epilogue = instrs[183:200]
  compact = prefix + [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))] + epilogue
  compact += [NOP()] * (len(instrs) - len(compact))
  return b"".join(compact)


def patch_half_f16_rpt3_tight(image: bytes) -> bytes:
  compact = patch_half_f16_rpt3_compact(image)
  instrs = [compact[i:i+8] for i in range(0, len(compact), 8)]
  loop_start = 22
  drop_loop_nops = {22, 40, 42, 45, 49, 51}
  loop = [instrs[i] for i in range(22, 69) if i not in drop_loop_nops]
  prefix = instrs[:22] + loop
  br_idx = len(prefix)
  epilogue = instrs[71:88]
  tight = prefix + [BR(2, inv=False), JUMP(loop_start - (br_idx + 1))] + epilogue
  tight += [NOP()] * (len(instrs) - len(tight))
  return b"".join(tight)


def patch_linear(linear: UOp, patch: str) -> tuple[UOp, int, str, dict[str, int|bool]]:
  main_idx = find_main_call(linear)
  call = linear.src[main_idx]
  prg = call.src[0]
  new_lib, asm, meta = patch_lib(prg.src[4].arg, patch)
  new_prg = prg.replace(src=prg.src[:4] + (prg.src[4].replace(arg=new_lib),))
  new_call = call.replace(src=(new_prg, *call.src[1:]))
  return linear.replace(src=tuple(new_call if i == main_idx else c for i, c in enumerate(linear.src))), main_idx, asm, meta


def override_main_local(linear: UOp, main_idx: int, local_size: tuple[int, int, int]) -> UOp:
  call = linear.src[main_idx]
  prg = call.src[0]
  if prg.arg.local_size is None: raise RuntimeError("main program has no local_size")
  total = tuple(g*l for g, l in zip(prg.arg.global_size, prg.arg.local_size))
  if any(t % l != 0 for t, l in zip(total, local_size)):
    raise RuntimeError(f"local_size {local_size} does not divide total launch {total}")
  new_global = tuple(t // l for t, l in zip(total, local_size))
  new_prg = prg.replace(arg=replace(prg.arg, global_size=new_global, local_size=local_size))
  new_call = call.replace(src=(new_prg, *call.src[1:]))
  return linear.replace(src=tuple(new_call if i == main_idx else c for i, c in enumerate(linear.src)))


def parse_local_size(s: str) -> tuple[int, int, int]:
  parts = tuple(int(x) for x in s.split(","))
  if len(parts) != 3: raise ValueError("local size must be X,Y,Z")
  return parts


def make_matmul(n: int, dtype, acc_dtype, ones: bool):
  make = Tensor.ones if ones else Tensor.empty
  a = make(n, n, dtype=dtype).realize()
  b = make(n, n, dtype=dtype).realize()
  c = a.matmul(b, dtype=acc_dtype)
  return c, compile_linear(c.schedule_linear())


def flat_tensor_data(t: Tensor):
  return t._buffer().as_memoryview().cast(t.dtype.base.fmt)  # noqa: SLF001 - debug harness


def check_all_ones(c: Tensor, n: int) -> bool:
  data = flat_tensor_data(c)
  expected = float(n)
  atol = 1e-3 if c.dtype.base is dtypes.float32 else 1.0
  for i, x in enumerate(data):
    if abs(float(x) - expected) > atol:
      print(f"CHECK FAIL expected={expected} first_mismatch idx={i} got={float(x)}")
      return False
  print(f"CHECK PASS all {len(data)} outputs are {expected}")
  return True


def print_meta(main_name: str, meta: dict[str, int|bool]):
  print(
    f"main={main_name} image_bytes={meta['image_bytes']} instrs={meta['instrs']} "
    f"fregs={meta['fregs']} hregs={meta['hregs']} repacked_same={meta['repacked_same']} "
    f"changed_bytes={meta['changed_bytes']} mad.f32={meta['mad_f32']} mad.f16={meta['mad_f16']} "
    f"rpt_mad={meta['rpt_mad']} isam={meta['isam']} stores={meta['stores']}"
  )


def bench_main(linear: UOp, main_idx: int, n: int, warmup: int, iters: int):
  if main_idx > 0:
    run_linear(linear.replace(src=linear.src[:main_idx]), jit=True, wait=True, update_stats=False)
  call = linear.src[main_idx]
  for _ in range(warmup): time_call(call)
  times = [time_call(call) for _ in range(iters)]
  best = min(t for t in times if t is not None)
  print(f"BENCH main {2*n*n*n / best / 1e9:.1f} GFLOPS ({best*1e3:.3f} ms)")


def bench_full(linear: UOp, n: int, warmup: int, iters: int):
  for _ in range(warmup): run_linear(linear, jit=True, wait=True, update_stats=False)
  times = []
  for _ in range(iters):
    st = time.perf_counter()
    run_linear(linear, jit=True, wait=True, update_stats=False)
    times.append(time.perf_counter() - st)
  best = min(times)
  print(f"BENCH full {2*n*n*n / best / 1e9:.1f} GFLOPS ({best*1e3:.3f} ms)")


def dtype_from_arg(name: str):
  return {"none": None, "float": dtypes.float, "half": dtypes.half}[name]


def run(args):
  dtype, acc_dtype = dtype_from_arg(args.dtype), dtype_from_arg(args.acc_dtype)
  c, linear = make_matmul(args.n, dtype, acc_dtype, ones=args.check)
  patched, main_idx, asm, meta = patch_linear(linear, args.patch)
  if args.main_local is not None:
    patched = override_main_local(patched, main_idx, parse_local_size(args.main_local))
  main_name = patched.src[main_idx].src[0].arg.name
  print_meta(main_name, meta)
  if args.disasm: print(asm)

  if args.stats_run:
    run_linear(patched, jit=True, wait=True, update_stats=True)
    if args.check and not check_all_ones(c, args.n): raise SystemExit(1)
  elif args.check:
    run_linear(patched, jit=True, wait=True, update_stats=False)
    if not check_all_ones(c, args.n): raise SystemExit(1)
  if args.bench: bench_main(patched, main_idx, args.n, args.warmup, args.iters)
  if args.bench_full: bench_full(patched, args.n, args.warmup, args.iters)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n", type=int, default=1024)
  parser.add_argument("--dtype", choices=("float", "half"), default="float")
  parser.add_argument("--acc-dtype", choices=("none", "float", "half"), default="float")
  parser.add_argument("--patch", default="noop")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--bench", action="store_true")
  parser.add_argument("--bench-full", action="store_true")
  parser.add_argument("--stats-run", action="store_true")
  parser.add_argument("--main-local")
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--iters", type=int, default=10)
  parser.add_argument("--disasm", action="store_true")
  run(parser.parse_args())
