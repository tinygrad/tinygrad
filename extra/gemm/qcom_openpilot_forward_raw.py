#!/usr/bin/env python3
"""Raw 8x8 FP16-accumulate projection for the padded OpenPilot vision layout."""
import argparse, os, pickle, struct
from dataclasses import replace

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import (ADD_S, ADD_S_REG, AND_B, BR, CMPS_S_EQ, COV_F16F32, END, ISAM_F16, MAD_F16, MOV_F32,
                               MOV_H_IMM, MOV_S32, NOP, NOP_SS, SHL_B, SHR_B, STIB_F32, assemble, inject)
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm.qcom_8x4_gemm import prologue_8x4
from extra.gemm.qcom_ir3_matmul_patch import plain_name
from extra.gemm.qcom_openpilot_forward_tile8 import SOURCE

TARGET = "r_32_192_4_4_64_4"


def build_raw_shader(dev) -> tuple[bytes, int, int]:
  instrs = prologue_8x4(dev, 128)
  # The donor produces row=gid1*32+(lid>>5)*8 and col=gid0*32+(lid&31).
  # Widen col to a two-col4 tile: gid0*64+tid, with the second column at +32.
  instrs += [MOV_F32("r12.x", "r51.w"), NOP(rpt=2), SHL_B("r12.x", "r12.x", 5), NOP(rpt=2),
             ADD_S_REG("r7.y", "r7.y", "r12.x"), NOP(rpt=2)]

  # Precompute the eight padded-A row bases. A is a 1D image laid out as
  # (row&31)*260 + (row>>5)*65 + k4.
  instrs += [SHR_B("r12.y", "r7.x", 5), AND_B("r12.z", "r7.x", 31), NOP(rpt=2),
             SHL_B("r12.w", "r12.y", 6), SHL_B("r13.x", "r12.z", 8), SHL_B("r13.y", "r12.z", 2),
             ADD_S_REG("r12.w", "r12.w", "r12.y"), ADD_S_REG("r13.x", "r13.x", "r13.y"), NOP(rpt=2),
             ADD_S_REG("r13.x", "r13.x", "r12.w"), MOV_S32("r13.y", 260), NOP(rpt=2)]
  row_bases = ("r13.x", "r13.z", "r13.w", "r14.x", "r14.y", "r14.z", "r14.w", "r15.x")
  for index, dst in enumerate(row_bases[1:], 1):
    instrs += [ADD_S_REG(dst, row_bases[index-1], "r13.y"), NOP(rpt=2)]

  acc0 = 12 * 4
  for base in range(acc0, acc0+16*4, 4): instrs.append(MOV_H_IMM(base, 0, rpt=3))
  instrs += [MOV_S32("r6.z", 0), MOV_S32("r6.y", 3, sy=True)]
  loop_start = len(instrs)

  b_pairs = tuple((f"r{16+i//2}.{'xz'[i&1]}", f"r{16+i//2}.{'yw'[i&1]}") for i in range(8))
  for component in range(4):
    for col in range(2):
      xreg, yreg = b_pairs[component*2+col]
      instrs.append(MOV_F32(xreg, "r6.y") if component == 3 else ADD_S(xreg, "r6.y", component-3))
      instrs.append(MOV_F32(yreg, "r7.y") if col == 0 else ADD_S(yreg, "r7.y", 32))
  instrs.append(NOP(rpt=3))
  for index, (xreg, _) in enumerate(b_pairs): instrs.append(ISAM_F16(index*4, xreg, 1))

  a_pairs = (("r20.x", "r20.y"), ("r20.z", "r20.w"), ("r21.x", "r21.y"), ("r21.z", "r21.w"))
  def load_a(first_row: int) -> None:
    nonlocal instrs
    for slot, ((xreg, yreg), base) in enumerate(zip(a_pairs, row_bases[first_row:first_row+4])):
      instrs += [ADD_S_REG(xreg, base, "r6.z"), MOV_S32(yreg, 0)]
    instrs.append(NOP(rpt=3))
    for slot, (xreg, _) in enumerate(a_pairs): instrs.append(ISAM_F16((8+slot)*4, xreg, 0))

  def mads(first_row: int) -> None:
    first = True
    for slot, row in enumerate(range(first_row, first_row+4)):
      for component in range(4):
        for col in range(2):
          acc = acc0+(row*2+col)*4
          instrs.append(MAD_F16(acc, (8+slot)*4+component, (component*2+col)*4, acc, rpt=3, r=True, sy=first))
          first = False

  load_a(0)
  mads(0)
  instrs.append(NOP_SS())
  load_a(4)
  mads(4)
  instrs += [ADD_S("r0.x", "r6.z", 1), ADD_S("r6.y", "r6.y", 4), CMPS_S_EQ("r6.z", 63, nop=1),
             MOV_F32("r6.z", "r0.x"), NOP(rpt=3)]
  loop_end = len(instrs)
  instrs.append(BR(loop_start-loop_end))

  if os.getenv("RAW_NO_STORE"):
    instrs.append(END())
    return assemble(instrs), 24, 28

  # Typed image stores. p=row>>5 is constant within a tile; output x is
  # col+p*192 and output y is row&31.
  instrs += [SHL_B("r12.w", "r12.y", 7), SHL_B("r13.x", "r12.y", 6),
             ADD_S_REG("r12.w", "r12.w", "r13.x"), ADD_S_REG("r12.w", "r12.w", "r7.y"), NOP(rpt=2)]
  for row in range(8):
    for col in range(2):
      instrs.append(MOV_F32("r22.x", "r12.w") if col == 0 else ADD_S("r22.x", "r12.w", 32))
      instrs.append(MOV_F32("r22.y", "r12.z") if row == 0 else ADD_S("r22.y", "r12.z", row))
      instrs += [COV_F16F32("r23.x", acc0+(row*2+col)*4, sy=True, rpt=3, r=True), NOP(rpt=5),
                 STIB_F32("r23.x", "r22.x"), NOP(rpt=8)]
  instrs.append(END())
  return assemble(instrs), 24, 28


def raw_lib(dev) -> bytes:
  lib = dev.compiler.compile_cached(SOURCE)
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  if os.getenv("RAW_GENERAL"):
    threads = int(os.getenv("RAW_THREADS", "128"))
    q8.K, q8.K4 = 256, 64
    shader, hregs, fregs, _ = q8.build_8x8_split_a_unroll_shader(
      dev, threads, k_unroll=8, b_coord_delay=0, fast_coords=True,
      prefetch_next_b=True, no_store=True)
  else:
    shader, fregs, hregs = build_raw_shader(dev)
  return inject(lib, image_off, image_size, reg_off, shader, fregs, hregs)


def patch_model(model) -> int:
  outer = model.captured.linear.src[0]
  batch, patched, lib = list(outer.src[0].src[0].src), 0, raw_lib(Device["QCOM"])
  threads = int(os.getenv("RAW_THREADS", "128")) if os.getenv("RAW_GENERAL") else 128
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET: continue
    program = call.src[0].replace(arg=replace(call.src[0].arg, global_size=(3, 512//threads, 1), local_size=(threads, 1, 1)),
                                  src=call.src[0].src[:3]+(call.src[0].src[3].replace(arg=lib),))
    batch[index] = call.replace(src=(program, *call.src[1:]))
    patched += 1
  if patched:
    model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return patched


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args = parser.parse_args()
  with open(args.input, "rb") as f: model = pickle.load(f)
  print("patched", patch_model(model))
  with open(args.output, "wb") as f: pickle.dump(model, f)
