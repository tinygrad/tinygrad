#!/usr/bin/env python3
"""Repair the lane order in packed half8 OpenPilot projection weights."""
import argparse
import pickle
import struct

import numpy as np

from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import BR, COV_S32S16, ISAM_F16, MAD_F16, SHRG_H
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET = "r_32_192_4_4_64_4"


def fix_repeat_mads(lib: bytes) -> bytes:
  image_offset = struct.unpack_from("<I", lib, 0xC0)[0]
  image_size = struct.unpack_from("<I", lib, 0x100)[0]
  instructions = [lib[x:x+8] for x in range(image_offset, image_offset+image_size, 8)]
  if instructions[71] != BR(25-71):
    raise RuntimeError("unexpected half8 loop layout")
  # Keep sampler outputs, accumulators, and activations in disjoint banks.
  instructions[35] = ISAM_F16("hr23.x", "r5.w", 0)
  instructions[36] = ISAM_F16("hr22.x", "r6.y", 0)
  instructions[37] = ISAM_F16("hr21.x", "r6.w", 0)
  instructions[39] = ISAM_F16("hr20.x", "r7.y", 0)
  unpack = []
  for destination, source in ((12, "r3.x"), (14, "r2.x"), (16, "r1.x"), (18, "r0.x")):
    unpack.append(COV_S32S16(f"hr{destination}.x", source, rpt=3, r=True, sy=not unpack))
    unpack.append(SHRG_H(f"hr{destination+1}.x", source, rpt=3, r=True))
  mads = []
  rows = (("hr10.x", "hr11.x", "hr23"), ("hr8.x", "hr9.x", "hr22"),
          ("hr6.x", "hr7.x", "hr21"), ("hr4.x", "hr5.x", "hr20"))
  for component, (weight0, weight1) in zip("xyzw", (("hr12.x", "hr13.x"), ("hr14.x", "hr15.x"),
                                                         ("hr16.x", "hr17.x"), ("hr18.x", "hr19.x"))):
    for accumulator0, accumulator1, activation in rows:
      mads.append(MAD_F16(accumulator0, f"{activation}.{component}", weight0, accumulator0,
                          rpt=3, r=True))
      mads.append(MAD_F16(accumulator1, f"{activation}.{component}", weight1, accumulator1, rpt=3, r=True))
  output = instructions[:48] + unpack + mads + instructions[70:]
  output[89] = BR(25-89)
  output = output[:len(instructions)]
  patched = bytearray(lib[:image_offset] + b"".join(output) + lib[image_offset+image_size:])
  register_offset = struct.unpack_from("<I", patched, 0x34)[0]
  old_hregs = struct.unpack_from("<I", patched, register_offset+0x18)[0]
  struct.pack_into("<I", patched, register_offset+0x18, (old_hregs & 0x80000000) | 24)
  return bytes(patched)


def patch_model(model, fix_weights: bool = True, recompile: bool = False, fix_mads: bool = False) -> int:
  outer = model.captured.linear.src[0]
  batch = list(outer.src[0].src[0].src)
  seen, patched = set(), 0
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET:
      continue
    if recompile or fix_mads:
      program = call.src[0]
      lib = Device["QCOM"].compiler.compile_cached(program.src[2].arg) if recompile else fix_repeat_mads(program.src[3].arg)
      program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=lib),))
      batch[index] = call.replace(src=(program, *call.src[1:]))
    weight = call.src[3].buffer
    if not fix_weights or id(weight) in seen or weight.dtype.itemsize != 4:
      patched += 1
      continue
    seen.add(id(weight))
    # The old pack transposed two adjacent float4 output channels before
    # bitcasting to uint4, producing a0,b0,a1,b1,... in each half8 pixel.
    # The kernel consumes half8.lo/hi as complete float4 channels.
    packed = weight.numpy().view(np.float16).reshape(-1, 8)
    corrected = np.ascontiguousarray(packed[:, (0, 2, 4, 6, 1, 3, 5, 7)])
    raw = memoryview(corrected).cast("B")
    if hasattr(weight, "copyin"):
      weight.copyin(raw)
    else:
      weight.copy_from(Buffer("PYTHON", weight.size, weight.dtype, opaque=raw))
    patched += 1
  if recompile or fix_mads:
    model.captured._linear = model.captured.linear.substitute({outer: create_graph_call(batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return patched


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--skip-weights", action="store_true")
  parser.add_argument("--recompile", action="store_true")
  parser.add_argument("--fix-mads", action="store_true")
  args = parser.parse_args()
  with open(args.input, "rb") as f:
    model = pickle.load(f)
  print("patched", patch_model(model, not args.skip_weights, args.recompile, args.fix_mads))
  with open(args.output, "wb") as f:
    pickle.dump(model, f)


if __name__ == "__main__":
  main()
