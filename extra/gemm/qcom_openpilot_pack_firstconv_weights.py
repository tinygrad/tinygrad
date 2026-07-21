#!/usr/bin/env python3
"""Repack driving-vision first-convolution weights for output-thread locality."""
import argparse, itertools, pickle, struct

import numpy as np

from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name
from extra.gemm.ir3asm import BR, MAD_F32, NOP

TARGET = "r_64_32_16_4_4_6_3_3_4"


def packed_weight(weight:UOp, group4:bool=False) -> UOp:
  original = np.asarray(weight.buffer.numpy()).reshape(-1)
  assert original.size == 16*216*4
  packed = np.empty_like(original)
  for out4 in range(16):
    for ky in range(3):
      for ic in range(6):
        for kx in range(3):
          old = (out4*216 + ky*72 + ic*12 + kx*4)*4
          tap = ky*18 + ic*3 + kx
          new_pixel = ((out4//4)*864 + tap*16 + (out4%4)*4) if group4 else (tap*16 + out4)*4
          new = new_pixel*4
          packed[new:new+16] = original[old:old+16]
  return UOp.from_buffer(Buffer("QCOM", packed.size, weight.dtype, initial_value=packed.tobytes()))


def repeat_pack(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  instrs = [lib[image_offset+i:image_offset+i+8] for i in range(0, image_size, 8)]
  if len(instrs) != 261: raise RuntimeError(f"expected 261 repacked first-conv instructions, got {len(instrs)}")
  out = instrs[:58]
  first = True
  for component, weight in zip("xyzw", ("r5", "r2", "r3", "r4")):
    out.append(MAD_F32("r11.x", f"r7.{component}", f"{weight}.x", "r11.x", sy=first, r=True))
    first = False
    out.append(MAD_F32("r12.y", f"r7.{component}", f"{weight}.y", "r12.y", rpt=2, r=True))
  for component, weight in zip("xyzw", ("r5.x", "r2.x", "r3.x", "r4.x")):
    for acc, activation in zip(("r13.x", "r14.x", "r15.x"), ("r6", "r1", "r0")):
      out.append(MAD_F32(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True))
  out += instrs[122:128]
  out.append(BR(29-len(out)))
  out += instrs[129:]
  out[91] = BR(24-91)
  out[98] = BR(22-98)
  if len(out) > len(instrs): raise RuntimeError(f"packed shader grew to {len(out)} instructions")
  out += [NOP()]*(len(instrs)-len(out))
  return lib[:image_offset]+b"".join(out)+lib[image_offset+image_size:]


def patch_model(model, rpt:bool=False, group4:bool=False) -> int:
  existing = [x.arg.slot for x in model.captured.linear.toposort()
              if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(existing, default=-1)+1)
  outer = model.captured.linear.src[0]
  batch, patched = list(outer.src[0].src[0].src), 0
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET: continue
    program, source = call.src[0], call.src[0].src[2].arg
    old = "int alu18 = ((Ridx0*12)+(Ridx3<<2)+(Ridx2*72)+(idx0*216));"
    new = ("int alu18 = ((idx0>>2)*864+(Ridx2*18+Ridx0*3+Ridx3)*16+(idx0&3)*4);" if group4 else
           "int alu18 = (((Ridx2*18+Ridx0*3+Ridx3)*16+idx0)*4);")
    if old not in source: raise RuntimeError("unexpected first-convolution source")
    source = source.replace(old, new)
    lib = Device["QCOM"].compiler.compile(source)
    if rpt: lib = repeat_pack(lib)
    program = program.replace(src=program.src[:2]+(program.src[2].replace(arg=source), program.src[3].replace(arg=lib)))
    batch[index] = call.replace(src=(program, call.src[1], call.src[2], packed_weight(call.src[3], group4), *call.src[4:]))
    patched += 1
  if patched:
    model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return patched


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output"); ap.add_argument("--rpt", action="store_true")
  ap.add_argument("--group4", action="store_true"); args=ap.parse_args()
  with open(args.input,"rb") as f: model=pickle.load(f)
  print("patched",patch_model(model, args.rpt, args.group4))
  with open(args.output,"wb") as f: pickle.dump(model,f)


if __name__ == "__main__": main()
