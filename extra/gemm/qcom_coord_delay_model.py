#!/usr/bin/env python3
"""Remove redundant coordinate-settle repeats from cached openpilot GEMMs."""
import argparse, itertools, pickle, struct

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.ir3asm import NOP
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def patch_lib(lib:bytes, keep_first:bool) -> bytes:
  image_off = struct.unpack_from("<I", lib, 0xc0)[0]
  image_size = struct.unpack_from("<I", lib, 0x100)[0]
  instrs = [lib[image_off+i:image_off+i+8] for i in range(0, image_size, 8)]
  if len(instrs) != 349: raise ValueError(f"expected 349 instructions, got {len(instrs)}")
  delay_indices = (55, 57, 59, 61, 71, 73, 75, 77)
  for position, index in enumerate(delay_indices):
    if instrs[index] != NOP(rpt=4): raise ValueError(f"unexpected instruction at delay {index}: {instrs[index].hex()}")
    if not (keep_first and position in (0, 4)): instrs[index] = NOP()
  ret = bytearray(lib)
  ret[image_off:image_off+image_size] = b"".join(instrs)
  return bytes(ret)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--global-size", default="12,8,1")
  parser.add_argument("--keep-first", action="store_true")
  args = parser.parse_args()
  target_global = tuple(int(x) for x in args.global_size.split(","))
  with open(args.input, "rb") as f: jit = pickle.load(f)
  slots = [x.arg.slot for x in jit.captured.linear.toposort()
           if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(slots, default=-1)+1)
  outer, cache, replacements = jit.captured.linear.src[0], {}, {}
  batch = outer.src[0].src[0].src
  for call in batch:
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    program = call.src[0]
    if plain_name(program.arg.name) != "gemm_h" or tuple(program.arg.global_size) != target_global: continue
    old_lib = program.src[3].arg
    new_lib = cache.setdefault(old_lib, patch_lib(old_lib, args.keep_first))
    new_program = program.replace(src=program.src[:3]+(program.src[3].replace(arg=new_lib),))
    replacements[call] = call.replace(src=(new_program, *call.src[1:]))
  if not replacements: raise ValueError(f"no gemm_h calls with global size {target_global}")
  new_outer = create_graph_call([replacements.get(call, call) for call in batch])
  jit.captured._linear = jit.captured.linear.substitute({outer:new_outer}, walk=True)
  jit.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(jit, f)
  print(f"patched {len(replacements)} calls across {len(cache)} binaries keep_first={args.keep_first}")


if __name__ == "__main__": main()
