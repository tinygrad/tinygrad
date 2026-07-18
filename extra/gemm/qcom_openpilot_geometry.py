#!/usr/bin/env python3
"""Apply validated equivalent launch geometries to OpenPilot QCOM kernels."""
import argparse, pickle
from dataclasses import replace

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


GEOMETRIES = {
  "r_32_192_4_4_64_4": ((16, 2, 1), (12, 16, 1)),
  "r_8_384_4_4_128_4": ((48, 1, 1), (8, 8, 1)),
  "r_64_32_16_4_4_6_3_3_4": ((4, 2, 32), (4, 16, 2)),
}


def patch_geometry(model) -> int:
  outer = model.captured.linear.src[0]
  batch, patched = list(outer.src[0].src[0].src), 0
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    if (geometry := GEOMETRIES.get(plain_name(call.src[0].arg.name))) is None: continue
    program = call.src[0].replace(arg=replace(call.src[0].arg, global_size=geometry[0], local_size=geometry[1]))
    batch[index] = call.replace(src=(program, *call.src[1:]))
    patched += 1
  if patched:
    model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return patched


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args = parser.parse_args()
  with open(args.input, "rb") as f: model = pickle.load(f)
  print("patched", patch_geometry(model))
  with open(args.output, "wb") as f: pickle.dump(model, f)


if __name__ == "__main__": main()
