#!/usr/bin/env python3
"""Experimental native-FP16 accumulator rewrite for driving_vision kernels."""
import argparse, pickle, re

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def half_acc_source(source:str) -> str:
  # buf0 is the matrix accumulator in these generated reduction kernels.  val0..7
  # are the two four-vector matrix operands; later values belong to the FP32
  # bias/residual epilogue and intentionally remain float.
  source = source.replace("float buf0[16];", "half buf0[16];")
  source = source.replace("float buf0[4];", "half buf0[4];")
  for index in range(8):
    source = re.sub(fr"float4 val{index} = read_imagef\(", fr"half4 val{index} = read_imageh(", source)
  return source


def patch_model(model, names:set[str]) -> int:
  outer = model.captured.linear.src[0]
  batch = list(outer.src[0].src[0].src)
  compiler, cache, patched = Device["QCOM"].compiler, {}, 0
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    program = call.src[0]
    if plain_name(program.arg.name) not in names or not any(x in program.src[2].arg for x in ("float buf0[16];", "float buf0[4];")): continue
    source = half_acc_source(program.src[2].arg)
    if source not in cache: cache[source] = compiler.compile(source)
    program = program.replace(src=program.src[:2]+(program.src[2].replace(arg=source), program.src[3].replace(arg=cache[source])))
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
  parser.add_argument("--names", required=True, help="comma-separated exact display names")
  args = parser.parse_args()
  with open(args.input, "rb") as f: model = pickle.load(f)
  print("patched", patch_model(model, set(args.names.split(","))))
  with open(args.output, "wb") as f: pickle.dump(model, f)


if __name__ == "__main__": main()
