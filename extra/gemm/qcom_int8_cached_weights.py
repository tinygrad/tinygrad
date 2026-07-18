#!/usr/bin/env python3
"""Quantize selected cached QCOM GEMM weights to normalized int8 textures."""
import argparse, itertools, pickle
from dataclasses import replace

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def graph_batch(model): return model.captured.linear.src[0].src[0].src[0].src


def adapt_aux_dtype(aux, index, dtype):
  if isinstance(aux, tuple) and len(aux) == 3 and aux[0] == index and isinstance(aux[0], int):
    return (aux[0], dtype, aux[2])
  return tuple(adapt_aux_dtype(x, index, dtype) for x in aux) if isinstance(aux, tuple) else aux


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--geometry", type=int, choices=(3, 12), required=True)
  parser.add_argument("--indices", default="", help="comma-separated occurrence indices; default is all")
  parser.add_argument("--per-channel", action="store_true", help="scale each output channel independently")
  args = parser.parse_args()
  with open(args.input, "rb") as f: model = pickle.load(f)
  batch = graph_batch(model)
  existing_slots = [x.arg.slot for x in model.captured.linear.toposort()
                    if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(existing_slots, default=-1) + 1)
  candidates = [(i, call) for i, call in enumerate(batch) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                plain_name(call.src[0].arg.name) == "gemm_h" and int(call.src[0].arg.global_size[0]) == args.geometry]
  selected = {int(x) for x in args.indices.split(",") if x} if args.indices else set(range(len(candidates)))
  replacements = {}
  for occurrence, (index, call) in enumerate(candidates):
    if occurrence not in selected: continue
    if index+1 >= len(batch): raise ValueError(f"GEMM {occurrence} has no epilogue")
    epi_call = batch[index+1]
    epi_name = plain_name(epi_call.src[0].arg.name)
    expected_epi = "epi_fp32" if args.geometry == 3 else "epi3_fp32"
    if epi_name != expected_epi: raise ValueError(f"GEMM {occurrence} is followed by {epi_name}, expected {expected_epi}")
    weights = np.asarray(call.src[2].buffer.numpy(), dtype=np.float32)
    k, n = ((1536, 384) if args.geometry == 3 else (384, 1536))
    scales = np.max(np.abs(weights.reshape(k, n)), axis=0) if args.per_channel else np.asarray([np.max(np.abs(weights))])
    if not np.isfinite(scales).all():
      raise ValueError(f"invalid scale range {scales.min()}..{scales.max()} for GEMM {occurrence}")
    scales[scales == 0] = 1.0
    quantized = np.clip(np.rint(weights.reshape(k, n)/scales.reshape(1, -1)*127.0), -127, 127).astype(np.int8)
    weight = UOp.new_buffer("QCOM", quantized.size, dtypes.int8)
    weight.buffer.ensure_allocated()
    weight.buffer.copyin(memoryview(quantized).cast("B"))
    program = call.src[0].replace(arg=replace(call.src[0].arg, aux=adapt_aux_dtype(call.src[0].arg.aux, 1, dtypes.int8)))
    replacements[index] = call.replace(src=(program, call.src[1], weight, *call.src[3:]))

    epi_program = epi_call.src[0]
    source = epi_program.src[2].arg
    needle = "float4 v=vload4(0,C+row*1024+col*4);" if args.geometry == 3 else \
             "float4 z=vload4(0,C+row*2048+col*4);"
    if args.per_channel:
      source = source.replace("__global float *C)", "__global float *C,__global float *Q)")
      replacement = needle + (" v*=vload4(0,Q+col*4);" if args.geometry == 3 else " z*=vload4(0,Q+col*4);")
    else:
      scale = float(scales[0])
      replacement = needle + (f" v*=(float4)({scale:.9g}f);" if args.geometry == 3 else f" z*=(float4)({scale:.9g}f);")
    if needle not in source: raise ValueError(f"epilogue source pattern missing for GEMM {occurrence}")
    source = source.replace(needle, replacement)
    lib = Device["QCOM"].compiler.compile(source)
    if args.per_channel:
      scale_buf = UOp.new_buffer("QCOM", n, dtypes.float)
      scale_buf.buffer.ensure_allocated()
      scale_buf.buffer.copyin(memoryview(np.ascontiguousarray(scales, dtype=np.float32)).cast("B"))
      info = epi_program.arg
      old_aux = info.aux[0]
      info = replace(info, globals=info.globals+(len(epi_call.src)-1,), ins=info.ins+(len(epi_call.src)-1,),
                     aux=(old_aux+(((len(epi_call.src)-1, dtypes.float, (n,)),),),))
      epi_program = epi_program.replace(arg=info, src=epi_program.src[:2] +
                                        (epi_program.src[2].replace(arg=source), epi_program.src[3].replace(arg=lib)))
      replacements[index+1] = epi_call.replace(src=(epi_program, *epi_call.src[1:], scale_buf))
      print(f"geometry={args.geometry} occurrence={occurrence} scale={scales.min():.8g}..{scales.max():.8g}")
    else:
      epi_program = epi_program.replace(src=epi_program.src[:2] +
                                        (epi_program.src[2].replace(arg=source), epi_program.src[3].replace(arg=lib)))
      replacements[index+1] = epi_call.replace(src=(epi_program, *epi_call.src[1:]))
      print(f"geometry={args.geometry} occurrence={occurrence} scale={scale:.8g}")

  outer = model.captured.linear.src[0]
  new_outer = create_graph_call([replacements.get(i, call) for i, call in enumerate(batch)])
  model.captured._linear = model.captured.linear.substitute({outer: new_outer}, walk=True)
  model.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(model, f)
  print(f"wrote {args.output} with {len(replacements)//2} int8 GEMMs")


if __name__ == "__main__": main()
