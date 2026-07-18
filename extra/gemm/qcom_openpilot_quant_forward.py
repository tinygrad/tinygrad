#!/usr/bin/env python3
"""Quantize the six vision MLP forward weights with per-output-lane scales."""
import argparse, itertools, pickle, re
from dataclasses import replace

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET = "r_32_192_4_4_64_4"


def upload(buffer:Buffer, data:np.ndarray) -> None:
  raw=memoryview(np.ascontiguousarray(data)).cast("B")
  if hasattr(buffer, "copyin"): buffer.copyin(raw)
  else: buffer.copy_from(Buffer("PYTHON", buffer.size, buffer.dtype, opaque=raw))


def adapt_aux_dtype(aux, index:int, dtype):
  if isinstance(aux, tuple) and len(aux) == 3 and aux[0] == index and isinstance(aux[0], int): return (aux[0], dtype, aux[2])
  return tuple(adapt_aux_dtype(x, index, dtype) for x in aux) if isinstance(aux, tuple) else aux


def patch_model(model) -> int:
  outer=model.captured.linear.src[0]
  batch=list(outer.src[0].src[0].src)
  existing=[x.arg.slot for x in model.captured.linear.toposort() if x.op is Ops.BUFFER and hasattr(x.arg,"slot") and x.arg.slot >= 0]
  UOp.unique_num=itertools.count(max(existing,default=-1)+1)
  cache,patched={},0
  for index,call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET: continue
    program,source=call.src[0],call.src[0].src[2].arg
    match=re.search(r"(__kernel void \w+\(.*?)(\)) \{",source,re.S)
    if match is None: raise RuntimeError("kernel signature not found")
    source=source[:match.start(2)]+", read_only image2d_t qscale"+source[match.start(2):]
    bias=re.search(r"(float4 (val\d+) = read_imagef\(data3_[^;]+;)",source)
    if bias is None: raise RuntimeError("bias load not found")
    source=source[:bias.end()]+"\n  float4 qs = read_imagef(qscale, smp, (int2)(idx0,0));"+source[bias.end():]
    bias_name=bias.group(2)
    for lane in range(16):
      component="xyzw"[lane&3]
      pattern=fr"\(\(\*\(buf0\+{lane}\)\)\+{bias_name}\.{component}\)"
      replacement=f"(((*(buf0+{lane}))*qs.{component})+{bias_name}.{component})"
      source,count=re.subn(pattern,replacement,source)
      if count != 1: raise RuntimeError(f"output lane {lane} matched {count} times")

    weight=np.asarray(call.src[3].buffer.numpy(),dtype=np.float32).reshape(192,320,4)
    scale=np.max(np.abs(weight),axis=1,keepdims=True).astype(np.float32)
    scale[scale == 0]=1.0
    quantized=np.clip(np.rint(weight/scale*127.0),-127,127).astype(np.int8)
    qweight=UOp.new_buffer("QCOM",quantized.size,dtypes.int8); qweight.buffer.ensure_allocated(); upload(qweight.buffer,quantized)
    qscale=UOp.new_buffer("QCOM",scale.size,dtypes.half); qscale.buffer.ensure_allocated(); upload(qscale.buffer,scale.astype(np.float16))

    info=program.arg
    aux=adapt_aux_dtype(info.aux,2,dtypes.int8)
    aux=(aux[0]+(((4,dtypes.half,(1,192,4)),),),)
    info=replace(info,globals=info.globals+(4,),ins=info.ins+(4,),aux=aux)
    if source not in cache: cache[source]=Device["QCOM"].compiler.compile(source)
    lib=cache[source]
    program=program.replace(arg=info,src=program.src[:2]+(program.src[2].replace(arg=source),program.src[3].replace(arg=lib)))
    batch[index]=call.replace(src=(program,call.src[1],call.src[2],qweight,call.src[4],qscale))
    patched+=1
  if patched:
    model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(batch)},walk=True)
    model.captured.__dict__.pop("linear",None)
  return patched


if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output"); args=ap.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("patched",patch_model(model))
  with open(args.output,"wb") as f:pickle.dump(model,f)
