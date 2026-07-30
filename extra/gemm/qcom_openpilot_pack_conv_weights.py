#!/usr/bin/env python3
"""Prepack static weights for the slow stride-2 openpilot 7x7 convolutions."""
import argparse, pickle, re

import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_openpilot_ir3 import plain_name

TARGETS={"r_16_8_16_2_4_4_7_7", "r_8_4_32_2_4_4_7_7", "r_4_2_64_2_4_4_7_7"}


def packed_source(source:str) -> str:
  start=source.index("    half val0")
  end=source.index("    int alu20",start)
  weight_name=re.search(r"__global half\* (data2_\d+)",source).group(1)  # type: ignore[union-attr]
  block="    int wp=(((alu0*2+alu2)*7+Ridx0)*28);\n"+"\n".join(
    f"    float4 w{i}=convert_float4(vload4(0,{weight_name}+wp+{i*4}));" for i in range(7))+"\n"
  source=source[:start]+block+source[end:]
  accum_start=source.index("    *(buf0+0)",start)
  loop_prefix=source[start:accum_start]
  casts=re.findall(r"    float (cast\d+) = \(\(float\)\(val(\d+)\)\);\n",loop_prefix)
  assert len(casts) == 28
  source=source[:start]+re.sub(r"    float cast\d+ = \(\(float\)\(val\d+\)\);\n", "", loop_prefix)+source[accum_start:]
  mapping={cast:("w0.x" if int(val) == 27 else f"w{int(val)%7+1}.x" if int(val) < 6 else
                       f"w{(int(val)-6)%7}.{'yzw'[(int(val)-6)//7]}") for cast,val in casts}
  for old,new in sorted(mapping.items(),key=lambda item:-len(item[0])):
    source=re.sub(rf"\b{old}\b",new,source)
  return source


def pack_weight_buffer(weight:UOp) -> UOp:
  original=np.asarray(weight.buffer.numpy()).reshape(-1)
  outputs=original.size//896
  assert outputs*896 == original.size
  packed=np.empty((outputs,2,7,7,4),dtype=np.float16)
  for output in range(outputs):
    for parity in range(2):
      for row in range(7):
        base=output*896+parity+row*28
        for tap in range(7):
          for component in range(4): packed[output,parity,row,tap,component]=original[base+component*224+tap*4]
  buf=Buffer("QCOM",packed.size,weight.dtype,initial_value=bytearray(packed.tobytes()))
  return UOp.from_buffer(buf)


def patch_conv(model) -> int:
  outer=model.captured.linear.src[0]
  batch=outer.src[0].src[0].src
  new_batch=[]
  replaced=0
  for call in batch:
    name=plain_name(call.src[0].arg.name) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM else ""
    if name in TARGETS:
      program=call.src[0]
      source=packed_source(program.src[2].arg)
      lib=Device["QCOM"].compiler.compile_cached(source)
      program=program.replace(src=program.src[:2]+(program.src[2].replace(arg=source),program.src[3].replace(arg=lib)))
      call=call.replace(src=(program,call.src[1],call.src[2],pack_weight_buffer(call.src[3]),*call.src[4:]))
      replaced+=1
    new_batch.append(call)
  model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(new_batch)},walk=True)
  model.captured.__dict__.pop("linear",None)
  return replaced


def main() -> None:
  parser=argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args=parser.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("patched",patch_conv(model))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__ == "__main__":main()
