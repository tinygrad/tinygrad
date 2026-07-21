#!/usr/bin/env python3
"""Replace selected QCOM GELU epilogues with a bounded polynomial approximation."""
import argparse, pickle, re

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def fast_gelu_source(source:str) -> tuple[str, int]:
  variables=set(re.findall(r"float (alu\d+) =",source))
  replaced=0
  for var in variables:
    old=f"((1/(1.0f+exp2((({var}+(0.044708251953125f*{var}*{var}*{var}))*-2.3021129851685216f))))*{var})"
    # Degree-10 approximation of the model's exact tanh-GELU on |x| < 4.
    # GELU(-x)=GELU(x)-x lets one polynomial cover both signs; outside this
    # interval ReLU differs from the source expression by less than 1.3e-4.
    coeffs=(1.95458887333,2.17220398188,-0.215882554761,-0.00733160096454,0.28997582181,-0.274775761974,
            0.0167240224759,0.132422938329,-0.0634056438625,-0.022554589963,0.0179724326925)
    t=f"(fabs({var})*0.5f-1.0f)"
    poly=f"{coeffs[-1]:.10g}f"
    for coefficient in reversed(coeffs[:-1]): poly=f"({coefficient:.10g}f+{t}*{poly})"
    new=f"((fabs({var})>=4.0f)?max({var},0.0f):({poly}+min({var},0.0f)))"
    if old in source:
      source=source.replace(old,new)
      replaced+=1
  return source,replaced


def patch_model(model,names:set[str]) -> int:
  outer=model.captured.linear.src[0]
  batch=list(outer.src[0].src[0].src)
  compiler,cache,patched=Device["QCOM"].compiler,{},0
  for index,call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) not in names: continue
    program=call.src[0]
    source,count=fast_gelu_source(program.src[2].arg)
    if not count: continue
    if source not in cache: cache[source]=compiler.compile(source)
    program=program.replace(src=program.src[:2]+(program.src[2].replace(arg=source),program.src[3].replace(arg=cache[source])))
    batch[index]=call.replace(src=(program,*call.src[1:]))
    patched+=1
  if patched:
    model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(batch)},walk=True)
    model.captured.__dict__.pop("linear",None)
  return patched


if __name__ == "__main__":
  ap=argparse.ArgumentParser();ap.add_argument("input");ap.add_argument("output");ap.add_argument("--names",required=True);args=ap.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("patched",patch_model(model,set(args.names.split(","))))
  with open(args.output,"wb") as f:pickle.dump(model,f)
