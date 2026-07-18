#!/usr/bin/env python3
"""Sensitivity experiment: alias selected vision MLP residual branches to identity."""
import argparse, pickle

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name

FORWARD, INVERSE = "r_32_192_4_4_64_4", "r_32_64_4_4_192_4"


def patch_model(model, selected:set[int]) -> int:
  outer=model.captured.linear.src[0]
  batch=list(outer.src[0].src[0].src)
  aliases, remove, pair = {}, set(), 0
  for index in range(len(batch)-1):
    a,b=batch[index:index+2]
    if not all(x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM for x in (a,b)): continue
    if (plain_name(a.src[0].arg.name),plain_name(b.src[0].arg.name)) != (FORWARD,INVERSE): continue
    if pair in selected:
      aliases[b.src[1]] = b.src[2]
      remove.update((index,index+1))
    pair += 1
  def representative(x):
    while x in aliases: x=aliases[x]
    return x
  new_batch=[call.replace(src=tuple(representative(x) for x in call.src)) for i,call in enumerate(batch) if i not in remove]
  model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(new_batch)},walk=True)
  model.captured.__dict__.pop("linear",None)
  return len(remove)//2


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output"); ap.add_argument("--indices",required=True); args=ap.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("skipped",patch_model(model,{int(x) for x in args.indices.split(",")}))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__ == "__main__":main()
