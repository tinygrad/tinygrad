#!/usr/bin/env python3
"""Group ready OpenPilot graph calls by program family without crossing dependency levels."""
import argparse, pickle
from collections import defaultdict

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_openpilot_ir3 import plain_name


def schedule_levels(model) -> int:
  outer=model.captured.linear.src[0]
  batch=list(outer.src[0].src[0].src)
  writer,levels={},{}
  grouped=defaultdict(list)
  for sequence,call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM:
      grouped[sequence].append((sequence,call))
      continue
    deps={writer[arg] for arg in call.src[1:] if arg in writer}
    level=1+max((levels[dep] for dep in deps),default=-1)
    levels[sequence]=level
    grouped[level].append((sequence,call))
    for output in call.src[0].arg.outs: writer[call.src[output+1]]=sequence
  scheduled=[]
  moved=0
  for entries in grouped.values():
    ordered=sorted(entries,key=lambda item:(plain_name(item[1].src[0].arg.name),item[0]))
    scheduled.extend(call for _index,call in ordered)
    moved+=sum(old_index!=entries[new_index][0] for new_index,(old_index,_call) in enumerate(ordered))
  if scheduled != batch:
    model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(scheduled)},walk=True)
    model.captured.__dict__.pop("linear",None)
  return moved


def main() -> None:
  ap=argparse.ArgumentParser()
  ap.add_argument("input")
  ap.add_argument("output")
  args=ap.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("moved",schedule_levels(model))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__=="__main__":main()
