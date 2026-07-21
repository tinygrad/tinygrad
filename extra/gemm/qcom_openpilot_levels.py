#!/usr/bin/env python3
"""Print dependency-level parallelism in a captured OpenPilot graph."""
import argparse, pickle
from collections import defaultdict

from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def main() -> None:
  ap = argparse.ArgumentParser(); ap.add_argument("model"); args = ap.parse_args()
  with open(args.model, "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  writer, level, groups = {}, {}, defaultdict(list)
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    deps = {writer[x] for x in call.src[1:] if x in writer}
    level[index] = 1 + max((level[x] for x in deps), default=-1)
    groups[level[index]].append((index, plain_name(call.src[0].arg.name), call.src[0].arg.global_size, call.src[0].arg.local_size))
    for out in call.src[0].arg.outs: writer[call.src[out+1]] = index
  for lev, calls in groups.items():
    if len(calls) > 1: print(f"level={lev} calls={len(calls)}", *calls, sep="\n  ")


if __name__ == "__main__": main()
