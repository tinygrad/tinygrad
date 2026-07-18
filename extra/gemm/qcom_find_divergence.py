#!/usr/bin/env python3
"""Locate the first call whose output differs between two compiled QCOM models."""
import argparse, pickle

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.jit import create_graph_call
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import resolve_params, run_linear
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def load(path: str):
  with open(path, "rb") as f: return pickle.load(f)


def batch(model): return model.captured.linear.src[0].src[0].src[0].src


def prepare(model, corpus, case: int):
  inputs = {}
  for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
    arr = corpus[f"case{case}:input:{name}"].astype(np.dtype(dtype.fmt), copy=False)
    inputs[name] = Tensor(arr, device=device).realize()
  return _prepare_jit_inputs((), inputs)[:2]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("reference")
  parser.add_argument("candidate")
  parser.add_argument("corpus")
  parser.add_argument("--case", type=int, default=9)
  parser.add_argument("--threshold", type=float, default=1e-3)
  parser.add_argument("--graph-prefix", type=int)
  args = parser.parse_args()
  reference, candidate = load(args.reference), load(args.candidate)
  corpus = np.load(args.corpus)
  rb, cb = batch(reference), batch(candidate)
  if len(rb) != len(cb): raise ValueError(f"batch lengths differ: {len(rb)} != {len(cb)}")
  ri, rv = prepare(reference, corpus, args.case)
  ci, cv = prepare(candidate, corpus, args.case)
  if args.graph_prefix is not None:
    index = args.graph_prefix
    rc, cc = rb[index], cb[index]
    run_linear(UOp(Ops.LINEAR, src=(create_graph_call(rb[:index+1]),)), rv, input_uops=ri, jit=True, wait=True)
    rargs = resolve_params(rc, tuple(ri))
    snapshots = {out_index: np.asarray(rargs[out_index].buffer.numpy(), dtype=np.float32).copy()
                 for out_index in rc.src[0].arg.outs}
    run_linear(UOp(Ops.LINEAR, src=(create_graph_call(cb[:index+1]),)), cv, input_uops=ci, jit=True, wait=True)
    cargs = resolve_params(cc, tuple(ci))
    for out_index, candidate_out_index in zip(rc.src[0].arg.outs, cc.src[0].arg.outs):
      delta = np.abs(snapshots[out_index]-np.asarray(cargs[candidate_out_index].buffer.numpy(), dtype=np.float32))
      print(f"{index}: graph-prefix output={out_index} max_abs={float(delta.max(initial=0)):.9g} mean_abs={float(delta.mean()):.9g}")
    return
  for index, (rc, cc) in enumerate(zip(rb, cb)):
    run_linear(UOp(Ops.LINEAR, src=(rc,)), rv, input_uops=ri, jit=True, wait=True)
    reference_outputs = {}
    if rc.op is Ops.CALL:
      rargs = resolve_params(rc, tuple(ri))
      reference_outputs = {out_index: (rargs[out_index].dtype, rargs[out_index].buffer.nbytes,
        np.asarray(rargs[out_index].buffer.numpy(), dtype=np.float32).copy()) for out_index in rc.src[0].arg.outs}
    run_linear(UOp(Ops.LINEAR, src=(cc,)), cv, input_uops=ci, jit=True, wait=True)
    if rc.op is not Ops.CALL or cc.op is not Ops.CALL: continue
    rn = plain_name(rc.src[0].arg.name) if rc.src[0].op is Ops.PROGRAM else str(rc.op)
    cn = plain_name(cc.src[0].arg.name) if cc.src[0].op is Ops.PROGRAM else str(cc.op)
    cargs = resolve_params(cc, tuple(ci))
    maximum = 0.0
    for out_index, candidate_out_index in zip(rc.src[0].arg.outs, cc.src[0].arg.outs):
      reference_dtype, reference_nbytes, ro = reference_outputs[out_index]
      cout = cargs[candidate_out_index]
      if reference_dtype != cout.dtype or reference_nbytes != cout.buffer.nbytes:
        print(f"{index}: {rn} -> {cn}: incompatible output {out_index}")
        continue
      co = np.asarray(cout.buffer.numpy(), dtype=np.float32)
      delta = np.abs(ro-co)
      out_maximum, mean = float(delta.max(initial=0)), float(delta.mean())
      maximum = max(maximum, out_maximum)
      if out_maximum > args.threshold or rn != cn:
        print(f"{index}: {rn} -> {cn}: output={out_index} max_abs={out_maximum:.9g} mean_abs={mean:.9g}")
    if maximum > args.threshold: break


if __name__ == "__main__": main()
