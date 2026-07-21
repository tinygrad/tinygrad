#!/usr/bin/env python3
"""Compare the first fused OpenPilot graph operation with its original call sequence."""
import argparse, pickle

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def load(path):
  with open(path, "rb") as f: return pickle.load(f)


def batch(model): return model.captured.linear.src[0].src[0].src[0].src


def prepared(model, corpus_path, case=0):
  corpus = np.load(corpus_path)
  legacy = "names" in corpus and "case0:output" not in corpus
  names = corpus["names"].tolist() if legacy else []
  inputs = {}
  for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
    key = f"s{case}_input_{names.index(name)}" if legacy else f"case{case}:input:{name}"
    arr = corpus[key].astype(np.dtype(dtype.fmt), copy=False)
    inputs[name] = Tensor(arr, device=device).realize()
  return _prepare_jit_inputs((), inputs)[:2]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("base")
  ap.add_argument("candidate")
  ap.add_argument("--fused", required=True)
  ap.add_argument("--original", required=True, help="comma-separated original program names")
  ap.add_argument("--occurrence", type=int, default=0, help="zero-based matching fusion occurrence")
  ap.add_argument("--corpus", default="/data/openpilot_validation_5seeds.npz")
  ap.add_argument("--case", type=int, default=0)
  ap.add_argument("--side", choices=("base", "candidate"), help="run and dump only one model in this process")
  ap.add_argument("--dump", help=".npy output path for --side")
  ap.add_argument("--dump-inputs", help="optional .npz path containing selected call arguments")
  args = ap.parse_args()
  originals = args.original.split(",")
  if args.side:
    if not args.dump: ap.error("--side requires --dump")
    model = load(args.base if args.side == "base" else args.candidate)
    calls = batch(model)
    if args.side == "base":
      indices = [i for i in range(len(calls)-len(originals)+1)
                 if [plain_name(x.src[0].arg.name) for x in calls[i:i+len(originals)]] == originals]
      end = indices[args.occurrence]+len(originals)
    else:
      indices = [i for i, x in enumerate(calls) if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
                 plain_name(x.src[0].arg.name) == args.fused]
      end = indices[args.occurrence]+1
    iu, vv = prepared(model, args.corpus, args.case)
    run_linear(UOp(Ops.LINEAR, src=tuple(calls[:end])), vv, input_uops=iu, jit=True, wait=True)
    last_call = calls[end-1]
    out_index = last_call.src[0].arg.outs[0]
    out = np.array(last_call.src[out_index+1].buffer.numpy(), copy=True)
    np.save(args.dump, out)
    if args.dump_inputs:
      selected = calls[indices[args.occurrence]:end]
      np.savez(args.dump_inputs, **{f"call{ci}_arg{ai}":np.array(arg.buffer.numpy(), copy=True)
                                    for ci, call in enumerate(selected) for ai, arg in enumerate(call.src[1:])
                                    if arg.op in (Ops.BUFFER, Ops.SLICE)})
    print(args.side, "index", end-1, "shape", out.shape, "min", float(out.min()), "max", float(out.max()))
    return
  base, cand = load(args.base), load(args.candidate)
  bb, cb = batch(base), batch(cand)
  bis = [i for i in range(len(bb)-len(originals)+1)
         if [plain_name(x.src[0].arg.name) for x in bb[i:i+len(originals)]] == originals]
  cis = [i for i, x in enumerate(cb) if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
         plain_name(x.src[0].arg.name) == args.fused]
  bi, ci = bis[args.occurrence], cis[args.occurrence]
  iu, vv = prepared(base, args.corpus, args.case)
  run_linear(UOp(Ops.LINEAR, src=tuple(bb[:bi+len(originals)])), vv, input_uops=iu, jit=True, wait=True)
  bo = np.array(bb[bi+len(originals)-1].src[1].buffer.numpy(), copy=True)
  iu, vv = prepared(cand, args.corpus, args.case)
  run_linear(UOp(Ops.LINEAR, src=tuple(cb[:ci+1])), vv, input_uops=iu, jit=True, wait=True)
  co = np.array(cb[ci].src[1].buffer.numpy(), copy=True)
  d = np.abs(bo.astype(np.float32)-co.astype(np.float32))
  at = np.unravel_index(int(d.argmax()), d.shape)
  print("indices", bi, ci, "shape", bo.shape, "max_abs", float(d[at]), "mean_abs", float(d.mean()),
        "at", at, "base", float(bo[at]), "candidate", float(co[at]))
  print("base_stats", float(bo.min()), float(bo.max()), float(np.mean(np.abs(bo))))
  print("candidate_stats", float(co.min()), float(co.max()), float(np.mean(np.abs(co))))


if __name__ == "__main__": main()
