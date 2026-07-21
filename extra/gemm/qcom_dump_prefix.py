#!/usr/bin/env python3
"""Run one cached model graph prefix and dump the selected call output."""
import argparse, pickle

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.jit import _prepare_jit_inputs, create_graph_call
from tinygrad.engine.realize import resolve_params, run_linear
from tinygrad.uop.ops import Ops, UOp


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("corpus")
  parser.add_argument("output")
  parser.add_argument("--case", type=int, default=9)
  parser.add_argument("--index", type=int)
  parser.add_argument("--indices", help="comma-separated indices; output must contain a {index} placeholder")
  parser.add_argument("--individual-last", action="store_true")
  args = parser.parse_args()
  if (args.index is None) == (args.indices is None): parser.error("pass exactly one of --index or --indices")
  indices = [args.index] if args.index is not None else [int(x) for x in args.indices.split(",")]
  if len(indices) > 1 and "{index}" not in args.output: parser.error("multi-index output must contain {index}")
  with open(args.model, "rb") as f: model = pickle.load(f)
  corpus = np.load(args.corpus)
  inputs = {}
  for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
    arr = corpus[f"case{args.case}:input:{name}"].astype(np.dtype(dtype.fmt), copy=False)
    inputs[name] = Tensor(arr, device=device).realize()
  input_uops, var_vals = _prepare_jit_inputs((), inputs)[:2]
  batch = model.captured.linear.src[0].src[0].src[0].src
  for index in indices:
    prefix_end = index if args.individual_last else index+1
    if prefix_end:
      run_linear(UOp(Ops.LINEAR, src=(create_graph_call(batch[:prefix_end]),)), var_vals,
                 input_uops=input_uops, jit=True, wait=True)
    if args.individual_last:
      run_linear(UOp(Ops.LINEAR, src=(batch[index],)), var_vals, input_uops=input_uops, jit=True, wait=True)
    call = batch[index]
    call_args = resolve_params(call, tuple(input_uops))
    out_buffer = call_args[call.src[0].arg.outs[0]]
    out = np.asarray(out_buffer.buffer.numpy()).copy()
    output = args.output.format(index=index)
    np.save(output, out)
    print(f"index={index} shape={out.shape} dtype={out.dtype} min={float(out.min())} max={float(out.max())} mean={float(out.mean())}")


if __name__ == "__main__": main()
