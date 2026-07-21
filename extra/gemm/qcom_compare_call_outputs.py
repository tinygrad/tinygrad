#!/usr/bin/env python3
"""Compare captured buffers after selected calls in two OpenPilot pickles."""
import argparse
import pickle

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.jit import _prepare_jit_inputs, create_graph_call
from tinygrad.engine.realize import resolve_params, run_linear
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def capture(model, corpus, name: str, arg_index: int) -> np.ndarray:
  inputs = {key: Tensor(corpus[key], device=device).realize()
            for key, (_view, _vars, _dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info)}
  input_uops, var_vals, _names, _info = _prepare_jit_inputs((), inputs)
  batch = model.captured.linear.src[0].src[0].src[0].src
  index, call = next((i, call) for i, call in enumerate(batch) if call.op is Ops.CALL and
                     call.src[0].op is Ops.PROGRAM and plain_name(call.src[0].arg.name) == name)
  run_linear(UOp(Ops.LINEAR, src=(create_graph_call(list(batch[:index+1])),)), var_vals,
             input_uops=input_uops, jit=True, wait=True)
  resolved = resolve_params(call, tuple(input_uops))
  output = resolved[call.src[0].arg.outs[0] if arg_index < 0 else arg_index]
  return output.buffer.numpy().copy()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("left")
  parser.add_argument("left_name")
  parser.add_argument("right")
  parser.add_argument("right_name")
  parser.add_argument("corpus")
  parser.add_argument("--left-arg", type=int, default=-1)
  parser.add_argument("--right-arg", type=int, default=-1)
  args = parser.parse_args()
  with open(args.left, "rb") as f:
    left = pickle.load(f)
  with open(args.right, "rb") as f:
    right = pickle.load(f)
  corpus = np.load(args.corpus)
  a = capture(left, corpus, args.left_name, args.left_arg)
  b = capture(right, corpus, args.right_name, args.right_arg)
  if a.size == 32*1088*4 and b.size == 2048*16*4:
    image, expected = a.reshape(32, 1088, 4), np.empty((2048, 16, 4), dtype=a.dtype)
    for row in range(2048):
      idx1, block = row >> 2, row & 3
      expected[row] = image[idx1 >> 4, (idx1 & 15)*68+block*17:(idx1 & 15)*68+block*17+16]
    a = expected.reshape(-1)
  delta = np.abs(a.astype(np.float32)-b.astype(np.float32))
  print("shape", a.shape, b.shape, "max", float(delta.max()), "mean", float(delta.mean()))
  print("left", a[:32])
  print("right", b[:32])


if __name__ == "__main__":
  main()
