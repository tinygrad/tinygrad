#!/usr/bin/env python3
"""Sweep QCOM compute texture/UAV partition registers on one captured model."""
import argparse, os, pickle, time

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.realize import graph_cache


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("corpus")
  parser.add_argument("--case", type=int, default=9)
  parser.add_argument("--pairs", default="128:64,1:1,1:64,64:1,32:32,64:32,128:32,64:64")
  parser.add_argument("--runs", type=int, default=5)
  args = parser.parse_args()
  with open(args.model, "rb") as f: model = pickle.load(f)
  corpus = np.load(args.corpus)
  inputs = {}
  for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
    key=f"case{args.case}:input:{name}"
    inputs[name] = Tensor(corpus[key if key in corpus else name].astype(np.dtype(dtype.fmt), copy=False), device=device).realize()
  output_key=f"case{args.case}:output"
  expected = corpus[output_key if output_key in corpus else "out"]
  for pair in args.pairs.split(","):
    tsize, usize = pair.split(":")
    os.environ["QCOM_TSIZE"], os.environ["QCOM_USIZE"] = tsize, usize
    graph_cache.clear()
    for _ in range(2): got = model(**inputs).numpy()
    start = time.perf_counter()
    for _ in range(args.runs): got = model(**inputs).numpy()
    elapsed = (time.perf_counter()-start)*1000/args.runs
    delta = np.abs(got.astype(np.float32)-expected.reshape(got.shape).astype(np.float32))
    print(f"tsize={tsize} usize={usize} ms={elapsed:.3f} max_abs={float(delta.max()):.9g}")


if __name__ == "__main__": main()
