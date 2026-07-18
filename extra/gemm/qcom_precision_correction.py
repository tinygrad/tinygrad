#!/usr/bin/env python3
"""Measure whether a fast model's output error admits a cross-validated correction."""
import argparse, pickle

import numpy as np

from tinygrad import Tensor


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("corpus")
  args = parser.parse_args()
  with open(args.model, "rb") as f: model = pickle.load(f)
  corpus = np.load(args.corpus)
  expected, actual = [], []
  for case, seed in enumerate(corpus["seeds"].tolist()):
    inputs = {}
    for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
      arr = corpus[f"case{case}:input:{name}"].astype(np.dtype(dtype.fmt), copy=False)
      inputs[name] = Tensor(arr, device=device).realize()
    got = np.asarray(model(**inputs).numpy(), dtype=np.float32).reshape(-1)
    ref = corpus[f"case{case}:output"].astype(np.float32).reshape(-1)
    actual.append(got.copy())
    expected.append(ref)
    print(f"captured case={case} seed={seed}")
  x, y = np.stack(actual), np.stack(expected)
  raw = np.abs(y-x)
  print(f"raw max={float(raw.max()):.9g} mean={float(raw.mean()):.9g}")
  for case in np.argsort(raw.max(axis=1))[-5:]:
    worst = int(raw[case].argmax())
    print(f"raw_case={int(case)} worst_index={worst} expected={float(y[case, worst]):.9g} "
          f"actual={float(x[case, worst]):.9g} max={float(raw[case, worst]):.9g} "
          f"bias_mean={float((y[case]-x[case]).mean()):.9g}")

  corrected_mean, corrected_affine = [], []
  for held_out in range(len(x)):
    train = np.arange(len(x)) != held_out
    bias = (y[train]-x[train]).mean(axis=0)
    corrected_mean.append(y[held_out]-(x[held_out]+bias))
    xm, ym = x[train].mean(axis=0), y[train].mean(axis=0)
    covariance = ((x[train]-xm)*(y[train]-ym)).sum(axis=0)
    variance = ((x[train]-xm)**2).sum(axis=0)
    slope = np.divide(covariance, variance, out=np.ones_like(covariance), where=variance > 1e-12)
    intercept = ym-slope*xm
    corrected_affine.append(y[held_out]-(slope*x[held_out]+intercept))
  for name, residual in (("mean", corrected_mean), ("affine", corrected_affine)):
    error = np.abs(np.stack(residual))
    maxima = error.max(axis=1)
    print(f"loo_{name} max={float(error.max()):.9g} mean={float(error.mean()):.9g} "
          f"passing={int((maxima <= 0.01).sum())}/{len(maxima)} per_case={maxima.tolist()}")


if __name__ == "__main__": main()
