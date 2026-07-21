#!/usr/bin/env python3
"""Validate a pickled openpilot TinyJit against saved ONNX Runtime outputs."""
import argparse, hashlib, pickle, time

import numpy as np

from tinygrad import Tensor


def sha256(path:str) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    while chunk := f.read(1 << 20): digest.update(chunk)
  return digest.hexdigest()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("pickle")
  ap.add_argument("datasets", nargs="+")
  ap.add_argument("--max-cases", type=int)
  ap.add_argument("--cases", help="comma-separated case indices")
  args = ap.parse_args()
  print(f"pickle_sha256={sha256(args.pickle)}")
  with open(args.pickle, "rb") as f: model = pickle.load(f)
  total, failures, worst_abs, worst_where = 0, 0, 0.0, ""
  for dataset_path in args.datasets:
    data = np.load(dataset_path)
    cases = sorted({int(k.split(":", 1)[0][4:]) for k in data.files if k.startswith("case")})
    if args.cases is not None:
      selected = {int(x) for x in args.cases.split(",")}
      cases = [x for x in cases if x in selected]
    if args.max_cases is not None: cases = cases[:args.max_cases]
    print(f"dataset={dataset_path} onnx_sha256={data['onnx_sha256'].item()} cases={len(cases)}")
    for case in cases:
      inputs = {name:Tensor(data[f"case{case}:input:{name}"], device="QCOM") for name in ("big_img", "img")}
      start = time.perf_counter()
      got = model(**inputs).numpy()
      elapsed_ms = (time.perf_counter()-start)*1e3
      expected = data[f"case{case}:output"]
      delta = np.abs(got-expected)
      max_abs, mean_abs = float(delta.max()), float(delta.mean())
      close = bool(np.allclose(got, expected, rtol=1e-2, atol=1e-2))
      hard = max_abs <= 1e-2
      passed = close and hard and bool(np.isfinite(got).all())
      failures += not passed
      total += 1
      if max_abs > worst_abs: worst_abs, worst_where = max_abs, f"{dataset_path}:case{case}"
      print(f"case={case:02d} ms={elapsed_ms:.3f} max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} "
            f"allclose={close} hard_max={hard} finite={bool(np.isfinite(got).all())} pass={passed}")
  print(f"SUMMARY total={total} failures={failures} worst_abs={worst_abs:.9g} worst={worst_where}")
  if failures: raise SystemExit(1)


if __name__ == "__main__": main()
