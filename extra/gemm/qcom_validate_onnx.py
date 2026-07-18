#!/usr/bin/env python3
"""Build a cached ONNX reference corpus and validate a compiled QCOM model against it."""
import argparse, hashlib, pickle, time

import numpy as np

DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,42,100,999,1234,65537"


def make_reference(onnx_path: str, output_path: str, seeds: list[int], scale: float) -> None:
  import onnxruntime as ort

  session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  ort_dtypes = {
    "tensor(float)": np.dtype(np.float32), "tensor(float16)": np.dtype(np.float16),
    "tensor(uint8)": np.dtype(np.uint8),
  }
  corpus: dict[str, np.ndarray] = {}
  with open(onnx_path, "rb") as f: corpus["onnx_sha256"] = np.asarray(hashlib.file_digest(f, "sha256").hexdigest())
  corpus["input_scale"] = np.asarray(scale, dtype=np.float32)
  for case, seed in enumerate(seeds):
    rng = np.random.default_rng(seed)
    inputs = {}
    for spec in session.get_inputs():
      shape = tuple(x if isinstance(x, int) else 1 for x in spec.shape)
      dtype = ort_dtypes[spec.type]
      if np.issubdtype(dtype, np.floating): arr = (rng.standard_normal(shape)*scale).astype(dtype)
      else: arr = rng.integers(0, 256, shape, dtype=dtype)
      inputs[spec.name] = arr
      corpus[f"case{case}:input:{spec.name}"] = arr
    corpus[f"case{case}:output"] = session.run(None, inputs)[0]
  corpus["seeds"] = np.asarray(seeds, dtype=np.int64)
  np.savez(output_path, **corpus)
  print(f"wrote {len(seeds)} cases to {output_path}")


def check_candidate(model_path: str, reference_path: str, rtol: float, atol: float, max_abs: float,
                    runs: int, cases: str = "", min_cases: int = 20, warmups: int = 2,
                    onnx_path: str = "") -> None:
  from tinygrad import Tensor

  with open(model_path, "rb") as f: model = pickle.load(f)
  corpus = np.load(reference_path)
  if onnx_path:
    with open(onnx_path, "rb") as f: actual_sha = hashlib.file_digest(f, "sha256").hexdigest()
    if "onnx_sha256" not in corpus: raise ValueError("reference corpus has no ONNX SHA-256")
    reference_sha = str(corpus["onnx_sha256"].item())
    if actual_sha != reference_sha:
      raise ValueError(f"reference corpus is for ONNX {reference_sha}, not {actual_sha}")
    print(f"onnx_sha256={actual_sha} reference_match=True")
  else:
    print("WARNING: ONNX identity not checked; pass --onnx for a correctness claim")
  all_seeds = corpus["seeds"].tolist()
  case_indices = [int(x) for x in cases.split(",")] if cases else list(range(len(all_seeds)))
  if len(case_indices) < min_cases:
    raise ValueError(f"refusing correctness claim from only {len(case_indices)} cases; need at least {min_cases}")
  seeds = [all_seeds[i] for i in case_indices]
  legacy = "names" in corpus and "case0:output" not in corpus
  legacy_names = corpus["names"].tolist() if legacy else []
  closes, max_errors, mean_errors, timings = [], [], [], []
  for case, seed in zip(case_indices, seeds):
    inputs = {}
    for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
      key = f"s{case}_input_{legacy_names.index(name)}" if legacy else f"case{case}:input:{name}"
      arr = corpus[key].astype(np.dtype(dtype.fmt), copy=False)
      assert arr.shape == view.shape, (name, arr.shape, view.shape)
      inputs[name] = Tensor(arr, device=device).realize()
    for _ in range(warmups): got = model(**inputs).numpy()
    start = time.perf_counter()
    for _ in range(runs): got = model(**inputs).numpy()
    timings.append((time.perf_counter()-start)*1000/runs)
    expected = corpus[f"s{case}_raw" if legacy else f"case{case}:output"].reshape(got.shape)
    delta = np.abs(expected.astype(np.float32)-got.astype(np.float32))
    max_errors.append(float(delta.max()))
    mean_errors.append(float(delta.mean()))
    relative_close = bool(np.allclose(expected, got, rtol=rtol, atol=atol))
    hard_abs_close = bool(np.isfinite(max_errors[-1]) and max_errors[-1] <= max_abs)
    closes.append(relative_close and hard_abs_close)
    print(f"seed={seed} ms={timings[-1]:.3f} max_abs={max_errors[-1]:.9g} "
          f"mean_abs={mean_errors[-1]:.9g} relative_close={relative_close} hard_abs_close={hard_abs_close} pass={closes[-1]}")
  print(f"candidate={model_path} mean_ms={np.mean(timings):.3f} max_abs={max(max_errors):.9g} "
        f"mean_abs={np.mean(mean_errors):.9g} allclose={all(closes)}")
  if not all(closes): raise SystemExit(1)


def main() -> None:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)
  make = subparsers.add_parser("make-reference")
  make.add_argument("onnx")
  make.add_argument("output")
  make.add_argument("--seeds", default=DEFAULT_SEEDS)
  make.add_argument("--scale", type=float, default=8.0)
  check = subparsers.add_parser("check-candidate")
  check.add_argument("model")
  check.add_argument("reference")
  check.add_argument("--rtol", type=float, default=1e-2)
  check.add_argument("--atol", type=float, default=1e-2)
  check.add_argument("--max-abs", type=float, default=1e-2,
                     help="hard maximum absolute error (in addition to allclose)")
  check.add_argument("--runs", type=int, default=1)
  check.add_argument("--warmups", type=int, default=2)
  check.add_argument("--onnx", default="", help="verify the reference corpus belongs to this exact ONNX file")
  check.add_argument("--cases", default="", help="comma-separated cached case indices")
  check.add_argument("--min-cases", type=int, default=20,
                     help="minimum number of cases required for a correctness claim (use 1 for focused diagnostics)")
  args = parser.parse_args()
  if args.command == "make-reference":
    make_reference(args.onnx, args.output, [int(x) for x in args.seeds.split(",")], args.scale)
  else: check_candidate(args.model, args.reference, args.rtol, args.atol, args.max_abs, args.runs, args.cases,
                        args.min_cases, args.warmups, args.onnx)


if __name__ == "__main__": main()
