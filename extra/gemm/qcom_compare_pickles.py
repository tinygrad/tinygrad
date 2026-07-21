#!/usr/bin/env python3
"""Compare two compiled-model pickles on identical deterministic inputs."""
import argparse, os, pickle, time

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.realize import graph_cache


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("reference")
  parser.add_argument("candidates", nargs="+")
  parser.add_argument("--seeds", default="123", help="comma-separated deterministic input seeds")
  parser.add_argument("--scale", type=float, default=1.0, help="scale applied to generated normal inputs")
  parser.add_argument("--rtol", type=float, default=1e-2)
  parser.add_argument("--atol", type=float, default=1e-2)
  parser.add_argument("--runs", type=int, default=5)
  parser.add_argument("--corpus", help="NPZ corpus with caseN:input:name arrays; seed values select case indices")
  parser.add_argument("--candidate-constlen", type=int)
  parser.add_argument("--corpus-output", action="store_true", help="compare with corpus out/caseN:output instead of rerun reference")
  args = parser.parse_args()
  with open(args.reference, "rb") as f: reference = pickle.load(f)
  seeds = [int(x) for x in args.seeds.split(",")]
  corpus = np.load(args.corpus) if args.corpus else None

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    inputs = {}
    for name, (view, _vars, dtype, device) in zip(reference.captured.expected_names, reference.captured.expected_input_info):
      corpus_key = f"case{seed}:input:{name}"
      arr = (corpus[corpus_key if corpus_key in corpus else name].astype(np.dtype(dtype.fmt), copy=False) if corpus is not None else
             (rng.standard_normal(view.shape)*args.scale).astype(np.dtype(dtype.fmt)))
      inputs[name] = Tensor(arr, device=device).realize()
    return inputs

  def run(model, inputs):
    for _ in range(2): out = model(**inputs).numpy()
    start = time.perf_counter()
    for _ in range(args.runs): out = model(**inputs).numpy()
    return np.array(out, copy=True), (time.perf_counter()-start)*1000.0/args.runs

  inputs_by_seed = [make_inputs(seed) for seed in seeds]
  refs, ref_times = zip(*(run(reference, inputs) for inputs in inputs_by_seed))
  if args.corpus_output:
    if corpus is None: raise ValueError("--corpus-output requires --corpus")
    refs = tuple(np.asarray(corpus[f"case{seed}:output" if f"case{seed}:output" in corpus else "out"]) for seed in seeds)
  print(f"reference_ms={np.mean(ref_times):.3f} seeds={seeds} scale={args.scale:g}")
  if args.candidate_constlen is not None:
    os.environ["QCOM_CONSTLEN"] = str(args.candidate_constlen)
    graph_cache.clear()
  failed = False
  for candidate_path in args.candidates:
    with open(candidate_path, "rb") as f: candidate = pickle.load(f)
    results = [run(candidate, inputs) for inputs in inputs_by_seed]
    got_times = [x[1] for x in results]
    deltas = [np.abs(ref-got) for ref, (got, _) in zip(refs, results)]
    closes = [np.allclose(ref, got, rtol=args.rtol, atol=args.atol) for ref, (got, _) in zip(refs, results)]
    worst_seed = int(np.argmax([x.max() for x in deltas]))
    worst = np.unravel_index(np.argmax(deltas[worst_seed]), deltas[worst_seed].shape)
    print(f"candidate={candidate_path} candidate_ms={np.mean(got_times):.3f}")
    print(f"max_abs={max(x.max() for x in deltas):.9g} mean_abs={np.mean([x.mean() for x in deltas]):.9g} "
          f"allclose={all(closes)} per_seed={closes}")
    print(f"worst_seed={seeds[worst_seed]} worst={worst} reference={refs[worst_seed][worst]!r} "
          f"candidate={results[worst_seed][0][worst]!r}")
    failed |= not all(closes)
  if failed: raise SystemExit(1)


if __name__ == "__main__": main()
