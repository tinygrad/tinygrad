#!/usr/bin/env python3
"""Sweep equivalent launch geometries for global-ID-only OpenPilot kernels."""
import argparse, itertools, pickle, statistics

from tinygrad import Tensor
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import get_runtime, resolve_params
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def divisors(value:int) -> list[int]: return [x for x in range(1, value+1) if value%x == 0]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("name")
  parser.add_argument("--runs", type=int, default=7)
  parser.add_argument("--max-threads", type=int, default=256)
  args = parser.parse_args()
  with open(args.model, "rb") as f: model = pickle.load(f)
  inputs = {name:Tensor.zeros(*view.shape, dtype=dtype, device=device).contiguous().realize()
            for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info)}
  input_uops, _var_vals, _names, _info = _prepare_jit_inputs((), inputs)
  model(**inputs).numpy()
  calls = [call for call in model.captured.linear.src[0].src[0].src[0].src
           if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and plain_name(call.src[0].arg.name) == args.name]
  call, program = calls[0], calls[0].src[0]
  source = program.src[2].arg
  if any(token in source for token in ("get_group_id", "get_local_id", "barrier(", "__local")):
    raise ValueError("kernel is not global-ID-only")
  resolved = resolve_params(call, tuple(input_uops))
  bufs = [u.buffer.ensure_allocated()._buf for u in resolved]
  runtime = get_runtime(resolved[0].device, program)
  total = tuple(int(g*l) for g,l in zip(program.arg.global_size, program.arg.local_size))
  candidates = []
  for local in itertools.product(*(divisors(x) for x in total)):
    threads = local[0]*local[1]*local[2]
    if threads > args.max_threads or threads < 16: continue
    global_size = tuple(total[i]//local[i] for i in range(3))
    candidates.append((local, global_size))
  results = []
  for local, global_size in candidates:
    for _ in range(2): runtime(*bufs, global_size=global_size, local_size=local, vals=(), wait=True)
    times = [runtime(*bufs, global_size=global_size, local_size=local, vals=(), wait=True)*1e3 for _ in range(args.runs)]
    results.append((statistics.median(times), min(times), local, global_size))
  for median, best, local, global_size in sorted(results)[:30]:
    print(f"median_ms={median:.5f} best_ms={best:.5f} local={local} global={global_size}")


if __name__ == "__main__": main()
