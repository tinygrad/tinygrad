#!/usr/bin/env python3
"""Benchmark Kimi-Linear load, prefill, and decode on its TP4 checkpoint."""
import argparse, resource, time
from tinygrad import Device, TinyJit
from tinygrad.llm.kimi import load_kimi

def sync(devices:int) -> None:
  for i in range(devices): Device[f"AMD:{i}"].synchronize()

def timed_next(gen, devices:int) -> tuple[int, float]:
  begin = time.perf_counter()
  token = next(gen)
  sync(devices)
  return token, time.perf_counter()-begin

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="converted Kimi-Linear-48B-A3B MXFP4-v2 directory")
  parser.add_argument("--devices", type=int, default=4)
  parser.add_argument("--max-context", type=int, default=128)
  parser.add_argument("--prompt-tokens", type=int, default=32)
  parser.add_argument("--decode-tokens", type=int, default=8)
  parser.add_argument("--chunk-size", type=int, default=32)
  parser.add_argument("--sweep-chunks", help="comma-separated prefill chunk sizes; uses the fastest for decode")
  args = parser.parse_args()
  if args.prompt_tokens < 1 or args.prompt_tokens + args.decode_tokens + 1 > args.max_context:
    raise ValueError("prompt and decode tokens must fit within --max-context")

  begin = time.perf_counter()
  model = load_kimi(args.model, max_context=args.max_context, devices=args.devices)
  sync(args.devices)
  print(f"load: {time.perf_counter()-begin:.3f}s", flush=True)

  prompt = [1] + [1000+i%1000 for i in range(args.prompt_tokens-1)]
  chunks = [int(x) for x in args.sweep_chunks.split(",")] if args.sweep_chunks else [args.chunk_size]
  if any(x < 1 or x > args.prompt_tokens for x in chunks): raise ValueError("prefill chunks must be between 1 and --prompt-tokens")
  timings:list[tuple[float, int]] = []
  prefill_jits:dict[int, TinyJit] = {}
  for chunk in chunks:
    # Recurrent prefill has a static token dimension. Give each swept shape its own capture;
    # the rollout JIT remains shared and independently benchmarks chunk 1/decode.
    if chunk != 1: model.prefill_jit = TinyJit(model.forward)
    cold = model.generate(prompt.copy(), chunk_size=chunk)
    first, cold_prefill = timed_next(cold, args.devices)
    print(f"chunk {chunk}: cold prefill {cold_prefill:.3f}s, token={first}", flush=True)
    warm = model.generate(prompt.copy(), chunk_size=chunk)
    warm_first, prefill = timed_next(warm, args.devices)
    if first != warm_first: raise RuntimeError(f"chunk {chunk} is not repeatable: cold={first}, warm={warm_first}")
    timings.append((prefill, chunk))
    if chunk != 1: prefill_jits[chunk] = model.prefill_jit
    print(f"chunk {chunk}: prefill {prefill:.3f}s ({args.prompt_tokens/prefill:.3f} tok/s), token={first}", flush=True)

  prefill, best_chunk = min(timings)
  if best_chunk != 1: model.prefill_jit = prefill_jits[best_chunk]
  warm = model.generate(prompt.copy(), chunk_size=best_chunk)
  first, replay_prefill = timed_next(warm, args.devices)
  _, cold_decode = timed_next(warm, args.devices)
  _, capture_decode = timed_next(warm, args.devices)
  print(f"selected chunk: {best_chunk}; prefill replay {replay_prefill:.3f}s "
        f"({args.prompt_tokens/replay_prefill:.3f} tok/s), token={first}", flush=True)
  print(f"cold decode: {cold_decode:.3f}s", flush=True)
  print(f"capture decode: {capture_decode:.3f}s", flush=True)
  begin = time.perf_counter()
  output = [next(warm) for _ in range(args.decode_tokens)]
  sync(args.devices)
  decode = time.perf_counter()-begin
  print(f"decode: {decode:.3f}s ({args.decode_tokens/decode:.3f} tok/s, {decode/args.decode_tokens*1e3:.3f} ms/tok), output={output}", flush=True)
  print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MiB", flush=True)

if __name__ == "__main__": main()
