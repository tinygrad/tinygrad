#!/usr/bin/env python3
"""Benchmark tinygrad LLM prefill and decode independently."""
from __future__ import annotations

import argparse, gc, json, statistics, time
from dataclasses import asdict, dataclass

from tinygrad import Context, Device, Tensor, UOp
from tinygrad.helpers import fetch, profile_marker
from tinygrad.llm.cli import models
from tinygrad.llm.model import Transformer

@dataclass
class Result:
  prompt_tokens: int
  decode_tokens: int
  time_to_first_token_s: float
  prefill_tokens_per_s: float
  decode_tokens_per_s: float
  decode_p50_ms: float
  decode_p95_ms: float
  output_tokens: list[int]

def percentile(values:list[float], percentile:float) -> float:
  ordered = sorted(values)
  return ordered[round((len(ordered) - 1) * percentile)]

def synthetic_prompt(length:int, vocab_size:int, salt:int) -> list[int]:
  assert length > 0 and vocab_size > 256
  return [256 + salt % (vocab_size - 256)] + [256 + (i * 7919) % (vocab_size - 256) for i in range(1, length)]

def benchmark(model:Transformer, prompt:list[int], decode_tokens:int, chunk_size:int) -> Result:
  gen = model.generate(prompt.copy(), chunk_size=chunk_size)
  profile_marker(f"prefill {len(prompt)} start")
  begin = time.perf_counter()
  output_tokens = [next(gen)]
  ttft = time.perf_counter() - begin
  profile_marker(f"prefill {len(prompt)} end")

  decode_times: list[float] = []
  profile_marker(f"decode {len(prompt)} start")
  for _ in range(decode_tokens):
    begin = time.perf_counter()
    output_tokens.append(next(gen))
    decode_times.append(time.perf_counter() - begin)
  profile_marker(f"decode {len(prompt)} end")

  return Result(len(prompt), decode_tokens, ttft, len(prompt) / ttft, decode_tokens / sum(decode_times),
                statistics.median(decode_times) * 1e3, percentile(decode_times, 0.95) * 1e3, output_tokens)

def benchmark_decode_position(model:Transformer, position:int, decode_tokens:int) -> Result:
  token = Tensor([[0]], dtype="int32", device=Device.DEFAULT).realize()
  temperature = Tensor([0.0], device=Device.DEFAULT).realize()
  decode_times, output_tokens = [], []
  for pos in range(position, position + decode_tokens):
    begin = time.perf_counter()
    output_tokens.append(int(model(token, UOp.variable("start_pos", 0, model.max_context-1).bind(pos), temperature).realize().item()))
    decode_times.append(time.perf_counter() - begin)
  return Result(position, decode_tokens, 0.0, 0.0, decode_tokens / sum(decode_times),
                statistics.median(decode_times) * 1e3, percentile(decode_times, 0.95) * 1e3, output_tokens)

def main() -> None:
  parser = argparse.ArgumentParser(description="Measure LLM prefill and steady-state decode speed")
  parser.add_argument("--model", default="qwen3:0.6b", help="Model preset or local GGUF path")
  parser.add_argument("--max-context", type=int, default=32768)
  parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[128, 2048, 8192])
  parser.add_argument("--decode-tokens", type=int, default=32)
  parser.add_argument("--decode-position", type=int, nargs="+")
  parser.add_argument("--chunk-size", type=int, default=256)
  parser.add_argument("--beam", type=int, default=2)
  parser.add_argument("--jit-batch-size", type=int, default=448)
  parser.add_argument("--parallel-compile", type=int, default=12)
  parser.add_argument("--realize", action="store_true")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  if args.decode_tokens < 1: parser.error("--decode-tokens must be positive")
  if args.chunk_size < 1: parser.error("--chunk-size must be positive")
  if args.decode_position is None and max(args.prompt_tokens) + args.decode_tokens >= args.max_context:
    parser.error("prompt plus decode tokens must fit within --max-context")
  if args.decode_position is not None and max(args.decode_position) + args.decode_tokens >= args.max_context:
    parser.error("decode position plus decode tokens must fit within --max-context")

  begin = time.perf_counter()
  path = fetch(models.get(args.model, args.model))
  fetched = time.perf_counter()
  model, kv = Transformer.from_gguf(path, args.max_context, realize=args.realize)
  loaded = time.perf_counter()
  vocab_size = len(kv["tokenizer.ggml.tokens"])
  print(f"startup: fetch={fetched-begin:.2f}s load={loaded-fetched:.2f}s", flush=True)
  with Context(BEAM=args.beam, JIT_BATCH_SIZE=args.jit_batch_size, PARALLEL_COMPILE=args.parallel_compile):
    model.warmup(args.chunk_size)
  startup = time.perf_counter() - begin
  print(f"startup: warmup={startup-(loaded-begin):.2f}s total={startup:.2f}s", flush=True)
  gc.freeze()

  results = [benchmark_decode_position(model, pos, args.decode_tokens) for pos in args.decode_position] if args.decode_position is not None else \
    [benchmark(model, synthetic_prompt(n, vocab_size, salt=i+1), args.decode_tokens, args.chunk_size)
     for i, n in enumerate(args.prompt_tokens)]
  if args.json:
    print(json.dumps({"model": args.model, "max_context": args.max_context, "chunk_size": args.chunk_size,
                      "beam": args.beam, "jit_batch_size": args.jit_batch_size, "parallel_compile": args.parallel_compile,
                      "realize": args.realize, "startup_s": startup, "results": [asdict(x) for x in results]}, indent=2))
    return

  print(f"model={args.model} max_context={args.max_context} chunk_size={args.chunk_size} beam={args.beam} "
        f"jit_batch_size={args.jit_batch_size} parallel_compile={args.parallel_compile} realize={args.realize} startup={startup:.2f}s")
  print(f"{'prompt':>8} {'TTFT':>10} {'prefill':>14} {'decode':>14} {'decode p50':>12} {'decode p95':>12}")
  for result in results:
    print(f"{result.prompt_tokens:8d} {result.time_to_first_token_s:9.3f}s {result.prefill_tokens_per_s:11.1f} t/s "
          f"{result.decode_tokens_per_s:11.1f} t/s {result.decode_p50_ms:9.2f} ms {result.decode_p95_ms:9.2f} ms")

if __name__ == "__main__": main()
