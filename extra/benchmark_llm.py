#!/usr/bin/env python3
"""Benchmark tinygrad LLM prefill and decode independently.

Examples:
  python -m extra.benchmark_llm --model qwen3:0.6b --max-context 32768
  python -m extra.benchmark_llm --model /path/to/model.gguf --prompt-tokens 8192 --realize
"""
from __future__ import annotations

import argparse, json, statistics, time
from dataclasses import asdict, dataclass

from tinygrad.helpers import fetch
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


def percentile(values:list[float], percentile:float) -> float:
  ordered = sorted(values)
  return ordered[round((len(ordered) - 1) * percentile)]


def synthetic_prompt(length:int, vocab_size:int, salt:int) -> list[int]:
  # Avoid tokenizer and chat-template work while exercising the same embedding/model path.
  # Changing token zero guarantees that Transformer.get_start_pos cannot reuse an earlier KV cache.
  assert length > 0 and vocab_size > 256
  return [256 + salt % (vocab_size - 256)] + [256 + (i * 7919) % (vocab_size - 256) for i in range(1, length)]


def benchmark(model:Transformer, prompt:list[int], decode_tokens:int, chunk_size:int) -> Result:
  gen = model.generate(prompt.copy(), chunk_size=chunk_size)
  begin = time.perf_counter()
  next(gen)
  ttft = time.perf_counter() - begin

  decode_times: list[float] = []
  for _ in range(decode_tokens):
    begin = time.perf_counter()
    next(gen)
    decode_times.append(time.perf_counter() - begin)

  return Result(len(prompt), decode_tokens, ttft, len(prompt) / ttft, decode_tokens / sum(decode_times),
                statistics.median(decode_times) * 1e3, percentile(decode_times, 0.95) * 1e3)


def main() -> None:
  parser = argparse.ArgumentParser(description="Measure LLM prefill and steady-state decode speed")
  parser.add_argument("--model", default="qwen3:0.6b", help="Model preset or local GGUF path")
  parser.add_argument("--max-context", type=int, default=32768)
  parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[128, 2048, 8192])
  parser.add_argument("--decode-tokens", type=int, default=32)
  parser.add_argument("--chunk-size", type=int, default=256)
  parser.add_argument("--realize", action="store_true", help="Unpack model weights once at load time")
  parser.add_argument("--json", action="store_true", help="Print machine-readable results")
  args = parser.parse_args()

  if args.decode_tokens < 1: parser.error("--decode-tokens must be positive")
  if args.chunk_size < 1: parser.error("--chunk-size must be positive")
  if max(args.prompt_tokens) + args.decode_tokens >= args.max_context:
    parser.error("prompt plus decode tokens must fit within --max-context")

  path = fetch(models.get(args.model, args.model))
  model, kv = Transformer.from_gguf(path, args.max_context, realize=args.realize)
  vocab_size = len(kv["tokenizer.ggml.tokens"])

  model.warmup(args.chunk_size)

  results = [benchmark(model, synthetic_prompt(n, vocab_size, salt=i+1), args.decode_tokens, args.chunk_size)
             for i, n in enumerate(args.prompt_tokens)]
  if args.json:
    print(json.dumps({"model": args.model, "max_context": args.max_context, "chunk_size": args.chunk_size,
                      "realize": args.realize, "results": [asdict(x) for x in results]}, indent=2))
    return

  print(f"model={args.model} max_context={args.max_context} chunk_size={args.chunk_size} realize={args.realize}")
  print(f"{'prompt':>8} {'TTFT':>10} {'prefill':>14} {'decode':>14} {'decode p50':>12} {'decode p95':>12}")
  for result in results:
    print(f"{result.prompt_tokens:8d} {result.time_to_first_token_s:9.3f}s {result.prefill_tokens_per_s:11.1f} t/s "
          f"{result.decode_tokens_per_s:11.1f} t/s {result.decode_p50_ms:9.2f} ms {result.decode_p95_ms:9.2f} ms")


if __name__ == "__main__": main()
