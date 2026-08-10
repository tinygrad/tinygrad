#!/usr/bin/env python3
"""Bounded correctness and load/prefill/decode benchmark for the official TP8 Kimi K3 checkpoint."""
import argparse, resource, time

from tinygrad import Device
from tinygrad.helpers import profile_marker
from tinygrad.llm.cli import KimiK3Template, SimpleTokenizer
from tinygrad.llm.kimi_k3 import load_kimi_k3, load_kimi_tokenizer_data

def sync(devices:int) -> None:
  for i in range(devices): Device[f"AMD:{i}"].synchronize()

def fresh_generate(model, prompt:list[int], chunk_size:int):
  # Never reuse a prefix or recurrent state across correctness/benchmark trials.
  model._cached_tokens = [-1] * len(prompt)
  return model.generate(prompt.copy(), chunk_size=chunk_size, temperature=0.0)

def timed_next(gen, devices:int) -> tuple[int, float]:
  begin = time.perf_counter()
  token = next(gen)
  sync(devices)
  return token, time.perf_counter()-begin

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="official unmodified Kimi K3 checkpoint directory")
  parser.add_argument("--devices", type=int, default=8)
  parser.add_argument("--max-context", type=int, default=128)
  parser.add_argument("--prompt", default="Reply with exactly: OK")
  parser.add_argument("--stable-tokens", type=int, default=8)
  parser.add_argument("--decode-tokens", type=int, default=8)
  parser.add_argument("--chunk-size", type=int, default=8)
  args = parser.parse_args()

  begin = time.perf_counter()
  model = load_kimi_k3(args.model, max_context=args.max_context, devices=args.devices)
  sync(args.devices)
  load_time = time.perf_counter()-begin
  print(f"load: {load_time:.3f}s", flush=True)

  normal, special, bos, eos = load_kimi_tokenizer_data(args.model)
  tok = SimpleTokenizer(normal, special, "kimi-k2", bos_id=bos, eos_id=eos, eot_id=eos)
  rendered = KimiK3Template().render(messages=[{"role":"user", "content":args.prompt}], add_generation_prompt=True)
  prompt = tok.encode(rendered)
  needed = len(prompt) + max(args.stable_tokens, args.decode_tokens+3)
  if needed > args.max_context: raise ValueError(f"prompt and output need {needed} tokens but max context is {args.max_context}")
  print(f"prompt: {len(prompt)} tokens, chunk={args.chunk_size}", flush=True)

  sequences:list[list[int]] = []
  # The first execution captures the prefill and rollout JITs. Correctness comparisons must use
  # identical replay paths, rather than comparing compilation/capture numerics to replay numerics.
  for trial in range(3):
    gen = fresh_generate(model, prompt, args.chunk_size)
    sequence:list[int] = []
    prefill = 0.0
    for step in range(args.stable_tokens):
      token, elapsed = timed_next(gen, args.devices)
      sequence.append(token)
      if step == 0: prefill = elapsed
    if trial: sequences.append(sequence)
    print(f"{'capture warmup' if trial == 0 else f'stable trial {trial}'}: prefill={prefill:.3f}s "
          f"({len(prompt)/prefill:.3f} tok/s), tokens={sequence}", flush=True)
  if sequences[0] != sequences[1]: raise RuntimeError(f"greedy output is not repeatable: {sequences}")
  print(f"stable text: {tok.decode(sequences[0])!r}", flush=True)

  gen = fresh_generate(model, prompt, args.chunk_size)
  profile_marker("kimi k3 steady prefill start")
  first, prefill = timed_next(gen, args.devices)
  profile_marker("kimi k3 steady prefill end")
  warmup = [timed_next(gen, args.devices)[0] for _ in range(2)]
  profile_marker("kimi k3 steady decode start")
  begin = time.perf_counter()
  output = [next(gen) for _ in range(args.decode_tokens)]
  sync(args.devices)
  decode = time.perf_counter()-begin
  profile_marker("kimi k3 steady decode end")
  print(f"prefill replay: {prefill:.3f}s ({len(prompt)/prefill:.3f} tok/s), token={first}", flush=True)
  print(f"decode after warmup {warmup}: {decode:.3f}s ({args.decode_tokens/decode:.3f} tok/s, "
        f"{decode/args.decode_tokens*1e3:.3f} ms/tok), output={output}", flush=True)
  print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MiB", flush=True)

if __name__ == "__main__": main()
