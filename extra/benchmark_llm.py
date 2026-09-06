import argparse, time
from tinygrad.llm.model import Transformer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, help="path to gguf model")
  parser.add_argument("--max-context", type=int, default=8192, help="max context length (default: %(default)s)")
  parser.add_argument("--prompt-tokens", type=int, default=1024, help="number of prompt tokens (default: %(default)s)")
  parser.add_argument("--decode-tokens", type=int, default=16, help="number of tokens to decode (default: %(default)s)")
  parser.add_argument("--chunk-size", type=int, default=32, help="chunk size for prefill (default: %(default)s)")
  parser.add_argument("--context-points", help="comma-separated context lengths to benchmark in one run")
  args = parser.parse_args()

  st = time.perf_counter()
  model, _ = Transformer.from_gguf(args.model, args.max_context)
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)

  st = time.perf_counter()
  model.warmup()
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)

  context_points = [int(x) for x in args.context_points.split(",")] if args.context_points else [args.prompt_tokens]
  if context_points != sorted(context_points) or any(x < 0 for x in context_points): raise ValueError("context points must be non-negative and sorted")

  output, prompt = [], [257]
  for context in context_points:
    prompt_tokens = max(context, 1)  # context 0 uses only the start token
    if len(prompt) > prompt_tokens: raise ValueError(f"context {context} overlaps the previous decode window ending at {len(prompt)}")
    if prompt_tokens + args.decode_tokens + 1 > args.max_context: raise ValueError(f"context {context} exceeds max context")
    prompt += [1000+i%1000 for i in range(len(prompt), prompt_tokens)]
    prefill_tokens = prompt_tokens - model.get_start_pos(prompt)
    gen = model.generate(prompt, chunk_size=args.chunk_size)
    st = time.perf_counter()
    # first token is time-to-first-token; counted as part of prefill
    output.append(next(gen))
    pt = time.perf_counter()
    print(f"context {context} prefill {prefill_tokens/(pt-st):.3f} tok/s ({prefill_tokens} tokens)", flush=True)

    for _ in range(args.decode_tokens): output.append(next(gen))
    et = time.perf_counter()
    print(f"context {context} decode {args.decode_tokens/(et-pt):.3f} tok/s", flush=True)
  print(f"output {output}" if not args.context_points else f"output checksum {sum((i+1)*x for i, x in enumerate(output))} ({len(output)} tokens)", flush=True)
