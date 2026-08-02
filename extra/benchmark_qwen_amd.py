import argparse, time
from tinygrad import Context
from tinygrad.llm.model import Transformer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="/raid/models/Qwen3.6-27B-IQ4_XS.gguf")
  parser.add_argument("--max-context", type=int, default=131072)
  parser.add_argument("--prompt-tokens", type=int, default=3000)
  parser.add_argument("--decode-tokens", type=int, default=16)
  parser.add_argument("--chunk-size", type=int, default=256)
  parser.add_argument("--skip-resume-check", action="store_true")
  args = parser.parse_args()

  st = time.perf_counter()
  model, _ = Transformer.from_gguf(args.model, args.max_context)
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)
  st = time.perf_counter()
  with Context(BEAM=0): model.warmup(args.chunk_size)
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)

  prompt = [257] + [1000+i%1000 for i in range(args.prompt_tokens-1)]
  gen, st = model.generate(prompt, chunk_size=args.chunk_size), time.perf_counter()
  output = [next(gen)]
  pt = time.perf_counter()
  print(f"prefill {args.prompt_tokens/(pt-st):.3f} tok/s", flush=True)
  for _ in range(args.decode_tokens): output.append(next(gen))
  et = time.perf_counter()
  print(f"decode {args.decode_tokens/(et-pt):.3f} tok/s output {output}", flush=True)

  if not args.skip_resume_check:
    follow = model._cached_tokens + [1234+i for i in range(8)]
    checkpoint_pos, gen, st = model._state_checkpoint_pos, model.generate(follow, chunk_size=args.chunk_size), time.perf_counter()
    resumed_token = next(gen)
    print(f"resume {len(follow)-1} tokens from {checkpoint_pos} in {time.perf_counter()-st:.3f}s token {resumed_token}", flush=True)
    full_prompt, model._cached_tokens = follow[:-1], [-1]
    st = time.perf_counter()
    full_token = next(model.generate(full_prompt, chunk_size=args.chunk_size))
    print(f"full {time.perf_counter()-st:.3f}s token {full_token} match {resumed_token == full_token}", flush=True)
