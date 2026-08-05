import argparse, time
from tinygrad.llm.cli import models
from tinygrad.llm.model import Transformer
from tinygrad.helpers import fetch

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="qwen3.5:0.8b", help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max-context", type=int, default=32768)
  parser.add_argument("--prompt-tokens", type=int, default=3072)
  parser.add_argument("--decode-tokens", type=int, default=16)
  parser.add_argument("--chunk-size", type=int, default=256)
  parser.add_argument("--expect-output", type=int, nargs="+", default=None, help="expected output tokens to assert on")
  parser.add_argument("--skip-resume-check", action="store_true")
  args = parser.parse_args()

  startup_st = st = time.perf_counter()
  model, _ = Transformer.from_gguf(fetch(models.get(args.model, args.model)), args.max_context)
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)
  st = time.perf_counter()
  model.warmup()
  # warm up the chunked prefill JIT too, then invalidate the prompt cache (forces the state reset path)
  if model.has_recurrent_block:
    for _ in range(2):
      warm = model.generate([0]*args.chunk_size, chunk_size=args.chunk_size)
      next(warm), next(warm)
      model._cached_tokens = [-1]
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)
  states = [getattr(block, name) for block in model.blk for name in ("cache_kv", "cache_k", "cache_v", "conv_state", "recurrent_state")
            if hasattr(block, name)]
  device = str(model.token_embd.weight.device)
  assert all(str(state.device) == device and state.uop.is_realized for state in states)
  assert all(block.cache_kv.shape[3] >= args.max_context for block in model.blk if hasattr(block, "cache_kv"))
  assert model.prefill_jit.cnt >= 2 and model.rollout_jit.cnt >= 2
  print(f"preallocated {sum(state.nbytes() for state in states)/2**30:.3f} GiB state on {device}", flush=True)

  prompt = [257] + [1000+i%1000 for i in range(args.prompt_tokens-1)]
  gen, st = model.generate(prompt, chunk_size=args.chunk_size), time.perf_counter()
  output = [next(gen)]
  pt = time.perf_counter()
  print(f"prefill {args.prompt_tokens/(pt-st):.3f} tok/s", flush=True)
  for _ in range(args.decode_tokens): output.append(next(gen))
  print(f"decode {args.decode_tokens/(time.perf_counter()-pt):.3f} tok/s output {output}", flush=True)
  if args.expect_output is not None: assert output == args.expect_output, f"expected {args.expect_output}, got {output}"

  if not args.skip_resume_check:
    follow = model._cached_tokens + [1234+i for i in range(8)]
    full_prompt = list(follow)
    resume_pos, gen, st = model.get_start_pos(follow), model.generate(follow, chunk_size=args.chunk_size), time.perf_counter()
    resumed_token = next(gen)
    print(f"resume {len(follow)-1} tokens from {resume_pos} in {time.perf_counter()-st:.3f}s token {resumed_token}", flush=True)
    model._cached_tokens = [-1]
    st = time.perf_counter()
    full_token = next(model.generate(full_prompt, chunk_size=args.chunk_size))
    print(f"full {time.perf_counter()-st:.3f}s token {full_token} match {resumed_token == full_token}", flush=True)
    assert resumed_token == full_token
