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

  startup_st = st = time.perf_counter()
  model, _ = Transformer.from_gguf(args.model, args.max_context)
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)
  st = time.perf_counter()
  with Context(BEAM=0): model.warmup(args.chunk_size)
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)
  assert time.perf_counter()-startup_st < 60
  states = [getattr(block, name) for block in model.blk for name in ("cache_kv", "cache_kv_scale", "conv_state", "recurrent_state")
            if hasattr(block, name)]
  assert all(str(state.device).startswith("AMD") and state.uop.is_realized for state in states)
  assert all(block.cache_kv.shape[3] >= args.max_context for block in model.blk if hasattr(block, "cache_kv"))
  assert model.prefill_jit.cnt >= 2 and model.rollout_jit.cnt >= 2
  print(f"preallocated {sum(state.nbytes() for state in states)/2**30:.3f} GiB state on AMD", flush=True)

  prompt = [257] + [1000+i%1000 for i in range(args.prompt_tokens-1)]
  gen, st = model.generate(prompt, chunk_size=args.chunk_size), time.perf_counter()
  output = [next(gen)]
  pt = time.perf_counter()
  prefill = args.prompt_tokens/(pt-st)
  print(f"prefill {prefill:.3f} tok/s", flush=True)
  for _ in range(args.decode_tokens): output.append(next(gen))
  et = time.perf_counter()
  decode = args.decode_tokens/(et-pt)
  print(f"decode {decode:.3f} tok/s output {output}", flush=True)
  if args.prompt_tokens == 3000 and args.decode_tokens == 16:
    assert prefill > 750 and decode > 40
    assert output == [13, 271, 248068, 198, 8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983]

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
