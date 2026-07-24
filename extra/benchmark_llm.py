#!/usr/bin/env python3
"""Fail-closed Kimi MLA decode benchmark.

Synthetic mode measures explicitly initialized MLA cache prefixes; it is never a
correctness result. Genuine-prefill mode populates caches through the public
Transformer.generate API and checks timed tokens against a fresh-cache oracle.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, sys, tempfile, time
from types import SimpleNamespace
from typing import Any, Callable


def percentile(values:list[float], fraction:float) -> float:
  if not values: raise ValueError("cannot compute a percentile of no samples")
  if not 0.0 <= fraction <= 1.0: raise ValueError("percentile must be in [0, 1]")
  ordered = sorted(values)
  point = (len(ordered)-1) * fraction
  lower, upper = int(point), min(int(point)+1, len(ordered)-1)
  return ordered[lower] + (ordered[upper]-ordered[lower]) * (point-lower)


def validate_range(position:int, decode_tokens:int, max_context:int) -> None:
  if max_context <= 0: raise ValueError("--max-context must be positive")
  if position < 0: raise ValueError("position/prompt length must not be negative")
  if decode_tokens <= 0: raise ValueError("--decode-tokens must be positive")
  if position > max_context or decode_tokens > max_context-position:
    raise ValueError("position + decode tokens must be <= max context")


def synthetic_prompt(length:int, vocab_size:int, seed:int) -> list[int]:
  if length <= 0: raise ValueError("--prompt-tokens values must be positive")
  if vocab_size <= 0: raise ValueError("model has an invalid vocabulary size")
  # No tokenizer or chat template is involved. This sequence is stable across Python processes.
  return [int((seed + i*7919) % vocab_size) for i in range(length)]


def summarize(samples_s:list[float], decode_tokens:int) -> dict[str, Any]:
  if len(samples_s) != decode_tokens: raise ValueError("one timing sample is required per decoded token")
  if any(x <= 0 for x in samples_s): raise ValueError("timing samples must be positive")
  total = sum(samples_s)
  return {
    "raw_samples_ms": [x*1e3 for x in samples_s],
    "p50_ms": statistics.median(samples_s)*1e3,
    "p95_ms": percentile(samples_s, 0.95)*1e3,
    "tokens_per_second": decode_tokens/total,
    "timed_decode_seconds": total,
  }


def _single_device_from_tensors(tensors:list[Any]) -> str:
  if not tensors: raise RuntimeError("could not identify a participating device")
  if any(isinstance(tensor.device, tuple) for tensor in tensors):
    raise RuntimeError("unsupported current-master MULTI model: tuple-device tensors cannot be benchmarked by this harness")
  devices = {str(tensor.device) for tensor in tensors}
  if len(devices) != 1:
    raise RuntimeError(f"unsupported current-master MULTI model: expected one participating device, got {sorted(devices)}")
  return next(iter(devices))


def participating_devices(model:Any) -> list[str]:
  from tinygrad import nn
  tensors = list(nn.state.get_parameters(model))
  for block in model.blk:
    for name in ("cache_k", "cache_kv", "conv_state", "recurrent_state"):
      if hasattr(block, name): tensors.append(getattr(block, name))
  return [_single_device_from_tensors(tensors)]


def synchronize(devices:list[str]) -> None:
  from tinygrad import Device
  for device in devices: Device[device].synchronize()


def device_memory(devices:list[str]) -> dict[str, int]:
  from tinygrad.helpers import GlobalCounters
  return {device: int(GlobalCounters.mem_used_per_device[device]) for device in devices}


def validate_kimi_mla(model:Any, kv:dict[str, Any], require_metadata:bool=True) -> None:
  from tinygrad.llm.model import MLATransformerBlock
  if require_metadata and kv.get("tokenizer.ggml.pre") != "kimi-k2":
    raise RuntimeError("unsupported model: Kimi tokenizer metadata is required")
  if not model.blk: raise RuntimeError("unsupported model: no transformer blocks")
  if getattr(model, "has_recurrent_block", True): raise RuntimeError("unsupported synthetic layout: recurrent blocks are present")
  if any(type(block) is not MLATransformerBlock for block in model.blk):
    raise RuntimeError("unsupported synthetic layout: every block must be homogeneous MLA")
  for block in model.blk:
    config = block.config
    if config.kv_lora_rank <= 0 or config.rope_dim <= 0 or config.ssm is not None:
      raise RuntimeError("unsupported synthetic layout: invalid MLA configuration")


def expected_cache_value(seed:int, layer:int) -> float:
  return ((seed + layer*104729) % 1021 - 510) / 2048.0


def initialize_mla_cache(model:Any, through_position:int, seed:int, chunk_size:int) -> int:
  """Allocate MLA state and deterministically realize cache_k[:, :, :through_position, :]."""
  from tinygrad import Tensor
  if through_position < 0 or through_position > model.max_context: raise ValueError("invalid cache initialization range")
  if chunk_size <= 0: raise ValueError("--chunk-size must be positive")
  parameter_device = participating_devices(model)[0]
  dummy = Tensor.zeros(1, 1, model.blk[0].config.dim, device=parameter_device).realize()
  total_bytes = 0
  for layer, block in enumerate(model.blk):
    block._init_state(dummy)
    if not hasattr(block, "cache_k") or hasattr(block, "cache_kv") or hasattr(block, "recurrent_state"):
      raise RuntimeError("unsupported synthetic state: expected cache_k only")
    cache = block.cache_k
    if len(cache.shape) != 4 or cache.shape[:2] != (1, 1) or cache.shape[2] != model.max_context:
      raise RuntimeError(f"unsupported MLA cache shape: {cache.shape}")
    expected_width = block.config.kv_lora_rank + block.config.rope_dim
    if cache.shape[3] != expected_width: raise RuntimeError(f"unsupported MLA cache width: {cache.shape[3]}")
    total_bytes += cache.nbytes()
    # A small bounded staging tensor avoids constructing a second context-sized cache.
    value = expected_cache_value(seed, layer)
    for start in range(0, through_position, chunk_size):
      end = min(start+chunk_size, through_position)
      fill = Tensor.full((1, 1, end-start, expected_width), value, dtype=cache.dtype, device=cache.device)
      cache[:, :, start:end, :].assign(fill).realize()
  return total_bytes


def restore_synthetic_cache(model:Any, through_position:int, seed:int, chunk_size:int) -> int:
  """Restore deterministic synthetic state after representative calls and before timing."""
  return initialize_mla_cache(model, through_position, seed, chunk_size)


def timed_call(call:Callable[[], Any], devices:list[str]) -> tuple[Any, float]:
  synchronize(devices)
  begin = time.perf_counter()
  output = call()
  output.realize()
  synchronize(devices)
  elapsed = time.perf_counter()-begin
  return output, elapsed


def _direct_decode(model:Any, token:Any, position:int, temperature:Any, start_pos_var:Any) -> Any:
  return model(token, start_pos_var.bind(position), temperature).realize()


def require_rollout_jit(model:Any) -> None:
  rollout_jit = getattr(model, "rollout_jit", None)
  if rollout_jit is None or rollout_jit.captured is None:
    raise RuntimeError("rollout TinyJit was not captured after two representative calls; set JIT=1 (eager timing is rejected)")


def benchmark_synthetic(model:Any, position:int, decode_tokens:int, chunk_size:int, seed:int,
                        metadata:dict[str, Any]) -> dict[str, Any]:
  from tinygrad import Tensor, UOp
  validate_range(position, decode_tokens, model.max_context)
  cache_bytes = initialize_mla_cache(model, position+decode_tokens, seed, chunk_size)
  devices = participating_devices(model)
  token = Tensor([[seed % model.blk[0].config.vocab_size]], dtype="int32", device=devices[0]).realize()
  temperature = Tensor([0.0], device=devices[0]).realize()
  # Match Transformer.generate's variable identity so an already-captured rollout graph can replay.
  start_pos = UOp.variable("start_pos", 0, model.max_context-1)

  # TinyJit call one is eager and call two captures. Both use representative tail shapes.
  warm_positions = (position, min(position+1, model.max_context-1))
  for warm_position in warm_positions:
    token = _direct_decode(model, token, warm_position, temperature, start_pos)
    token.item()  # explicitly outside every measured interval
  require_rollout_jit(model)
  # Warm calls wrote cache state. Restore every prefix element that measured calls can consume.
  restore_synthetic_cache(model, position+decode_tokens, seed, chunk_size)

  samples_s, sampled_tokens = [], []
  token = Tensor([[seed % model.blk[0].config.vocab_size]], dtype="int32", device=devices[0]).realize()
  for current_position in range(position, position+decode_tokens):
    token, elapsed = timed_call(lambda p=current_position, t=token: _direct_decode(model, t, p, temperature, start_pos), devices)
    samples_s.append(elapsed)
    sampled_tokens.append(int(token.item()))  # scalar readback is outside timing
  return {
    **metadata,
    "mode": "synthetic-position", "position": position, "prompt_tokens": None,
    "decode_tokens": decode_tokens, "cache_strategy": "synthetic_initialized_kv",
    "correctness_acceptance": False, "acceptance_status": "not_applicable_synthetic",
    "seed": seed, "devices": devices, "tensor_parallel_degree": 1, "multi_device_support": "unsupported_current_master",
    "execution_mode": "tinyjit_replay", "rollout_jit_capture_verified": True, "post_warmup_cache_restored": True,
    "device_memory_bytes": device_memory(devices), "allocated_cache_bytes": cache_bytes,
    "warmup_policy": "two_untimed_representative_rollout_calls_then_deterministic_cache_restore",
    "timing_boundaries": {
      "included": "model rollout realize between participating-device synchronizations",
      "excluded": ["model load", "model identity hashing", "cache allocation", "cache initialization", "JIT preparation", "state restoration", ".item()"],
    },
    "sampled_token_ids": sampled_tokens, **summarize(samples_s, decode_tokens),
  }


def _fresh_public_generation(model:Any, prompt:list[int], count:int, chunk_size:int, seed:int) -> list[int]:
  from tinygrad import Tensor
  model._cached_tokens = []
  Tensor.manual_seed(seed)
  generator = model.generate(prompt.copy(), chunk_size=chunk_size, temperature=0.0)
  return [next(generator) for _ in range(count)]


def _restore_genuine_prefix(model:Any, prompt:list[int], chunk_size:int, seed:int) -> None:
  """Populate the cache preceding the final prompt token through public generate."""
  from tinygrad import Tensor
  model._cached_tokens = []
  Tensor.manual_seed(seed)
  if len(prompt) > 1:
    # The yielded token is ignored: generate has genuinely populated prompt[:-1].
    next(model.generate(prompt[:-1].copy(), chunk_size=chunk_size, temperature=0.0))


def require_token_oracle_match(actual:list[int], expected:list[int]) -> None:
  if actual != expected: raise RuntimeError(f"genuine-prefill acceptance failed: timed tokens {actual} != public generate oracle {expected}")


def benchmark_genuine(model:Any, prompt_length:int, decode_tokens:int, chunk_size:int, seed:int,
                      metadata:dict[str, Any]) -> dict[str, Any]:
  from tinygrad import Tensor, UOp
  validate_range(prompt_length, decode_tokens, model.max_context)
  devices = participating_devices(model)
  prompt = synthetic_prompt(prompt_length, model.blk[0].config.vocab_size, seed)

  # Public generate is the acceptance oracle and genuinely writes every prompt KV entry.
  expected = _fresh_public_generation(model, prompt, decode_tokens, chunk_size, seed)

  # Explicitly prepare two representative rollout calls, regardless of prior JIT state.
  _restore_genuine_prefix(model, prompt, chunk_size, seed)
  temperature = Tensor([0.0], device=devices[0]).realize()
  start_pos = UOp.variable("start_pos", 0, model.max_context-1)
  token = Tensor([[prompt[-1]]], dtype="int32", device=devices[0]).realize()
  for warm_position in (prompt_length-1, min(prompt_length, model.max_context-1)):
    token = _direct_decode(model, token, warm_position, temperature, start_pos)
    token.item()
  require_rollout_jit(model)

  # Deterministically restore genuine state through the public API before replay timing. The final
  # prompt token is the first timed input, so D calls produce exactly the D public-oracle tokens,
  # including when prompt_length + D == max_context.
  _restore_genuine_prefix(model, prompt, chunk_size, seed)
  token = Tensor([[prompt[-1]]], dtype="int32", device=devices[0]).realize()
  samples_s, actual = [], []
  for current_position in range(prompt_length-1, prompt_length+decode_tokens-1):
    token, elapsed = timed_call(lambda p=current_position, t=token: _direct_decode(model, t, p, temperature, start_pos), devices)
    samples_s.append(elapsed)
    actual.append(int(token.item()))
  require_token_oracle_match(actual, expected)
  return {
    **metadata,
    "mode": "genuine-prefill", "position": prompt_length, "prompt_tokens": prompt_length,
    "decode_tokens": decode_tokens, "cache_strategy": "genuine_prefill_via_Transformer.generate",
    "correctness_acceptance": True, "acceptance_status": "passed_public_generate_oracle",
    "seed": seed, "devices": devices, "tensor_parallel_degree": 1, "multi_device_support": "unsupported_current_master",
    "execution_mode": "tinyjit_replay", "rollout_jit_capture_verified": True,
    "device_memory_bytes": device_memory(devices), "allocated_cache_bytes": sum(block.cache_k.nbytes() for block in model.blk),
    "warmup_policy": "two_untimed_representative_rollout_calls_then_public_generate_state_restore",
    "timing_boundaries": {
      "included": "model rollout realize from final prompt token through decoded tokens, between participating-device synchronizations",
      "excluded": ["model load", "model identity hashing", "public generate oracle", "genuine prefill", "JIT preparation", "state restoration", ".item()"],
    },
    "sampled_token_ids": actual, "oracle_token_ids": expected, **summarize(samples_s, decode_tokens),
  }


def gguf_part_metadata(resolved:Any, kv:dict[str, Any]) -> list[dict[str, Any]]:
  from tinygrad.llm.gguf import _gguf_split_paths
  first_path = pathlib.Path(str(resolved)).expanduser().resolve()
  parts = []
  for path in _gguf_split_paths(first_path, kv):
    canonical = path.resolve(strict=True)
    stat = canonical.stat()
    parts.append({"path": str(canonical), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
  if len(parts) != int(kv.get("split.count", 1)):
    raise RuntimeError("GGUF split enumeration did not match split.count")
  return parts


def model_metadata(requested:str, resolved:Any, kv:dict[str, Any], model:Any) -> dict[str, Any]:
  from tinygrad import nn
  parts = gguf_part_metadata(resolved, kv)
  identity = {
    "resolved_path": parts[0]["path"], "parts": parts, "split_count": len(parts),
    "size_bytes": sum(part["size_bytes"] for part in parts),
    "name": kv.get("general.name") or kv.get("general.basename"),
    "architecture": kv.get("general.architecture"), "quantization_file_type": kv.get("general.file_type"),
  }
  identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  return {
    "schema_version": 1, "model": {"requested": requested, **identity, "identity_sha256": identity_hash,
      "hash_kind": "sha256_of_canonical_identity_metadata_not_file_content",
      "parameter_dtypes": sorted({str(parameter.dtype) for parameter in nn.state.get_parameters(model)}),
      "weight_realization_policy": "lazy_from_gguf; first-use realization and TinyJit capture complete before timing"},
    "effective_max_context": model.max_context,
  }


def load_model(requested:str, max_context:int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
  from tinygrad.helpers import fetch
  from tinygrad.llm.cli import models
  from tinygrad.llm.model import Transformer
  resolved = fetch(models.get(requested, requested))
  model, kv = Transformer.from_gguf(resolved, max_context)
  validate_kimi_mla(model, kv)
  participating_devices(model)  # Reject current-master MULTI before any benchmark setup or timing.
  return model, kv, model_metadata(requested, resolved, kv, model)


def self_test() -> dict[str, Any]:
  checks: list[str] = []
  validate_range(7, 1, 8)
  for bad in ((-1, 1, 8), (7, 2, 8), (0, 0, 8), (0, 1, 0)):
    try: validate_range(*bad)
    except ValueError: pass
    else: raise AssertionError(f"range validation accepted {bad}")
  checks.append("inclusive_bounds_and_negative_overflow_rejection")
  summary = summarize([0.001, 0.002, 0.004, 0.008], 4)
  if summary["p50_ms"] != 3.0 or abs(summary["p95_ms"]-7.4) > 1e-9: raise AssertionError(summary)
  try: summarize([0.001], 2)
  except ValueError: pass
  else: raise AssertionError("mismatched timing sample count was accepted")
  checks.append("statistics_and_sample_count_validation")

  from tinygrad import Tensor
  from tinygrad.llm.model import Transformer, TransformerConfig
  config = TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
    vocab_size=32, head_dim=8, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=8, kv_lora_rank=4)
  model = Transformer(config)
  validate_kimi_mla(model, {}, require_metadata=False)
  initialize_mla_cache(model, 4, 123, 2)
  cache = model.blk[0].cache_k
  expected = expected_cache_value(123, 0)
  if not bool((cache[:, :, :4, :] == expected).all().item()): raise AssertionError("cache does not equal its closed-form value")
  cache[:, :, :1, :].assign(Tensor.full((1, 1, 1, cache.shape[3]), 0.75, dtype=cache.dtype, device=cache.device)).realize()
  if bool((cache[:, :, :4, :] == expected).all().item()): raise AssertionError("cache mutation did not take effect")
  restore_synthetic_cache(model, 4, 123, 3)
  if not bool((cache[:, :, :4, :] == expected).all().item()): raise AssertionError("synthetic cache restoration failed")
  checks.append("closed_form_cache_initialization_and_restoration")
  require_token_oracle_match([1, 2], [1, 2])
  try: require_token_oracle_match([1, 3], [1, 2])
  except RuntimeError: pass
  else: raise AssertionError("intentional token-oracle mismatch was accepted")
  checks.append("token_oracle_match_and_mismatch")
  try: _single_device_from_tensors([SimpleNamespace(device=("CPU", "CPU:1"))])
  except RuntimeError as error:
    if "MULTI" not in str(error): raise
  else: raise AssertionError("tuple-device model was not rejected")
  try: _single_device_from_tensors([SimpleNamespace(device="CPU"), SimpleNamespace(device="NV")])
  except RuntimeError as error:
    if "MULTI" not in str(error): raise
  else: raise AssertionError("multiple-device model was not rejected")
  checks.append("current_master_multi_fail_closed")
  with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)/"tiny-00001-of-00003.gguf"
    split_paths = [pathlib.Path(directory)/f"tiny-{part:05d}-of-00003.gguf" for part in range(1, 4)]
    for part, path in enumerate(split_paths, 1): path.write_bytes(bytes([part])*part)
    parts = gguf_part_metadata(root, {"split.count": 3, "split.no": 0})
    if len(parts) != 3 or sum(part["size_bytes"] for part in parts) != 6: raise AssertionError("split identity is incomplete")
    split_paths[-1].unlink()
    try: gguf_part_metadata(root, {"split.count": 3, "split.no": 0})
    except FileNotFoundError: pass
    else: raise AssertionError("missing GGUF split was not rejected")
  checks.append("all_split_gguf_parts_identified_and_required")
  metadata = {"schema_version": 1, "model": {"name": "self-test"}, "effective_max_context": 8}
  synthetic_model = Transformer(config)
  synthetic_result = benchmark_synthetic(synthetic_model, 7, 1, 2, 123, metadata)
  if synthetic_result["correctness_acceptance"] or synthetic_result["cache_strategy"] != "synthetic_initialized_kv":
    raise AssertionError("synthetic mode mislabeled its result")
  if synthetic_result["acceptance_status"] != "not_applicable_synthetic": raise AssertionError("synthetic acceptance status is invalid")
  if not synthetic_result["post_warmup_cache_restored"]: raise AssertionError("synthetic restoration was not reported")
  if synthetic_result["tensor_parallel_degree"] != 1 or not synthetic_result["rollout_jit_capture_verified"]:
    raise AssertionError("synthetic execution metadata is not honest")
  checks.append("synthetic_benchmark_jit_restoration_status_and_inclusive_boundary")
  genuine_model = Transformer(config)
  genuine_result = benchmark_genuine(genuine_model, 7, 1, 2, 123, metadata)
  if genuine_result["acceptance_status"] != "passed_public_generate_oracle": raise AssertionError("genuine oracle failed")
  if genuine_result["tensor_parallel_degree"] != 1 or not genuine_result["rollout_jit_capture_verified"]:
    raise AssertionError("genuine execution metadata is not honest")
  checks.append("genuine_benchmark_jit_public_oracle_and_inclusive_boundary")
  return {"self_test": "passed", "checks": checks, "downloads": False}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Benchmark Kimi MLA decode with synthetic initialized KV or genuine public-API prefill")
  parser.add_argument("--mode", choices=("synthetic-position", "genuine-prefill"), default="synthetic-position")
  parser.add_argument("--model", default="kimi-k2.6.gguf", help="local Kimi GGUF path (or a tinygrad model alias)")
  parser.add_argument("--max-context", type=int, default=262144)
  parser.add_argument("--positions", type=int, nargs="+", default=[0, 8192, 65536, 131072, 262016])
  parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[128])
  parser.add_argument("--decode-tokens", type=int, default=32)
  parser.add_argument("--chunk-size", type=int, default=256)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--json", action="store_true")
  parser.add_argument("--self-test", action="store_true", help="run dependency-free validation without model downloads")
  return parser


def emit(payload:dict[str, Any], as_json:bool) -> None:
  if as_json:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return
  if "self_test" in payload:
    print(f"self-test: {payload['self_test']} ({', '.join(payload['checks'])})")
    return
  model = payload["model"]
  print(f"model={model['name'] or model['requested']} mode={payload['mode']} context={payload['effective_max_context']}")
  for result in payload["results"]:
    print(f"position={result['position']} cache={result['cache_strategy']} acceptance={result['acceptance_status']} "
          f"p50={result['p50_ms']:.3f} ms p95={result['p95_ms']:.3f} ms tok/s={result['tokens_per_second']:.2f}")


def main(argv:list[str]|None=None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  try:
    if args.self_test:
      emit(self_test(), args.json)
      return 0
    if args.chunk_size <= 0: raise ValueError("--chunk-size must be positive")
    if args.seed < 0: raise ValueError("--seed must not be negative")
    selected = args.positions if args.mode == "synthetic-position" else args.prompt_tokens
    if not selected: raise ValueError("at least one position/prompt length is required")
    for position in selected: validate_range(position, args.decode_tokens, args.max_context)
    model, _kv, metadata = load_model(args.model, args.max_context)
    results = [benchmark_synthetic(model, position, args.decode_tokens, args.chunk_size, args.seed, metadata) if args.mode == "synthetic-position"
      else benchmark_genuine(model, position, args.decode_tokens, args.chunk_size, args.seed, metadata) for position in selected]
    emit({**metadata, "mode": args.mode, "chunk_size": args.chunk_size, "results": results}, args.json)
    return 0
  except (AssertionError, ValueError, RuntimeError, OSError, StopIteration) as error:
    print(f"benchmark_llm: error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__": raise SystemExit(main())
