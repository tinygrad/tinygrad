#!/usr/bin/env python3
"""Fast exact-shape K3 KDA/layer benchmark using bounded fake weights instead of the 1.56 TB checkpoint."""
from __future__ import annotations
import argparse, statistics, time
from dataclasses import replace

from tinygrad import Device, Tensor, TinyJit, dtypes, nn
from tinygrad.helpers import profile_marker
from tinygrad.llm.kimi_k3 import kimi_k3_config
from tinygrad.llm.model import GatedDeltaNetBlock

def tp_axis(name:str) -> int|None:
  if "ffn_gate_exps.weight" in name or "ffn_up_exps.weight" in name: return 1
  if "ffn_gate_exps.weight_scale" in name or "ffn_up_exps.weight_scale" in name: return 1
  if "ffn_down_exps.weight" in name or "ffn_down_exps.weight_scale" in name: return 2
  if name.endswith(("ffn_gate_shexp.weight", "ffn_up_shexp.weight")): return 0
  if name.endswith(("ffn_down_shexp.weight", "ffn_routed_down.weight", "ffn_routed_up.weight", "ssm_out.weight")): return 1
  if name.endswith(("attn_q.weight", "attn_k.weight", "attn_v.weight", "ssm_g_full.weight", "ssm_f_b.weight", "ssm_beta.weight")): return 0
  if name.endswith(("ssm_q_conv1d.weight", "ssm_k_conv1d.weight", "ssm_v_conv1d.weight", "ssm_dt.bias")): return 0
  return None

def fake_value(name:str) -> tuple[int|float, object]:
  if name.endswith("weight_scale"): return 120, dtypes.uint8
  if name.endswith("_exps.weight"): return 0x11, dtypes.uint8
  if name.endswith("ssm_a"): return -0.1, dtypes.float32
  if name.endswith("ssm_dt.bias"): return 0.1, dtypes.float32
  if "conv1d.weight" in name: return 0.1, dtypes.float32
  if name.endswith("exp_probs_b.bias"): return 0.0, dtypes.float32
  if name.endswith("norm.weight"): return 1.0, dtypes.bfloat16
  return 0.001, dtypes.bfloat16

def fake_tp_tensor(shape:tuple[int, ...], value:int|float, dtype, devices:tuple[str, ...], axis:int|None) -> Tensor:
  if axis is not None and shape[axis] % len(devices): raise ValueError(f"shape {shape} is not TP{len(devices)} divisible on axis {axis}")
  source = Tensor.full(shape, value, dtype=dtype, device=devices[0]).clone().realize()
  return source.shard(devices, axis=axis).realize()

def sync(devices:tuple[str, ...]) -> None:
  for device in devices: Device[device].synchronize()

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--devices", type=int, default=8)
  parser.add_argument("--mode", choices=("attention", "block"), default="attention")
  parser.add_argument("--iterations", type=int, default=20)
  args = parser.parse_args()
  devices = tuple(f"AMD:{i}" for i in range(args.devices))
  # One exact-width KDA layer, but only 16 fake routed experts. This retains top-k 16 and every
  # official per-GPU matrix/state shape while keeping fake expert storage below 300 MB per layer.
  config = replace(kimi_k3_config(4), num_blocks=1, num_experts=16, num_experts_per_tok=16, ssm_layers=(True,),
                   attn_res_block_size=0)
  block = GatedDeltaNetBlock(config, config.ssm)
  begin = time.perf_counter()
  for name,tensor in nn.state.get_state_dict(block).items():
    if args.mode == "attention" and name.startswith(("ffn_", "exp_probs_")): continue
    value, dtype = fake_value(name)
    tensor.replace(fake_tp_tensor(tuple(int(x) for x in tensor.shape), value, dtype, devices, tp_axis(name)))
  sync(devices)
  print(f"fake weights: {time.perf_counter()-begin:.3f}s", flush=True)
  x_source = (((Tensor.arange(config.dim, dtype=dtypes.float32).reshape(1, 1, config.dim) % 31) / 31) \
    .cast(dtypes.bfloat16).to(devices[0])).clone().realize()
  x = x_source.shard(devices, axis=None).realize()
  block._init_state(x)
  # Use direct buffer-backed state shards. The production path reaches this form after prefill;
  # the fake harness begins immediately at decode and must not feed lazy clone graphs to TinyJit.
  for state,axis in ((block.conv_state_q, 2), (block.conv_state_k, 2), (block.conv_state_v, 2), (block.recurrent_state, 1)):
    state.replace(Tensor.zeros(*state.shape, dtype=state.dtype, device=devices[0]).shard(devices, axis=axis).realize())

  @TinyJit
  def run(inp:Tensor) -> Tensor:
    if args.mode == "attention": return block._attention(block.attn_norm(inp), 0).realize()
    return block(inp, 0).realize()

  # uncaptured, capture, then replay only
  run(x); sync(devices)
  run(x); sync(devices)
  samples:list[float] = []
  profile_marker(f"fake K3 {args.mode} start")
  for _ in range(args.iterations):
    begin = time.perf_counter(); out = run(x); sync(devices); samples.append((time.perf_counter()-begin)*1e3)
  profile_marker(f"fake K3 {args.mode} end")
  print(f"{args.mode}: median={statistics.median(samples):.3f} ms/layer, min={min(samples):.3f} ms/layer, "
        f"projected_93_layer_rate={1000/(statistics.median(samples)*93):.3f} tok/s, finite={out.float().isfinite().all().item()}")

if __name__ == "__main__": main()
