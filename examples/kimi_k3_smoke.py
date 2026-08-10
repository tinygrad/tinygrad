#!/usr/bin/env python3
"""Run a reduced, architecture-complete K3 prefill/decode on tensor-parallel devices."""
import argparse, time
from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.llm.kimi_k3 import _shard_kimi_k3, kimi_k3_smoke_config
from tinygrad.llm.model import Transformer

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--devices", type=int, default=8)
  args = parser.parse_args()
  if args.devices not in (1, 2, 4, 8): raise ValueError("the K3 admission smoke test supports 1, 2, 4, or 8 devices")
  devices = tuple(f"AMD:{i}" for i in range(args.devices))
  model = Transformer(kimi_k3_smoke_config())
  for name,value in nn.state.get_state_dict(model).items():
    fill = 127 if name.endswith("weight_scale") else 0
    dtype = value.dtype if value.dtype is dtypes.uint8 else dtypes.bfloat16
    value.replace(Tensor.full(value.shape, fill, dtype=dtype, device="CPU"))
  _shard_kimi_k3(model, devices)
  temperature = Tensor([0.0], device=devices)
  for label,tokens,start in (("prefill", [[1, 2]], 0), ("decode", [[3]], 2), ("decode replay", [[4]], 3)):
    begin = time.perf_counter()
    out = model(Tensor(tokens, dtype=dtypes.int32, device=devices), start, temperature).realize()
    for device in devices: Device[device].synchronize()
    print(f"{label}: shape={out.shape}, {time.perf_counter()-begin:.3f}s")

if __name__ == "__main__": main()
