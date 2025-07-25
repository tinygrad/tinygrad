from tinygrad.nn.state import safe_load
from tinygrad import Tensor, dtypes

def md(a:Tensor, b:Tensor):
  diff = (a - b).abs()
  max_diff = diff.max()
  mean_diff = diff.mean()
  assert a.dtype in (dtypes.float16, dtypes.float32)
  eps = 2e-5 if a.dtype is dtypes.float16 else 1e-8
  eps_str = "2e-5" if a.dtype is dtypes.float16 else "1e-8"
  ratio = (diff / (a.abs() + eps)).mean()
  print(f"diff.abs().mean(): {mean_diff.item()}")
  print(f"a.abs().mean(): {a.abs().mean().item()}")
  print(f"diff.abs().max(): {max_diff.item()}")
  print()
  return mean_diff.item(), ratio.item(), max_diff.item()

d = safe_load("/home/hooved/train-sd/training/stable_diffusion/checkpoints/mixed.safetensors")
for v in d.values(): v.to_("NV").realize()

measure_layernorm=True