from tinygrad.nn.state import safe_load, get_parameters
from tinygrad import Tensor, dtypes
#from tinygrad.helpers import getenv

def md(a:Tensor, b:Tensor):
  diff = (a - b).abs()
  max_diff = diff.max()
  mean_diff = diff.mean()
  assert a.dtype in (dtypes.float16, dtypes.float32, dtypes.float64)
  print(f"diff.abs().mean(): {mean_diff.item()}")
  print(f"a.abs().mean(): {a.abs().mean().item()}")
  print(f"diff.abs().max(): {max_diff.item()}")
  print()
  return mean_diff.item(), a.abs().mean().item(), max_diff.item()

base = "/home/hooved/stable_diffusion/checkpoints"
unet = safe_load(f"{base}/train0_init_unet.safetensors")

data = safe_load(f"{base}/train_20_steps.safetensors")
for v in get_parameters(data):
  v.to_("CPU").realize()

with open(f"{base}/train_20_steps_prompts.txt") as f:
  prompts = f.read()
  prompts = prompts.split("\n")

pause = 1