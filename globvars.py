from tinygrad.nn.state import safe_load, get_parameters
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv

def md(a:Tensor, b:Tensor):
  diff = (a - b).abs()
  max_diff = diff.max()
  mean_diff = diff.mean()
  assert a.dtype in (dtypes.float16, dtypes.float32)
  print(f"diff.abs().mean(): {mean_diff.item()}")
  print(f"a.abs().mean(): {a.abs().mean().item()}")
  print(f"diff.abs().max(): {max_diff.item()}")
  print()
  return mean_diff.item(), a.abs().mean().item(), max_diff.item()

GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
for x in GPUS: Device[x]

data = safe_load("/home/hooved/stable_diffusion/checkpoints/val0.safetensors")
for d in (data,):
  for v in get_parameters(d):
    v.to_("CPU").realize()
#data['t_in'] = data['t_in'].cast(dtypes.int).realize()

with open("/home/hooved/stable_diffusion/checkpoints/val0prompt.txt", "r") as f:
  prompt = f.read()
