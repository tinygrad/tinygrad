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

#clip = safe_load("/home/hooved/stable_diffusion/checkpoints/val2_clip.safetensors")
#fid = safe_load("/home/hooved/stable_diffusion/checkpoints/val2_fid_score.safetensors")
#images = safe_load("/home/hooved/stable_diffusion/checkpoints/val2_images.safetensors")
#inception = safe_load("/home/hooved/stable_diffusion/checkpoints/val2_inception.safetensors")
#for d in (clip, fid, images, inception):
t0 = safe_load(f"{base}/train0.safetensors")
t1 = safe_load(f"{base}/train1.safetensors")
for d in (t0, t1):
  for v in get_parameters(d):
    v.to_("CPU").realize()

prompts = []
fns = (f"{base}/train0_prompt0.txt", f"{base}/train0_prompt1.txt")
for fn in fns:
  with open(fn) as f:
    p = f.read()
    prompts.append(p)

pause = 1
