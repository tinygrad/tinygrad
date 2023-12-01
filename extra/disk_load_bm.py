from tinygrad.helpers import fetch
from tinygrad.device import Device
from tinygrad.nn.state import torch_load, load_state_dict
from examples.stable_diffusion import StableDiffusion
import time

# Run sudo purge between tests for the most accurate results

# prefetch dataset incase it is not downloaded
fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt')

model = StableDiffusion()

st = time.perf_counter()
load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)
Device[Device.DEFAULT].synchronize()
print(f"TOTAL LOAD TIME: {time.perf_counter() - st:.2f} seconds")