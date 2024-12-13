import os
from extra.export_model import export_model
from extra.f16_decompress import u32_to_f16
from examples.stable_diffusion import StableDiffusion
from tinygrad.nn.state import get_state_dict, safe_save, safe_load_metadata, torch_load, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad import Device, dtypes
from tinygrad.helpers import fetch
from typing import NamedTuple, Any, List
import requests
import argparse
import numpy as np
from pathlib import Path

def convert_f32_to_f16(input_file, output_file):
  with open(input_file, 'rb') as f:
    metadata_length_bytes = f.read(8)
    metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little', signed=False)
    metadata_json_bytes = f.read(metadata_length)
    values = np.fromfile(f, dtype=np.float32)

  f16_values = values.astype(np.float16)

  with open(output_file, 'wb') as f:
    f.write(metadata_length_bytes)
    f.write(metadata_json_bytes)
    f16_values.tofile(f)

def fetch_dep(file, url):
  with open(file, "w", encoding="utf-8") as f:
    f.write(requests.get(url).text.replace("https://huggingface.co/wpmed/tinygrad-sd-f16/raw/main/bpe_simple_vocab_16e6.mjs", "./bpe_simple_vocab_16e6.mjs"))

if __name__ == "__main__":
  fetch_dep(os.path.join(os.path.dirname(__file__), "clip_tokenizer.js"), "https://huggingface.co/wpmed/tinygrad-sd-f16/raw/main/clip_tokenizer.js")
  fetch_dep(os.path.join(os.path.dirname(__file__), "bpe_simple_vocab_16e6.mjs"), "https://huggingface.co/wpmed/tinygrad-sd-f16/raw/main/bpe_simple_vocab_16e6.mjs")
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--remoteweights', action='store_true', help="Use safetensors from Huggingface, or from local")
  args = parser.parse_args()
  Device.DEFAULT = "WEBGPU"

  Tensor.no_grad = True
  model = StableDiffusion()
  load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)

  class ModelPart(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  model_parts = [
    ModelPart(name = "textModel", input = [Tensor.randn(1, 77)], forward = model.cond_stage_model.transformer.text_model),
    ModelPart(name = "diffusor", input = [
      Tensor.randn(1, 77, 768), Tensor.randn(1, 77, 768), Tensor.randn(1,4,64,64), 
      Tensor.rand(1), Tensor.randn(1), Tensor.randn(1), Tensor.randn(1)
    ], forward = model),
    ModelPart(name = "decoder", input = [Tensor.randn(1,4,64,64)], forward = model.decode),
    ModelPart(name = "f16tof32", input = [Tensor.randn(2097120, dtype=dtypes.uint32)], forward = u32_to_f16)
  ]

  for model in model_parts:
    prg, inp_sizes, out_sizes, state = export_model(model, Device.DEFAULT.lower(), *model.input)
    dirname = Path(__file__).parent
    weight_loc =  (dirname / f"net_{model.name}.safetensors").as_posix()
    safe_save(state, weight_loc)
    if model.name == "diffusor":
      convert_f32_to_f16(weight_loc, (dirname / f"net_diffuso_f16.safetensors").as_posix())

    with open(dirname / f"net_{model.name}.js", "w") as text_file:
        text_file.write(prg)
