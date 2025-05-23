import os
from extra.f16_decompress import u32_to_f16
from examples.stable_diffusion import StableDiffusion
from tinygrad.nn.state import get_state_dict, safe_save, safe_load_metadata, torch_load, load_state_dict
from tinygrad import Device, dtypes, Tensor, TinyJit
from tinygrad.helpers import fetch
from typing import NamedTuple, Any, List
import requests
import argparse
import numpy as np

def convert_f32_to_f16(input_file, output_file):
  with open(input_file, 'rb') as f:
    metadata_length_bytes = f.read(8)
    metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little', signed=False)
    metadata_json_bytes = f.read(metadata_length)
    float32_values = np.fromfile(f, dtype=np.float32)

  first_text_model_offset = 3772703308
  num_elements = int((first_text_model_offset)/4)
  front_float16_values = float32_values[:num_elements].astype(np.float16)
  rest_float32_values = float32_values[num_elements:]

  with open(output_file, 'wb') as f:
    f.write(metadata_length_bytes)
    f.write(metadata_json_bytes)
    front_float16_values.tofile(f)
    rest_float32_values.tofile(f)

def split_safetensor(fn):
  _, data_start, metadata = safe_load_metadata(fn)
  text_model_offset = 3772703308
  f16_num_bytes = int(text_model_offset / 2)
  decode_chunk_size = 8388608
  chunk_size = decode_chunk_size * 64
  with open(fn, 'rb') as r:
    for c in range(0, f16_num_bytes, chunk_size):
      with open(os.path.join(os.path.dirname(__file__), f'./net_part{int(c/chunk_size)}.safetensors'), "wb+") as w:
        if int(c/chunk_size) == 0:
          w.write(r.read(data_start))
        actual = min(chunk_size, f16_num_bytes - c)
        w.write(r.read(actual))
    with open(os.path.join(os.path.dirname(__file__), f'./net_textmodel.safetensors'), "wb+") as w:
      w.write(r.read())

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

  # load in weights
  load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  sub_steps = [
    Step(name = "textModel", input = [Tensor.randn(1, 77)], forward = model.cond_stage_model.transformer.text_model),
    Step(name = "diffusor", input = [Tensor.randn(1, 77, 768), Tensor.randn(1, 77, 768), Tensor.randn(1,4,64,64), Tensor.rand(1), Tensor.randn(1), Tensor.randn(1), Tensor.randn(1)], forward = model),
    Step(name = "decoder", input = [Tensor.randn(1,4,64,64)], forward = model.decode),
    Step(name = "f16tof32", input = [Tensor.randn(2097152, dtype=dtypes.uint32)], forward = u32_to_f16)
  ]

  prg = ""
  state_dict = get_state_dict(model)

  safe_save(state_dict, os.path.join(os.path.dirname(__file__), "net.safetensors"))
  convert_f32_to_f16(os.path.join(os.path.dirname(__file__), "./net.safetensors"), os.path.join(os.path.dirname(__file__), "./net_conv.safetensors"))
  offsets = split_safetensor(os.path.join(os.path.dirname(__file__), "./net_conv.safetensors"))
  os.remove(os.path.join(os.path.dirname(__file__), "net.safetensors"))
  os.remove(os.path.join(os.path.dirname(__file__), "net_conv.safetensors"))

  for step in sub_steps:
    print(f'Executing step={step.name}')
    prg, state = TinyJit(step.forward).export_webgpu(*step.input, tensor_names=state_dict)
    with open(os.path.join(os.path.dirname(__file__), f"{step.name}.js"), "w") as f: f.write(prg)
