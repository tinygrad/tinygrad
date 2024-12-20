# based on ./examples/webgpu/stable_diffusion/compile.py

import os, sys
from extra.export_model import compile_net, jit_model, dtype_to_js_type
from examples.llama3 import build_transformer
from tinygrad.nn.state import get_state_dict, safe_save, load_state_dict, safe_load
from tinygrad.tensor import Tensor
from tinygrad import Device, dtypes
from tinygrad.helpers import fetch
from typing import NamedTuple, Any, List

if __name__=="__main__":
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
  model_size="1B"
  Tensor.no_grad = True
  f32_fn = "llama3_1B_f32.safetensors"

  if not os.path.exists(f32_fn):
    # this is ugly, but wgpu adapter doesn't support f16 (they're working on it), throws exception on loading llama3 1B weights
    # the tinygrad llama code just converts the f16 to f32 anyway, we let that happen, then transfer the weights to WEBGPU device
    # TODO clean this up when wgpu supports f16, or maybe use dawn if it supports f16 (cc wpmed92)
    model = build_transformer(model_path, model_size=model_size)
    state_dict = get_state_dict(model)
    safe_save(state_dict, f32_fn)
    print(f"f32 weights saved to {f32_fn}, exiting to free idle GPU memory, restart program as-is to resume")
    # TODO force free all the currently used GPU memory after loading state_dict into the WEBGPU-initialized model
    #  this doesn't happen by default below, so we restart execution to clear ~3GB of GPU memory
    #  maybe extra.models.llama.Transformer.forward_jit is the culprit?
    sys.exit()

  Device.DEFAULT = "WEBGPU"
  model = build_transformer(model_path, model_size=model_size, load_weights=False)
  state_dict = safe_load(f32_fn)
  load_state_dict(model, state_dict, consume=True)