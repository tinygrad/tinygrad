# TODO: merge with examples/tinychat/compile.py

import os
from extra.export_model import export_model
from examples.llama3 import build_transformer
from tinygrad import Device, Variable, Tensor, dtypes
from tinygrad.helpers import fetch, Context
from typing import List, Any, NamedTuple

if __name__=="__main__":
  Device.DEFAULT = "CPU"
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf", "Llama-3.2-1B-Instruct-f16.gguf", subdir="llama3-1b-instruct")
  Tensor.no_grad = True
  max_context=1024
  model = build_transformer(model_path, model_size="1B", quantize="int8", scale_dtype=dtypes.float32, device=Device.DEFAULT, max_context=max_context)
  model.output.weight = model.tok_embeddings.weight
  model.output.scale = model.tok_embeddings.scale

  tok = 128000
  TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P = 0.95, 0, 0.0, 0.0, 0.0
  model_input = [Tensor([[tok]]), 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P]
  out = model.forward(*model_input)

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None
    model: Any = {}

  start_pos = Variable("start_pos", 0, max_context).bind(0)
  sub_steps = [
    Step(name = "transformer", input = [Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P], forward = model.forward),
  ]

  for step in sub_steps:
    print(f'Executing step={step.name}')
    with Context(BEAM=3):
      cprog, js_wrapper = export_model(model, "wasm", *step.input, model_name=step.name)
      if step.name == "transformer":
        # ensure consistency with exported weights
        js_wrapper = js_wrapper.replace("output.weight", "tok_embeddings.weight").replace("output.scale", "tok_embeddings.scale")

    with open(os.path.join(os.path.dirname(__file__), f"{step.name}.c"), "w") as text_file:
      text_file.write(cprog)

    with open(os.path.join(os.path.dirname(__file__), "net_clang.js"), "w") as text_file:
      text_file.write(js_wrapper)