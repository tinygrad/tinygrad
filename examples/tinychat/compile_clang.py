# TODO: merge with examples/tinychat/compile.py

import os
from extra.export_model import compile_net, jit_model, export_model
from examples.llama3 import build_transformer
from tinygrad import Device, Variable, Tensor, dtypes
from tinygrad.helpers import fetch
from typing import List, Tuple, Any, NamedTuple
from tinygrad.nn.state import get_state_dict

if __name__=="__main__":
  Device.DEFAULT = "CLANG"
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
  model_size="1B"
  Tensor.no_grad = True
  f32_fn = os.path.join(os.path.dirname(__file__), "llama3_1B_f32.safetensors")
  max_context=1024
  model = build_transformer(model_path, model_size=model_size, quantize="int8", max_context=max_context)

  TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P = 0.95, 0, 0.0, 0.0, 0.0

  tok = 128000
  model_input = [Tensor([[tok]]), 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P]
  out = model.forward(*model_input)

  #model_input[1] = Variable("start_pos", 0, max_context).bind(0)
  #prg, a, b, state = export_model(model, "clang", *model_input)
  #with open(os.path.join(os.path.dirname(__file__), "llama3_1B_fp16qint8_BEAM2.c"), "w") as text_file:
    #text_file.write(prg)

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  start_pos = Variable("start_pos", 0, max_context).bind(0)
  sub_steps = [
    Step(name = "transformer", input = [Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P], forward = model.forward),
    #Step(name = "q6k_to_f32", input = [Tensor.randn(3_144_960, dtype=dtypes.uint8)], forward = q6k_to_f32),
  ]

  prg = ""

  # TODO: refactor to move some corrected CLANG rendering to export_model.py
  def compile_step(model, step: Step):
    run, special_names = jit_model(step, *step.input)
    functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
    state = get_state_dict(model)
    weightbuf_to_name = {id(x.lazydata.base.realized): name for name, x in state.items()}
    input_names = [name for _,name in special_names.items() if "input" in name]
    output_names = [name for _,name in special_names.items() if "output" in name]
    # this omits saving the random seeds, which therefore will be set in client by default to 0,0 (2x uint32)
    bufs_to_save = {k:v for k,v in bufs.items() if v[2] in weightbuf_to_name}

    cprog = ["#include <tgmath.h>"]
    # TODO: add symbolic variable (start_pos) as arg to net function and enclosed functions that use it

    # declare buffers that we'll load weights into from javascript
    wasm_to_js_pairs = []
    buf_casts = []
    # TODO: import the same type names used in each function declaration. Below mapping is not comprehensive, and may go out of date
    dtype_map = {dtypes.int: "int", dtypes.float: "float", dtypes.char: "signed char", dtypes.half: "__fp16", dtypes.uint: "unsigned int"}
    for name,data in bufs_to_save.items():
      n_bytes, dtype, weightbuf_id = data
      cprog.append(f"unsigned char {name}_data[{n_bytes}];")
      c_dtype = dtype_map[dtype]
      buf_casts.append(f"{c_dtype}* {name} = ({c_dtype}*){name}_data;")
      wasm_to_js_pairs.append((f"{name}_data", f"{weightbuf_to_name[weightbuf_id]}"))

    # wasm_to_js_pairs must have unchanged order from hereafter. We will map downloaded weights in javascript to buffers in wasm using array index
    cprog.append(f"unsigned char* buffers[] = {{\n{"\n".join([wasm_name for wasm_name, js_name in wasm_to_js_pairs])}\n}}")

    # declare zero-filled intermediate buffers
    for name in set(bufs.keys()) - set(bufs_to_save.keys()):
      if 'output' in name or 'input' in name: continue
      n_bytes, dtype, weightbuf_id = bufs[name]
      cprog += [f"{dtype_map[dtype]} {name}[{n_bytes // dtype.itemsize}];"]

    # TODO: function to be exposed to JS for loading weights

    # TODO: are inputs/outputs always arrays?
    inputs = ", ".join([f'{dtype_map[bufs[input][1]]}* {input}' for input in input_names])
    outputs = ", ".join([f'{dtype_map[bufs[output][1]]}* {output}' for output in output_names])
    cprog += list(functions.values())
    # TODO: move buf_casts into a separate function called once before inference, or instead setup weight buffers with final types?
    cprog += [f"void net({inputs}, {outputs}) {{"] + buf_casts
    cprog += [f"{name}({', '.join(args)});" for (name, args, _global_size, _local_size) in statements] + ["}"]
    cprog = "\n".join(cprog)

    return cprog

  for step in sub_steps:
    print(f'Executing step={step.name}')
    prg += compile_step(model, step)

  with open(os.path.join(os.path.dirname(__file__), "transformer.c"), "w") as text_file:
    text_file.write(prg)

  # TODO: JS code for loading in weights based on wasm_to_js_pairs mapping
  # TODO: JS code for setting up inference function, analogous to net.js for WebGPU implementation
  done = 1