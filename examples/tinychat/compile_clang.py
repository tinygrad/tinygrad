# TODO: merge with examples/tinychat/compile.py

import os
from extra.export_model import compile_net, jit_model, export_model
from examples.llama3 import build_transformer
from tinygrad import Device, Variable, Tensor, dtypes
from tinygrad.helpers import fetch
from typing import List, Tuple, Any, NamedTuple
from tinygrad.nn.state import get_state_dict
from tinygrad.ops import Ops

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

  # TODO: refactor to move some corrected CLANG rendering to export_model.py
  def compile_step(model, step: Step):
    run, special_names = jit_model(step, *step.input)
    functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
    state = get_state_dict(model)
    weightbuf_to_name = {id(x.lazydata.base.realized): name for name, x in state.items()}
    # this omits saving the random seeds, which therefore will be set in client by default to 0,0 (2x uint32)
    bufs_to_save = {k:v for k,v in bufs.items() if v[2] in weightbuf_to_name}

    cprog = ["#include <tgmath.h>"]

    # declare buffers that we'll load weights into from javascript
    buf_to_name = []
    # TODO: import the same type names used in each function declaration. Below mapping is not comprehensive, and may go out of date
    dtype_map = {dtypes.int: "int", dtypes.float: "float", dtypes.char: "signed char", dtypes.half: "__fp16", dtypes.uint: "unsigned int"}
    for name,data in bufs_to_save.items():
      n_bytes, dtype, weightbuf_id = data
      cprog += [f"{dtype_map[dtype]} {name}[{n_bytes // dtype.itemsize}];"]
      buf_to_name.append((f"{name}", f"{weightbuf_to_name[weightbuf_id]}"))

    # buf_to_name must have unchanged order from hereafter. We rely on integer index of void* buffers to load weights from javascript
    cprog.append(f"void* buffers[] = {{\n{",\n".join([buf_name for buf_name, weight_name in buf_to_name])}\n}};")

    # declare zero-filled intermediate buffers
    for name in set(bufs.keys()) - set(bufs_to_save.keys()) - set(special_names.values()):
      n_bytes, dtype, weightbuf_id = bufs[name]
      cprog += [f"{dtype_map[dtype]} {name}[{n_bytes // dtype.itemsize}];"]

    # TODO: function to be exposed to JS for getting pointer from void* buffers based on index

    inputs = [f"{dtype_map[bufs[input][1]]}* {input}" for input in special_names.values() if "input" in input]
    symbolic_vars = set()
    for i, (_, args, _, _) in enumerate(statements):
      for j, var in enumerate(args):
        if getattr(var, "op", None) is Ops.DEFINE_VAR and isinstance(getattr(var, "arg", None), tuple) and isinstance(var.arg[0], str):
          symbolic_vars.add(var)
          statements[i][1][j] = var.arg[0] # name assigned in Variable(name, ...), e.g. "start_pos"

    inputs = ", ".join(inputs + [f"{dtype_map[var.dtype]} {var.arg[0]}" for var in symbolic_vars])
    outputs = ", ".join([f'{dtype_map[bufs[output][1]]}* {output}' for output in special_names.values() if "output" in output])
    cprog += list(functions.values())
    cprog += [f"void net({inputs}, {outputs}) {{"]
    cprog += [f"{name}({', '.join(args)});" for (name, args, _global_size, _local_size) in statements] + ["}"]
    cprog = "\n".join(cprog)

    with open(os.path.join(os.path.dirname(__file__), f"{step.name}.c"), "w") as text_file:
      text_file.write(cprog)

    return f"""\nvar {step.name} = function() {{

  // load wasm module, get handle

    return {{
      "setup": async (safetensor, metadata) => {{

      // fetch pointers to empty weight buffers on wasm heap
      // allocate weights from JS memory to above pointers

      // allocate wasm memory for input/output arrays

        return async ({",".join([f'data{i}' for i,(k,v) in enumerate(special_names.items()) if v != "output0"])}) => {{

          // set input buffer
          // call net function
          // set and return result
        }}
      }}
    }}
  }}"""

  prg = ""

  for step in sub_steps:
    print(f'Executing step={step.name}')
    prg += compile_step(model, step)

  with open(os.path.join(os.path.dirname(__file__), "net.js"), "w") as text_file:
    text_file.write(prg)

  # TODO: JS code for loading in weights based on buf_to_name mapping
  # TODO: JS code for setting up inference function, analogous to net.js for WebGPU implementation
  done = 1