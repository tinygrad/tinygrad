from typing import Optional, cast
from tinygrad import Device, Tensor, UOp
from tinygrad.nn.state import get_state_dict
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.export import extract_model, export_webgpu, ExportSpec, resolve_gidx
import json, types
from collections import OrderedDict

EXPORT_SUPPORTED_DEVICE = ["WEBGPU", "CPU", "CUDA", "GPU"]

<<<<<<< HEAD
def export_model_clang(ex: ExportSpec, buf_names: dict[Buffer|UOp, str], weight_names: dict[Buffer|UOp, str]|None=None,
                       model_name="model", wasm=False):
=======
def compile_net(run:TinyJit, special_names:Dict[int,str]) -> Tuple[Dict[str,str],List[Tuple[str,List[str],List[int]]],Dict[str,Tuple[int,DType,int]],Dict[str,Tensor]]:
  functions, bufs, bufs_to_save, statements, bufnum = {}, {}, {}, [], 0
  for ji in run.jit_cache:
    fxn: ProgramSpec = ji.prg.p
    functions[fxn.function_name] = fxn.src   # NOTE: this assumes all with the same name are the same
    cargs = []
    for i,arg in enumerate(ji.bufs):
      key = id(arg)
      if key not in bufs:
        if key in special_names:
          bufs[key] = (special_names[key], arg.size*arg.dtype.itemsize, arg.dtype, key)
        else:
          bufs[key] = (f"buf_{bufnum}", arg.size*arg.dtype.itemsize, arg.dtype, key)
          bufnum += 1
          if i > 0: bufs_to_save[bufs[key][0]] = arg   # if first usage of a buffer is not an output, and it's not a special name
      cargs.append(bufs[key][0])
    cargs += [var for var in fxn.vars if getattr(var, "op", None) is Ops.DEFINE_VAR] # symbolic vars; is it necessary or sufficient to check for DEFINE_VAR?
    statements.append((fxn.function_name, cargs, fxn.global_size, fxn.local_size))

  return functions, statements, {name:(size, dtype, key) for (name,size,dtype,key) in bufs.values()}, bufs_to_save

def jit_model(model, *args) -> Tuple[TinyJit,Dict[int,str]]:
  assert hasattr(model, "forward") or callable(model), "model needs a forward function"
  @TinyJit
  def run(*x):
    out = model.forward(*x) if hasattr(model, "forward") else model(*x)
    assert isinstance(out, (tuple, list, Tensor)), "model output must be a Tensor, tuple, or a list of Tensors for export"
    out = [out] if isinstance(out, Tensor) else out
    return [o.realize() for o in out]

  # twice to run the JIT
  for _ in range(2): the_output = run(*args)
  special_names = {}

  # hack to put the inputs back
  for (j,i),idx in run.input_replace.items():
    realized_input = args[idx].lazydata.base.realized
    run.jit_cache[j].bufs[i] = realized_input
    special_names[id(realized_input)] = f'input{idx}'

  # TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
  for i, output in enumerate(the_output):
    special_names[id(output.lazydata.base.realized)] = f'output{i}'
  return run, special_names

def export_model_clang(functions:Dict[str,str], statements:Dict[str,Tuple[str,int,int]], bufs:Dict[str,Tuple[str,int,int]],
  bufs_to_save:Dict[str,Tensor], input_names:List[str], output_names:List[str], weight_names={}, model_name="model", symbolic_vars={}, wasm=False) -> str:
>>>>>>> master
  headers = ["#include <tgmath.h>"]
  cprog = list(ex.kernels.values())
  dtype_map = {dtypes.int: "int", dtypes.float: "float", dtypes.uchar: "unsigned char", dtypes.char: "signed char", dtypes.half: "__fp16",
               dtypes.uint: "unsigned int"}
  forward_args = ",".join(f"{dtype_map[var.dtype]}{'*' if isinstance(var,Buffer) else ''} {buf_names[var] if isinstance(var, Buffer) else var.arg[0]}"
                          for var in (ex.outputs+ex.inputs if wasm else ex.inputs+ex.outputs))
  if not wasm:
    for buf in ex.weight_bufs:
      weight = ''.join(["\\x%02X"%x for x in bytes(buf.as_buffer())])
      cprog.append(f"unsigned char {buf_names[buf]}_data[] = \"{weight}\";")
    cprog += [f"{dtype_map[buf.dtype]} {buf_names[buf]}[{buf.size}];" for buf in ex.empty_bufs]
    cprog += [f"{dtype_map[buf.dtype]} *{buf_names[buf]} = ({dtype_map[buf.dtype]} *){buf_names[buf]}_data;" for buf in ex.weight_bufs]
    def render_args(args: list[Buffer|UOp]): return ", ".join(buf_names[arg] if isinstance(arg, Buffer) else arg.arg[0] for arg in args)
    cprog += [f"void net({forward_args}) {{"] + [f"{kc.kernel_name}({render_args(kc.args)});" for kc in ex.kernel_calls] + ["}"]
    return '\n'.join(headers + cprog)
  else:
    assert weight_names is not None
    if ex.weight_bufs:
      headers += ["#include <stddef.h>"]
      # TODO: fix random seeds mapping when weights are exported from a separate webgpu model
      buf_to_name = OrderedDict((buf_names[buf], {"name": weight_names[buf], "idx": i}) for i, buf in enumerate(ex.weight_bufs))
      cprog.append(f"void* bufs[{len(buf_to_name)}];")
      cprog.append(f"""void set_buf(size_t index, void* ptr) {{\n  bufs[index] = ptr;\n}}""")

    for buf in ex.empty_bufs:
      cprog += [f"{dtype_map[buf.dtype]} {buf_names[buf]}[{buf.size}];"]

    cprog += [f"void net({forward_args})"] + ["{"]
    def render_arg(arg: Buffer|UOp):
      if arg in ex.weight_bufs: return f"({dtype_map[arg.dtype]} *)bufs[{buf_to_name[buf_names[arg]]['idx']}]"
      elif isinstance(arg, Buffer): return buf_names[arg]
      else: return arg.arg[0]

    cprog += [f"  {kc.kernel_name}({', '.join(map(render_arg, kc.args))});" for kc in ex.kernel_calls] + ["}"]
    weightMapping = "" if not ex.weight_bufs else \
f"""\nconst weightNames = [{", ".join([f'"{weight_name}"' for weight_name in [v["name"] for v in buf_to_name.values()]])}];
const {model_name}_name_to_id = Object.fromEntries(weightNames.map((name, index) => [name, index]));\n"""
    top = f"""import {model_name}Module from './{model_name}.js'{weightMapping}"""
    whitespace = "\n  "
    js_wrapper = f"""{top}\nvar {model_name} = async function() {{
  const wasm = await {model_name}Module();
  {whitespace.join(f"const {buf_names[arg]}Ptr = wasm._malloc({arg.nbytes});" for arg in ex.outputs+ex.inputs if isinstance(arg, Buffer))}
  return {{
    run: ({", ".join(buf_names[arg] if isinstance(arg, Buffer) else arg.arg[0] for arg in ex.inputs)}) => {{
      {(whitespace + "    ").join(f"wasm.HEAPU8.set({buf_names[arg]}, {buf_names[arg]}Ptr);" for arg in ex.inputs if isinstance(arg, Buffer))}
      wasm._net({", ".join(f"{buf_names[arg] if isinstance(arg, Buffer) else arg.arg[0]}{'Ptr' if isinstance(arg, Buffer) else ''}"
                            for arg in ex.outputs+ex.inputs)});
      {(whitespace + "    ").join(f"const {(name:=buf_names[buf])} = wasm.HEAPU8.slice({name}Ptr, {name}Ptr + {buf.nbytes});" for buf in ex.outputs)}
      return [{", ".join(f"{buf_names[buf]}" for buf in ex.outputs)}];
    }},
    wasm: wasm
  }}
}}\nexport {{ {model_name}, {model_name}_name_to_id }};"""

    return '\n'.join(headers + cprog), js_wrapper

def export_model(model, target:str, *inputs, model_name: Optional[str] = "model"):
  assert Device.DEFAULT in EXPORT_SUPPORTED_DEVICE, "only WEBGPU, CPU, CUDA, GPU, METAL are supported"
  if hasattr(model, "forward"): model = model.forward
  else: assert callable(model)

  if isinstance(model, types.MethodType): weights_holder = model.__self__
  elif hasattr(model, "__call__") and not isinstance(model, types.FunctionType): weights_holder = model
  else: weights_holder = None
  state_dict = get_state_dict(weights_holder)

  # TODO: there is some clunkiness for reverse-compatibility with tests
  ex: ExportSpec = extract_model(model, inputs)

  buf_names = {buf: f"buf_{i}" for i,buf in enumerate(cast(list[Buffer|UOp], ex.inputs + ex.outputs + ex.empty_bufs + ex.weight_bufs))}
  weight_names = {cast(Buffer, t.lazydata.base.realized): name for name, t in state_dict.items()} if state_dict else buf_names
  for buf in ex.weight_bufs:
    name = weight_names[buf] = weight_names.get(buf, buf_names[buf])
    state_dict[name] = state_dict.get(name, Tensor(bytes(buf.as_buffer()), dtype=buf.dtype, device=buf.device).realize())

  if target == "webgpu": prg, state_dict = export_webgpu(model, inputs, model_name=model_name)
  elif target == "clang":
    prg = export_model_clang(ex, buf_names)
  elif target == "wasm":
    return export_model_clang(ex, buf_names, weight_names, model_name, wasm=True)
  else:
    bufs = {buf_names[buf]: {"size": buf.nbytes, "dtype": buf.dtype.name, "id": weight_names[buf]} for buf in ex.empty_bufs + ex.weight_bufs}
    bufs.update({f"input{i}": {"size": (arg.nbytes if isinstance(arg, Buffer) else arg.dtype.itemsize),
      "dtype": arg.dtype.name, "id": ""} for i, arg in enumerate(ex.inputs)})
    bufs.update({f"output{i}": {"size": buf.nbytes, "dtype": buf.dtype.name, "id": ""} for i, buf in enumerate(ex.outputs)})
    prg = json.dumps({
      "backend": Device.DEFAULT,
      "inputs": [{
        "name": buf_names[var] if isinstance(var, Buffer) else var.arg[0],
        "arg_type": "symbolic" if isinstance(var, UOp) else "buffer",
        "size": var.nbytes if isinstance(var, Buffer) else var.dtype.itemsize,
        "dtype": var.dtype.name
      } for var in ex.inputs],
      "outputs": [{"size": buf.nbytes, "dtype": buf.dtype.name} for buf in ex.outputs],
      "functions": ex.kernels,
      "statements": [{
        "kernel": kc.kernel_name,
        "args": [buf_names[arg] if isinstance(arg, Buffer) else arg.arg[0] for arg in kc.args],
        "global_size": ', '.join(resolve_gidx(x) for x in kc.global_size),
        "local_size": ', '.join(resolve_gidx(x) for x in kc.local_size)
      } for kc in ex.kernel_calls],
    # TODO: probably rewrite the tests that fail without the dict.update calls
      "buffers": bufs
    })

  # TODO: update tests that make assertions about input_sizes and output_sizes
  input_sizes = {f"input{i}": (arg.nbytes if isinstance(arg, Buffer) else arg.dtype.itemsize) for i, arg in enumerate(ex.inputs)}
  return prg, input_sizes, {f"output{i}": buf.nbytes for i, buf in enumerate(ex.outputs)}, state_dict
