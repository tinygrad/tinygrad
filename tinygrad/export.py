# export a model for use outside of tinygrad: including data, transforms, and runtime wrapper

# TODO: is there a better place in tinygrad for this?
# TODO: refactor in progress, simplify everything

from typing import Callable, Sequence, Union, Any, Optional, cast
from dataclasses import dataclass
from collections import OrderedDict
import types
from tinygrad import Tensor, dtypes, TinyJit, Device, Variable, UOp
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.ops import Ops, Variable
from tinygrad.renderer import ProgramSpec
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, get_parameters
from tinygrad.helpers import Context
from tinygrad.dtype import DType
from tinygrad.device import Buffer

gidxT = Union[int, UOp]

@dataclass(frozen=True)
class KernelCall:
  kernel_name: str
  args: list[Union[Buffer, Variable]]
  global_size: Optional[tuple[gidxT, gidxT, gidxT]]=None
  local_size: Optional[tuple[int, int, int]]=None

@dataclass(frozen=True)
class ExportSpec:
  inputs: list[Union[Buffer, Variable]]
  outputs: list[Buffer]
  empty_bufs: set[Buffer]
  weight_bufs: set[Buffer]
  kernels: dict[str, str]
  kernel_calls: list[KernelCall]

def extract_model(model: Callable, args: Sequence) -> ExportSpec:
  @TinyJit
  def run(args) -> list[Tensor]:
    out:list[Tensor]|tuple[Tensor] = returned if isinstance((returned := model(*args)), (list, tuple)) else [returned]
    assert all(isinstance(x, Tensor) for x in out), "must return a Tensor, or a list or tuple of Tensors"
    return [t.realize() for t in out] # TODO: or if we allow non-tensors, change to nn.state.get_parameters and then realize

  for _ in range(2): run(args)
  input_bufs, var_vals, names, st_vars_dtype_device = _prepare_jit_inputs(tuple(args), {}) # TODO: enable kwargs?
  for (j,i),idx in run.captured.input_replace.items(): run.jit_cache[j].bufs[i] = input_bufs[idx]
  output_bufs: list[Buffer] = [t.lazydata.base.realized for t in cast(list[Tensor], run.captured.ret)]
  
  kernels, empty_bufs, weight_bufs, seen, kernel_calls = {}, set(), set(), set(input_bufs + output_bufs), []
  for ei in run.jit_cache:
    fxn: ProgramSpec = ei.prg.p
    kernels[fxn.function_name] = fxn.src
    for i, buf in enumerate(ei.bufs):
      if buf not in seen and i==0: empty_bufs.add(buf)
      elif buf not in seen: weight_bufs.add(buf)
      seen.add(buf)
    kernel_calls.append(KernelCall(fxn.function_name, ei.bufs + fxn.vars, fxn.global_size, fxn.local_size))

  resolve = lambda x: x if isinstance(x, Variable) else cast(Tensor, x).lazydata.base.realized
  return ExportSpec([resolve(x) for x in args if isinstance(x, (Variable, Tensor))], output_bufs, empty_bufs, weight_bufs, kernels, kernel_calls)

def compile_net(run:TinyJit, special_names:dict[int,str]) -> tuple[dict[str,str], list[tuple[str,list[str],list[int]]], dict[str,tuple[int,DType,int]]]:
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

def jit_model(model:Callable[..., Tensor|list[Tensor]|tuple[Tensor]], args) -> tuple[TinyJit,dict[int,str]]:
  @TinyJit
  def run(x) -> list[Tensor]:
    out:list[Tensor]|tuple[Tensor] = returned if isinstance((returned := model(*x)), (list, tuple)) else [returned]
    # TODO do we want this assertion? the non-Tensor ret values aren't modified, and may be tricky to export - are they ever useful?
    assert all(isinstance(x, Tensor) for x in out), "must return a Tensor, or a list or tuple of Tensors"
    return [t.realize() for t in out] # TODO: or if we allow non-tensors, change to nn.state.get_parameters and then realize
  for _ in range(2): run(*args) # generate run.captured:CapturedJit by calling run twice

  input_buffers, var_vals, names, st_vars_dtype_device = _prepare_jit_inputs(args, {}) # TODO: handle kwargs
  # TODO: requires that run.captured.ret is list[Tensor]; instead can use nn.state.get_parameters, but ensure canonical ordering
  special_names = {id(t.lazydata.base.realized): f"output{i}" for i,t in enumerate(cast(list[Tensor], run.captured.ret))}
  special_names.update({id(k): f"input{v}" for k,v in zip(input_buffers, names)})
  for (j,i),idx in run.captured.input_replace.items(): run.jit_cache[j].bufs[i] = input_buffers[idx]
  
  return run, special_names

def dtype_to_js_type(dtype: DType) -> str:
  return f"{'Uint' if dtype in dtypes.uints else 'Int' if (dtype in dtypes.sints or dtype == dtypes.bool) else 'Float'}{8*dtype.itemsize}Array"

def export_webgpu(model:Callable[..., Tensor|list[Tensor]|tuple[Tensor]], inputs:Sequence[Any], js_outfile:Optional[str]=None,
                  state_dict:Optional[dict[str,Tensor]]=None, model_name="model", save_weights=True, fix_contiguous=True) -> tuple[str, dict[str, Tensor]]:
  """
  Exports a javascript WebGPU implementation of a tinygrad model.
  """
  if isinstance(model, types.MethodType): weights_holder = model.__self__
  elif hasattr(model, "__call__") and not isinstance(model, types.FunctionType): weights_holder = model
  else: weights_holder = None

  # TODO: get rid of this _state_dict / contiguous stuff
  # torch_load sometimes loads non-contiguous tensors, which when saved with safe_save, can cause exported WebGPU inference to fail
  # TODO: investigate contiguity in torch_load, and in safe_save/safe_load cycle (which enforces contiguity)
  _state_dict = get_state_dict(weights_holder)
  if _state_dict and fix_contiguous:
    for k,v in _state_dict.items():
      _state_dict[k] = v.contiguous().to("WEBGPU").realize()
    load_state_dict(weights_holder, _state_dict)
  if not state_dict: state_dict = _state_dict

  with Context(JIT=2): ex = extract_model(model, inputs)
  # TODO: broken hereafter; refactor to wrap state/compute from ex in JS runtime
  #with Context(JIT=2): run, special_names = jit_model(model, *inputs)
  #functions, statements, bufs, bufs_to_save = compile_net(run, special_names)

  # TODO: if user passes incomplete state_dict, make it still work by filling in gaps with bufs_to_save
  if state_dict:
    weight_names = {id(x.lazydata.base.realized): name for name, x in state_dict.items()}
  else:
    weight_names = {id(buf): name for name, buf in bufs_to_save.items()}
  max_buf_nbytes = max(v[0] for k,v in bufs.items())
  input_names = [name for _,name in special_names.items() if "input" in name]
  output_names = [name for _,name in special_names.items() if "output" in name]

  # set(sum((ji.prg.p.vars for ji in run.jit_cache), []))
  # self.simplify().render()

  # handle symbolic variables; TODO: refactor to fix some of this stuff upstream in tinygrad
  symbolic_vars = OrderedDict()
  for i, (_, args, global_size, _) in enumerate(statements):
    for j, var in enumerate(args):
      if getattr(var, "op", None) is Ops.DEFINE_VAR and isinstance(getattr(var, "arg", None), tuple) and isinstance(var.arg[0], str):
        if var not in symbolic_vars:
          symbolic_vars[var] = var.arg[0]
          bufs[symbolic_vars[var]] = (var.dtype.itemsize, var.dtype, symbolic_vars[var])
        statements[i][1][j] = symbolic_vars[var]

    if global_size:
      for j, dim in enumerate(global_size):
        if getattr(dim, "op", None) is Ops.ADD and len(dim.src) == 2 and {dim.src[0].op, dim.src[1].op} == {Ops.DEFINE_VAR, Ops.CONST}:
          name, val = dim.src if dim.src[1].op is Ops.CONST else reversed(dim.src)
          global_size[j] = f"_{name.arg[0]}[0] + {val.arg}"

  kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
  kernel_names = ', '.join([name for (name, _, _, _) in statements])
  input_names += list(symbolic_vars.values())
  input_buffer_types = [dtype_to_js_type(bufs[inp_name][1]) for inp_name in input_names]
  output_buffer_types = [dtype_to_js_type(bufs[out_name][1]) for out_name in output_names]

  buf_type = lambda x: "uniform" if x in set(symbolic_vars.values()) else "storage"
  create_bind_group_layouts = ",".join([
    "device.createBindGroupLayout({{entries: [{{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {{ type: 'uniform' }}}}, {}]}})".format(
        ",".join([f"{{binding: {argIdx+1}, visibility: GPUShaderStage.COMPUTE, buffer: {{ type: '{buf_type(argName)}' }} }}" for argIdx, argName in enumerate(args)])
    )
    for _, (_, args, _, _) in enumerate(statements)
  ])
  layouts = f"const layouts=[{create_bind_group_layouts}]"
  kernel_calls = '\n        '.join([f"addComputePass(device, commandEncoder, pipelines[{i}], layouts[{i}], infinityBuf, [{', '.join(args)}], [{', '.join(str(x) for x in global_size)}]);" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])

  buf_type = lambda x: "createUniformBuf" if x in set(uop.arg[0] for uop in symbolic_vars) else "createEmptyBuf"
  map_to_external_weight = lambda _key: f"state_dict['{weight_names[_key]}']" if not save_weights else f"getTensorBuffer(safetensor, metadata['{weight_names[_key]}'])"
  _bufs =  '\n    '.join([f"const {name} = " + (f"{buf_type(_key)}(device, {size});" if _key not in weight_names else f"createWeightBuf(device, {size}, {map_to_external_weight(_key)})") + ";"  for name,(size,dtype,_key) in bufs.items()])
  gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:{input_name}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,input_name in enumerate(input_names)])
  input_writers = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n        new {input_buffer_types[i]}(gpuWriteBuffer{i}.getMappedRange()).set(" + f'_{inp_name});' + f"\n        gpuWriteBuffer{i}.unmap();\n        commandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, {inp_name}, 0, gpuWriteBuffer{i}.size);"  for i,inp_name in enumerate(input_names)])
  gpu_read_bufs = '\n    '.join([f"const gpuReadBuffer{i} = device.createBuffer({{size:{output_name}.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});" for i,output_name in enumerate(output_names)])
  outbuf_copies = '\n        '.join([f"commandEncoder.copyBufferToBuffer({output_name}, 0, gpuReadBuffer{i}, 0, output{i}.size);" for i,output_name in enumerate(output_names)])
  output_readers = '\n        '.join([f"await gpuReadBuffer{i}.mapAsync(GPUMapMode.READ);\n        const resultBuffer{i} = new {output_buffer_types[i]}(gpuReadBuffer{i}.size/{bufs[output_names[i]][1].itemsize});\n        resultBuffer{i}.set(new {output_buffer_types[i]}(gpuReadBuffer{i}.getMappedRange()));\n        gpuReadBuffer{i}.unmap();" for i in range(len(output_names))])
  output_return = '[{}]'.format(",".join([f'resultBuffer{i}' for i in range(len(output_names))]))
  getTensorMetadata = f"""\nconst getTensorMetadata = (safetensorBuffer) => {{
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
}};\n""" if save_weights else ""
  prg = f"""
if (!navigator.gpu) throw new Error("WebGPU not supported.");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({{
	requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [],
	powerPreference: "high-performance",
  requiredLimits: {{maxStorageBufferBindingSize: {max_buf_nbytes}, maxBufferSize: {max_buf_nbytes},
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup}},
}});

const {model_name} = (() => {{
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {{
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
}};
{getTensorMetadata}
const createEmptyBuf = (device, size) => {{
    return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
}};

const createUniformBuf = (device, size) => {{
  return device.createBuffer({{size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}})
}}

const createInfinityUniformBuf = (device) => {{
  const size = 4;
  const buf = device.createBuffer({{
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  }});
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
}};

const createWeightBuf = (device, size, data) => {{
  const buf = device.createBuffer({{ size, usage: GPUBufferUsage.STORAGE{" | GPUBufferUsage.COPY_DST" if not save_weights else ", mappedAtCreation: true"} }});
  {"data.bytes = buf;" if not save_weights else "new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();"}
  return buf;
}};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {{
  const bindGroup = device.createBindGroup({{
    layout: layout,
    entries: [
      {{ binding: 0, resource: {{ buffer: infinityUniformBuf }} }},
      ...bufs.map((buffer, index) => ({{ binding: index + 1, resource: {{ buffer }} }}))
    ]
  }});

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
}};

{kernel_code}

const setupNet = async ({"state_dict" if not save_weights else "safetensor"}) => {{
    {"const metadata = getTensorMetadata(safetensor);" if not not save_weights else ""}
    const infinityBuf = createInfinityUniformBuf(device);

    {layouts}

    {_bufs}

    {gpu_write_bufs}

    {gpu_read_bufs}

    const kernels = [{kernel_names}];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {{
      return await device.createComputePipelineAsync({{
          layout: device.createPipelineLayout({{
              bindGroupLayouts: [layouts[i]],
          }}),
          compute: {{
              module: device.createShaderModule({{
                  code: name,
              }}),
              entryPoint: "main",
          }},
      }});
  }}))

    return [async ({",".join([f"_{input_name}" for input_name in input_names])}) => {{
        const commandEncoder = device.createCommandEncoder();
        {input_writers}
        {kernel_calls}
        {outbuf_copies}
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        {output_readers}
        return {output_return};
    }}, device]
}}
const load = async (weight_path) => {{ return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(new Uint8Array(x))).then(x => x[0]); }}
return {{ load, setupNet }};
}})();
export default {model_name};
"""

  if js_outfile:
    with open(js_outfile, "w") as f: f.write(prg)
    if not state_dict:
      state_dict = {name: Tensor(bytes(buf.as_buffer()), dtype=buf.dtype, device=buf.device).realize() for name, buf in bufs_to_save.items()}
    safe_save(state_dict, f"{js_outfile}.safetensors")

  return prg, state_dict