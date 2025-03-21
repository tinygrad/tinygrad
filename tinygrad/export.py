# export a model for use outside of tinygrad: including data, transforms, and runtime wrapper

# TODO: is there a better place in tinygrad for this?
# TODO: refactor in progress, simplify everything

from typing import Callable, Sequence, Union, Any, Optional, cast
from dataclasses import dataclass
import types
from tinygrad import Tensor, dtypes, TinyJit, UOp
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.renderer import ProgramSpec
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save
from tinygrad.helpers import Context
from tinygrad.dtype import DType
from tinygrad.device import Buffer

gidxT = Union[int, UOp]

@dataclass(frozen=True)
class KernelCall:
  kernel_name: str
  args: list[Union[Buffer, UOp]]
  global_size: Optional[tuple[gidxT, gidxT, gidxT]]=None

@dataclass(frozen=True)
class ExportSpec:
  inputs: list[Union[Buffer, UOp]]
  outputs: list[Buffer]
  empty_bufs: list[Buffer]
  weight_bufs: list[Buffer]
  kernels: dict[str, str]
  kernel_calls: list[KernelCall]

def extract_model(model: Callable, args: Sequence) -> ExportSpec:
  """
  Extracts data and transforms that specify a realized tinygrad model, for use in external runtimes.
  """
  @TinyJit
  def run(args) -> list[Tensor]:
    out:list[Tensor]|tuple[Tensor] = returned if isinstance((returned := model(*args)), (list, tuple)) else [returned]
    assert all(isinstance(x, Tensor) for x in out), "must return a Tensor, or a list or tuple of Tensors"
    return [t.realize() for t in out]

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
    kernel_calls.append(KernelCall(fxn.function_name, ei.bufs + fxn.vars, fxn.global_size))

  res = lambda x: x if isinstance(x, UOp) else cast(Tensor, x).lazydata.base.realized
  return ExportSpec([res(x) for x in args if isinstance(x, (UOp, Tensor))], output_bufs, list(empty_bufs), list(weight_bufs), kernels, kernel_calls)

def dtype_to_js_type(dtype: DType) -> str:
  return f"{'Uint' if dtype in dtypes.uints else 'Int' if (dtype in dtypes.sints or dtype == dtypes.bool) else 'Float'}{8*dtype.itemsize}Array"

def export_webgpu(model:Callable, inputs:Sequence, js_outfile:Optional[str]=None, state_dict:Optional[dict[str,Tensor]]=None,
                  model_name="model", save_weights=True, fix_contiguous=True) -> tuple[str, dict[str, Tensor]]:
  """
  Exports a javascript WebGPU implementation of a tinygrad model.
  """
  if isinstance(model, types.MethodType): weights_holder = model.__self__
  elif hasattr(model, "__call__") and not isinstance(model, types.FunctionType): weights_holder = model
  else: weights_holder = None

  # TODO: get rid of this _state_dict / contiguous stuff
  # for the WebGPU efficientnet, torch_load loads non-contiguous tensors, which when not safe saved/loaded, gives incorrect inference
  # TODO: investigate contiguity in torch_load, and in safe_save/safe_load cycle (which enforces contiguity)
  _state_dict = get_state_dict(weights_holder)
  if _state_dict and fix_contiguous:
    for k,v in _state_dict.items():
      _state_dict[k] = v.contiguous().to("WEBGPU").realize()
    load_state_dict(weights_holder, _state_dict)
  if not state_dict: state_dict = _state_dict

  with Context(JIT=2): ex = extract_model(model, inputs)

  # TODO: if user passes incomplete state_dict, make it still work by filling in gaps with bufs_to_save
  if state_dict:
    weight_names: dict[Buffer, str] = {x.lazydata.base.realized: name for name, x in state_dict.items()}
  #else:
    #weight_names = {id(buf): name for name, buf in bufs_to_save.items()}
  max_buf_nbytes = max(buf.nbytes for buf in ex.weight_bufs + ex.empty_bufs)
  kernel_code = '\n\n'.join([f"const {name} = `{code.replace(name, 'main')}`;" for name, code in ex.kernels.items()])

  buf_names = {buf: f"buf_{i}" for i,buf in enumerate(ex.inputs + ex.outputs + ex.empty_bufs + ex.weight_bufs) if isinstance(buf, Buffer)}
  kernel_calls = "\n        ".join(f"""
addComputePass(device, commandEncoder, pipelines[{i}], infinityBuf, [{', '.join(buf_names[arg] for arg in kc.args)}], [{', '.join(str(x) for x in kc.global_size)}], kernels[{i}].split("INFINITY").length > 2); 
""" for i, kc in enumerate(ex.kernel_calls))
  empty_bufs = '\n    '.join(f"const {buf_names[buf]} = createEmptyBuf(device, {buf.nbytes});" for buf in ex.inputs + ex.outputs + ex.empty_bufs if isinstance(buf, Buffer))
  map_to_external_weight = lambda _key: f"state_dict['{weight_names[_key]}']" if not save_weights else f"getTensorBuffer(safetensor, metadata['{weight_names[_key]}'])"
  weight_bufs = '\n    '.join(f"const {buf_names[buf]} = createWeightBuf(device, {buf.nbytes}, {map_to_external_weight(buf)});" for i,buf in enumerate(ex.weight_bufs))
  input_writers = "\n".join(f"""
  device.queue.writeBuffer(gpuWriteBuffer{i}, 0, _input{i});
  commandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, {buf_names[buf]}, 0, gpuWriteBuffer{i}.size);
  """ for i, buf in enumerate(ex.inputs) if isinstance(buf, Buffer))
  outbuf_copies = '\n        '.join([f"commandEncoder.copyBufferToBuffer({buf_names[buf]}, 0, gpuReadBuffer{i}, 0, {buf.nbytes});" for i,buf in enumerate(ex.outputs)])

  output_buffer_types = [dtype_to_js_type(buf.dtype) for buf in ex.outputs]
  output_readers = "\n".join(f"""
await gpuReadBuffer{i}.mapAsync(GPUMapMode.READ);
const resultBuffer{i} = new {output_buffer_types[i]}(gpuReadBuffer{i}.size/{buf.dtype.itemsize});
resultBuffer{i}.set(new {output_buffer_types[i]}(gpuReadBuffer{i}.getMappedRange()));
gpuReadBuffer{i}.unmap();
""" for i,buf in enumerate(ex.outputs))

  output_return = '[{}]'.format(",".join([f'resultBuffer{i}' for i in range(len(ex.outputs))]))

  getTensorMetadata = f"""\nconst getTensorMetadata = (safetensorBuffer) => {{
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
}};\n""" if save_weights else ""

  nl="\n"
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

const addComputePass = (device, commandEncoder, pipeline, infinityUniformBuf, bufs, workgroup, useInfinity) => {{
  const entries = [];
  if (useInfinity) entries.push({{ binding: 0, resource: {{ buffer: infinityUniformBuf }} }});
  entries.push(...bufs.map((buffer, index) => ({{
    binding: index + 1,
    resource: {{ buffer }}
  }})));
  const bindGroup = device.createBindGroup({{
    layout: pipeline.getBindGroupLayout(0),
    entries
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
    {empty_bufs + weight_bufs}

    {(nl+'    ').join(f"const gpuWriteBuffer{i} = device.createBuffer({{size:{buf.nbytes}, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST}});" for i,buf in enumerate(ex.inputs) if isinstance(buf, Buffer))}
    {(nl+'    ').join(f"const gpuReadBuffer{i} = device.createBuffer({{size:{buf.nbytes}, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}});" for i,buf in enumerate(ex.outputs))}

    const kernels = [{",".join(kc.kernel_name for kc in ex.kernel_calls)}];
    const pipelines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{
      layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

    return [async ({",".join([f"_input{i}" for i in range(len(ex.inputs))])}) => {{
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