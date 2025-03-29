from typing import Callable, Sequence, Union, Optional, cast
from dataclasses import dataclass, field
import types
from tinygrad import Tensor, dtypes, TinyJit, UOp
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import CompiledRunner
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save
from tinygrad.dtype import DType
from tinygrad.device import Buffer

gidxT = Union[int, UOp]
def resolve_gidx(x): return x.simplify().render() if isinstance(x, UOp) else str(x)

@dataclass(frozen=True)
class KernelCall:
  kernel_name: str
  args: list[Union[Buffer, UOp]]
  global_size: list[gidxT]=field(default_factory=list)
  local_size: list[gidxT]=field(default_factory=list) # not currently used

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

  for _ in range(2): out = run(args)
  assert run.captured is not None
  input_bufs = _prepare_jit_inputs(tuple(args), {})[0] # TODO: enable kwargs?
  for (j,i),idx in run.captured.input_replace.items(): run.jit_cache[j].bufs[i] = input_bufs[idx]
  output_bufs = [t.lazydata.base.realized for t in out if t.lazydata.base.realized is not None]

  kernels, empty_bufs, weight_bufs, seen, calls = {}, set(), set(), set(input_bufs + output_bufs), []
  for ei in run.jit_cache:
    assert isinstance(ei.prg, CompiledRunner) and all(isinstance(buf, Buffer) for buf in ei.bufs)
    fxn = ei.prg.p
    global_size, local_size = map(lambda x: [] if not x else cast(list[gidxT], x), (fxn.global_size, fxn.local_size))
    kernels[fxn.function_name] = fxn.src

    for i, buf in enumerate(cast(list[Buffer], ei.bufs)):
      if buf not in seen and i==0: empty_bufs.add(buf)
      elif buf not in seen: weight_bufs.add(buf)
      seen.add(buf)
    calls.append(KernelCall(fxn.function_name, cast(list[Buffer], ei.bufs) + fxn.vars, global_size, local_size))

  def res(x): return x.unbind()[0] if isinstance(x, UOp) else cast(Tensor, x).lazydata.base.realized
  return ExportSpec([res(x) for x in args if isinstance(x, (UOp, Tensor))], output_bufs, list(empty_bufs), list(weight_bufs), kernels, calls)

def js_type(dtype: DType) -> str:
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
  for t in state_dict.values(): assert isinstance(t.lazydata.base.realized, Buffer)

  # Extract model data/transforms, sufficient for export to external runtime
  ex: ExportSpec = extract_model(model, inputs)

  # Map buffers to their names in the state_dict and handle weight buffers that are missing from the state_dict
  buf_names = {buf: f"buf_{i}" for i,buf in enumerate(cast(list[Buffer|UOp], ex.inputs + ex.outputs + ex.empty_bufs + ex.weight_bufs))}
  weight_names = {cast(Buffer, t.lazydata.base.realized): name for name, t in state_dict.items()} if state_dict else buf_names
  for buf in ex.weight_bufs:
    name = weight_names[buf] = weight_names.get(buf, buf_names[buf])
    state_dict[name] = state_dict.get(name, Tensor(bytes(buf.as_buffer()), dtype=buf.dtype, device=buf.device).realize())

  # Render model data
  empty_bufs = [f"const {buf_names[b]} = createEmptyBuf({b.nbytes});" for b in ex.inputs+ex.outputs+ex.empty_bufs if isinstance(b, Buffer)]
  symbolic_bufs = [f"const {buf_names[var]} = createUniformBuf({var.dtype.itemsize});" for var in ex.inputs if isinstance(var, UOp)]
  def map_wt(buf): return f"state_dict['{weight_names[buf]}']" if not save_weights else f"getTensorBuffer(safetensor,metadata['{weight_names[buf]}'])"
  weight_bufs = [f"const {buf_names[buf]} = createWeightBuf({buf.nbytes}, {map_wt(buf)});" for buf in ex.weight_bufs]

  # Render model transforms
  kernel_declarations = '\n\n'.join([f"const {name} = `{code.replace(name, 'main')}`;" for name, code in ex.kernels.items()])
  kernel_calls = [f"addComputePass(commandEncoder, pipelines[{i}], [{', '.join(buf_names[arg] for arg in kc.args)}], " + \
    f"[{', '.join(resolve_gidx(x) for x in kc.global_size)}], kernels[{i}].split('INFINITY').length > 2);" for i, kc in enumerate(ex.kernel_calls)]

  # Render runtime-specific operations
  # TODO: validate symbolic var ranges against input args at runtime in JS?
  args = [f"_input{i}" if isinstance(var, Buffer) else f"{var.arg[0]}" for i,var in enumerate(ex.inputs)]
  input_validation = [f"""if (!({ f'{args[i]} instanceof {js_type(ex.inputs[i].dtype)}' if isinstance(ex.inputs[i], Buffer) else
    f'typeof {args[i]} === "number"'})) {{ throw new Error(`arg {i} type: ${{typeof {args[i]}}} is not as expected`) }}""" for i in range(len(args))]
  input_writer_bufs = [f"""const gpuWriteBuffer{i} = device.createBuffer({{size:{buf_names[buf]}.size,
                usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST}});""" for i,buf in enumerate(ex.inputs)]
  input_writers = [f"""
      device.queue.writeBuffer(gpuWriteBuffer{i}, 0, {f"_input{i}" if isinstance(var, Buffer) else f"new {js_type(var.dtype)}([{var.arg[0]}])"});
      commandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, {buf_names[var]}, 0, gpuWriteBuffer{i}.size);""" for i, var in enumerate(ex.inputs)]

  output_reader_bufs = [f"""const gpuReadBuffer{i} = device.createBuffer({{size:{buf_names[buf]}.size,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}});""" for i,buf in enumerate(ex.outputs)]
  output_readers = [f"""await gpuReadBuffer{i}.mapAsync(GPUMapMode.READ);
      const resultBuffer{i} = new {js_type(buf.dtype)}(gpuReadBuffer{i}.size/{buf.dtype.itemsize});
      resultBuffer{i}.set(new {js_type(buf.dtype)}(gpuReadBuffer{i}.getMappedRange()));
      gpuReadBuffer{i}.unmap();""" for i,buf in enumerate(ex.outputs)]

  outbuf_copies = [f"commandEncoder.copyBufferToBuffer({buf_names[buf]}, 0, gpuReadBuffer{i}, 0, {buf.nbytes});" for i,buf in enumerate(ex.outputs)]
  output_return = '[{}]'.format(",".join([f'resultBuffer{i}' for i in range(len(ex.outputs))]))

  getTensorMetadata = """\n  const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(
      ([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
  };\n""" if save_weights else ""
  max_buf_nbytes = max([buf.nbytes for buf in ex.weight_bufs + ex.empty_bufs] + [4])

  def j(to_join: list, num_indents: int): return ("\n" + num_indents * "  ").join(to_join)
  prg = f"""if (!navigator.gpu) throw new Error("WebGPU not supported.");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({{
	requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [], powerPreference: "high-performance",
  requiredLimits: {{maxStorageBufferBindingSize: {max_buf_nbytes}, maxBufferSize: {max_buf_nbytes},
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup}},
}});

const {model_name} = (() => {{
  const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {{return safetensorBuffer.subarray(...tensorMetadata.data_offsets);}};
  {getTensorMetadata}
  const createEmptyBuf = (size) => {{
    return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
  }};
  const createUniformBuf = (size) => {{ return device.createBuffer({{size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}}) }}
  const infinityBuf = createUniformBuf(4);
  device.queue.writeBuffer(infinityBuf, 0, new Float32Array([Infinity]));
  const createWeightBuf = (size, data) => {{
    const buf = device.createBuffer(
      {{ size, usage: GPUBufferUsage.STORAGE{" | GPUBufferUsage.COPY_DST" if not save_weights else ", mappedAtCreation: true"} }});
    {"data.bytes = buf;" if not save_weights else "new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();"}
    return buf;
  }};
  const addComputePass = (commandEncoder, pipeline, bufs, workgroup, useInfinity) => {{
    const entries = [];
    if (useInfinity) entries.push({{ binding: 0, resource: {{ buffer: infinityBuf }} }});
    entries.push(...bufs.map((buffer, index) => ({{ binding: index + 1, resource: {{ buffer }} }})));
    const bindGroup = device.createBindGroup({{ layout: pipeline.getBindGroupLayout(0), entries }});
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...workgroup);
    passEncoder.end();
  }};
  {kernel_declarations}
  const setupNet = async ({"state_dict" if not save_weights else "safetensor"}) => {{
    {"const metadata = getTensorMetadata(safetensor);" if not not save_weights else ""}
    {j(symbolic_bufs + empty_bufs + weight_bufs + input_writer_bufs + output_reader_bufs, 2)}
    const kernels = [{",".join(kc.kernel_name for kc in ex.kernel_calls)}];
    const pipelines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{
      layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

    return [async ({",".join(args)}) => {{
      {j(input_validation, 3)}
      const commandEncoder = device.createCommandEncoder();
      {j(input_writers + kernel_calls + outbuf_copies, 3)}
      const gpuCommands = commandEncoder.finish();
      device.queue.submit([gpuCommands]);
      {j(output_readers, 3)}
      return {output_return};
    }}, device]
  }}
  const load = async(path) => {{return await fetch(path).then(x => x.arrayBuffer()).then(x => setupNet(new Uint8Array(x))).then(x => x[0]);}}
  return {{ load, setupNet }};
}})();
export default {model_name};
"""

  if js_outfile:
    with open(js_outfile, "w") as f: f.write(prg)
    safe_save(state_dict, f"{js_outfile}.safetensors")

  return prg, state_dict