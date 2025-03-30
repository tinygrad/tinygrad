from typing import cast
from tinygrad.helpers import merge_dicts
from tinygrad.engine.jit import GraphRunner, GraphException, CapturedJit
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.runtime.ops_webgpu import WebGPUProgram, execute_commands
from tinygrad.device import Buffer
from tinygrad.dtype import DType, dtypes
from tinygrad.ops import Variable
from tinygrad import Tensor

class WebGPUGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[Variable, int]):
    # TODO: capture this more cleanly?
    self._dev, self.timestamp_supported = (_prg:=jit_cache[0].prg._prg).dev, _prg.timestamp_supported

    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg._prg, WebGPUProgram) for ji in jit_cache): raise GraphException

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False) -> float|None:
    for (j,i),idx in self.input_replace.items(): self.jit_cache[j].bufs[i] = rawbufs[idx]
    wait = wait and self.timestamp_supported

    def callback(command_encoder, comp_pass_desc):
      for ji in self.jit_cache:
        _prg = cast(WebGPUProgram, (prg:=cast(CompiledRunner, ji.prg))._prg)
        vals = tuple(var_vals[k] for k in prg.p.vars)
        _prg.add_compute_pass(command_encoder, comp_pass_desc, *[b._buf for b in ji.bufs], global_size=prg.p.launch_dims(var_vals)[0], vals=vals)

    return execute_commands(self._dev, callback, wait)

def js_type(dtype: DType) -> str:
  return f"{'Uint' if dtype in dtypes.uints else 'Int' if (dtype in dtypes.sints or dtype == dtypes.bool) else 'Float'}{8*dtype.itemsize}Array"

def render_idx(idx:Variable|int): return idx.simplify().render() if isinstance(idx, Variable) else str(idx)

def render_js(cj: CapturedJit, in_bufs:dict[Buffer, int], in_vars:dict[Variable, int], weight_names:dict[Buffer, str],
              model_name="model", save_weights=True) -> str:
  names:dict[Buffer|Variable:str] = merge_dicts([cj.buf_names, {b:f"in_buf_{i}" for b,i in in_bufs.items()}, {v:f"sym_{i}" for v,i in in_vars.items()}])
  assert isinstance(cj.ret, list) and all(isinstance(t, Tensor) and t.lazydata.base.is_realized for t in cj.ret)
  out_bufs = cast(dict[Buffer, int], {t.lazydata.base.realized: i for i, t in enumerate(cj.ret)})
  names.update({b:f"out_buf_{i}" for b,i in out_bufs.items()})

  # Render model data
  empty_bufs = [f"const {names[b]} = createEmptyBuf({b.nbytes});" for b in list(in_bufs.keys()) + list(out_bufs.keys()) + list(cj.empty_bufs)]
  symbolic_bufs = [f"const {names[var]} = createUniformBuf({var.dtype.itemsize});" for var in in_vars.keys()]
  def map_wt(buf): return f"state_dict['{weight_names[buf]}']" if not save_weights else f"getTensorBuffer(safetensor,metadata['{weight_names[buf]}'])"
  weight_bufs = [f"const {names[buf]} = createWeightBuf({buf.nbytes}, {map_wt(buf)});" for buf in cj.weight_bufs]

  # TODO: condense this code; use one command encoder per graph runner
  # Render model transforms
  kernels: dict[str, str] = {}
  all_eis: list[ExecItem] = []
  rendered_calls, ctr = [], 0
  def render_call(ei: ExecItem, ctr) -> str:
    arg_names = ", ".join([names[buf] for buf in ei.bufs] + [names[var] for var in ei.prg.p.vars])
    return f"addComputePass(commandEncoder, pipelines[{ctr}], [{arg_names}], " + \
    f"[{', '.join(render_idx(x) for x in ei.prg.p.global_size)}], kernels[{ctr}].split('INFINITY').length > 2);"

  for ji in cj._jit_cache:
    if isinstance(ji.prg, WebGPUGraph):
      for graphed_ji in ji.prg.jit_cache:
        all_eis.append(graphed_ji)
        kernels[graphed_ji.prg.p.function_name] = graphed_ji.prg.p.src
        rendered_calls.append(render_call(graphed_ji, ctr))
        ctr += 1
    else:
      all_eis.append(ji)
      kernels[ji.prg.p.function_name] = ji.prg.p.src
      rendered_calls.append(render_call(ji, ctr))
      ctr += 1
  kernel_declarations = '\n\n'.join([f"const {name} = `{code.replace(name, 'main')}`;" for name, code in kernels.items()])

  # Render runtime-specific operations
  # TODO: validate symbolic var ranges against input args at runtime in JS?
  arg_idx: dict[Buffer|Variable: int] = merge_dicts([in_bufs, in_vars])
  args = [f"_input{i}" if isinstance(arg, Buffer) else f"{arg.expr}" for arg, i in arg_idx.items()]
  def check(arg:Buffer|Variable):
    return f"{args[in_bufs[arg]]} instanceof {js_type(arg.dtype)}" if isinstance(arg, Buffer) else f'typeof {args[in_vars[arg]]} === "number"'
  validation = [f"if (!({check(arg)})) {{ throw new Error(`arg {i} type: ${{typeof {args[i]}}} is not as expected`) }}" for arg, i in arg_idx.items()]

  input_writer_bufs = [f"""const gpuWriteBuffer{i} = device.createBuffer({{size:{names[buf]}.size,
                usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST}});""" for buf, i in in_bufs.items()]
  input_writers = [f"""
      device.queue.writeBuffer(gpuWriteBuffer{i}, 0, {f"_input{i}" if isinstance(var, Buffer) else f"new {js_type(var.dtype)}([{var.expr}])"});
      commandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, {names[var]}, 0, gpuWriteBuffer{i}.size);""" for var, i in arg_idx.items()]

  output_reader_bufs = [f"""const gpuReadBuffer{i} = device.createBuffer({{size:{names[buf]}.size,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}});""" for buf, i in out_bufs.items()]
  output_readers = [f"""await gpuReadBuffer{i}.mapAsync(GPUMapMode.READ);
      const resultBuffer{i} = new {js_type(buf.dtype)}(gpuReadBuffer{i}.size/{buf.dtype.itemsize});
      resultBuffer{i}.set(new {js_type(buf.dtype)}(gpuReadBuffer{i}.getMappedRange()));
      gpuReadBuffer{i}.unmap();""" for buf, i in out_bufs.items()]

  outbuf_copies = [f"commandEncoder.copyBufferToBuffer({names[buf]}, 0, gpuReadBuffer{i}, 0, {buf.nbytes});" for buf, i in out_bufs.items()]
  output_return = '[{}]'.format(",".join([f'resultBuffer{i}' for i in range(len(out_bufs))]))

  getTensorMetadata = """\n  const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(
      ([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
  };\n""" if save_weights else ""
  max_buf_nbytes = max([buf.nbytes for buf in cj.weight_bufs.union(cj.empty_bufs)] + [4])

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
    const kernels = [{",".join(ei.prg.p.function_name for ei in all_eis)}];
    const pipelines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{
      layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

    return [async ({",".join(args)}) => {{
      {j(validation, 3)}
      const commandEncoder = device.createCommandEncoder();
      {j(input_writers + rendered_calls + outbuf_copies, 3)}
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
  return prg