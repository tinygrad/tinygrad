from tinygrad import dtypes
from tinygrad.device import Buffer
from tinygrad.ops import Ops, Variable
from tinygrad.runtime.ops_webgpu.js import init_device, init_encoder, alloc, copyin, copy, copyout, create_layout, create_pipeline, \
  create_bind_group, begin_compute_pass
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import merge_dicts
from typing import cast

safe_load_state_dict = f"""const safeLoadStateDict = async (modelStateDict, safetensorPath) => {{
  const safetensorBuffer = await fetch(safetensorPath).then(x => x.arrayBuffer()).then(x => new Uint8Array(x));
  const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
  const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
  for (const [key, info] of Object.entries(metadata)) {{
    if (key === "__metadata__") continue;
    const src = safetensorBuffer.subarray(8 + metadataLength + info.data_offsets[0], 8 + metadataLength + info.data_offsets[1]);
    {copyin("modelStateDict[key]", "src")}
  }}
}};\n"""

empty = (state:="GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST") + " | GPUBufferUsage.COPY_SRC"
uniform, map_read = "GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST", "GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ"
buf_usages = [f"const {label} = {usage};" for label, usage in zip(("empty", "state", "uniform", "map_read"), (empty, state, uniform, map_read))]

def indent(strings:list[str], indents:int) -> list[str]: return [indents*"  " + s for s in strings]

class WebGPUJSRenderer(GraphRenderer):
  def render_graph(self) -> str:
    # render I/O between host/WebGPU
    arg_names, copyins, sym_var_bufs, validators, copyout_bufs, output_copies, ret_bufs, copyouts, ret_names = [], [], [], [], [], [], [], [], []
    kernel_bufs = cast(dict[Buffer|Variable, str], merge_dicts([self.empty_bufs, {k: f'stateDict["{v}"]' for k,v in self.state_bufs.items()}]))
    # render inputs
    for i, uop in enumerate(self.inputs):
      if uop.base.op is Ops.BUFFER: # from Tensor input
        arg_names.append(f"bufArg_{i}")
        copyins.append(copyin((buf:=kernel_bufs[uop.base.buffer]), f"bufArg_{i}"))
        validators.append(f"if (bufArg_{i}.byteLength !== {buf}.size) throw new Error(`byteLength mismatch between argument and WebGPU buffer`);")
      elif uop.op is Ops.BIND: # from symbolic variable input
        arg_names.append(f"intArg_{i}")
        kernel_bufs[uop.unbind()[0]] = f"input_{i}"
        sym_var_bufs.append(f'const input_{i} = {alloc("4", "uniform")};')
        copyins.append(copyin(f"input_{i}", f'new Int32Array([{f"intArg_{i}"}])'))

    # render outputs
    command_encoder = [f"const commandEncoder = {init_encoder};"]
    for i, uop in enumerate(self.outputs):
      copyout_bufs.append(f'const gpuReadBuffer{i} = {alloc(str(uop.base.buffer.nbytes), "map_read")};')
      output_copies.append(copy("commandEncoder", (out_buf := kernel_bufs[uop.base.buffer]), f"gpuReadBuffer{i}", f"{out_buf}.size"))
      array_type = f"{'Uint' if (dt:=uop.dtype) in dtypes.uints else 'Int' if dt in (dtypes.sints+(dtypes.bool,)) else 'Float'}{8*dt.itemsize}Array"
      ret_bufs.append(f"const resultBuffer{i} = new {array_type}(gpuReadBuffer{i}.size / {dt.itemsize});")
      copyouts += copyout(f"resultBuffer{i}", f"gpuReadBuffer{i}")
      ret_names.append(f"resultBuffer{i}")
    args, ret = ", ".join(arg_names), ", ".join(ret_names)

    # render setup of WebGPU buffers
    empty_bufs = [f'const {name} = {alloc(str(buf.nbytes), "empty")};' for buf, name in self.empty_bufs.items()]
    state_dict_kv_pairs = [f'"{name}": {alloc(str(buf.nbytes), "state")},' for buf, name in self.state_bufs.items()]
    state_dict = ["const stateDict = {"] + indent(state_dict_kv_pairs, 1) + ["};"]
    # representing Infinity with a runtime buffer is the most correct way known, see https://github.com/tinygrad/tinygrad/pull/10179
    declare_infinity = f'const infinityBuf = {alloc("4", "uniform")};'
    write_infinity = f'{copyin("infinityBuf", "new Float32Array([Infinity])")}'

    # render WebGPU compute
    add_compute_pass = ["const addComputePass = (pipeline, buffers, workgroupDims, commandEncoder, layout) => {",
      "  const entries = [...[infinityBuf].concat(buffers).map((buffer, index) => ({ binding: index, resource: { buffer } }))];",
      f'  const bindGroup = {create_bind_group("layout", "entries")};',
      "  \n".join(begin_compute_pass("commandEncoder", "pipeline", "bindGroup", "workgroupDims")),
    "};\n"]

    layouts, kernels, kernel_name_sequence, compute_passes = [], {}, [], []
    for i, (ei, p) in enumerate((ei, ei.prg.p) for ei in self.eis if isinstance(ei.prg, CompiledRunner)):
      # first buf in every kernel is infinityBuf, a uniform buffer
      layouts.append(create_layout(['"uniform"'] + ['"storage"'] * len(ei.bufs) + ['"uniform"'] * len(p.vars)))
      kernels[p.function_name] = p.src.replace(p.function_name, "main")
      # kernel_name_sequence becomes pipelines: a JS array of {p.function_name: GPUComputePipeline}
      kernel_name_sequence.append(f'"{p.function_name}"')
      buf_names = ", ".join(kernel_bufs[cast(Buffer|Variable, arg)] for arg in ei.bufs + p.vars)
      assert p.global_size is not None and len(p.global_size) == 3
      global_size = ', '.join(idx.simplify().render() if isinstance(idx, Variable) else str(idx) for idx in p.global_size)
      # deliberately display p.function_name in every addComputePass for easier debugging/understanding
      compute_passes.append(f'addComputePass(pipelines[{i}]["{p.function_name}"], [{buf_names}], [{global_size}], commandEncoder, layouts[{i}]);')

    layouts = [f'const layouts = [{", ".join(layouts)}];']
    kernel_obj = ["const kernels = {"] + indent([f'"{k}": `{v}`,' for k,v in kernels.items()], 1) + ["};\n"]
    make_pipelines = [f'const kernelNameSequence = [{", ".join(kernel_name_sequence)}];',
      f'let pipelines = await Promise.all(kernelNameSequence.map((name, i) => {create_pipeline("layouts[i]", "kernels[name]")}));',
      'pipelines = pipelines.map((pipeline, i) => { return {[kernelNameSequence[i]] : pipeline} });']
    trigger_gpu = ["const gpuCommands = commandEncoder.finish();", "device.queue.submit([gpuCommands]);"]
    load = ["const load = async (fn) => { await safeLoadStateDict(stateDict, fn); };"]

    prg: list[str] = init_device + safe_load_state_dict.split("\n") + buf_usages + kernel_obj
    prg += ["const createGraph = async () => {"]
    prg += indent(sym_var_bufs + empty_bufs + state_dict + copyout_bufs + ret_bufs + [declare_infinity, write_infinity], 1)
    prg += indent(add_compute_pass + layouts + make_pipelines + load, 1)
    prg += indent([f"const run = async ({args}) => {{"], 1)
    prg += indent(validators + command_encoder + copyins + compute_passes + output_copies + trigger_gpu + copyouts + [f"return [{ret}];"], 2)
    prg += indent(["};", "return { device, stateDict, load, run };"], 1)
    prg += ["};"]
    prg += ["export default createGraph;"]
    return "\n".join(prg)