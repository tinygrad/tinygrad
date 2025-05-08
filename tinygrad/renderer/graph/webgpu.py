from tinygrad import Tensor
from tinygrad.ops import Ops, Variable
from tinygrad.runtime.ops_webgpu import js_init_device, js_init_encoder, js_alloc, js_copyin, js_copy, js_copyout, js_create_layout, \
  js_create_pipeline, js_create_bind_group, js_begin_compute_pass
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import merge_dicts
from typing import Callable, Sequence

safe_load_state_dict = f"""const safeLoadStateDict = async (modelStateDict, safetensorPath) => {{
  const safetensorBuffer = await fetch(safetensorPath).then(x => x.arrayBuffer()).then(x => new Uint8Array(x));
  const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
  const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
  for (const [key, info] of Object.entries(metadata)) {{
    if (key === "__metadata__") continue;
    const src = safetensorBuffer.subarray(8 + metadataLength + info.data_offsets[0], 8 + metadataLength + info.data_offsets[1]);
    {js_copyin("modelStateDict[key]", "src")}
  }}
}};\n"""

empty = (state:="GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST") + " | GPUBufferUsage.COPY_SRC"
uniform, copyout = "GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST", "GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ"
buf_usages = [f"const {label} = {usage};" for label, usage in zip(("empty", "state", "uniform", "copyout"), (empty, state, uniform, copyout))]

def indent(strings:list[str], indents:int) -> list[str]: return [indents*"  " + s for s in strings]

class WebGPUJSRenderer(GraphRenderer):
  def render_graph(self) -> str:
    # render I/O between host/WebGPU
    kernel_bufs = merge_dicts(self.empty_bufs, self.state_bufs)
    arg_names, input_bufs, input_copyins, output_bufs, output_read_bufs, output_copies, ret_bufs, output_copyouts = [], [], [], [], [], [], [], []
    # render inputs
    for i, uop in enumerate(self.inputs):
      if uop.base.op is Ops.BUFFER: # from Tensor input
        arg_names.append(f"bufArg_{i}")
        input_bufs.append(f"const inputBuf_{i} = {js_alloc(str(uop.buffer.nbytes), "empty")};")
        input_copyins.append(js_copyin(f"inputBuf_{i}", f"bufArg_{i}"))
        kernel_bufs[uop.buffer] = f"inputBuf_{i}"
      elif uop.op is Ops.BIND: # from symbolic variable input
        arg_names.append(f"intArg_{i}")
        input_bufs.append(f"const inputBuf_{i} = {js_alloc("4", "uniform")};")
        input_copyins.append(js_copyin(f"inputBuf_{i}", f"new Int32Array([{f"intArg_{i}"}])"))
        kernel_bufs[uop.unbind()[0]] = f"inputBuf_{i}"

    # render outputs
    command_encoder = [f"const commandEncoder = {js_init_encoder};"]
    for i, uop in enumerate(self.outputs):
      output_bufs.append(f"const output_{i} = {js_alloc(str(uop.base.buffer.nbytes), "empty")};")
      output_read_bufs.append(f"const outputReader_{i} = {js_alloc(str(uop.base.buffer.nbytes), "copyout")};")
      output_copies.append(js_copy("commandEncoder", f"output_{i}", f"outputReader_{i}", f"output_{i}.size"))
      ret_bufs.append(f"const ret_{i} = new Uint8Array(output_{i}.size);")
      output_copyouts.append(js_copyout(f"ret_{i}", f"outputReader_{i}"))
    args, ret = ", ".join(arg_names), ", ".join(ret_bufs)

    # render setup of buffers used only by WebGPU kernels
    empty_bufs = [f"const {name} = {js_alloc(str(buf.nbytes), "empty")};" for buf, name in self.empty_bufs.items()]
    state_dict_kv_pairs = [f'"{name}": {js_alloc(str(buf.nbytes), "state")}' for buf, name in self.state_bufs.items()]
    state_dict = ["const stateDict = {", indent(state_dict_kv_pairs, 1), "};"]
    # representing Infinity with a runtime buffer is the most correct way known, see https://github.com/tinygrad/tinygrad/pull/10179
    declare_infinity = f'const infinityBuf = {js_alloc("4", "uniform")};'
    write_infinity = f'{js_copyin(f"infinityBuf", f"new Float32Array([Infinity])")}'

    # render WebGPU compute
    add_compute_pass = ["const addComputePass = (pipeline, buffers, workgroupDims, commandEncoder, layout) => {",
      "  const entries = [...[infinityBuf].concat(buffers).map((buffer, index) => ({ binding: index, resource: { buffer } })))];",
      f'  const bindGroup = {js_create_bind_group("layout", "entries")};', 
      "  \n".join(js_begin_compute_pass("commandEncoder", "pipeline", "bindGroup", "workgroupDims")),
    "};\n"]

    layouts, kernels, kernel_name_sequence, compute_passes = [], {}, [], []
    for i, (ei, p) in enumerate((ei, ei.prg.p) for ei in self.eis if isinstance(ei.prg, CompiledRunner)):
      # first buf in every kernel is infinity_buf, a uniform buffer
      layouts.append(js_create_layout(["uniform"] + ["storage"] * len(ei.bufs) + ["uniform"] * len(p.vars)))
      kernels[p.function_name] = p.src.replace(p.function_name, "main")
      # kernel_name_sequence becomes pipelines: a JS array of {p.function_name: GPUComputePipeline}
      kernel_name_sequence.append(f'"{p.function_name}"') 
      buf_names = ", ".join(kernel_bufs[arg] for arg in ei.bufs + p.vars)
      global_size = ', '.join(idx.simplify().render() if isinstance(idx, Variable) else str(idx) for idx in p.global_size)
      # deliberately display p.function_name in every addComputePass for easier debugging/understanding
      compute_passes.append(f'addComputePass(pipelines[{i}]["{p.function_name}"], [{buf_names}], [{global_size}], commandEncoder, layouts[{i}]);')
    
    layouts = [f'const layouts = [{", ".join(layouts)}];']
    kernels = ["const kernels = {"] + indent([f'"{k}": {v}' for k,v in kernels.items()], 1) + ["};\n"]
    make_pipelines = [f'const kernelNameSequence = [{", ".join(kernel_name_sequence)}];',
      f'let pipelines = await Promise.all(kernelNameSequence.map((name, i) => {js_create_pipeline("layouts[i]", "kernels[name]")}));',
      'pipelines = pipelines.map((pipeline, i) => { kernelNameSequence[i] : pipeline });']
    trigger_gpu = ["const gpuCommands = commandEncoder.finish();", "device.queue.submit([gpuCommands]);"]

    prg: list[str] = js_init_device + safe_load_state_dict.split("\n") + buf_usages + kernels
    prg += ["const createGraph = async () => {"]
    prg += indent(input_bufs + empty_bufs + state_dict + output_bufs + output_read_bufs + ret_bufs + [declare_infinity, write_infinity], 1)
    prg += indent(add_compute_pass + layouts + kernels + make_pipelines, 1)
    prg += indent([f"const run = async ({args}) {{"], 1)
    prg += indent(command_encoder + input_copyins + compute_passes + output_copies + trigger_gpu + output_copyouts + [f"return [{ret}];"], 2)
    prg += indent(["};", "return { stateDict, run };"], 1)
    prg += ["};"]
    prg += ["export { createGraph, safeLoadStateDict, device };"]
    return "\n".join(prg)

def export_webgpu(fxn:Callable, args:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Generates a kernel graph, renders the graph into JavaScript, and exports the graph's state as a `state_dict`.
  """
  renderer = WebGPUJSRenderer(fxn, args)
  state_dict = {v: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in renderer.state_bufs.items()}
  return renderer.render_graph(), state_dict