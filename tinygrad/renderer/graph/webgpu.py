from tinygrad import dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops, Variable
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.engine.realize import CompiledRunner
from typing import cast

init_device = """if (!navigator.gpu) throw new Error("WebGPU not supported.");
const adapter = await navigator.gpu.requestAdapter();
const { maxStorageBufferBindingSize, maxBufferSize, maxComputeInvocationsPerWorkgroup } = adapter.limits;
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [], powerPreference: "high-performance",
  requiredLimits: { maxStorageBufferBindingSize, maxBufferSize, maxComputeInvocationsPerWorkgroup }
});\n""".split("\n")

empty = (state:="GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST") + " | GPUBufferUsage.COPY_SRC"
uniform, map_read = "GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST", "GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ"
buf_usages = [f"const {label} = {usage};" for label, usage in zip(("empty", "state", "uniform", "map_read"), (empty, state, uniform, map_read))]

# TODO: handle 4-byte alignment
#if src.nbytes % 4: pad_src = f"const padded = new Uint8Array({src}.length + (4 - {src}.length % 4) % 4); padded.set({src});"
def alloc(size:str, usage:str) -> str: return f"device.createBuffer({{size: {size}, usage: {usage}}})"

def copyin(dest:str, src:str) -> str: return f"device.queue.writeBuffer({dest}, 0, {src});"

pipes = """let pipelines = await Promise.all(kernelSequence.map((name, i) => device.createComputePipelineAsync({layout: device.createPipelineLayout(
  { bindGroupLayouts: [layouts[i]]}), compute: {module: device.createShaderModule({code: kernels[name]}), entryPoint: "main" }\n})));
pipelines = pipelines.map((pipeline, i) => { return {[kernelSequence[i]] : pipeline} });""".split("\n")

add_compute_pass = """const addComputePass = (commandEncoder, idx, kernelName, buffers, workgroupDims) => {
  const entries = [...[infinityBuf].concat(buffers).map((buffer, index) => ({ binding: index, resource: { buffer } }))];
  const bindGroup = device.createBindGroup({ layout: layouts[idx], entries });
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipelines[idx][kernelName]);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroupDims);
  passEncoder.end();\n}\n""".split("\n")

trigger_gpu = ["const gpuCommands = commandEncoder.finish();", "device.queue.submit([gpuCommands]);"]

safe_load_state_dict = f"""const load = async (safetensorPath) => {{
  const safetensorBuffer = await fetch(safetensorPath).then(x => x.arrayBuffer()).then(x => new Uint8Array(x));
  const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
  const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
  for (const [key, info] of Object.entries(metadata)) {{
    if (key === "__metadata__" || !(key in stateDict)) continue;
    const src = safetensorBuffer.subarray(8 + metadataLength + info.data_offsets[0], 8 + metadataLength + info.data_offsets[1]);
    {copyin("stateDict[key]", "src")}
  }}
}};\n""".split("\n")

def indent(strings:list[str], indents:int) -> list[str]: return [indents*"  " + s for s in strings]

class WebGPUJSRenderer(GraphRenderer):
  def render_graph(self) -> str:
    # render I/O between host/WebGPU
    arg_names, copyins, sym_bufs, validators, copyout_bufs, output_copies, ret_bufs, copyouts, ret_names = [], [], {}, [], [], [], [], [], []
    # render inputs
    for i, uop in enumerate(self.inputs):
      if uop.base.op is Ops.BUFFER: # from Tensor input
        arg_names.append(f"bufArg_{i}")
        copyins.append(copyin((dest_buf:=self.bufs[cast(Buffer, uop.base.buffer)]), f"bufArg_{i}"))
        validators.append(f"if (bufArg_{i}.byteLength !== {dest_buf}.size) throw new Error(`bufArg_{i} does not have expected number of bytes`);")
      elif uop.op is Ops.BIND: # from symbolic variable input
        arg_names.append(var_name:=uop.unbind()[0].simplify().render())
        sym_bufs[uop.unbind()[0]] = f"input_{i}"
        copyins.append(copyin(f"input_{i}", f'new Int32Array([{var_name}])'))

    # render outputs
    command_encoder = ["const commandEncoder = device.createCommandEncoder();"]
    for i, uop in enumerate(self.outputs):
      copyout_bufs.append(f'const gpuReadBuffer{i} = {alloc(str((buf:=cast(Buffer, uop.base.buffer)).nbytes), "map_read")};')
      output_copies.append(f"commandEncoder.copyBufferToBuffer({(out_buf:=self.bufs[buf])}, 0, gpuReadBuffer{i}, 0, {out_buf}.size);")
      array_type = f"{'Uint' if (dt:=uop.dtype) in dtypes.uints else 'Int' if dt in (dtypes.sints+(dtypes.bool,)) else 'Float'}{8*dt.itemsize}Array"
      ret_bufs.append(f"const resultBuffer{i} = new {array_type}(gpuReadBuffer{i}.size / {dt.itemsize});")
      copyouts += [f"await gpuReadBuffer{i}.mapAsync(GPUMapMode.READ);",
        f"resultBuffer{i}.set(new resultBuffer{i}.constructor(gpuReadBuffer{i}.getMappedRange()));",
        f'gpuReadBuffer{i}.unmap();']
      ret_names.append(f"resultBuffer{i}")
    args, ret = ", ".join(arg_names), ", ".join(ret_names)

    # render setup of WebGPU buffers
    state_dict, empty_buf_allocs, state_buf_allocs = ["const stateDict = {};"], [], []
    for buf, name in self.bufs.items():
      if buf in self.state_bufs: state_buf_allocs.append(f'const {name} = stateDict["{self.state_bufs[buf]}"] = {alloc(str(buf.nbytes), "state")};')
      else: empty_buf_allocs.append(f'const {name} = {alloc(str(buf.nbytes), "empty")};')
    sym_uniforms = [f'const {name} = {alloc("4", "uniform")};' for name in sym_bufs.values()]
    # representing Infinity with a runtime buffer is the most correct way known, see https://github.com/tinygrad/tinygrad/pull/10179
    create_infinity = [f'const infinityBuf = {alloc("4", "uniform")};']
    create_infinity += [f'{copyin("infinityBuf", "new Float32Array([Infinity])")}']

    # render WebGPU compute
    layouts, kernels, kernel_sequence, compute_passes = [], {}, [], []
    for i, (ei, p) in enumerate((ei, ei.prg.p) for ei in self.eis if isinstance(ei.prg, CompiledRunner)):
      # first buf in every kernel is infinityBuf, a uniform buffer
      layouts.append(f"""device.createBindGroupLayout({{entries: [{",".join(f"{{binding:{i}, visibility:GPUShaderStage.COMPUTE, buffer:{{type:{t}}}}}"
                     for i, t in enumerate(['"uniform"'] + ['"storage"'] * len(ei.bufs) + ['"uniform"'] * len(p.vars)))}]}})""")
      kernels[p.function_name] = p.src.replace(p.function_name, "main")
      # kernel_sequence becomes pipelines: a JS array of {p.function_name: GPUComputePipeline}
      kernel_sequence.append(f'"{p.function_name}"')
      for arg in ei.bufs + p.vars: assert arg is not None
      buf_names = ", ".join(self.bufs[arg] if isinstance(arg, Buffer) else sym_bufs[cast(Variable, arg)] for arg in ei.bufs + p.vars)
      assert p.global_size is not None and len(p.global_size) == 3
      global_size = ', '.join(idx.simplify().render() if isinstance(idx, Variable) else str(idx) for idx in p.global_size)
      # deliberately display p.function_name in every addComputePass for easier debugging/understanding
      compute_passes.append(f'addComputePass(commandEncoder, {i}, "{p.function_name}", [{buf_names}], [{global_size}]);')

    kernel_sequence = [f'const kernelSequence = [{", ".join(kernel_sequence)}];']
    layouts = [f'const layouts = [{", ".join(layouts)}];']
    kernel_obj = ["const kernels = {"] + indent([f'"{k}": `{v}`,' for k,v in kernels.items()], 1) + ["};\n"]

    prg: list[str] = buf_usages + kernel_obj
    prg += ["const createGraph = async () => {"]
    prg += indent(init_device + sym_uniforms + empty_buf_allocs + state_dict + state_buf_allocs + copyout_bufs + create_infinity, 1)
    prg += indent(kernel_sequence + layouts + pipes + add_compute_pass + safe_load_state_dict, 1)
    prg += indent([f"const run = async ({args}) => {{"], 1)
    prg += indent(validators + command_encoder + copyins + compute_passes + output_copies + trigger_gpu + ret_bufs + copyouts +[f"return [{ret}];"],2)
    prg += indent(["};", "return { device, stateDict, load, run };"], 1)
    prg += ["};"]
    prg += ["export default createGraph;"]
    return "\n".join(prg)