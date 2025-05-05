from tinygrad import Tensor
from tinygrad.ops import Ops
from tinygrad.runtime.ops_webgpu import js_init_device, js_alloc, js_copyin, js_copy, js_copyout
from tinygrad.renderer import ProgramSpec
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.engine.realize import CompiledRunner
from typing import Callable, Sequence

def declare_kernel(p:ProgramSpec) -> str: return f"const {p.function_name} = `{p.src.replace(p.function_name, 'main')}`;"

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

class WebGPUJSRenderer(GraphRenderer):
  def render_graph(self) -> str:
    prg: list[str] = js_init_device + safe_load_state_dict.split("\n") + buf_usages
    prg.append("const setupNet = async (safetensorPath) => {")

    # render I/O between host/WebGPU
    input_bufs, input_copyins, output_bufs, output_read_bufs, output_copies, ret_bufs, output_copyouts = [], [], [], [], [], [], []
    for i, uop in enumerate(self.inputs):
      if uop.base.op is Ops.BUFFER: # from Tensor input
        input_bufs.append(f"const input_{i} = {js_alloc(str(uop.base.buffer.nbytes), "empty")};")
        input_copyins.append(js_copyin(f"input_{i}", f"_input_{i}"))

      elif uop.op is Ops.BIND: # from symbolic variable input
        input_bufs.append(f"const input_{i} = {js_alloc("4", "uniform")};")
        input_copyins.append(js_copyin(f"input_{i}", f"new Int32Array([{uop.unbind()[0].arg[0]}])"))

    for i, uop in enumerate(self.outputs):
      output_bufs.append(f"const output_{i} = {js_alloc(str(uop.base.buffer.nbytes), "empty")};")
      output_read_bufs.append(f"const outputReader_{i} = {js_alloc(str(uop.base.buffer.nbytes), "copyout")};")
      output_copies.append(js_copy(f"output_{i}", f"outputReader_{i}", f"output_{i}.size"))
      ret_bufs.append(f"const ret_{i} = new Uint8Array(output_{i}.size);")
      output_copyouts.append(js_copyout(f"ret_{i}", f"outputReader_{i}"))

    # render WebGPU buffer setup
    empty_bufs = [f"const {name} = {js_alloc(str(buf.nbytes), "empty")};" for buf, name in self.empty_bufs.items()]
    state_dict_kv_pairs = [f'"{name}": {js_alloc(str(buf.nbytes), "state")}' for buf, name in self.state_bufs.items()]
    state_dict = f"const stateDict = {{ {",\n".join(state_dict_kv_pairs)} }};"
    load_state_dict = f"await safeLoadStateDict(stateDict, safeTensorPath);"

    # render WebGPU compute
    kernels = [declare_kernel(ei.prg.p) for ei in self.eis if isinstance(ei.prg, CompiledRunner)]

    # TODO: complete rendering
    return "\n".join(prg)

def export_webgpu(fxn:Callable, args:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Generates a kernel graph, renders the graph into JavaScript, and exports the graph's state as a `state_dict`.
  """
  renderer = WebGPUJSRenderer(fxn, args)
  state_dict = {v: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in renderer.state_bufs.items()}
  return renderer.render_graph(), state_dict