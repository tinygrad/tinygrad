from tinygrad import Tensor
from tinygrad.runtime import ops_webgpu
from tinygrad.renderer import ProgramSpec
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.engine.realize import CompiledRunner
from typing import Callable, Sequence

def declare_kernel(p:ProgramSpec) -> str: return f"const {p.function_name} = `{p.src.replace(p.function_name, 'main')}`;"

safe_load_state_dict = f"""const safeLoadStateDict = async (modelStateDict, safeTensorPath) => {{
  const safetensorBuffer = await fetch(safetensorPath).then(x => x.arrayBuffer()).then(x => new Uint8Array(x));
  const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
  const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
  for (const [key, info] of Object.entries(metadata)) {{
    if (key === "__metadata__") continue;
    const src = safetensorBuffer.subarray(8 + metadataLength + info.data_offsets[0], 8 + metadataLength + info.data_offsets[1]);
    {ops_webgpu.js_copyin("modelStateDict[key]", "src")}
  }}
}};"""
empty_buf_flags = (state_buf_flags:="GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST") + "GPUBufferUsage.COPY_SRC"

class WebGPUJSRenderer(GraphRenderer):
  def render_graph(self) -> str:
    prg: list[str] = ops_webgpu.js_init_device + safe_load_state_dict.split("\n")
    kernel_declares = [declare_kernel(ei.prg.p) for ei in self.eis if isinstance(ei.prg, CompiledRunner)]
    empty_buf_declares = [f"const {name} = {ops_webgpu.js_alloc(str(buf.nbytes), empty_buf_flags)};" for buf, name in self.empty_bufs.items()]
    state_dict_kv_pairs = [f'"{name}": {ops_webgpu.js_alloc(str(buf.nbytes), state_buf_flags)}' for buf, name in self.state_bufs.items()]
    state_dict_declare = f"const stateDict = {{ {",\n".join(state_dict_kv_pairs)} }};"
    state_dict_load = f"await safeLoadStateDict(stateDict, safeTensorPath);"
    # TODO: complete rendering
    return "\n".join(prg)

def export_webgpu(fxn:Callable, args:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Generates a kernel graph, renders the graph into JavaScript, and exports the graph's state as a `state_dict`.
  """
  renderer = WebGPUJSRenderer(fxn, args)
  state_dict = {v: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in renderer.state_bufs.items()}
  return renderer.render_graph(), state_dict