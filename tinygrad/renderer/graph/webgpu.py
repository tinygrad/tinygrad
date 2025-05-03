from tinygrad import Tensor
from tinygrad.runtime import ops_webgpu
from tinygrad.renderer.graph import GraphRenderer, create_graph_with_io
from tinygrad.engine.realize import ExecItem, CompiledRunner
from typing import Callable, Sequence

# helper functions that make the output JavaScript easier to read
helpers = ["const createEmptyBuf = (size) => {",
  f'  return {ops_webgpu.js_alloc("size", "GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST")};'
  "};\n",

  "const createWeightBuf = (size, data) => {",
  f'  const buf = {ops_webgpu.js_alloc("size", "GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST")};',
  f"  {ops_webgpu.js_copyin("buf", "data")}",
  "};\n"]

class WebGPUJSRenderer(GraphRenderer):
  def declare_kernel(self, ei:ExecItem) -> str:
    return f"const {(fn:=ei.prg.p.function_name)} = `{ei.prg.p.src.replace(fn, 'main')}`;" if isinstance(ei.prg, CompiledRunner) else ""

  def render_graph(self) -> str:
    prg: list[str] = ops_webgpu.js_init_device + helpers
    kernels = [self.declare_kernel(ei) for ei in self.eis if isinstance(ei.prg, CompiledRunner)]
    bufs = [f"const {name} = createEmptyBuf({buf.nbytes})" for buf,name in self.empty_bufs.items()]
    return "\n".join(prg)

def export_webgpu(fxn:Callable, args:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Generates a kernel graph, renders the graph into JavaScript, and exports the graph's state as a `state_dict`.
  """
  renderer = WebGPUJSRenderer(*create_graph_with_io(fxn, args))
  state_dict = {v: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in renderer.state_bufs.items()}
  return renderer.render_graph(), state_dict