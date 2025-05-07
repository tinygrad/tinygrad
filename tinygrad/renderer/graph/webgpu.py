from tinygrad import Tensor, dtypes
from tinygrad.ops import Ops, UOp, UPat
from tinygrad.runtime.ops_webgpu import js_init_device, js_alloc, js_copyin, js_copy, js_copyout, js_create_pipeline
from tinygrad.renderer import ProgramSpec
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.engine.realize import CompiledRunner
from typing import Callable, Sequence
import math

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
    # representing Infinity with a runtime buffer is the most correct way known, see https://github.com/tinygrad/tinygrad/pull/10179
    infinity_buf = f'const infinity_buf = {js_alloc("4", "uniform")};'
    write_infinity = f'{js_copyin(f"infinity_buf", f"new Float32Array([Infinity])")}'

    # render WebGPU compute
    declare_kernels = list(set(declare_kernel(ei.prg.p) for ei in self.eis if isinstance(ei.prg, CompiledRunner)))
    invocation_order = f"const kernels = [{", ".join(ei.prg.p.function_name for ei in self.eis if isinstance(ei.prg, CompiledRunner))}];"
    # group pipelines by kernel name, so developers can see a kernel name on every added compute pass, making debugging easier
    pipelines = [f'const pending = kernels.reduce((r, k) => ((r[k] = r[k] || []).push({js_create_pipeline("k")})), r), {{}});',
"const pipelines = Object.fromEntries(await Promise.all(Object.entries(pending).map(async([k,ps]) => [k, await Promise.all(ps)])));"]
    pos_inf, neg_inf = UPat(Ops.CONST, arg=math.inf, name="x"), UPat(Ops.CONST, arg=-math.inf, name="x")
    def p_has_inf(p:ProgramSpec): return "true" if any(pos_inf.match(u, {}) or neg_inf.match(u, {}) for u in p.uops) else "false"
    uses_inf = f'const usesInf = [{", ".join(p_has_inf(ei.prg.p) for ei in self.eis if isinstance(ei.prg, CompiledRunner))}];'

    # TODO: complete rendering
    return "\n".join(prg)

def export_webgpu(fxn:Callable, args:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Generates a kernel graph, renders the graph into JavaScript, and exports the graph's state as a `state_dict`.
  """
  renderer = WebGPUJSRenderer(fxn, args)
  state_dict = {v: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in renderer.state_bufs.items()}
  return renderer.render_graph(), state_dict