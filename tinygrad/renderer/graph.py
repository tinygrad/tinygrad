# render the kernel graph for execution outside tinygrad
from tinygrad import Tensor
from tinygrad.device import Buffer
from tinygrad.ops import UOp, Variable, Ops
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_webgpu import JavaScriptRenderer
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule, BufferCopy, CompiledRunner
from tinygrad.nn.state import get_parameters
from typing import Callable, Sequence
import itertools

def is_partial_write(ast:UOp, buf_idx:int) -> bool:
  for u in ast.toposort():
    if u.op is Ops.STORE and (buf:=u.src[0]).arg == buf_idx and buf.dtype.size > u.src[1].arg.size: return True
  return False

# Common logic regardless of render target (e.g. WebGPU-JS, C)
class GraphRenderer(Renderer):
  def __init__(self, inputs:list[UOp], outputs:list[UOp]):
    # linearize the kernel graph
    schedule, _, becomes_map = create_schedule_with_vars(UOp.sink(*outputs))
    outputs = [becomes_map[uop.base] for uop in outputs]
    self.inputs, self.outputs, schedule = inputs, outputs, memory_planner(schedule)

    # render kernels, render buffer names
    # mark which buffers have state
    self.kernels: dict[str, str] = dict()
    self.empty_bufs: dict[Buffer, str] = dict()
    self.state_bufs: dict[Buffer, str] = dict()
    seen_bufs, ctr = set([i.base.buffer for i in inputs if i.base.op is Ops.BUFFER]), itertools.count()
    for si, ei in lower_schedule(schedule):
      if isinstance(ei.prg, CompiledRunner): self.kernels[ei.prg.p.function_name] = ei.prg.p.src
      out_idxs = set([0]) if isinstance(ei.prg, BufferCopy) else ei.prg.p.outs if isinstance(ei.prg, CompiledRunner) else []
      for i, buf in enumerate(ei.bufs):
        if buf not in seen_bufs and (i not in out_idxs or is_partial_write(si.ast, i)): self.state_bufs[buf] = f"buf_{next(ctr)}"
        elif buf not in seen_bufs and i in out_idxs: self.empty_bufs[buf] = f"buf_{next(ctr)}"
        seen_bufs.add(buf)

class JSGraphRenderer(GraphRenderer, JavaScriptRenderer):
  def render_graph(self) -> str: pass

def create_graph_with_io(fxn:Callable, args:Sequence) -> tuple[list[UOp], list[UOp]]:
  # TODO: assert no realizes happen within model, only kernelize is allowed
  inputs = [(x.realize().lazydata if isinstance(x, Tensor) else x) for x in args if isinstance(x, (Tensor, Variable))]
  outputs = [t.kernelize().lazydata for t in get_parameters(fxn(*args))]
  assert len(outputs) > 0, "The function argument must return at least one Tensor so that the kernel graph can be accessed."

  return inputs, outputs

def export_webgpu(fxn:Callable, args:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Generates a kernel graph, renders the graph into JavaScript, and exports the graph's state as a `state_dict`.
  """
  renderer = JSGraphRenderer(*create_graph_with_io(fxn, args))
  state_dict = {v: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in renderer.state_bufs.items()}
  return renderer.render_graph(), state_dict