# render the kernel graph for execution outside tinygrad
from tinygrad.tensor import Tensor
from tinygrad.device import Buffer
from tinygrad.ops import UOp, Variable, UPat, Ops, PatternMatcher
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_webgpu import JavaScriptRenderer
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule, BufferCopy, CompiledRunner
from tinygrad.nn.state import get_parameters
from typing import Callable, Sequence
import itertools

def is_partial_write(ast:UOp, i:int) -> bool:
  for u in ast.toposort():
    if u.op is Ops.STORE and (buf:=u.src[0]).arg == i and buf.dtype.size > u.src[1].arg.size: return True
  return False

# Common logic regardless of render target (e.g. WebGPU-JS, C)
class GraphRenderer(Renderer):
  def __init__(self, inputs:list[UOp], outputs:list[UOp]):
    self.graph, self.inputs, self.outputs = UOp.sink(*outputs), inputs, outputs

    # linearize the kernel graph
    self.schedule = memory_planner(create_schedule_with_vars(self.graph)[0])

    # render kernels, render buffer names
    # mark which buffers have state
    self.kernels, self.empty_bufs, self.state_bufs, seen_bufs, ctr = dict(), dict(), dict(), set(), itertools.count()
    for si, ei in lower_schedule(self.schedule):
      if isinstance(ei.prg, CompiledRunner): self.kernels[ei.prg.p.function_name] = ei.prg.p.src
      out_idxs = set([0]) if isinstance(ei.prg, BufferCopy) else ei.prg.p.outs if isinstance(ei.prg, CompiledRunner) else []
      for i, buf in enumerate(ei.bufs):
        if buf not in seen_bufs and (i not in out_idxs or is_partial_write(si.ast, i)): self.state_bufs[buf] = f"buf_{next(ctr)}"
        elif buf not in seen_bufs and i in out_idxs: self.empty_bufs[buf] = f"buf_{next(ctr)}"
        seen_bufs.add(buf)

class JSGraphRenderer(GraphRenderer, JavaScriptRenderer):
  def render_graph(self) -> str: pass

def create_graph_with_io(fxn:Callable, inputs:Sequence) -> tuple[list[UOp], list[UOp]]:

  # Inputs
  # dynamic_inputs are the input args to the exported function
  dynamic_inputs = [(x.realize().lazydata if isinstance(x, Tensor) else x) for x in inputs if isinstance(x, (Tensor, Variable))]
  # - Tensor input becomes typed array of closest type, e.g. dtypes.int32 -> Int32Array (JS) or int* (C)
  # - Variable input becomes signed integer type, e.g. number (JS), int (C)
  # TODO: should not all args be positional? instead use Object w/ names for Variable args, or for everything?
  
  # Outputs
  # exported function output is an array of typed arrays whose order matches below outputs
  outputs = [t.kernelize().lazydata for t in get_parameters(fxn(*inputs))]
  # TODO: assert no realizes happen within model, only kernelize is allowed
  # TODO: if only one output, return a typed array instead of array of typed arrays?
  assert len(outputs) > 0, "The input function must return at least one Tensor so that the kernel graph can be accessed."

  return dynamic_inputs, outputs

def export_webgpu(model:Callable, inputs:Sequence) -> tuple[str, dict[str, Tensor]]:
  """
  Renders the model's kernel graph into JavaScript and exports the model's state.
  """
  renderer = JSGraphRenderer(create_graph_with_io(model, inputs))
  state = renderer.
  return renderer.render_graph(), renderer