# render the kernel graph for execution outside tinygrad
from tinygrad import Tensor
from tinygrad.device import Buffer
from tinygrad.ops import UOp, Variable, Ops
from tinygrad.renderer import Renderer
from tinygrad.nn.state import get_parameters
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule, CompiledRunner, ExecItem
from typing import Callable, Sequence
import itertools

def is_partial_write(ast:UOp, buf_idx:int) -> bool:
  for u in ast.toposort():
    if u.op is Ops.STORE and (buf:=u.src[0]).arg == buf_idx and buf.dtype.size > u.src[1].arg.size: return True
  return False

# Common logic regardless of render target (e.g. JavaScript, C)
class GraphRenderer(Renderer):
  def __init__(self, fxn:Callable, args:Sequence):

    # Ensure random seeds are on-device
    Tensor.randn(1).realize()

    # construct the kernel graph
    # designate the input and output nodes
    inputs = [(x.realize().lazydata if isinstance(x, Tensor) else x) for x in args if isinstance(x, (Tensor, Variable))]
    # TODO: assert no realizes happen here in function call, only kernelize is allowed
    outputs = [t.kernelize().lazydata for t in get_parameters(fxn(*args))]
    assert len(outputs) > 0, "The function argument must return at least one Tensor so that the kernel graph can be accessed."

    # linearize the kernel graph
    schedule, _, becomes_map = create_schedule_with_vars(UOp.sink(*outputs))
    outputs = [becomes_map[uop.base] for uop in outputs]
    self.inputs, self.outputs, schedule = inputs, outputs, memory_planner(schedule)

    # render kernels, render buffer names
    # mark which buffers have state
    self.eis: list[ExecItem] = []
    self.empty_bufs: dict[Buffer, str] = dict()
    self.state_bufs: dict[Buffer, str] = dict()
    seen, ctr = set([i.base.buffer for i in inputs if i.base.op is Ops.BUFFER]), itertools.count()
    for si, ei in lower_schedule(schedule):
      assert isinstance(ei.prg, CompiledRunner), "BufferCopy not yet supported, ensure all Tensors are on the same device"
      self.eis.append(ei)
      for i, buf in enumerate(ei.bufs):
        if buf not in seen and (i not in ei.prg.p.outs or i in ei.prg.p.ins or is_partial_write(si.ast, i)): self.state_bufs[buf] = f"buf_{next(ctr)}"
        elif buf not in seen: self.empty_bufs[buf] = f"buf_{next(ctr)}"
        seen.add(buf)

  def render_graph(self) -> str: raise NotImplementedError("Implement a language-specific GraphRenderer")