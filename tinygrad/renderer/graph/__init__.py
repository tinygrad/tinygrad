# render the kernel graph for execution outside tinygrad
from tinygrad import Tensor
from tinygrad.device import Buffer
from tinygrad.ops import UOp, Variable, Ops
from tinygrad.renderer import Renderer
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule, ExecItem, BufferCopy
from typing import Callable, Sequence
import itertools

def is_partial_write(ast:UOp, buf_idx:int) -> bool:
  for u in ast.toposort():
    if u.op is Ops.STORE and (buf:=u.src[0]).arg == buf_idx and buf.dtype.size > u.src[1].arg.size: return True
  return False

# Common logic regardless of render target (e.g. JavaScript, C)
class GraphRenderer(Renderer):
  def __init__(self, fxn:Callable, args:Sequence):
    # realize state_dict and use state_dict names for exported state_bufs; TODO: enable more general state_dict handling
    buf_names = {buf: k for k,v in get_state_dict(getattr(fxn, "__self__", fxn)).items() if (buf:=v.realize().lazydata.base.realized) is not None}
    # ensure random seeds are on-device
    Tensor.randn(1).realize()

    # construct the kernel graph
    # designate the input and output nodes
    self.inputs: list[UOp] = [(x.realize().lazydata if isinstance(x, Tensor) else x) for x in args if isinstance(x, (Tensor, Variable))]
    # TODO: assert no realizes happen here in function call, only kernelize is allowed
    outputs = [t.kernelize().lazydata for t in get_parameters(fxn(*args))]
    assert len(outputs) > 0, "The function argument must return at least one Tensor so that the kernel graph can be accessed."

    # linearize the kernel graph
    schedule, _, becomes_map = create_schedule_with_vars(UOp.sink(*outputs))
    self.outputs: list[UOp] = [becomes_map[uop.base] for uop in outputs]

    # render kernels, render buffer names
    # mark which buffers have state
    ctr = itertools.count()
    self.eis: list[ExecItem] = []
    self.empty_bufs: dict[Buffer, str] = {u.base.buffer: f"buf_{next(ctr)}" for u in self.inputs if u.base.op is Ops.BUFFER}
    self.state_bufs: dict[Buffer, str] = dict()
    seen = set(self.empty_bufs.keys())
    for si, ei in lower_schedule(memory_planner(schedule)):
      if isinstance(ei.prg, BufferCopy):
        ei.run()
        continue
      self.eis.append(ei)
      for i, buf in enumerate(ei.bufs):
        if buf not in seen and (i not in ei.prg.p.outs or i in ei.prg.p.ins or is_partial_write(si.ast, i)): self.state_bufs[buf] = f"buf_{next(ctr)}"
        elif buf not in seen: self.empty_bufs[buf] = f"buf_{next(ctr)}"
        seen.add(buf)

    self.state_bufs.update({k: v for k,v in buf_names.items() if k in self.state_bufs})
    # TODO: we need to ensure the self.state_bufs have been realized with actual data, before now
    self.state_dict: dict[str, Tensor] = {v: Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in self.state_bufs.items()}

  def render_graph(self) -> str: raise NotImplementedError("Implement a language-specific GraphRenderer")