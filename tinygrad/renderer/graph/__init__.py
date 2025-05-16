# render the kernel graph for execution outside tinygrad
from tinygrad import Tensor, dtype
from tinygrad.tensor import no_realize_uops, all_tensors
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from tinygrad.ops import UOp, Variable, Ops
from tinygrad.renderer import Renderer
from tinygrad.nn.state import get_state_dict
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule, ExecItem, CompiledRunner
from tinygrad.helpers import Context
from typing import Callable, cast
import itertools

def is_partial_write(ast:UOp, i:int) -> bool:
  stores = [u for u in ast.toposort() if u.op is Ops.STORE and isinstance(u.src[0].dtype, dtype.PtrDType) and isinstance(u.src[1].arg, ShapeTracker)]
  return True if any((b:=u.src[0]).arg == i and cast(dtype.PtrDType, b.dtype).size > cast(ShapeTracker, u.src[1].arg).size for u in stores) else False

# Common logic regardless of render target (e.g. JavaScript, C)
class GraphRenderer(Renderer):
  def __init__(self, fxn:Callable, *args, tensor_names:dict[str, Tensor]|None=None, **kwargs):
    assert len(get_state_dict(args)) == len([x for x in args if isinstance(x, Tensor)]) and len(get_state_dict(kwargs)) == 0, \
      "All Tensor (and Variable) function arguments must be positional, whose order will match the order of the rendered function's arguments."

    # construct the kernel graph
    # designate the input and output nodes
    self.inputs: list[UOp] = [(x.realize().lazydata if isinstance(x, Tensor) else x) for x in args if isinstance(x, (Tensor, Variable))]
    assert len(no_realize_uops) == 0, "having GraphRenderer inside another GraphRenderer is not supported"
    no_realize_uops.update(self.inputs)
    with Context(LIMIT_REALIZE=1):
      ret_tensors: list[Tensor] = [*filter(lambda t:isinstance(t, Tensor), r if isinstance(r:=fxn(*args, **kwargs), (list, tuple)) else [r])]
    no_realize_uops.clear()
    assert (l:=len(ret_tensors)) and l == len(get_state_dict(r)), "One or more Tensors must be returned as a singleton or elements of a list/tuple."
    compute_device = ret_tensors[0].device

    # linearize the kernel graph
    schedule, var_vals, becomes_map = create_schedule_with_vars(UOp.sink(*(out_uops:=[t.kernelize().lazydata for t in ret_tensors])))
    assert set(var_vals.keys()) == set(u.unbind()[0] for u in self.inputs if u.op is Ops.BIND), "Variables must be positional arguments."
    self.outputs: list[UOp] = [becomes_map[uop.base] for uop in out_uops]

    # render kernels, render buffer names
    # mark which compute buffers have state
    ctr = itertools.count()
    self.eis: list[ExecItem] = []
    self.bufs: dict[Buffer, str] = {u.base.buffer: f"input_{i}" for i, u in enumerate(self.inputs) if u.base.op is Ops.BUFFER}
    self.bufs.update({u.base.buffer: f"output_{i}" for i, u in enumerate(self.outputs)})
    self.state_bufs: dict[Buffer, str] = dict()
    for si, ei in lower_schedule(memory_planner(schedule)):
      if isinstance(ei.prg, CompiledRunner):
        for buf in ei.bufs: assert buf is not None and buf.device == compute_device, "All compute and returned Tensor(s) must be on the same device"
        self.eis.append(ei)
        for i, buf in enumerate(cast(list[Buffer], ei.bufs)):
          if buf not in self.bufs:
            self.bufs[buf] = name = f"buf_{next(ctr)}"
            if i not in ei.prg.p.outs or i in ei.prg.p.ins or is_partial_write(si.ast, i): self.state_bufs[buf] = name

    state_makers = [t for tref in all_tensors if (t:=tref()) is not None and (new_uop:=becomes_map.get(t.lazydata.base)) is not None \
                    and new_uop.op is Ops.BUFFER and new_uop.buffer in self.state_bufs]
    if state_makers: Tensor.realize(*state_makers)
    if tensor_names: self.state_bufs.update({b:k for k,v in tensor_names.items() if (b:=v.lazydata.base.realized) and b in self.state_bufs})
    self.state_dict: dict[str, Tensor] = {v:Tensor(bytes(k.as_buffer()), dtype=k.dtype, device=k.device).realize() for k,v in self.state_bufs.items()}

  def render_graph(self) -> str: raise NotImplementedError("Implement a language-specific GraphRenderer")