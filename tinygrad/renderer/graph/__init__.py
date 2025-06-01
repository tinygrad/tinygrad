from tinygrad.tensor import Tensor, all_tensors
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Variable, Ops
from tinygrad.renderer import Renderer
from tinygrad.nn.state import get_state_dict, get_parameters
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.realize import lower_schedule, ExecItem, CompiledRunner
from tinygrad.helpers import Context
from typing import Callable, cast
import itertools

# Common logic regardless of render target (e.g. JavaScript, C)
class GraphRenderer(Renderer):
  def __init__(self, graph_constructor:Callable, *args, tensor_names:dict[str, Tensor]|None=None, **kwargs):
    assert len(get_state_dict(args)) == len([x for x in args if isinstance(x, Tensor)]) and len(get_state_dict(kwargs)) == 0, \
      "All Tensor (and Variable) function arguments must be positional, whose order will match the order of the rendered function's arguments"
    self.inputs: list[UOp] = [(x.realize().lazydata.base if isinstance(x, Tensor) else x) for x in args if isinstance(x, (Tensor, Variable))]
    Tensor.rand(1).realize() # Don't capture PYTHON-->device BufferCopy of random seeds/counter

    # Detect Tensors that are new or mutated as a result of calling graph_constructor
    precall: dict[Tensor, UOp] = {tref: tref.kernelize().lazydata for t in list(all_tensors) if (tref:=t()) is not None}
    with Context(BAN_REALIZE=1): ret_tensors = get_parameters(graph_constructor(*args, **kwargs))
    postcall: dict[Tensor, UOp] = {tref: tref.kernelize().lazydata for t in list(all_tensors) if (tref:=t()) is not None}
    affected: dict[Tensor, dict[UOp, None]] = {t: t.lazydata.toposort() for t in postcall if t not in precall or precall[t].key != postcall[t].key}
    assert len(affected), "The exported function did not create or change any Tensors, no graph can be captured"
    device = next(iter(affected)).device
    if not isinstance(device, str): raise RuntimeError(f"Multiple devices not supported: {device}")
    self.bufs: dict[Buffer, str] = {cast(Buffer, u.base.buffer): f"input_{i}" for i, u in enumerate(self.inputs) if u.base.op is Ops.BUFFER}
    self.outputs: list[UOp] = [t.lazydata.base.buf_uop for t in ret_tensors]
    self.bufs.update({cast(Buffer, u.base.buffer): f"output_{i}" for i, u in enumerate(self.outputs)})

    # Realize implicit inputs as they were before calling graph_constructor
    # Capture implicit inputs in self.state_dict
    ctr = itertools.count()
    self.state_dict: dict[str, Tensor] = {}
    tensor_name_lookup = {t: name for name, t in tensor_names.items()} if tensor_names else {}
    for t, u in precall.items():
      if u.base not in self.inputs and u.base.op in (Ops.BUFFER, Ops.ASSIGN) and any(u.base in toposort for toposort in affected.values()):
        (precall_implicit_input:=Tensor(0)).lazydata = u.base
        if (b:=cast(Buffer, precall_implicit_input.realize().lazydata.base.realized)) not in self.bufs: self.bufs[b] = name = f"buf_{next(ctr)}"
        self.state_dict[tensor_name_lookup.get(t, name)] = precall_implicit_input

    sink = UOp.sink(*[t.lazydata.base for t in affected])

    # Assigns on implicit input can cause the final buffer to be different from the original buffer, so we need to copy the data back
    # Holding the end buffer's UOp prevents the memory planner from reassigning the end buffer's Buffer, so the copy will stay correct
    self.implicit_input_copies: list[tuple[UOp, UOp]] = []
    for t in set(affected).intersection(set(precall)):
      if (end := t.lazydata.base.buf_uop).key != (start := precall[t].base.buf_uop).key and start in t.lazydata.toposort():
        self.implicit_input_copies.append((start, end))
        # Ensure implicit i/o Tensors point at the correct data, but don't leave unrealized compute in them
        t.lazydata = start
      else: t.lazydata = end

    # Linearize the kernel graph
    schedule, var_vals = create_schedule_with_vars(sink)
    assert set(var_vals.keys()) == set(u.unbind()[0] for u in self.inputs if u.op is Ops.BIND), "Variables must be positional arguments"
    self.eis: list[ExecItem] = []
    del precall, postcall, ret_tensors, affected, sink, t
    for _, ei in lower_schedule(memory_planner(schedule)):
      assert isinstance(ei.prg, CompiledRunner), f"Export only supported for CompiledRunner\nei.prg: {ei.prg}\n\nei.bufs: {ei.bufs}"
      for buf in ei.bufs: assert buf is not None and buf.device == device, "All compute must be on the same device"
      self.eis.append(ei)
      self.bufs.update({b: f"buf_{next(ctr)}" for b in cast(list[Buffer], ei.bufs) if b not in self.bufs})

  def render_graph(self) -> str: raise NotImplementedError("Implement a language-specific GraphRenderer")