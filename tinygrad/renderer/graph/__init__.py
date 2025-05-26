# render the kernel graph for execution outside tinygrad
from tinygrad import Tensor, dtype
from tinygrad.tensor import no_realize_uops
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Variable, Ops
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

    # disallow realization of Tensors whose lazydata contains inputs, to preserve all dependence of compute on input in the constructed graph
    assert len(no_realize_uops) == 0, "having GraphRenderer inside another GraphRenderer is not supported"
    no_realize_uops.update(self.inputs)
    with Context(LIMIT_REALIZE=1):
      ret_tensors: list[Tensor] = [*filter(lambda t:isinstance(t, Tensor), r if isinstance(r:=fxn(*args, **kwargs), (list, tuple)) else [r])]
    no_realize_uops.clear()
    assert (l:=len(ret_tensors)) and l == len(get_state_dict(r)), "One or more Tensors must be returned as a singleton or elements of a list/tuple."
    if not isinstance(device:=ret_tensors[0].device, str): raise RuntimeError(f"Multiple devices not supported: {device}")
    rng_tensors = [Tensor._device_seeds[device].realize(), Tensor._device_rng_counters[device].realize()] if device in Tensor._device_seeds else []

    # linearize the kernel graph
    schedule, var_vals = create_schedule_with_vars(sink:=UOp.sink(*[t.kernelize().lazydata for t in ret_tensors]))
    assert set(var_vals.keys()) == set(u.unbind()[0] for u in self.inputs if u.op is Ops.BIND), "Variables must be positional arguments."
    remove_assign_map = {u:u.buf_uop for u in sink.toposort() if u.op is Ops.ASSIGN}
    self.outputs: list[UOp] = [remove_assign_map[uop.base] for uop in sink.src]

    # render kernels, render buffer names
    # mark which buffers used in computation have state
    self.eis: list[ExecItem] = []
    self.bufs: dict[Buffer, str] = {cast(Buffer, u.base.buffer): f"input_{i}" for i, u in enumerate(self.inputs) if u.base.op is Ops.BUFFER}
    self.bufs.update({cast(Buffer, u.base.buffer): f"output_{i}" for i, u in enumerate(self.outputs)})
    self.state_bufs: dict[Buffer, str] = dict()
    ctr = itertools.count()

    del r, ret_tensors, sink, remove_assign_map # GC the UOps indirectly referencing our buffers so memory planning works
    for si, ei in lower_schedule(memory_planner(schedule)):
      assert isinstance(ei.prg, CompiledRunner), f"Unsupported data xfer between bufs:\n{ei.bufs}\n\nDid you realize all state Tensors before export?"
      for buf in ei.bufs: assert buf is not None and buf.device == device, "All compute and returned Tensor(s) must be on the same device"
      self.eis.append(ei)
      for i, buf in enumerate(cast(list[Buffer], ei.bufs)):
        if buf not in self.bufs:
          self.bufs[buf] = name = f"buf_{next(ctr)}"
          if i not in ei.prg.p.outs or i in ei.prg.p.ins or is_partial_write(si.ast, i): self.state_bufs[buf] = name

    # build complete state_dict, rename state bufs with meaningful names from tensor_names
    assert not any(unreal:=[b for b in self.state_bufs if not b.is_allocated()]), f"Are all state Tensors realized? Unrealized state bufs:\n{unreal}"
    self.state_dict = {k:v for k,v in tensor_names.items() if (b:=v.lazydata.base.realized) and b in self.state_bufs} if tensor_names else {}
    if rng_tensors and all((b:=t.lazydata.base.realized) and b in self.state_bufs for t in rng_tensors):
      self.state_dict.update({"random_seeds": rng_tensors[0], "random_counter": rng_tensors[1]})
    for k,v in self.state_dict.items(): v.lazydata = v.lazydata.base # non-contiguous views cause data permutation in safe_save
    self.state_bufs.update({cast(Buffer, v.lazydata.base.realized):k for k,v in self.state_dict.items()})
    self.state_dict.update({k:Tensor(bytes(b.as_buffer()), "CPU", b.dtype).realize() for b,k in self.state_bufs.items() if k not in self.state_dict})

  def render_graph(self) -> str: raise NotImplementedError("Implement a language-specific GraphRenderer")