# render the kernel graph for execution outside tinygrad
from tinygrad import Tensor, dtype
from tinygrad.tensor import no_realize_uops
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Variable, Ops
from tinygrad.renderer import Renderer
from tinygrad.nn.state import get_state_dict
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.memory import _internal_memory_planner
from tinygrad.engine.realize import lower_schedule, ExecItem, CompiledRunner
from tinygrad.engine.grouper import Kernel
from tinygrad.helpers import Context
from typing import Callable, cast
import itertools

def is_partial_write(ast:UOp, i:int) -> bool:
  stores = [u for u in ast.toposort() if u.op is Ops.STORE and isinstance(u.src[0].dtype, dtype.PtrDType) and isinstance(u.src[1].arg, ShapeTracker)]
  return True if any((b:=u.src[0]).arg == i and cast(dtype.PtrDType, b.dtype).size > cast(ShapeTracker, u.src[1].arg).size for u in stores) else False

def is_store_kernel(u:UOp) -> bool: return True if u.op is Ops.KERNEL and any(v.op is Ops.STORE for v in cast(Kernel,u.arg).ast.toposort()) else False

# Common logic regardless of render target (e.g. JavaScript, C)
class GraphRenderer(Renderer):
  def __init__(self, fxn:Callable, *args, tensor_names:dict[str, Tensor]|None=None, **kwargs):
    assert len(get_state_dict(args)) == len([x for x in args if isinstance(x, Tensor)]) and len(get_state_dict(kwargs)) == 0, \
      "All Tensor (and Variable) function arguments must be positional, whose order will match the order of the rendered function's arguments."

    # Ensure random seeds are realized and in device memory
    Tensor.randn(1).realize()

    # construct the kernel graph
    # designate the input and output nodes
    self.inputs: list[UOp] = [(x.realize().lazydata if isinstance(x, Tensor) else x) for x in args if isinstance(x, (Tensor, Variable))]
    assert len(no_realize_uops) == 0, "having GraphRenderer inside another GraphRenderer is not supported"
    # disallow realization of Tensors whose lazydata contains inputs, to preserve all dependence of compute on input in the constructed graph
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
    # mark which buffers used in computation have state
    ctr = itertools.count()
    self.eis: list[ExecItem] = []
    self.bufs: dict[Buffer, str] = {cast(Buffer, u.base.buffer): f"input_{i}" for i, u in enumerate(self.inputs) if u.base.op is Ops.BUFFER}
    self.bufs.update({cast(Buffer, u.base.buffer): f"output_{i}" for i, u in enumerate(self.outputs)})
    self.state_bufs: dict[Buffer, str] = dict()

    # TODO: this is a modified version of stuff in the jit, deduplicate this stuff when the jit is refactored
    buffer_replace: dict[Buffer, Buffer] = {b:b for b in self.bufs}
    def add_buffer(b:Buffer, stateful:bool=False) -> Buffer:
      if found:=buffer_replace.get(b, None): return found
      buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, options=b.options) if not stateful else b
      return ret

    for si, ei in lower_schedule(schedule):
      assert isinstance(ei.prg, CompiledRunner), f"Unsupported data transfer between bufs:\n{ei.bufs}\n\nEnsure all state is realized"
      for buf in ei.bufs: assert buf is not None and buf.device == compute_device, "All compute and returned Tensor(s) must be on the same device"
      for i, buf in enumerate(cast(list[Buffer], ei.bufs)):
        if buf not in buffer_replace:
          if i not in ei.prg.p.outs or i in ei.prg.p.ins or is_partial_write(si.ast, i): self.state_bufs[add_buffer(buf, True)] = f"buf_{next(ctr)}"
          else: self.bufs[add_buffer(buf)] = ""
      self.eis.append(ExecItem(ei.prg, [add_buffer(buf) for buf in ei.bufs if buf is not None], ei.metadata, ei.fixedvars))
    self.bufs.update(self.state_bufs)

    assigned = _internal_memory_planner(cast(list[list[Buffer]], [ei.bufs for ei in self.eis]))
    for old, new in assigned.items():
      del self.bufs[old]
      if self.bufs[new] == "": self.bufs[new] = f"buf_{next(ctr)}"
    for buf,name in self.bufs.items():
      if name == "": self.bufs[buf] = f"buf_{next(ctr)}"

    for i, ei in enumerate(self.eis): self.eis[i] = ExecItem(ei.prg, [assigned.get(cast(Buffer, b), b) for b in ei.bufs])

    assert all(b.is_allocated() for b in self.state_bufs)
    # build complete state_dict, rename state bufs with meaningful names from tensor_names
    self.state_dict = {k:v for k,v in tensor_names.items() if (b:=v.lazydata.base.realized) and b in self.state_bufs} if tensor_names else {}
    for k,v in self.state_dict.items(): v.lazydata = v.lazydata.base # non-contiguous views cause data permutation in safe_save
    self.state_bufs.update({cast(Buffer, v.lazydata.base.realized):k for k,v in self.state_dict.items()})
    self.state_dict.update({k:Tensor(bytes(b.as_buffer()), "CPU", b.dtype).realize() for b,k in self.state_bufs.items() if k not in self.state_dict})

  def render_graph(self) -> str: raise NotImplementedError("Implement a language-specific GraphRenderer")