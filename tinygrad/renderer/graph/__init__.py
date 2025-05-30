# render the kernel graph for execution outside tinygrad
from tinygrad import Tensor, dtype
from tinygrad.tensor import all_tensors
from tinygrad.shape.shapetracker import ShapeTracker
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

def is_partial_write(ast:UOp, i:int) -> bool:
  stores = [u for u in ast.toposort() if u.op is Ops.STORE and isinstance(u.src[0].dtype, dtype.PtrDType) and isinstance(u.src[1].arg, ShapeTracker)]
  return True if any((b:=u.src[0]).arg == i and cast(dtype.PtrDType, b.dtype).size > cast(ShapeTracker, u.src[1].arg).size for u in stores) else False

# Common logic regardless of render target (e.g. JavaScript, C)
class GraphRenderer(Renderer):
  def __init__(self, fxn:Callable, *args, tensor_names:dict[str, Tensor]|None=None, **kwargs):
    assert len(get_state_dict(args)) == len([x for x in args if isinstance(x, Tensor)]) and len(get_state_dict(kwargs)) == 0, \
      "All Tensor (and Variable) function arguments must be positional, whose order will match the order of the rendered function's arguments."
    self.inputs: list[UOp] = [(x.realize().lazydata if isinstance(x, Tensor) else x) for x in args if isinstance(x, (Tensor, Variable))]
    Tensor.rand(1).realize() # Don't capture host-->GPU BufferCopy of random seeds/counter

    # Detect Tensors that are new or mutated as a result of calling fxn
    original: dict[Tensor, UOp] = {tref: tref.kernelize().lazydata for t in all_tensors if (tref:=t()) is not None}
    with Context(BAN_REALIZE=1): ret_tensors = get_parameters(fxn(*args, **kwargs))
    postcall: dict[Tensor, UOp] = {tref: tref.kernelize().lazydata for t in all_tensors if (tref:=t()) is not None}
    affected: dict[Tensor, dict] = {t: t.lazydata.toposort() for t in postcall if t not in original or original[t].key != postcall[t].key}
    assert len(affected), "The exported function did not create or change any Tensors, no graph can be captured"
    device = next(iter(affected)).device
    if not isinstance(device, str): raise RuntimeError(f"Multiple devices not supported: {device}")

    # For each implicit input Tensor, realize the portion of its UOp graph that existed before calling fxn
    self.impl_ins: dict[Buffer, str] = dict()
    ctr = itertools.count()
    exported_data: dict[Tensor, dict] = {}
    all_names = {"random_seeds": ds[device], "random_counter": Tensor._device_rng_counters[device]} if device in (ds:=Tensor._device_seeds) else {}
    if tensor_names: all_names.update(tensor_names)
    name_lookup = {v:k for k,v in all_names.items()} if all_names else {}
    for t, u in original.items():
      if u.base not in self.inputs and u.base.op in (Ops.BUFFER, Ops.ASSIGN) and any(u.base in toposort for toposort in affected.values()):
        (precall_implicit_input:=Tensor(0)).lazydata = u.base
        self.impl_ins[cast(Buffer, precall_implicit_input.realize().lazydata.base.realized)] = name = f"buf_{next(ctr)}"
        exported_data[precall_implicit_input] = {"default_name": name}
        if t in name_lookup: exported_data[precall_implicit_input]["tensor_name"] = name_lookup[t]

    # TODO: does scheduling always respect chronological order of assigns, e.g. when a buffer is mutated in one branch and used in another?
    sink = UOp.sink(*[t.lazydata.base for t in affected])
    remove_assign_map = {u:u.buf_uop for u in sink.toposort() if u.op is Ops.ASSIGN}
    self.outputs: list[UOp] = [remove_assign_map[t.lazydata.base] for t in ret_tensors]
    self.bufs: dict[Buffer, str] = {cast(Buffer, u.base.buffer): f"input_{i}" for i, u in enumerate(self.inputs) if u.base.op is Ops.BUFFER}
    self.bufs.update({cast(Buffer, u.base.buffer): f"output_{i}" for i, u in enumerate(self.outputs)})

    # assigns on implicit input can cause the final buffer to be different than the original buffer, so we need to copy the data back
    self.extra_copies: list[tuple[Buffer, Buffer]] = []
    for t in affected:
      if t in original and (new_buf := t.lazydata.base.buf_uop.buffer) != (old_buf := original[t].base.buf_uop.buffer):
        self.extra_copies.append((cast(Buffer, old_buf), cast(Buffer, new_buf)))

    # linearize the kernel graph
    schedule, var_vals = create_schedule_with_vars(sink)
    assert set(var_vals.keys()) == set(u.unbind()[0] for u in self.inputs if u.op is Ops.BIND), "Variables must be positional arguments."

    self.eis: list[ExecItem] = []
    del original, postcall, ret_tensors, sink, remove_assign_map, affected, t
    for si, ei in lower_schedule(memory_planner(schedule)):
      assert isinstance(ei.prg, CompiledRunner), f"Export only supported for CompiledRunner\nei.prg: {ei.prg}\n\nei.bufs: {ei.bufs}"
      for buf in ei.bufs: assert buf is not None and buf.device == device, "All compute and returned Tensor(s) must be on the same device"
      self.eis.append(ei)
      for i, buf in enumerate(cast(list[Buffer], ei.bufs)):
        if buf not in self.bufs:
          if buf in self.impl_ins and not (i not in ei.prg.p.outs or i in ei.prg.p.ins or is_partial_write(si.ast, i)):
            self.bufs[buf] = self.impl_ins.pop(buf)
          else: self.bufs[buf] = f"buf_{next(ctr)}"

    self.bufs.update(self.impl_ins)
    self.state_dict = {v.get("tensor_name", v["default_name"]):k for k,v in exported_data.items() if k.lazydata.base.realized in self.impl_ins}
    for k,v in self.state_dict.items(): v.lazydata = v.lazydata.base # non-contiguous views cause data permutation in safe_save
    self.impl_ins.update({cast(Buffer, v.lazydata.base.realized): k for k,v in self.state_dict.items()})

  def render_graph(self) -> str: raise NotImplementedError("Implement a language-specific GraphRenderer")