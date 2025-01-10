from dataclasses import dataclass
from tinygrad.ops import UOp, Ops, Variable, type_verify, PatternMatcher, UPat, graph_rewrite_map, symbolic_simple, merge_views, track_rewrites
from tinygrad.ops import graph_rewrite, GroupOp, view_left, buffers, identity_element
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata, all_int, unwrap, DEBUG, prod
from tinygrad.dtype import dtypes
from tinygrad.shape.shapetracker import ShapeTracker

def cooking(*args, **kwargs): raise Exception(f"cooking! {args} {kwargs}")

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

tensor_uop_spec = PatternMatcher([
  (UPat(Ops.DEVICE, name="root", src=()), lambda root: isinstance(root.arg, str)),
  (UPat(Ops.BUFFER, name="root", src=(UPat(Ops.DEVICE))), lambda root: isinstance(root.arg, tuple) and len(root.arg) == 2 and all_int(root.arg)),
  (UPat(Ops.COPY, name="root", src=(UPat(Ops.DEVICE), UPat.var("x"))), lambda root,x: isinstance(root.arg, bool) and root.dtype == x.dtype),
  (UPat({Ops.CONTIGUOUS, Ops.DETACH}, name="root", src=(UPat.var("x"),), arg=None), lambda root,x: root.dtype == x.dtype),
  (UPat(GroupOp.Movement, name="root", src=(UPat.var("x"),)), lambda root,x: root.dtype == x.dtype),
  (UPat(Ops.BUFFER_VIEW, name="root", src=(UPat.var("x"),)), lambda root,x: True),
])

def collapse_size0_op(root:UOp):
  if root.base.st is None or root.size != 0: return None
  return None if root.base.op is Ops.CONST and root.const_arg == 0 else root.const_like(0)

def collapse_const_reduce(root:UOp, x:UOp):
  if not all_int(x.shape): return None
  prshape = prod(unwrap(x.st).shape[i] for i in root.arg[1])
  ret = x.const_arg
  match root.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass # NOTE: Ops.MAX is passthrough
    case _: return None
  return root.const_like(ret)

sym = symbolic_simple+PatternMatcher([
  (UPat(set(Ops), name="root"), collapse_size0_op),

  # reduce folding
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat(Ops.CONST, arg=0),)),
   lambda root: root.const_like(identity_element(root.arg[0], root.dtype)) if root.size != 0 else None),
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat(Ops.CONST, name="x"),)), collapse_const_reduce),

  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.VIEW, name="vm", src=(UPat.var("x"),)),)),
   lambda vm,x:x.contiguous().view(vm.st) if vm.st.contiguous and vm.size == x.size and x.op is not Ops.CONST else None),
])

def mv_const(x:UOp, vm1:UOp, vm2:UOp):
  if (new_st:=unwrap(vm1.st)+unwrap(vm2.st)).views[0].mask is None: return x.replace(src=(vm1.replace(arg=new_st),))
  x = x.replace(src=(vm1.replace(arg=ShapeTracker.from_shape(()).reshape((1,)*len(new_st.shape)).expand(new_st.shape)),))
  return UOp(Ops.VALID, dtypes.bool, (new_st.to_uop(),)).where(x, 0)

remove_movement_ops = merge_views+PatternMatcher([
  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)), lambda mv,x: x.view(mv.st)),
  (UPat(Ops.VIEW, name="vm2", src=(UPat(Ops.CONST, name="x", src=(UPat.var("vm1"),)),)), mv_const),
  (UPat(Ops.VIEW, name="vm", src=(UPat.var("x"),)), lambda vm,x:x if vm.st.contiguous and x.st is not None and vm.shape == x.shape else None),
])

def create_copy_kernel(ctx:dict[UOp, UOp], root:UOp, copyin:UOp):
  ctx[output_buf:=UOp.new_buffer(root.device, copyin.size, copyin.dtype)] = root
  return output_buf.view(unwrap(copyin.st))

def create_kernel(ctx:dict[UOp, UOp], root:UOp):
  ctx[output_buf:=UOp.new_buffer(root.device, root.size, root.dtype)] = root
  return output_buf.view(unwrap(root.st))

def create_buffer_view(ctx:dict[UOp, UOp], root:UOp, src:UOp):
  if src.base.op is Ops.BUFFER:
    ctx[output_buf:=UOp.new_buffer(root.device, root.size, root.dtype)] = root
    buffers[output_buf] = src.base.buffer.view(root.size, root.dtype, unwrap(src.st).views[0].offset*src.dtype.itemsize)
    assert output_buf.realized._base is not None
    return output_buf.view(unwrap(root.st))
  return root.replace(op=Ops.CONTIGUOUS)

bufferize = PatternMatcher([
  (UPat(Ops.COPY, name="root", src=(UPat(), UPat.var("copyin"),)), create_copy_kernel),
  (UPat(Ops.BUFFER_VIEW, name="root", src=(UPat.var("src"),)), create_buffer_view),
  (UPat(Ops.CONTIGUOUS, name="root"), create_kernel),
  (UPat(Ops.REDUCE_AXIS, name="root"), create_kernel),
])

def add_buffer(ctx:list[UOp], buf:UOp):
  glbl = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(size=buf.size), (), len(ctx))
  ctx.append(buf)
  return UOp(Ops.LOAD, buf.dtype, (glbl, unwrap(buf.st).to_uop()))

load_buffers = PatternMatcher([
  (UPat(Ops.BUFFER, name="buf"), add_buffer),
])

def add_store(root:UOp):
  new_src: list[UOp] = []
  for i,x in enumerate(root.src):
    if x.op is Ops.STORE: new_src.append(x)
    else:
      glbl = UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=x.size), (), i)
      new_src.append(UOp.store(glbl, ShapeTracker.from_shape(x.shape).to_uop(), x))
  return root.replace(src=tuple(new_src)) if tuple(new_src) != root.src else None

to_ast = PatternMatcher([
  (UPat(Ops.SINK, src=(UPat({Ops.COPY, Ops.BUFFER_VIEW}, name="x"),)), lambda x:x),
  (UPat(Ops.SINK, name="root"), add_store),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x:x),
])

def make_schedule_item(sink:UOp, output_buf:UOp) -> ScheduleItem:
  sink = graph_rewrite(sink, load_buffers+view_left+remove_movement_ops+to_ast, bufs:=[output_buf])
  return ScheduleItem(sink, tuple(b.buffer for b in bufs), ())

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  sink = UOp.sink(*outs)
  type_verify(list(sink.toposort), tensor_uop_spec)

  realizes: dict[UOp, UOp] = {}
  tensor_map = graph_rewrite_map(sink, remove_movement_ops+sym)
  buffer_map = graph_rewrite_map(tensor_map[sink], remove_movement_ops+sym+bufferize, realizes)

  buffer_tensors: dict[UOp, list[UOp]] = {}
  for k,v in buffer_map.items():
    if (b:=v.base).op is not Ops.BUFFER: continue
    if b not in buffer_tensors: buffer_tensors[b] = []
    buffer_tensors[b] = [t for t,v2 in tensor_map.items() if v2 is k]

  becomes_map: dict[UOp, UOp] = {}
  schedule: list[ScheduleItem] = []
  for k,v in realizes.items():
    si = make_schedule_item(v.sink(), k)
    schedule.append(si)
    for buffer in si.bufs: buffer.ref(1)
    for tensor in buffer_tensors[k]: becomes_map[tensor] = k.view(unwrap(tensor.st))
  var_vals: dict[Variable, int] = {}
  return schedule, var_vals, becomes_map
