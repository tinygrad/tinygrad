import sys, atexit, pickle
from collections import deque
from dataclasses import dataclass
from tinygrad.ops import UOp, Variable, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, track_rewrites, buffers
from tinygrad.ops import identity_element, resolve, view_left, merge_views
from tinygrad.codegen.symbolic import symbolic_simple
from tinygrad.helpers import Context, ContextVar, Metadata, all_int, all_same, colored, diskcache_put, prod, dedup, unwrap, getenv, pluralize
#from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, CAPTURE_PROCESS_REPLAY, DONT_REALIZE_EXPAND, DONT_GROUP_REDUCES, SPLIT_REDUCEOP
from tinygrad.helpers import DEBUG, CAPTURE_PROCESS_REPLAY, SPLIT_REDUCEOP
from tinygrad.dtype import ImageDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer
from tinygrad.spec import type_verify, kernel_spec

# creation can recurse a lot
sys.setrecursionlimit(10000)

# **** schedule simplifier

def simplify_stride0_reduce(reduce:UOp, x:UOp):
  # must be unmasked (NOTE: can be relaxed if not masked on stride 0 axis)
  if any(v.mask is not None for v in unwrap(x.st).views): return None
  # must have all stride 0 in the relevant axis (NOTE: can do partial)
  if not all(unwrap(x.st).views[-1].strides[axis] == 0 for axis in reduce.arg[1]) or not all_int(x.shape): return None
  prshape = prod(x.shape[i] for i in reduce.arg[1])
  ret = x.shrink(tuple((0,s) if i not in reduce.arg[1] else (0,1) for i,s in enumerate(x.shape)))
  match reduce.arg[0]:
    case Ops.ADD: return ret*prshape
    case Ops.MUL: return ret.pow(prshape)
    case Ops.MAX: return ret # NOTE: Ops.MAX is passthrough

def split_reduceop(reduce:UOp, x:UOp):
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.
  real_strides = unwrap(x.st).real_strides(ignore_valid=True)
  if not (split_candidates:=[(i,d) for i in reduce.arg[1] for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and real_strides[i]!=0]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted.r(*reduce.arg).r(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape)

sym = symbolic_simple+PatternMatcher([
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
   and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
  # DETACH and CONTIGUOUS_BACKWARD are NOOPs here
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # reduce on stride 0 is collapsed
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), simplify_stride0_reduce),
  # split_reduceop
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), split_reduceop),
  # COPY(CONST) creates a new CONST on the destination device
  (UPat(Ops.COPY, name="root", src=(UPat(), UPat.cvar("x"),)), lambda root,x: root.const_like(x.arg)),
  # no COPY to same device, except clone (arg is True)
  (UPat(Ops.COPY, src=(UPat(), UPat.var("copyin")), name="copy"),
   lambda copyin,copy: copyin if copyin.device == copy.device and copy.arg is not True else None),
  # copyin must be base
  (UPat(Ops.COPY, src=(UPat(), UPat(Ops.VIEW, name="v")), name="copy"), lambda copy,v: v.contiguous().copy_to_device(copy.device) \
    if prod(v.shape) < prod(v.base.shape) else v.base.copy_to_device(copy.device, clone=copy.arg).view(v.st)),
  # remove cast to image when it's already a contiguous image
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm", src=(UPat(Ops.CONTIGUOUS, name="base"))),)),
   lambda cast,base,vm: base.view(vm.st) if isinstance(cast.dtype, ImageDType) and isinstance(base.dtype, ImageDType) else None),
  # put CAST to smaller dtype before EXPAND
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm"),)), lambda cast,vm: vm.base.cast(cast.dtype).view(vm.st)
     if (not getenv("CAST_AFTER_EXPAND") or vm.base.op is not Ops.BUFFER) and cast.dtype.itemsize <= vm.dtype.itemsize
     and resolve(prod(vm.shape) > vm.st.real_size()) else None),
  # make things that can't be images not images
  (UPat(GroupOp.All-{Ops.BUFFER, Ops.VIEW, Ops.CONST, Ops.DEVICE}, name="u"), lambda u: u.replace(dtype=dt.base) if isinstance(dt:=u.dtype,ImageDType)
   and (prod(u.shape) != prod(dt.shape) or not any(u.shape[x]%4 == 0 for x in u.st.unit_stride_axes())) else None),
  # remove contiguous if we can just view the buffer
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),)),
   lambda root,view,buf: view if view.st.contiguous and view.size == buf.size else None),
  # contiguous/buffer/copy is already contiguous
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat((Ops.CONTIGUOUS, Ops.BUFFER, Ops.COPY)),)), lambda root: root.src[0]),
  # substitute BITCAST/CONTIGUOUS with BUFFER_VIEW on DISK
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), src=(UPat.var("x"),), name="t"),
   lambda x,t: UOp(Ops.BUFFER_VIEW, t.dtype, (x.base,), (t.size, x.st.views[0].offset)).reshape(t.shape) if x.device.startswith("DISK") else None),
  # put UnaryOps before EXPANDs
  (UPat(GroupOp.Unary, src=UPat(Ops.VIEW, src=(UPat.var("inp"),), name="v"), name="alu"),
   lambda inp,v,alu: inp.alu(alu.op).view(v.st) if resolve(prod(alu.shape) > v.st.real_size()) else None),
  # put CAST after expanding BUFFER
  (UPat(Ops.VIEW, src=(UPat(Ops.CAST, src=(UPat.var("x"),)),), name="v"), lambda x,v: x.view(x.st+v.st).cast(v.dtype) if getenv("CAST_AFTER_EXPAND")
    and x.base.op is Ops.BUFFER and resolve(prod(v.shape) > prod(x.shape)) else None),
  # remove CONST/BIND/VIEW from SINK
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=new_src)
    if (new_src:=tuple(dedup(s.base for s in x.src if s.op not in {Ops.CONST,Ops.BIND}))) != x.src else None),
])

# support for using a contiguous permuted view instead of the parent view if one exists

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  if (sti:=unwrap(src.st).invert(src.base.shape)) is not None: ctx[src.base] = contig.view(sti)

replace_contiguous = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.VIEW, name="src"),), name="contig"), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
])

# **** UOp realization

DONT_PUSH_VIEWS = {Ops.BUFFER, Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.ASSIGN, Ops.SINK, Ops.CONTIGUOUS, Ops.COPY}

insert_contiguous = PatternMatcher([
  (UPat(Ops.SINK, name="s"),
   lambda s: s.replace(src=new_src) if (new_src:=tuple(x if x.base.op in DONT_PUSH_VIEWS else x.contiguous() for x in s.src)) != s.src else None),
  (UPat(Ops.VIEW, src=(UPat(GroupOp.All-DONT_PUSH_VIEWS, name="x"),), name="view"), lambda x,view: x.contiguous().view(view.arg)),
])

# **** create kernels

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  metadata: tuple[Metadata, ...] = ()
  def __repr__(self):
    return f"<Kernel {len(list(self.ast.toposort))} {[s.op for s in self.ast.src] if self.ast.op is Ops.SINK else self.ast.op} {self.metadata}>"

@dataclass(frozen=True)
class KernelContext:
  ops_metadata: dict[UOp, Metadata]

def create_kernel(ctx:KernelContext, x:UOp, b:UOp):
  kernel = UOp(Ops.KERNEL, src=(b,)+x.src, arg=Kernel(x.sink(), (m,) if (m:=ctx.ops_metadata.get(x)) else ()))
  buffer = b.base if b.size == b.base.size else UOp(Ops.BUFFER_VIEW, b.dtype, (b.base,), (b.size, b.arg.views[0].offset))
  return UOp(Ops.ASSIGN, x.dtype, (buffer, kernel)).reshape(x.shape)

DONT_PLACE_IN_KERNEL = {Ops.KERNEL, Ops.ASSIGN, Ops.BUFFER}

def append_to_kernel(ctx:KernelContext, x:UOp):
  new_srcs: list[UOp] = []
  metadata = dict.fromkeys(x.arg.metadata)
  for s in x.src:
    if s.op in DONT_PLACE_IN_KERNEL: new_srcs.append(s)
    else:
      new_srcs.extend(s.src)
      if (m:=ctx.ops_metadata.get(s)) is not None: metadata[m] = None
  if (new_src:=tuple(dedup(new_srcs))) != x.src: return x.replace(src=new_src, arg=Kernel(x.arg.ast, tuple(metadata)))

create_kernels = PatternMatcher([
  # always give assign/contiguous a kernel
  (UPat.assign(UPat.var("b"), UPat(GroupOp.All-{Ops.KERNEL}), name="x"), create_kernel),
  (UPat(Ops.CONTIGUOUS, name="x"), lambda ctx,x: create_kernel(ctx, x, UOp.new_buffer(x.device, x.size, x.dtype))),
  # create a buffer for COPY on the new device
  (UPat(Ops.COPY, src=(UPat(Ops.DEVICE, name="d"), UPat()), name="x"), lambda ctx,d,x: create_kernel(ctx, x, UOp.new_buffer(d.arg, x.size, x.dtype))),
  # walk back the local graph until we reach a buffer/assign parent
  (UPat(Ops.KERNEL, name="x"), append_to_kernel),
  # remove downstream reshapes from SINK
  (UPat(Ops.SINK, name="x"), lambda x:x.replace(src=tuple(s.base for s in x.src)) if any(s.op is Ops.VIEW for s in x.src) else None),
])

# **** swizzler

def apply_swizzle(u:UOp) -> UOp:
  with Context(TRACK_MATCH_STATS=0): return graph_rewrite(u, view_left)

def swizzle_reduceop(ctx, r:UOp, src:UOp, view:UOp):
  ctx.update_children()
  if r in ctx.children and len(ctx.children[r]) > 1: return r.contiguous().view(view.arg)
  if (st:=unwrap(view.st)).contiguous: return None
  input_st = ShapeTracker.from_shape(src.shape)
  tmp = input_st.permute(tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg)
  prshape = prod(rshape:=tmp.shape[-len(r.axis_arg):])
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in st.views]
  # create a new reduceop for the swizzled input
  new_input_st = tmp + ShapeTracker(tuple(nv))
  new_axis = tuple(range(len(st.shape), len(st.shape) + len(r.axis_arg)))
  return UOp(Ops.REDUCE_AXIS, r.dtype, (apply_swizzle(src.view(src.arg+new_input_st if src.op is Ops.VIEW else new_input_st)),),
             (r.arg[0], new_axis)).view(ShapeTracker.from_shape(st.shape))

def reduceop_view_right(src:UOp, v:UOp, r:UOp):
  assert unwrap(v.st).contiguous and v.size == src.size, f"can't compute new axis for {src.shape} -> {r.shape}"
  return src.r(r.arg[0], tuple(i for i,(s,u) in enumerate(zip(src.shape, r.shape)) if s != u)).view(ShapeTracker.from_shape(r.shape))

def elementwise_view_right(root:UOp):
  if not (swizzles:=[x for x in root.src if x.op is Ops.VIEW and x.base.op not in DONT_PUSH_VIEWS]): return None
  assert all_same([x.base.size for x in swizzles]), f"swizzle inputs must have the same size {swizzles}"
  if swizzles[0].base.size != root.size: return None
  # place view after applying the elementwise op
  new_st = ShapeTracker.from_shape(swizzles[0].base.shape)
  new_src = [x.base if x.base.shape==new_st.shape else apply_swizzle(x.view(x.arg+new_st) if x.op is Ops.VIEW else x.view(new_st)) for x in root.src]
  # reshape to match downstream shapes
  return root.replace(src=tuple(new_src)).reshape(root.shape)

def merge_double_reduce(root:UOp, first_reduce:UOp):
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  return first_reduce.replace(arg=(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg))

# push VIEW to children
view_right = merge_views+PatternMatcher([
  # push a non contiguous ShapeTracker through reduceop
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
  # apply view after reduceops
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.VIEW, src=(UPat(GroupOp.All-DONT_PUSH_VIEWS, name="src"),), name="v"),), name="r"), reduceop_view_right),
  # apply view after elementwise ops
  (UPat(GroupOp.All-DONT_PUSH_VIEWS, name="root"), elementwise_view_right),
  # double reduce op collapses to a single reduce op
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

# **** unbind variables

def unbind_shapetracker(ctx:tuple[dict[Variable, int], tuple[UOp, ...]], x:UOp):
  st = unwrap(x.st).simplify()
  if any(x.op is Ops.BIND for x in st.vars()):
    st, var_vals = st.unbind()
    ctx[0].update(var_vals)
  return x.replace(arg=st) if st != x.st else None

def unbind_variable(ctx:tuple[dict[Variable, int], tuple[UOp, ...]], var:UOp, val:UOp):
  ctx[0][var.replace(src=())] = val.arg
  return var

# **** fix kernel AST

add_buffer_ops = PatternMatcher([
  # LOAD
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x:UOp.load(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), ctx[1].index(x)), x.st.to_uop(), dtype=x.dtype)),
  (UPat(Ops.ASSIGN, name="x"), lambda x:x.src[0]),
  # STORE (except for COPY/BUFFER_VIEW)
  (UPat(Ops.SINK, src=(UPat((Ops.COPY, Ops.BUFFER_VIEW), name="x"),)), lambda x:x),
  # partial assign can store to a non-contiguous ShapeTracker
  (UPat(Ops.SINK, src=(UPat(Ops.ASSIGN, name="x"),)),
   lambda x: UOp.store(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), 0), x.src[0].st.to_uop(), x.src[1]).sink()),
  # otherwise the store is contiguous
  (UPat(Ops.SINK, src=(UPat(GroupOp.All-{Ops.STORE}, name="x"),)),
   lambda x: UOp.store(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), 0), ShapeTracker.from_shape(x.shape).to_uop(), x).sink()),
  # VALID
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="x"),), name="view"), lambda x,view: x.valid(view.arg)),
  # if the last child is a VIEW we merge the ShapeTrackers and store the base
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat(Ops.VIEW, src=(UPat(GroupOp.All-DONT_PUSH_VIEWS, name="x"),)))),
   lambda x,b,st: UOp.store(b, (st.arg+x.st).to_uop(), x)),
])

def check_load_st(glbl:UOp, view:UOp):
  if glbl.arg != 0 or (st:=unwrap(view.st)).contiguous: return
  # if it has a single view and it becomes contiguous when you shrink expanded axes, it's fine
  if len(st.views) == 1 and st.shrink(tuple((0,1) if st == 0 else (0,s) for s,st in zip(st.shape, st.views[0].strides))).contiguous: return
  # if it has a single view and it's equal when you shrink a contig, it's fine
  if len(st.views) == 1 and (mask:=st.views[0].mask) is not None and ShapeTracker.from_shape(st.shape).shrink(mask) == st.shrink(mask): return
  # otherwise, it's not fine
  raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                     +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))

fix_kernel_ops = PatternMatcher([
  # remove CONTIGUOUS/DEVICE from kernel AST
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),), name="view"), lambda view: view.replace(src=())),
  # BIND in shapetracker becomes DEFINE_VAR
  (UPat(Ops.VIEW, name="x"), unbind_shapetracker),
  (UPat(Ops.BIND, src=(UPat.var("var"), UPat.cvar("val"))), unbind_variable),
  # no ImageDType after load
  (UPat(GroupOp.All-{Ops.DEFINE_GLOBAL}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  # if this kernel also assigns to the loaded buffer, ensure we can index it correctly
  (UPat(Ops.LOAD, src=(UPat.var("glbl"), UPat.var("view"))), check_load_st),
])

def fix_kernel_ast(k:UOp, var_vals:dict[Variable, int]) -> UOp:
  assert k.op is Ops.KERNEL, f"kernel isn't kernel, it's {k}"
  # add buffer ops + fix_kernel_ops
  ast = graph_rewrite(k.arg.ast, merge_views+add_buffer_ops+fix_kernel_ops, ctx=(var_vals, bufs:=tuple(s.buf_uop for s in k.src)), bottom_up=True)
  if ast.op is Ops.SINK and not all_same(dev:=[x.device for x in bufs]): raise RuntimeError(f"all buffers must be on the same device: {dev}")
  # create subbuffer (TODO: this does not belong here)
  if ast.op is Ops.BUFFER_VIEW: buffers[bufs[0]] = (base:=bufs[1].buffer).view(ast.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
  return k.replace(arg=Kernel(ast, k.arg.metadata))

PROCESS_REPLAY_CAPTURE:dict[str, bytes] = {}
if CAPTURE_PROCESS_REPLAY:
  @atexit.register
  def save_process_replay():
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

# **** schedule creation and toposort

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

@track_rewrites(name_fxn=lambda r: f"Schedule {pluralize('Kernel', len(r[0]))}"+(f" (with_{pluralize('Var', len(r[1]))})" if len(r[1]) != 0 else ""))
def create_schedule_with_vars(big_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  # merge_views + sym + replace_contiguous
  tensor_map = graph_rewrite_map(big_sink, merge_views+sym+replace_contiguous, ctx={})

  # display the cleaned up tensor graph
  if getenv("VIZ"): graph_rewrite(tensor_map[big_sink], PatternMatcher([]), name="View Tensor Graph")

  # swizzler + create_kernels
  view_left_map = graph_rewrite_map(sink:=tensor_map[big_sink], view_left, name="view_left")
  view_right_map = graph_rewrite_map(vl_sink:=view_left_map[sink], view_right, track_children=True, name="view_right")
  contiguous_map = graph_rewrite_map(vr_sink:=view_right_map[vl_sink], merge_views+insert_contiguous+create_kernels, ctx=KernelContext({}))
  # map sink sources back to the kernel op
  assert len(vr_sink.src) == len(contiguous_map[vr_sink].src)
  for vs,ks in zip(vr_sink.src, contiguous_map[vr_sink].src): contiguous_map[vs] = ks
  # walk the maps forward and map tensors to kernels
  # childless tensors won't have a key in one of the maps, this is fine, just skip those
  kernel_map: dict[UOp, UOp] = {}
  for k,v in view_left_map.items():
    if v not in view_right_map: continue
    vr = view_right_map[v]
    if vr not in contiguous_map: continue
    ct = contiguous_map[vr]
    kernel_map[k] = ct
  sched_sink = kernel_map[sink]
  type_verify(list(sched_sink.toposort), kernel_spec)

  # map tensors to buffer/const, optionally apply a VIEW on top
  becomes_map: dict[UOp, UOp] = {}
  for k,v in tensor_map.items():
    # ASSIGN always becomes the target buffer
    if v.op is Ops.ASSIGN: becomes_map[k] = v.src[0]
    # if we created a new buffer for this tensor, map it to the assigned buffer
    elif (a:=kernel_map.get(v.base)) is not None and (a:=a.base).op is Ops.ASSIGN:
      becomes_map[k] = a.src[0] if a.src[0].st == v.st else a.src[0].view(unwrap(v.st))
    # tensors can also simplify to an existing buffer/const
    else:
      if k is v: continue
      if v.base.op is Ops.BUFFER: becomes_map[k] = v
      if v.base.op is Ops.CONST and all_int(v.shape): becomes_map[k] = v

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in sched_sink.toposort:
    if u.op is not Ops.ASSIGN: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.ASSIGN and x.buf_uop is s for x in u.toposort):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on ASSIGN or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep:
    sched_sink = sched_sink.substitute(assign_rep)
    type_verify(list(sched_sink.toposort), kernel_spec)

  # display the final graph
  if getenv("VIZ"): graph_rewrite(sched_sink, PatternMatcher([]), name="View Kernel Graph")

  # final toposort (bfs)
  children: dict[UOp, list[UOp]] = {}
  in_degree: dict[UOp, int] = {}
  for u in sched_sink.toposort:
    if u.op is not Ops.ASSIGN: continue
    in_degree[u] = 0
    for s in u.src[1].src:
      if s.op is not Ops.ASSIGN: continue
      children.setdefault(s, []).append(u)
      in_degree[u] += 1

  queue = deque(k for k,v in in_degree.items() if v == 0)
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  while queue:
    u = queue.popleft()
    # TODO: move this to create_kernels
    k = fix_kernel_ast(u.src[1], var_vals)
    schedule.append(ScheduleItem(k.arg.ast, tuple(s.buf_uop.buffer for s in k.src), k.arg.metadata))
    # increment the refcount of the target buf (this is required by the JIT and memory planner) TODO: this does not belong here
    k.src[0].buffer.ref(1)
    for x in children.get(u, []):
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if len(schedule) != (kc:=len(in_degree)): raise RuntimeError(f"cycle detected in graph, created {kc} kernels but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  # capture process replay
  if CAPTURE_PROCESS_REPLAY:
    with Context(PICKLE_BUFFERS=0): PROCESS_REPLAY_CAPTURE[str(big_sink.key)] = pickle.dumps((big_sink, ContextVar._cache, [x.ast for x in schedule]))
  return schedule, var_vals, becomes_map
