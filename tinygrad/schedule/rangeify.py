from typing import cast
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, PtrDType, ImageDType, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, _substitute, ssimplify, KernelInfo
from tinygrad.uop.ops import track_rewrites, graph_rewrite, identity_element, sint, AxisType, BottomUpGate
from tinygrad.uop.symbolic import symbolic_flat
from tinygrad.helpers import argsort, prod, all_same, pluralize, getenv, flatten, dedup, all_int, DEBUG, SPLIT_REDUCEOP, Metadata, DEBUG_RANGEIFY
from tinygrad.helpers import PCONTIG, partition
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_simplify
from tinygrad.codegen.opt import Opt
from tinygrad.schedule.indexing import run_rangeify, BufferizeOpts, ALWAYS_CONTIGUOUS, IndexingContext, apply_movement_op

# creation can recurse a lot
import sys
sys.setrecursionlimit(10000)

# movement op on INDEX as a PatternMatcher
pm_mops = PatternMatcher([
  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"),
   lambda r,idx: r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idx.src[1:]), dtype=idx.dtype, arg=idx.arg)),  # type: ignore
])

# *****************
# 0. do some cleanup rewrites, mostly copied from the old stuff

def find_permutes(a:UOp, b:UOp, assign:UOp):
  if not (permutes:=[s for s in b.toposort(gate=lambda s:s.op not in ALWAYS_CONTIGUOUS)
                     if s.op in GroupOp.Movement and s.op not in {Ops.RESHAPE, Ops.EXPAND, Ops.PAD, Ops.SHRINK}]): return
  target = a.base
  for p in permutes:
    if any(s is target for s in p.toposort(gate=lambda s:s.op not in ALWAYS_CONTIGUOUS-{Ops.BUFFER})): return assign.replace(src=(a, b.contiguous()))

def split_reduceop(reduce:UOp, x:UOp):
  if prod(reduce.shape) == 0: return None
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.

  # get expanded by rangeifying the UOp x
  indexed = x.index(*[UOp.range(s, i) if resolve(s>1) else UOp.const(dtypes.index, 0) for i,s in enumerate(x.shape)])
  range_nums = [y.arg[0] for y in indexed.substitute({x.base:UOp(Ops.NOOP)}, extra_pm=pm_mops).ranges]
  is_expanded = [i not in range_nums for i in range(len(x.shape))]

  if not (split_candidates:=[(i,d) for i in reduce.arg[1] for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and not is_expanded[i]]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted.r(*reduce.arg).contiguous().r(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape).replace(tag=reduce.tag)

mop_cleanup = PatternMatcher([
  # merge adjacent RESHAPES, safe because they are not tagged
  (UPat(Ops.RESHAPE, name="x2").f(Ops.RESHAPE, allow_any_len=True, name="x"),
   lambda x,x2: x.replace(src=(x2.src[0], x.src[1])) if x.tag is None and x2.tag is None else None),
])

earliest_rewrites = mop_cleanup+PatternMatcher([
  # just removing it works...
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.FUSE), name="x"), lambda x: x.src[0]),

  # remove CONTIGUOUS if the BUFFER is already contiguous
  (UPat(Ops.BUFFER).f(Ops.RESHAPE, allow_any_len=True, name="r").f(Ops.CONTIGUOUS, name="c"), lambda r,c: r.replace(tag=c.tag)),

  # split_reduceop
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), split_reduceop),

  # preserve tags?
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),

  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x._shape is not None and x.size == 0 else None),

  # remove contiguous on movement ops before a copy on disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.CONTIGUOUS).f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: copy.replace(src=(x,)+copy.src[1:]) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # push copy past movement ops to disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: x.replace(src=(copy.replace(src=(x.src[0],)+copy.src[1:], tag=None),)+x.src[1:], tag=copy.tag) \
      if isinstance(x.device, str) and x.device.startswith("DISK") else None),

  # ** copy rules **

  # early fixup const copy
  (UPat(Ops.COPY, src=(UPat.var("s"), UPat()), name="c"), lambda c,s: c.const_like(ss.arg) if (ss:=s.base).op is Ops.CONST else None),

  # COPY and source size need to match
  # TODO: expand after copy creates issues with tagging
  (UPat(Ops.COPY, src=(UPat(GroupOp.Movement, name="r"), UPat(name="d")), name="c"),
   lambda c,r,d: c.replace(src=(r.contiguous(), d)) if r.size != r.base.size else None),

  # copy only to different device
  (UPat(Ops.COPY, src=(UPat.var("x"), UPat()), name="copy"), lambda x,copy: x.f(Ops.NOOP, tag=copy.tag) if x.device == copy.device else None),

  # ** assign rules **

  # assign only to buffer, otherwise make it a CONTIGUOUS
  (UPat(Ops.ASSIGN, src=(UPat(GroupOp.All-{Ops.BUFFER}, name="target"), UPat(name="x")), name="assign"),
   lambda x,target,assign: x.f(Ops.CONTIGUOUS, tag=assign.tag) if ((t:=target.base).op is not Ops.BUFFER and \
       not (t.op is Ops.MSTACK and all(s.op is Ops.BUFFER for s in t.src))) else None),

   # realize before assign if input permutes the target buffer
   (UPat(Ops.ASSIGN, src=(UPat.var("a"), UPat.var("b")), name="assign"), find_permutes),
])

# *****************
# 3.5 cleanups

# Ops.NOOP happens when we have a COPY to the device the Tensor is already on. We treat it like COPY here for MSTACK.
ALWAYS_RUN_OPS = {Ops.CONTIGUOUS, Ops.COPY, Ops.ASSIGN, Ops.NOOP}

# you don't know in the first pass if axes are going to die, this happens if there's an EXPAND to the left
def cleanup_dead_axes(b:UOp):
  # don't optimize ALWAYS_RUN_OPS
  if b.src[0].op in ALWAYS_RUN_OPS: return None

  new_rng = []
  hit = False
  reshape: list[sint] = []
  for s,rng in zip(b.shape, b.src[1:]):
    # skip for symbolic. TODO: fix this
    if rng.op is Ops.RANGE and rng.src[0].op is not Ops.CONST: return None
    # CONSTs are already dead axes
    if rng.op is Ops.CONST or (rng.op is Ops.RANGE and rng not in b.src[0].ranges):
      reshape.append(1)
      hit = True
    else:
      reshape.append(s)
      new_rng.append(rng)
  if hit:
    # move the tag to the expand. NOTE: this expand tag might not survive
    return b.replace(src=b.src[0:1]+tuple(new_rng), tag=None).reshape(tuple(reshape)).expand(b.shape).replace(tag=b.tag)

def gate_substitute(ctx, b:UOp) -> None:
  if not any(r in b.ranges for r in ctx.keys()): raise BottomUpGate()
pm_gate_substitute = PatternMatcher([(UPat(GroupOp.All, name="b"), gate_substitute)], compiled=False)
# if a buffer is being stored just for permutes or something, remove it
# we want to reexpress the indexes of idx2 in terms of the implied b1
def remove_bufferize(src:UOp, buf:UOp, idx:UOp):
  # see if we can't do it, should this ever hit?
  assert len(buf.src) == len(idx.src), f"index on wrong bufferize, {len(buf.src)} != {len(idx.src)}"
  assert all(x.op in {Ops.RANGE, Ops.CONST} for x in buf.src[1:])

  # if it's user contiguous, we never remove it
  if src.op in ALWAYS_RUN_OPS: return None

  # we don't want to bufferize threefry, also causes problems because not all platforms support long
  if src.op is not Ops.THREEFRY:
    # *** here is where we compute the cost ***
    # if we return None, the bufferize is kept

    accessed_buffers: list[UOp] = []
    indexes: list[UOp] = []
    reduces: list[UOp] = []
    def red_gate(x:UOp):
      if x.op is Ops.BUFFERIZE and x.arg.addrspace == AddrSpace.GLOBAL:
        accessed_buffers.append(x)
        return False
      if x.op is Ops.BUFFER:
        accessed_buffers.append(x)
      if x.op is Ops.INDEX:
        indexes.append(x)
      if x.op is Ops.REDUCE: reduces.append(x)
      return True
    src.toposort(gate=red_gate)
    del red_gate
    accessed_buffers = dedup(accessed_buffers)

    # if this is generated from multiple buffers, don't remove this buffer
    if len(accessed_buffers) > 2 and not (PCONTIG > 2): return None

    # if any reduces access a buffer, don't remove this buffer
    buffer_in_reduce = False
    def buf_gate(x:UOp):
      nonlocal buffer_in_reduce
      if x.op in {Ops.BUFFER, Ops.BUFFERIZE}: buffer_in_reduce = True
      return not buffer_in_reduce
    UOp.sink(*[x.src[0] for x in reduces]).toposort(gate=buf_gate)
    del buf_gate
    if buffer_in_reduce:
      if PCONTIG > 2:
        out_in_ratio = (prod(buf.shape)+1) / (sum([x.size for x in accessed_buffers])+1)
        if out_in_ratio < 10: return None
        # here we have to check the indexes, we might do a partial contig here
        local_indexes = [x for x in indexes if x.src[0].op is Ops.BUFFERIZE and x.src[0].arg.addrspace == AddrSpace.LOCAL]
        exclude_ranges = UOp.group(*[UOp.group(*x.src[1:]) for x in local_indexes]).ranges
        subs = [(k,v) for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST]
        # if it's bufferized or a reduce, it's pcontig
        is_pcontig, is_subs = partition(subs, lambda x: x[0] in exclude_ranges or any([r.arg[-1] == AxisType.REDUCE for r in x[1].ranges]))
        if not len(is_subs):
          return None
        if len(is_pcontig):
          ret = src.substitute(dict(is_subs), extra_pm=pm_gate_substitute)
          return ret.bufferize(*[x[0] for x in is_pcontig], arg=BufferizeOpts(None, AddrSpace.LOCAL)).index(*[x[1] for x in is_pcontig])
      else:
        return None

  # if it makes it here, the bufferize is removed
  # this is the ranges replaced
  # NOTE: if buf src is a const, we don't replace it
  return src.substitute({k:v for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST}, extra_pm=pm_gate_substitute)

def remove_noop_bufferize(idx,b2):
  if idx.src[1:] != b2.src[1:] or idx.src[0].op is Ops.BUFFER_VIEW: return None
  new_tag = (idx.src[0].tag or ()) + (b2.tag or ()) or None
  return idx.src[0].rtag(new_tag).shrink(tuple((0, s) for s in b2.shape)) if b2.shape else idx.src[0].rtag(new_tag)

pm_const_buffer_folding = pm_mops+PatternMatcher([
  (UPat(Ops.BUFFERIZE, name="b"), cleanup_dead_axes),
  (UPat(GroupOp.All-{Ops.BUFFERIZE, Ops.BUFFER}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  (UPat((Ops.BUFFERIZE), name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType)
    and (resolve(prod(x.dtype.shape)!=prod(x.shape)) or x.shape[-1]%4!=0) else None),
  # remove noop buffers. if we look at the next index we can remove even more of these
  (UPat(Ops.INDEX, name="idx").f(Ops.BUFFERIZE, allow_any_len=True, name="b2"), remove_noop_bufferize),
  # dont bufferize an arange
  (UPat.any((r:=UPat(dtype=dtypes.index).cast()).named("src"), r.eq(UPat()).named("src")).f(Ops.BUFFERIZE,
    allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_bufferize),
  # no buffers for const
  (UPat(Ops.CONST, name='c').f(Ops.BUFFERIZE, allow_any_len=True, name="b"), lambda c,b: b.const_like(c.arg).rtag(b.tag)),
  # indexing a const is a const
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST, name="c"),),), lambda c: c),
  # copy on CONST is CONST
  (UPat(Ops.COPY, src=(UPat.cvar("x"), UPat()), name="copy"), lambda copy,x: copy.const_like(x.arg)),
  # hack if a noop turned to a const
  (UPat.cvar("c").f(Ops.NOOP).f(Ops.BUFFERIZE, allow_any_len=True, name="buf"), lambda c,buf: buf.replace(src=(c,)+buf.src[1:])),
  # mstack on CONST is CONST
  (UPat(Ops.MSTACK, src=(UPat.var("s"),), allow_any_len=True).f(Ops.INDEX, allow_any_len=True),
   lambda s: UOp.const(c.dtype, c.arg) if (c:=s.base).op is Ops.CONST else None),
])

def pre_bufferize(b:UOp, x:UOp, copy:UOp):
  nb = b.replace(src=(b.src[0].contiguous(),)+b.src[1:])
  return copy.replace(src=(x.replace(src=(nb,)+x.src[1:]), copy.src[1]))
pm_remove_bufferize = PatternMatcher([
  # hack so remove_bufferize doesnt remove the buffer before a copy
  (UPat(Ops.COPY, src=(UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.COPY}).f(Ops.BUFFERIZE, allow_any_len=True, name="b")
                      .f(Ops.INDEX, allow_any_len=True, name="x"), UPat()), name="copy"), pre_bufferize),
  # remove reindexing with cost function
  (UPat.var("src").f(Ops.BUFFERIZE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_bufferize),
])

def late_buffer_view(t:UOp, b:UOp):
  if isinstance(b.device, str) and (b.device.startswith("DISK") or b.device.startswith("TINYFS")):
    shape = b.shape
    size = prod(shape)

    # walk up for the INDEX
    x = t
    while not any(u.op is Ops.INDEX for u in x.src):
      assert x.op not in GroupOp.Elementwise, "can't buffer view elementwise"
      x = x.src[0]
    x = next(u for u in x.src if u.op is Ops.INDEX)

    if len(shape) == 0: offset = x.src[1].arg
    else: offset = max(sum(idx.vmin for idx in x.src[1:]), 0)

    return b.replace(src=(UOp(Ops.BUFFER_VIEW, t.dtype, (x.base,), (size, offset), tag=t.tag),) + b.src[1:])
  return b
to_bufferview = PatternMatcher([
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), name="t").f(Ops.BUFFERIZE, allow_any_len=True, name="b"), late_buffer_view),
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS)).f(Ops.BUFFER_VIEW, name="b"), lambda b: b.replace(src=b.src[0].src)),
])

DEVICE_MAX_BUFS = {"METAL": 31, "WEBGPU": 8} # TODO: get from device?
def limit_bufs(ctx:IndexingContext, root:UOp):
  if (device:=root._device) is None: return None # no device, index related calculations
  device = device if isinstance(device, str) else device[0].split(":")[0]
  if not (MAX_BUFS:=getenv("MAX_KERNEL_BUFFERS", DEVICE_MAX_BUFS.get(device, 0))): return None

  bufs: set[UOp] = set()
  def gate_input(u:UOp):
    # TODO: add cache to fix n^2
    if is_load:=(u.op in {Ops.BUFFERIZE, Ops.AFTER, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK, Ops.DEFINE_VAR}): bufs.add(u)
    return not is_load
  root.toposort(gate=gate_input)

  if len(bufs) > MAX_BUFS - 1: # NOTE: this -1 is for the output buffer
    srcs = []
    for s in root.src:
      if s.op in GroupOp.Elementwise:
        # Insert bufferize: all AxisType.REDUCE before bufferize are AxisType.LOOP
        orig_ranges, end_ranges = s.ranges, [x.replace(arg=(next(ctx.range_idx), AxisType.LOOP)) if x.op is Ops.RANGE else x for x in s.ranges]
        s = s.substitute(dict(zip(orig_ranges, end_ranges))).bufferize(*end_ranges, arg=BufferizeOpts(device=s.device)).index(*orig_ranges)
      srcs.append(s)
    return root.replace(src=tuple(srcs))
pm_limit_bufs = PatternMatcher([(UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), limit_bufs)])

# *****************
# 4. put in buffers for bufferize
# TODO: should BUFFERIZE look a lot more like STORE
# BUFFERIZE has device in arg
# BUFFERIZE doesn't have indexing, that's implied by the ranges it closes
# BUFFERIZE returns the BUFFER ready for INDEXing (doing this will make splitting a lot easier)
# NOTE: this has been fixed up a bit

def bufferize_to_store(x:UOp, allow_locals=True):
  rngs = x.src[1:]
  shape = x.shape
  size = prod(shape)
  assert size > 0 and isinstance(size, int), f"no zero sized or symbolic sized buffers {shape}"

  sdtype = x.dtype.ptr(size=size, addrspace=x.arg.addrspace)
  if x.src[0].op is Ops.ASSIGN:
    assign_target, assign_src, assign_mops = x.src[0].src
    assert assign_target.op is Ops.INDEX, f"{assign_target.op} is not index"
    # in assign, this is the buffer size, not the bufferize size
    # TODO: assign_mops here
    do_store = assign_target.replace(dtype=sdtype).store(assign_src, tag=x.tag).end(*[x for x in rngs if x.op is Ops.RANGE])
    ret = assign_target.src[0].after(do_store)
    mops = []
    walk = assign_mops
    while walk is not assign_mops.base:
      mops.append((walk.op, walk.marg))
      walk = walk.src[0]
    for m in mops[::-1]: ret = ret._mop(*m)
    return ret.forced_reshape(shape).replace(tag=x.tag)

  # NOTE: the DEFINE_LOCAL needs to be disambiguated here
  if sdtype.addrspace == AddrSpace.GLOBAL:
    buf = UOp.new_buffer(x.arg.device, size, x.dtype)
    do_store = buf.reshape(shape).index(*rngs, dtype=sdtype).store(x.src[0], tag=x.tag).end(*[x for x in rngs if x.op is Ops.RANGE])
    ret = buf.after(do_store).forced_reshape(shape)
    # TODO: is this right? what if it's offset
    if any(r.op is Ops.RANGE and r.src[0].op is not Ops.CONST for r in rngs):
      sym_shape = tuple([ssimplify(r.src[0]) if r.op is not Ops.CONST else 1 for r in rngs])
      ret = ret.shrink(tuple([(0,x) for x in sym_shape]))
    return ret.replace(tag=x.tag)

  if allow_locals:
    # handle locals
    tag = x.arg.device
    if tag is None: tag = UOp.unique().arg # TODO: hack
    buf = UOp(Ops.DEFINE_LOCAL, sdtype, arg=tag)
    do_store = buf.reshape(shape).index(*rngs, dtype=sdtype).store(x.src[0]).end(*[x for x in rngs if x.op is Ops.RANGE])
    return buf.after(do_store.barrier()).reshape(shape)

pm_add_buffers = pm_mops+to_bufferview+PatternMatcher([
  (UPat(Ops.BUFFERIZE, name="x"), lambda x: bufferize_to_store(x, allow_locals=False)),

  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src]), tag=None).reshape(m.shape).rtag(m.tag)),
])

pm_add_buffers_local = pm_mops+to_bufferview+PatternMatcher([
  (UPat(Ops.BUFFERIZE, name="x"), bufferize_to_store),
])

# *****************
# 5. split into kernels

@dataclass
class LocalAddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)
  vars:dict = field(default_factory=dict)
  range:int = 0
  parent_tags:list = field(default_factory=list)
  opts:tuple|None = None

def debuf(ctx:LocalAddBufferContext, buf:UOp):
  ret = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(buf.arg), arg=ctx.dg)
  if buf not in ctx.map: ctx.map[buf] = buf
  ctx.dg += 1
  return ret

def unbind_kernel(ctx:LocalAddBufferContext, b:UOp):
  ctx.vars[b] = None
  return b.src[0]

def handle_after(ctx:LocalAddBufferContext, after:UOp):
  if isinstance(after.dtype, PtrDType) and after.ptrdtype.addrspace == AddrSpace.LOCAL: return None
  buf = after.as_buf()
  # HACK to put the buffer in the MAP instead of MSTACK/MSELECT
  if buf.op in {Ops.MSTACK, Ops.MSELECT}: buf = buf.src[0]
  assert buf not in ctx.map
  ctx.map[buf] = after
  return buf

def renumber_range(ctx:LocalAddBufferContext, r:UOp):
  if r.tag is not None: return None
  ret = r.replace(arg=(ctx.range,)+r.arg[1:], tag=())
  ctx.range += 1
  return ret

def find_bufs(x:UOp):
  idxs = [s for s in x.toposort(gate=lambda x: x.op is not Ops.AFTER) if s.op is Ops.INDEX]
  read_from: dict[UOp, Ops] = {}
  if any((buf:=idx.as_buf()).op is Ops.BUFFER and read_from.setdefault(buf, op:=idx.src[0].op) is not op for idx in idxs):
    raise RuntimeError(f"cycle detected while indexing {buf}")

to_define_global = PatternMatcher([
  (UPat(Ops.STORE, name="x"), find_bufs),
  (UPat(Ops.BUFFER, name="buf"), debuf),
  (UPat(Ops.BIND, name="b"), unbind_kernel),
  (UPat((Ops.MSTACK, Ops.MSELECT, Ops.AFTER), name="after"), handle_after),

  # HACK in case any CONSTs were replaced
  # this is only needed if you are using symbolic
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"), lambda c: c.replace(src=()) if len(c.src) else None),

  # remove RANGE with 0 size
  (UPat(Ops.RANGE, name="r"), lambda r: UOp.const(dtypes.index, 0) if r.vmax == 0 else None),

  # renumber the ranges starting with 0 so that kernel deduping works
  (UPat(Ops.RANGE, name="r"), renumber_range),
])

def get_contiguous(ctx:LocalAddBufferContext, x:UOp):
  if isinstance(x.arg, tuple) and all(isinstance(y, Opt) for y in x.arg): ctx.opts = x.arg
  return x.src[0]

rangeify_codegen = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="x"), get_contiguous),

  # no NOOP in the kernel graph
  # TODO: this can be moved into codegen?
  (UPat(Ops.NOOP, name="x"), lambda x: x.src[0]),

  # strip the arg from store
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(arg=None) if x.arg is not None else None),

  # add loads to non ptr indexes
  # TODO: this can be moved into codegen?
  (UPat.any(UPat(Ops.DEFINE_GLOBAL, name="dg"), UPat(Ops.DEFINE_LOCAL).f(Ops.AFTER, allow_any_len=True, name="dg"))
   .f(Ops.INDEX, name="idx", allow_any_len=True),
    lambda dg,idx: None if isinstance(idx.dtype, (PtrDType, ImageDType)) else idx.replace(dtype=dg.dtype, arg=None).load()),
])

def remove_metadata_tags(ctx:LocalAddBufferContext, x:UOp):
  if x.tag is None or x.tag == (): return None
  ctx.parent_tags += list(x.tag)
  return x.replace(tag=None)

pm_remove_tags = PatternMatcher([
  # remove all the tags
  (UPat(GroupOp.All, name="x"), remove_metadata_tags),
])

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  metadata: tuple[Metadata, ...] = ()
  def __repr__(self):
    ast_rep = f"SINK{tuple(s.op for s in self.ast.src)}" if self.ast.op is Ops.SINK else repr(self.ast.op)
    return f"<Kernel {len(list(self.ast.toposort()))} {ast_rep} {self.metadata}>"

def split_store(ctx:list[UOp], x:UOp) -> UOp|None:
  if len(x.ranges): return None

  # local kernel rewrite
  lctx = LocalAddBufferContext()
  ret = graph_rewrite(x, to_define_global+pm_flatten_range+rangeify_codegen+pm_remove_tags, ctx=lctx, name="kernel split", bottom_up=True)

  # gather the metadata
  metadatas = [ctx[y].metadata for y in lctx.parent_tags]

  # NOTE: the hack for COPY is here
  for u in ret.toposort():
    # TODO: this can be wrong if there's multiple of these
    if u.op in {Ops.COPY, Ops.BUFFER_VIEW}:
      ret = u
      break
  else:
    ret = ret.sink(arg=KernelInfo(opts_to_apply=lctx.opts) if lctx.opts is not None else None)

  kernel_arg = Kernel(ret,tuple(dedup(flatten([x for x in metadatas if x is not None])))[::-1])
  kernel = UOp(Ops.KERNEL, src=tuple(lctx.map.values())+tuple(lctx.vars.keys()), arg=kernel_arg)
  if ret.op is Ops.SINK and not all_same([x.device for x in kernel.src if x.op is not Ops.BIND]):
    raise RuntimeError(f"all buffers must be on the same device: {tuple(b.buf_uop.buffer for b in kernel.src)}")
  return kernel

split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="x"), split_store),
])

def tag_uop(ctx:list[UOp], x:UOp):
  if x.tag is not None: return None
  if x.dtype.scalar() == dtypes.index: return None
  ctx.append(x)
  return x.replace(tag=(len(ctx)-1,))
add_tags = PatternMatcher([
  # don't tag BUFFERs, they are global
  (UPat(GroupOp.All-{Ops.BUFFER, Ops.CONST, Ops.DEVICE, Ops.UNIQUE, Ops.DEFINE_VAR, Ops.BIND,
                     Ops.MSTACK, Ops.MSELECT, Ops.RANGE}.union(GroupOp.Movement), name="x"), tag_uop),
  (UPat({Ops.MSTACK, Ops.MSELECT}, name="x"), lambda ctx,x: None if all(s.op is Ops.BUFFER for s in x.src) else tag_uop(ctx, x)),
])

# support for using a contiguous permuted view instead of the parent view if one exists
# modified from kernelize.py to not use ShapeTracker

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  x = src
  while x is not src.base:
    if x.op is Ops.PERMUTE: contig = contig.permute(argsort(x.marg))
    elif x.op is Ops.RESHAPE: contig = contig.reshape(x.src[0].shape)
    else: return None
    x = x.src[0]
  ctx[src.base] = contig
replace_contiguous = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, src=(UPat(GroupOp.Movement, name="src"),), name="contig"), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
])

@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len([u for u in UOp.sink(*ret.values()).toposort() if u.op is Ops.KERNEL]))}", True)
def get_rangeify_map(sink:UOp) -> dict[UOp, UOp]:
  if getenv("VIZ"): graph_rewrite(sink, PatternMatcher([]), name="View Input Graph")
  uop_list: list[UOp] = []
  tsink = graph_rewrite(sink, add_tags, ctx=uop_list, bottom_up=True, name="number the uops")

  tsink = graph_rewrite(tsink, earliest_rewrites+replace_contiguous, ctx={}, name="earliest rewrites")

  # convert movement ops to ranges
  tsink, rctx = run_rangeify(tsink, DEBUG_RANGEIFY)

  tsink = graph_rewrite(tsink, symbolic_flat+pm_reduce_simplify+pm_const_buffer_folding, name="symbolic+reduce_collapse")  # this does const folding
  tsink = graph_rewrite(tsink, pm_remove_bufferize, bottom_up=True, name="remove bufferize with cost function")
  tsink = graph_rewrite(tsink, pm_limit_bufs, ctx=rctx, name="limit buffers")

  # rebuild the sink with all the BUFFERIZEs with tags, this is what's ending up in the tensor graph
  # MSTACK stacks multiple BUFFERIZEs in one tagged tensor
  # if it's not tagged by here, it's out
  tsink = UOp.sink(*[x for x in tsink.backward_slice if x.base.op in {Ops.BUFFERIZE, Ops.MSTACK, Ops.CONST, Ops.BUFFER} and \
                     x.tag is not None and len(x.tag)])

  if getenv("VIZ"): graph_rewrite(tsink, PatternMatcher([]), name="View Tagged Rangeify")

  # bufferize -> store
  tsink = graph_rewrite(tsink, pm_add_buffers, bottom_up=True, name="bufferize to store")
  tsink = graph_rewrite(tsink, split_kernels, ctx=uop_list, name="split kernels")

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in tsink.toposort():
    if u.op is not Ops.AFTER: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      # TODO: this is probably broken for MSELECT/MSTACK
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.AFTER and x.buf_uop is s for x in u.toposort()):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on AFTER or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep: tsink = graph_rewrite(tsink, _substitute, ctx=assign_rep, bottom_up=True, name="fix_assign")

  if getenv("VIZ"): graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")

  becomes_map: dict[UOp, UOp] = {}
  for s in tsink.src:
    assert s.tag is not None
    for a in s.tag:
      if a is None: continue
      becomes_map[uop_list[cast(int, a)]] = s.replace(tag=None)
  return becomes_map
