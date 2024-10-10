# this is the new scheduler
from typing import Dict, List, Tuple, cast
import functools
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.ops import REDUCE_ALU, MetaOps, PatternMatcher, ReduceOps, UOp, UOps, UPat, UnaryOps, graph_rewrite
from tinygrad.shape.shapetracker import ShapeTracker

new_sched = PatternMatcher([
  # remove non swizzled load/stores
  (UPat.load(UPat(UOps.BUFFER, name="buf"), UPat(UOps.SHAPETRACKER, name="st"),
             UPat.store(UPat(UOps.BUFFER, name="buf"), UPat(UOps.SHAPETRACKER, name="st"),
                        UPat(UOps.ALU, name='inp'))), lambda buf,st,inp: inp),
  # rewrite const load/store to valid (or just const)
  (UPat.load(UPat(UOps.BUFFER, name="buf"), UPat(UOps.SHAPETRACKER, name="st1"),
             UPat.store(UPat(UOps.BUFFER, name="buf"), UPat(UOps.SHAPETRACKER, name="st2"),  # assuming st2 is contig
                        UPat.cvar('c'))),
    lambda buf,st1,st2,c: UOp.where(UOp(UOps.VALID, dtypes.bool, (st1,)), c, c.const_like(0)) if any(x.mask for x in st1.arg.views) else c),
])

def append_kernel(k:List[UOp], base:UOp):
  # rewrite store of size 0 to nothing
  if base.op is UOps.STORE and base.st_arg.size == 0: return None
  k.append(base.sink() if base.op is UOps.STORE else base)
break_sched = PatternMatcher([
  (UPat((UOps.EXT, UOps.STORE), name="base"), append_kernel),
  (UPat(UOps.LOAD, src=(UPat(), UPat(), UPat()), name="ld"), lambda k,ld: UOp.load(ld.src[0], ld.src[1], dtype=ld.dtype)),
])

def _lazy_to_uop(outs:List[LazyBuffer]) -> Tuple[UOp, List[Buffer], Dict[Buffer, List[LazyBuffer]]]:
  buf_uops:Dict[Buffer, UOp] = {}
  bufs_by_number:List[Buffer] = []
  buf_to_lbs:Dict[Buffer, List[LazyBuffer]] = {}
  @functools.lru_cache(None)
  def __st_to_uop(st:ShapeTracker) -> UOp: return st.to_uop()
  @functools.lru_cache(None)
  def __lazy_to_uop(lb:LazyBuffer) -> UOp:
    # assign a buffer (should be deduped to remove lazycache!)
    lbuf = lb.base.buffer
    if lb.base not in buf_to_lbs.setdefault(lbuf, []): buf_to_lbs[lbuf].append(lb.base)
    if lbuf not in buf_uops:
      buf_uops[lbuf] = ubuf = UOp(UOps.BUFFER, lb.dtype.ptr(), (), (len(buf_uops), (lbuf.device, lbuf.size, lbuf.dtype)))
      bufs_by_number.append(lbuf)
    else: ubuf = buf_uops[lbuf]
    if lb.is_realized(): return ubuf
    if lb._base is None:
      # this is a base
      if lb.op in MetaOps:
        usrcs = (ubuf,) + tuple(__lazy_to_uop(x) for x in lb.srcs)
        if lb.op is MetaOps.CONST: out = UOp.const(lb.dtype, lb.arg)
        elif lb.op is MetaOps.CONTIGUOUS:
          # TODO: mark graph to not be broken here
          x = lb.srcs[0]
          x_uop = __lazy_to_uop(x)
          out = UOp.load(buf_uops[x.base.buffer], __st_to_uop(x.st), x_uop, dtype=x.dtype)
        else:
          return UOp(UOps.EXT, lb.dtype, usrcs, (lb.op, lb.arg))
      else:
        # load all the inputs
        uop_srcs = []
        for x in lb.srcs:
          x_uop = __lazy_to_uop(x)
          uop_srcs.append(UOp.load(buf_uops[x.base.buffer], __st_to_uop(x.st), x_uop, dtype=x.dtype))
        if lb.op in ReduceOps:
          # reduce node
          out = UOp(UOps.REDUCE_AXIS, lb.dtype, tuple(uop_srcs), (REDUCE_ALU[cast(ReduceOps, lb.op)], lb.arg))
        else:
          if lb.op is UnaryOps.CAST: out = UOp(UOps.CAST, lb.dtype, tuple(uop_srcs))
          elif lb.op is UnaryOps.BITCAST: out = UOp(UOps.BITCAST, lb.dtype, tuple(uop_srcs))
          else: out = UOp(UOps.ALU, lb.dtype, tuple(uop_srcs), lb.op)
      return UOp.store(ubuf, __st_to_uop(lb.st), out)
    else:
     return __lazy_to_uop(lb.base) # NOOP for a view
  return UOp.sink(*[__lazy_to_uop(x) for x in outs]), bufs_by_number, buf_to_lbs

number_bufs = PatternMatcher([(UPat(UOps.BUFFER, name="x"), lambda ctx,x: UOp(UOps.DEFINE_GLOBAL, x.dtype, (), ctx.index(x.arg[0])))])

def create_schedule(outs:List[LazyBuffer]) -> Tuple[List[UOp], List[Tuple[Buffer, ...]]]:
  # rewrite LazyBuffers into the big graph!
  sink, bufs_by_number, buf_to_lbs = _lazy_to_uop(outs)
  sink = graph_rewrite(sink, new_sched)

  # break on STORE/EXT boundaries (no multioutput). implied DFS from the graph_rewrite
  graph_rewrite(sink, break_sched, kernels:=[])

  # this describes the compute
  uops: List[UOp] = []
  # this describes the data
  bufs: List[Tuple[Buffer, ...]] = []
  for k in kernels:
    # delete srcs of the buffers we schedule
    store_number = k.src[0].arg[0] if k.op is UOps.EXT else k.src[0].src[0].arg[0]
    for lb in buf_to_lbs[bufs_by_number[store_number]]: del lb.srcs

    numbered = tuple(x.arg[0] for x in k.sparents if x.op is UOps.BUFFER)
    ast = graph_rewrite(k, number_bufs, numbered)
    if ast.op is UOps.EXT: ast = ast.replace(src=())
    uops.append(ast)
    bufs.append(tuple(bufs_by_number[x] for x in numbered))
  return uops, bufs
