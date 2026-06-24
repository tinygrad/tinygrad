from typing import Any
import itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, AddrSpace, Invalid, ImageDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp
from tinygrad.helpers import getenv, IMAGE
from tinygrad.renderer import Renderer

pm_imageh_store = PatternMatcher([
  # store<imageh>(idx, x) is actually store(idx, x.cast(half)) so we can pull the cast into the store
  (UPat.var("x", dtypes.float).cast(dtypes.half), lambda x: x),
  # store(imageh, a.where(b.half(), c).float()) -> store(imageh, a.where(b, c.float()))
  (UPat(Ops.WHERE, src=(UPat.var("a"), UPat.var("b", dtypes.float).cast(dtypes.half), UPat.var("c"))), lambda a,b,c: a.where(b,c.cast(dtypes.float))),
  # otherwise, we cast to float
  (UPat(GroupOp.All, name="x"), lambda x: x.cast(dtypes.float))
])

def memory_coalesing(sink:UOp, ctx:Renderer) -> UOp:
  if getenv("DMC"): return sink

  # collect
  memory: defaultdict[tuple[Ops, UOp, Any, Any], dict[int, list[UOp]]] = defaultdict(dict)
  for u in sink.toposort():
    if u.op in {Ops.LOAD, Ops.STORE}:
      assert len(u.src) == (2 if u.op is Ops.STORE else 1), "memory coalesing does not support gated loads/stores"
      if u.src[0].op is not Ops.INDEX or len(u.src[0].src) != 2: continue
      buf, idx_u = u.src[0].src
      if buf.addrspace == AddrSpace.REG: continue
      idx: Any = idx_u.src[1] if idx_u.op is Ops.WHERE and idx_u.src[2].arg is Invalid else idx_u
      valid: Any = idx_u.src[0] if idx_u.op is Ops.WHERE and idx_u.src[2].arg is Invalid else None
      if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
      elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: root_src, arg = idx.src[1], idx.src[0].arg
      elif idx.op is Ops.CONST and idx.arg is Invalid: root_src, arg = "INVALID", 0
      elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
      else: root_src, arg = idx, 0
      memory[(u.op, buf, root_src, valid)].setdefault(arg, []).append(u)

  image_bufs = {}
  if IMAGE and ctx is not None and ctx.target.device in {"QCOM", "CL", "PYTHON", "NULL"}:
    for _,buf,_,_ in memory:
      if buf in image_bufs: continue
      ibuf = buf if isinstance(buf.dtype, ImageDType) else None
      if ibuf is None and buf.op is Ops.PARAM and buf.addrspace is AddrSpace.GLOBAL and (dims:=ImageDType.valid_dims(buf.dtype, ctx.target.arch)):
        ibuf = buf.replace(dtype=(dtypes.imageh if buf.dtype.base == dtypes.half else dtypes.imagef)((*dims[0], 4)))
      if ibuf is None: continue
      can_image = True
      for (_,membuf,base,_), offsets in memory.items():
        if membuf is not buf: continue
        for full_grp in [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]:
          while full_grp:
            offset = (base+full_grp[0]) if isinstance(base, UOp) else UOp.const(dtypes.int, full_grp[0])
            if len(full_grp) < 4 or offset.divides(4) is None:
              can_image = False
              break
            full_grp = full_grp[4:]
          if not can_image: break
        if not can_image: break
      if can_image: image_bufs[buf] = ibuf

  # build replacements
  replacements = {}
  for (op,buf,base,valid),offsets in memory.items():
    if isinstance(buf.dtype, ImageDType) and buf not in image_bufs: continue
    # allowed lengths (copied in)
    lengths = []
    must_divide = True
    if buf in image_bufs:
      lengths = [4]
    elif ctx is not None and ctx.target.device == "DSP":
      lengths = [128,64,32,16,8,4]
      must_divide = False
    elif buf.dtype.base not in (dtypes.float, dtypes.half, *dtypes.fp8s) and not isinstance(buf.dtype, ImageDType):
      pass
    elif buf.addrspace == AddrSpace.REG:
      pass
    elif ctx is not None and ctx.supports_float4:
      # TODO: a better way to get this than ctx
      lengths = [8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else [4,2]
    if buf not in image_bufs: lengths.append(1)  # worst case, it's not folded
    # do the grouping
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for full_grp in grouped_offsets:
      while len(full_grp):
        offset = (base+full_grp[0]) if isinstance(base, UOp) else UOp.const(dtypes.int, full_grp[0])
        length = [l for l in lengths if l <= len(full_grp) and (not must_divide or offset.divides(l) is not None)][0]
        grp = full_grp[:length]
        ibuf = image_bufs.get(buf) if length == 4 else None
        if ibuf is not None:
          pix = offset // 4
          idx = ibuf.index(pix // ibuf.dtype.shape[1], pix % ibuf.dtype.shape[1], ptr=True)
        else:
          idx = buf._mop(Ops.SHRINK, arg=[(offset, len(grp))]) if len(grp) > 1 else buf.index(offset)
        if op == Ops.STORE:
          datas = []
          for i,g in enumerate(grp):
            assert len(offsets[g]) == 1, f"attempting multiple stores: {len(offsets[g])}"
            datas.append(offsets[g][0].src[1])
          data = UOp.vectorize(*datas) if len(datas) > 1 else datas[0]
          if ibuf is not None and ibuf.dtype.itemsize == 2: data = pm_imageh_store.rewrite(data)
          store = idx.store(data, valid) if valid is not None else idx.store(data)
          for i,g in enumerate(grp): replacements[offsets[g][0]] = store
        else:
          ld = idx.load(idx.vconst_like(0), valid) if valid is not None else idx.load()
          if ibuf is not None and buf.dtype.base != dtypes.float: ld = ld.cast(buf.dtype.base)
          for i,g in enumerate(grp):
            for oo in offsets[g]:
              replacements[oo] = ld.index(UOp.const(dtypes.int, i)) if len(grp) > 1 else ld
        full_grp = full_grp[length:]

  # apply
  return sink.substitute(replacements, name="memory coalesing")
