from typing import Any, cast
import itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, AddrSpace, Invalid, ImageDType, PtrDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp
from tinygrad.helpers import getenv, IMAGE
from tinygrad.renderer import Renderer
from tinygrad.uop.symbolic import uop_given_valid
from tinygrad.codegen.late.devectorizer import simplify_valid_image_load, _drop_valid_stmts

pm_imageh_store = PatternMatcher([
  # store<imageh>(idx, x) is actually store(idx, x.cast(half)) so we can pull the cast into the store
  (UPat(Ops.CAST, src=(UPat.var("x"),), name="c"), lambda x,c: x if c.dtype.scalar() == dtypes.half and x.dtype.scalar() == dtypes.float else None),
  # store(imageh, a.where(b.half(), c).float()) -> store(imageh, a.where(b, c.float()))
  (UPat(Ops.WHERE, src=(UPat.var("a"), UPat.var("b", dtypes.float).cast(dtypes.half), UPat.var("c"))), lambda a,b,c: a.where(b,c.cast(dtypes.float))),
  (UPat(Ops.STACK, name="x"), lambda x: UOp.vectorize(*(s.cast(dtypes.float) for s in x.src)) if x.dtype.scalar() == dtypes.half else None),
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
      if ibuf is None and buf.op is Ops.PARAM and buf.addrspace is AddrSpace.GLOBAL and isinstance(buf.dtype, PtrDType) and \
         (dims:=ImageDType.valid_dims(buf.dtype, ctx.target.arch)):
        ibuf = buf.replace(dtype=(dtypes.imageh if buf.dtype.base == dtypes.half else dtypes.imagef)((*dims[0], 4)))
      if ibuf is not None: image_bufs[buf] = ibuf

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
    lengths.append(1)  # worst case, it's not folded
    # do the grouping
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for full_grp in grouped_offsets:
      while len(full_grp):
        offset = (base+full_grp[0]) if isinstance(base, UOp) else UOp.const(dtypes.int, full_grp[0])
        length = [l for l in lengths if l <= len(full_grp) and (not must_divide or offset.divides(l) is not None)][0]
        grp = full_grp[:length]
        ibuf, lane0 = image_bufs.get(buf) if length == 4 else None, None
        if ibuf is None and op is Ops.LOAD and buf in image_bufs and (lane:=(offset%4).simplify()).op is Ops.CONST:
          ibuf, lane0 = image_bufs[buf], lane.arg
        idx_valid = valid
        if ibuf is not None:
          if valid is not None and not isinstance(buf.dtype, ImageDType) and isinstance(buf.dtype, PtrDType):
            best_drop, cands = -1, []
            for h,w in ImageDType.valid_dims(buf.dtype, ctx.target.arch):
              cidx = uop_given_valid(valid, UOp.vectorize((offset // 4) % w, offset // (4*w)))
              if (dropped:=len(_drop_valid_stmts(valid, cidx, h, w))) > best_drop: best_drop, cands = dropped, [(h, w, cidx)]
              elif dropped == best_drop: cands.append((h, w, cidx))
            if cands:
              h, w, _ = cands[0] if len(cands) == 1 else min(cands, key=lambda cand: len(cand[2].gep(1).simplify().backward_slice))
              ibuf = buf.replace(dtype=(dtypes.imageh if buf.dtype.base == dtypes.half else dtypes.imagef)((h, w, 4)))
          idt = cast(ImageDType, ibuf.dtype)
          idx_y, idx_x = (offset // (4*idt.shape[1])).simplify(), ((offset // 4) % idt.shape[1]).simplify()
          idx = simplify_valid_image_load(ibuf, idx_y, idx_x, valid) if valid is not None else None
          if idx is None: idx = ibuf.index(idx_y, idx_x, ptr=True)
          idx_valid = None
        else:
          idx = buf._mop(Ops.SHRINK, arg=[(offset, len(grp))]) if len(grp) > 1 else buf.index(offset)
        if op == Ops.STORE:
          datas = []
          for i,g in enumerate(grp):
            assert len(offsets[g]) == 1, f"attempting multiple stores: {len(offsets[g])}"
            datas.append(offsets[g][0].src[1])
          data = UOp.vectorize(*datas) if len(datas) > 1 else datas[0]
          if ibuf is not None and ibuf.dtype.itemsize == 2: data = pm_imageh_store.rewrite(data)
          store = idx.store(data, idx_valid) if idx_valid is not None else idx.store(data)
          for i,g in enumerate(grp): replacements[offsets[g][0]] = store
        else:
          ld = idx.load(idx.vconst_like(0), idx_valid) if idx_valid is not None else idx.load()
          for i,g in enumerate(grp):
            for oo in offsets[g]:
              ret = ld.index(UOp.const(dtypes.int, lane0 if lane0 is not None else i)) if ibuf is not None or len(grp) > 1 else ld
              replacements[oo] = ret.cast(buf.dtype.base) if ibuf is not None and buf.dtype.base != dtypes.float else ret
        full_grp = full_grp[length:]

  # apply
  return sink.substitute(replacements, name="memory coalesing")
