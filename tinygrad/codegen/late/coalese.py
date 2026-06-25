from typing import Any
import itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, AddrSpace, Invalid, ImageDType
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import getenv
from tinygrad.renderer import Renderer

def memory_coalesing(sink:UOp, ctx:Renderer) -> UOp:
  if getenv("DMC"): return sink

  # collect
  memory: defaultdict[tuple[Ops, UOp, Any, Any], dict[int, list[UOp]]]  = defaultdict(dict)
  for u in sink.toposort():
    # TODO: this should handle images too, it's just memory coalesing
    if u.op in {Ops.LOAD, Ops.STORE} and not isinstance(u.src[0].src[0].dtype, ImageDType):
      assert len(u.src) == (2 if u.op is Ops.STORE else 1), "memory coalesing does not support gated loads/stores"
      if u.src[0].op is not Ops.INDEX: continue
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

  # build replacements
  replacements = {}
  for (op,buf,base,valid),offsets in memory.items():
    # allowed lengths (copied in)
    lengths = []
    must_divide = True
    if ctx is not None and ctx.target.device == "DSP":
      lengths = [128,64,32,16,8,4]
      must_divide = False
    elif buf.dtype.base not in (dtypes.float, dtypes.half, *dtypes.fp8s) and not isinstance(buf.dtype, ImageDType):
      pass
    elif buf.addrspace == AddrSpace.REG:
      pass
    elif isinstance(buf.dtype, ImageDType):
      lengths = [4]
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
        # NOTE: we apply the valid again after we determine the length
        offset = valid.where(offset, UOp(Ops.CONST, offset.dtype, arg=Invalid)) if valid is not None else offset
        idx = UOp(Ops.SHRINK, dtype=buf.dtype, src=(buf, offset, UOp.const(dtypes.int, len(grp)))) if len(grp) > 1 else buf.index(offset)
        if op == Ops.STORE:
          datas = []
          for i,g in enumerate(grp):
            assert len(offsets[g]) == 1, f"attempting multiple stores: {len(offsets[g])}"
            datas.append(offsets[g][0].src[1])
          store = idx.store(UOp._stack(*datas) if len(datas) > 1 else datas[0])
          for i,g in enumerate(grp): replacements[offsets[g][0]] = store
        else:
          ld = idx.load()
          for i,g in enumerate(grp):
            for oo in offsets[g]:
              replacements[oo] = ld.index(UOp.const(dtypes.int, i)) if len(grp) > 1 else ld
        full_grp = full_grp[length:]

  # apply
  return sink.substitute(replacements, name="memory coalesing")
