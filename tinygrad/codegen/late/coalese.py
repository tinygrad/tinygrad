from typing import Any
import itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, AddrSpace, Invalid
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import getenv

def memory_coalesing(sink:UOp):
  if getenv("DMC"): return sink

  # collect
  memory: defaultdict[tuple[UOp, UOp, UOp], dict[int, list[UOp]]]  = defaultdict(dict)
  for u in sink.toposort():
    if u.op in {Ops.LOAD, Ops.STORE} and u.src[0].addrspace != AddrSpace.REG:
      assert u.src[0].op is Ops.INDEX
      buf,idx_u = u.src[0].src
      idx: Any = idx_u.src[1] if idx_u.op is Ops.WHERE and idx_u.src[2].arg is Invalid else idx_u
      valid: Any = idx_u.src[0] if idx_u.op is Ops.WHERE and idx_u.src[2].arg is Invalid else None
      if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
      elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: root_src, arg = idx.src[1], idx.src[0].arg
      elif idx.op is Ops.CONST and idx.arg is Invalid: root_src, arg = "INVALID", 0
      elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
      else: root_src, arg = idx, 0
      memory[(u.op, buf, root_src, valid)].setdefault(arg, []).append(u)

  # allowed lengths
  lengths = [8,4,2,1]

  # build replacements
  replacements = {}
  for (op,buf,base,valid),offsets in memory.items():
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for full_grp in grouped_offsets:
      while len(full_grp):
        offset = (base+full_grp[0]) if isinstance(base, UOp) else UOp.const(dtypes.weakint, full_grp[0])
        length = [l for l in lengths if l <= len(full_grp) and offset.divides(l) is not None][0]
        grp = full_grp[:length]
        idx = buf._mop(Ops.SHRINK, arg=[(offset, len(grp))]) if len(grp) > 1 else buf.index(offset)
        if op is Ops.STORE:
          datas = []
          for i,g in enumerate(grp):
            assert len(offsets[g]) == 1
            datas.append(offsets[g][0].src[1])
          data = UOp.vectorize(*datas) if len(datas) > 1 else datas[0]
          store = idx.store(data, valid) if valid is not None else idx.store(data)
          for i,g in enumerate(grp): replacements[offsets[g][0]] = store
        else:
          ld = idx.load(idx.vconst_like(0), valid) if valid is not None else idx.load()
          for i,g in enumerate(grp):
            for oo in offsets[g]:
              replacements[oo] = ld.index(UOp.const(dtypes.int, i)) if len(grp) > 1 else ld
        full_grp = full_grp[length:]

  # apply
  return sink.substitute(replacements, name="memory coalesing")

