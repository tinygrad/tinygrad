from typing import Any
import itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, AddrSpace, Invalid, ImageDType, PtrDType
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, GroupOp
from tinygrad.helpers import getenv, IMAGE, all_same
from tinygrad.renderer import Renderer
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.codegen.late.devectorizer import image_valid_dims, _drop_valid_stmts, uop_given_valid

def do_devectorize(b:UOp):
  if b.shape == (): return None
  # broadcasting needs to be already unpacked
  if not all_same([x.shape for x in b.src]): return None
  src = []
  for idx in itertools.product(*[range(x) for x in b.shape]):
    idx_c = [UOp.const(dtypes.weakint, i) for i in idx]
    src.append(b.replace(src=tuple([x.index(*idx_c) for x in b.src])))
  return UOp._stack(*src).reshape(b.shape) if b.op is not Ops.STORE else UOp.group(*src)

devectorizer2 = PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise, name="b"), do_devectorize),
])

def transform_to_image(ctx, buf:UOp, x:UOp) -> UOp|None:
  shapes, ren = ctx
  if not IMAGE or ren.target.device not in {"QCOM", "CL", "PYTHON", "NULL"}: return None
  valid = UOp.const(dtypes.bool, True)
  if x.op == Ops.WHERE and x.src[2].op == Ops.CONST and x.src[2].arg == Invalid: valid,x,_= x.src
  # search for dims that drop the most valid statements
  best_drop, cands = -1, []
  for ch, cw in [shapes[buf.arg.slot]] if buf.arg.slot in shapes else image_valid_dims(buf.dtype, buf.max_numel(), ren.target.arch):
    cidx = uop_given_valid(valid, UOp.vectorize((x//4)%cw, x//(4*cw)))
    dropped = len(_drop_valid_stmts(valid, cidx, ch, cw))
    if dropped > best_drop: best_drop, cands = dropped, [(ch, cw, cidx)]
    elif dropped == best_drop: cands.append((ch, cw, cidx))
  # if no candidates, we don't rewrite
  if len(cands) == 0: return None
  # and tiebreak with indexing complexity (ie. number of nodes)
  h, w, cidx = cands[0] if len(cands) == 1 else min(cands, key=lambda cand: len(cand[2].gep(1).simplify().backward_slice))
  buf = buf.replace(dtype=(dtypes.imageh if buf.dtype.itemsize == 2 else dtypes.imagef)((h, w, 4)))
  shapes[buf.arg.slot] = (h, w)
  if valid.op is not Ops.CONST or valid.arg is not True:
    return buf.index(valid.where(cidx.src[1], cidx.src[1].const_like(Invalid)),
                     valid.where(cidx.src[0], cidx.src[0].const_like(Invalid)))
  else:
    return buf.index(cidx.src[1], cidx.src[0])

pm_simplify_add_image = PatternMatcher([
  (UPat(Ops.SHRINK, src=(UPat(Ops.PARAM, name="buf"), UPat(name="x"), UPat(arg=4))), transform_to_image),
  # image load/store is always float
  (UPat(Ops.INDEX, dtype=dtypes.float, name="x").load(dtype=dtypes.half), lambda x: x.load().cast(dtypes.half)),
  (UPat(Ops.INDEX, dtype=dtypes.float, name="x").store(UPat(name="d", dtype=dtypes.half)), lambda x,d: x.store(d.cast(dtypes.float))),
  (UPat.var("x", dtype=dtypes.float).cast(dtypes.half).cast(dtypes.float), lambda x: x),
])+devectorizer2+symbolic_simple

def memory_coalesing(sink:UOp, ctx:Renderer) -> UOp:
  if getenv("DMC"): return sink

  # collect
  memory: defaultdict[tuple[Ops, UOp, Any, Any], dict[int, list[UOp]]]  = defaultdict(dict)
  uops = sink.toposort()
  def base_scalar(dt): return dt.base.scalar() if isinstance(dt, PtrDType) else dt.scalar()
  if ctx is not None and ctx.float4_dtypes is not None and any(base_scalar(u.dtype) in (dtypes.bfloat16, *dtypes.fp8s) for u in uops):
    return sink
  uses: defaultdict[UOp, list[UOp]] = defaultdict(list)
  for u in uops:
    for s in u.src: uses[s].append(u)
  # Some ISA backends can only preserve vector memory semantics for a narrow dtype set.
  # If a kernel mixes other storage dtypes, leave the whole memory graph scalar.
  float4_safe = True
  if ctx is not None and ctx.float4_dtypes is not None:
    for u in uops:
      if u.op not in {Ops.LOAD, Ops.STORE} or isinstance(u.src[0].src[0].dtype, ImageDType) or u.src[0].op is not Ops.INDEX: continue
      value_dtype = u.src[1].dtype if u.op is Ops.STORE else u.dtype
      if base_scalar(u.src[0].src[0].dtype) not in ctx.float4_dtypes or value_dtype.scalar() not in ctx.float4_dtypes:
        float4_safe = False
        break
  for u in uops:
    # TODO: this should handle images too, it's just memory coalesing
    if u.op in {Ops.LOAD, Ops.STORE} and not isinstance(u.src[0].src[0].dtype, ImageDType):
      assert len(u.src) == (2 if u.op is Ops.STORE else 1), "memory coalesing does not support gated loads/stores"
      if u.src[0].op is not Ops.INDEX: continue
      buf, idx_u = u.src[0].src
      if buf.addrspace == AddrSpace.REG: continue
      value_dtype = u.src[1].dtype if u.op is Ops.STORE else u.dtype
      if ctx is not None and ctx.float4_dtypes is not None and not float4_safe: continue
      if ctx is not None and ctx.float4_dtypes is not None and base_scalar(buf.dtype) != value_dtype.scalar(): continue
      if ctx is not None and ctx.float4_dtypes is not None and u.op is Ops.LOAD and any(v.op in {Ops.CAST, Ops.BITCAST} for v in uses[u]): continue
      if ctx is not None and ctx.float4_dtypes is not None and u.op is Ops.STORE and u.src[1].op is Ops.BITCAST: continue
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
    elif buf.dtype not in (dtypes.float, dtypes.half, *dtypes.fp8s) and not isinstance(buf.dtype, ImageDType):
      pass
    elif buf.addrspace == AddrSpace.REG:
      pass
    elif isinstance(buf.dtype, ImageDType):
      lengths = [4]
    elif ctx is not None and ctx.supports_float4 and (ctx.float4_dtypes is None or buf.dtype.scalar() in ctx.float4_dtypes):
      # TODO: a better way to get this than ctx
      lengths = [8,4,2] if buf.dtype == dtypes.half and getenv("ALLOW_HALF8") else [4,2]
    lengths.append(1)  # worst case, it's not folded
    # do the grouping
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for full_grp in grouped_offsets:
      while len(full_grp):
        offset = (base+full_grp[0]) if isinstance(base, UOp) else UOp.const(dtypes.weakint, full_grp[0])
        length = [l for l in lengths if l <= len(full_grp) and (not must_divide or offset.divides(l) is not None)][0]
        grp = full_grp[:length]
        # NOTE: we apply the valid again after we determine the length
        offset = valid.where(offset, UOp(Ops.CONST, offset.dtype, arg=Invalid)) if valid is not None else offset
        idx = UOp(Ops.SHRINK, dtype=buf.dtype, src=(buf, offset, UOp.const(dtypes.weakint, len(grp)))) if len(grp) > 1 else buf.index(offset)
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
              replacements[oo] = ld.index(UOp.const(dtypes.weakint, i)) if len(grp) > 1 else ld
        full_grp = full_grp[length:]

  # apply
  return sink.substitute(replacements, name="memory coalesing")
