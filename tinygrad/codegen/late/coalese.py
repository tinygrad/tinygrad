from typing import Any
import itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, AddrSpace, Invalid, ImageDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp
from tinygrad.helpers import getenv, IMAGE
from tinygrad.renderer import Renderer

_Memory = dict[tuple[Ops, UOp, Any, Any], dict[int, list[UOp]]]

pm_imageh_store = PatternMatcher([
  # store<imageh>(idx, x) is actually store(idx, x.cast(half)) so we can pull the cast into the store
  (UPat.var("x", dtypes.float).cast(dtypes.half), lambda x: x),
  # store(imageh, a.where(b.half(), c).float()) -> store(imageh, a.where(b, c.float()))
  (UPat(Ops.WHERE, src=(UPat.var("a"), UPat.var("b", dtypes.float).cast(dtypes.half), UPat.var("c"))), lambda a,b,c: a.where(b,c.cast(dtypes.float))),
  # otherwise, we cast to float
  (UPat(GroupOp.All, name="x"), lambda x: x.cast(dtypes.float))
])

def _grouped_offsets(offsets:dict[int, list[UOp]]) -> list[list[int]]:
  return [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]

def _valid_idx(idx:UOp) -> tuple[UOp, UOp|None]:
  return (idx.src[1], idx.src[0]) if idx.op is Ops.WHERE and idx.src[2].arg is Invalid else (idx, None)

def _base_offset(idx:UOp) -> tuple[Any, int]:
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return idx.src[0], idx.src[1].arg
  if idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: return idx.src[1], idx.src[0].arg
  if idx.op is Ops.CONST and idx.arg is Invalid: return "INVALID", 0
  if idx.op is Ops.CONST: return "CONST", idx.arg
  return idx, 0

def _offset(base:UOp|str, off:int) -> UOp: return (base+off) if isinstance(base, UOp) else UOp.const(dtypes.int, off)

def _image_buf(buf:UOp, ctx:Renderer|None) -> UOp|None:
  if isinstance(buf.dtype, ImageDType): return buf
  if ctx is None or not IMAGE or ctx.target.device not in {"QCOM", "CL", "PYTHON", "NULL"}: return None
  if buf.op is Ops.PARAM and buf.addrspace is AddrSpace.GLOBAL and (dims:=ImageDType.valid_dims(buf.dtype, ctx.target.arch)):
    return buf.replace(dtype=(dtypes.imageh if buf.dtype.base == dtypes.half else dtypes.imagef)((*dims[0], 4)))
  return None

def _can_image(buf:UOp, memory:_Memory) -> bool:
  # A PARAM slot is image-backed only if every access can be emitted as aligned vec4 pixels.
  for (_,membuf,base,_), offsets in memory.items():
    if membuf is not buf: continue
    for full_grp in _grouped_offsets(offsets):
      while full_grp:
        if len(full_grp) < 4 or _offset(base, full_grp[0]).divides(4) is None: return False
        full_grp = full_grp[4:]
  return True

def _image_idx(buf:UOp, offset:UOp) -> UOp:
  pix = offset // 4
  return buf.index(pix // buf.dtype.shape[1], pix % buf.dtype.shape[1], ptr=True)

def _coalesced_idx(buf:UOp, ibuf:UOp|None, offset:UOp, length:int) -> UOp:
  if ibuf is not None: return _image_idx(ibuf, offset)
  return buf._mop(Ops.SHRINK, arg=[(offset, length)]) if length > 1 else buf.index(offset)

def _fold_lengths(buf:UOp, ctx:Renderer|None, use_image:bool) -> tuple[list[int], bool]:
  if use_image: return [4], True
  if ctx is not None and ctx.target.device == "DSP": return [128,64,32,16,8,4,1], False
  if buf.addrspace == AddrSpace.REG or buf.dtype.base not in (dtypes.float, dtypes.half, *dtypes.fp8s): return [1], True
  if ctx is not None and ctx.supports_float4:
    return ([8,4,2,1] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else [4,2,1]), True
  return [1], True

def _choose_length(lengths:list[int], group:list[int], offset:UOp, must_divide:bool) -> int:
  return next(l for l in lengths if l <= len(group) and (not must_divide or offset.divides(l) is not None))

def _image_buffers(memory:_Memory, ctx:Renderer|None) -> dict[UOp, UOp]:
  image_bufs = {}
  for _,buf,_,_ in memory:
    if buf in image_bufs: continue
    if (ibuf:=_image_buf(buf, ctx)) is not None and _can_image(buf, memory): image_bufs[buf] = ibuf
  return image_bufs

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
      idx, valid = _valid_idx(idx_u)
      root_src, arg = _base_offset(idx)
      memory[(u.op, buf, root_src, valid)].setdefault(arg, []).append(u)

  image_bufs = _image_buffers(memory, ctx)

  # build replacements
  replacements = {}
  for (op,buf,base,valid),offsets in memory.items():
    if isinstance(buf.dtype, ImageDType) and buf not in image_bufs: continue
    lengths, must_divide = _fold_lengths(buf, ctx, buf in image_bufs)
    for full_grp in _grouped_offsets(offsets):
      while full_grp:
        offset = _offset(base, full_grp[0])
        length = _choose_length(lengths, full_grp, offset, must_divide)
        grp = full_grp[:length]
        ibuf = image_bufs.get(buf) if length == 4 else None
        idx = _coalesced_idx(buf, ibuf, offset, length)
        if op == Ops.STORE:
          datas = []
          for g in grp:
            assert len(offsets[g]) == 1, f"attempting multiple stores: {len(offsets[g])}"
            datas.append(offsets[g][0].src[1])
          data = UOp.vectorize(*datas) if len(datas) > 1 else datas[0]
          if ibuf is not None and ibuf.dtype.itemsize == 2: data = pm_imageh_store.rewrite(data)
          store = idx.store(data, valid) if valid is not None else idx.store(data)
          for g in grp: replacements[offsets[g][0]] = store
        else:
          ld = idx.load(idx.vconst_like(0), valid) if valid is not None else idx.load()
          if ibuf is not None and buf.dtype.base != dtypes.float: ld = ld.cast(buf.dtype.base)
          for i,g in enumerate(grp):
            for oo in offsets[g]:
              replacements[oo] = ld.index(UOp.const(dtypes.int, i)) if len(grp) > 1 else ld
        full_grp = full_grp[length:]

  # apply
  return sink.substitute(replacements, name="memory coalesing")
