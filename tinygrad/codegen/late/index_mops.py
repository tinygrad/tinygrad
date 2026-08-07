from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat

# INDEX through RESHAPE/PERMUTE — codegen-only (not mop_cleanup; breaks symbolic/schedule/JIT).

def _index_through_reshape(r:UOp, idx:UOp) -> UOp|None:
  if len(idx.src) - 1 != len(r.shape): return None
  idxs, shape = idx.src[1:], r.shape
  if any(not isinstance(s, int) for s in shape): return None
  src = r.src[0]
  if not idxs: return src if src._shape is not None and src.shape == () else None
  flat: UOp = idxs[0]
  for i in range(1, len(shape)): flat = flat * shape[i] + idxs[i]
  if src._shape is None: return src.index(flat)
  if src.shape == (): return src
  src_idxs: list[UOp] = []
  rem = flat
  for dim in reversed(src.shape):
    src_idxs.append(rem % dim)
    rem = rem // dim
  src_idxs.reverse()
  return src.index(*src_idxs)

def _index_through_permute(p:UOp, idx:UOp) -> UOp|None:
  perm, idxs = p.arg, idx.src[1:]
  if len(idxs) != len(perm): return None
  old_idxs: list[UOp|None] = [None] * len(perm)
  for i, pi in enumerate(perm): old_idxs[pi] = idxs[i]
  if any(x is None for x in old_idxs): return None
  return p.src[0].index(*old_idxs)  # type: ignore[arg-type]

pm_index_mops = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat(Ops.RESHAPE, name="r"),), allow_any_len=True, name="idx"), _index_through_reshape),
  (UPat(Ops.INDEX, src=(UPat(Ops.PERMUTE, name="p"),), allow_any_len=True, name="idx"), _index_through_permute),
])
