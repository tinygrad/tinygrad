#!/usr/bin/env python3
import json, sys
from collections import Counter

from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, UOp
from tinygrad.uop.spec import eval_pyrender

BANKS = 32
BANK_BYTES = 4

def strip_view(u:UOp) -> UOp:
  while u.op is Ops.AFTER: u = u.src[0]
  return u

def prod(xs):
  ret = 1
  for x in xs: ret *= int(x)
  return ret

def strides(shape):
  ret, acc = [], 1
  for s in reversed(shape):
    ret.append(acc)
    acc *= int(s)
  return tuple(reversed(ret))

def unravel(flat:int, shape:tuple[int, ...]) -> tuple[int, ...]:
  ret = []
  for st,s in zip(strides(shape), shape):
    ret.append(flat // st)
    flat %= st
  return tuple(ret)

def eval_int(u:UOp|int, env:dict[tuple[int, AxisType], int]) -> int:
  if isinstance(u, int): return u
  if u.op is Ops.CONST: return int(u.arg)
  if u.op is Ops.RANGE: return env[(int(u.arg[0]), u.arg[-1])]
  if u.op is Ops.ADD: return eval_int(u.src[0], env) + eval_int(u.src[1], env)
  if u.op is Ops.MUL: return eval_int(u.src[0], env) * eval_int(u.src[1], env)
  if u.op in {Ops.IDIV, Ops.FLOORDIV}: return eval_int(u.src[0], env) // eval_int(u.src[1], env)
  if u.op in {Ops.MOD, Ops.FLOORMOD}: return eval_int(u.src[0], env) % eval_int(u.src[1], env)
  if u.op is Ops.SHR: return eval_int(u.src[0], env) >> eval_int(u.src[1], env)
  if u.op is Ops.SHL: return eval_int(u.src[0], env) << eval_int(u.src[1], env)
  if u.op is Ops.XOR: return eval_int(u.src[0], env) ^ eval_int(u.src[1], env)
  if u.op is Ops.AND: return eval_int(u.src[0], env) & eval_int(u.src[1], env)
  if u.op is Ops.CAST: return eval_int(u.src[0], env)
  raise RuntimeError(f"unsupported int op {u.op}: {u}")

def linear_offset(view:UOp, idx:tuple[int, ...]) -> int:
  view = strip_view(view)
  if view.op in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.PARAM, Ops.BUFFER}:
    assert len(idx) == 1, (view.op, view.shape, idx)
    return idx[0]
  if view.op is Ops.RESHAPE:
    old_shape = tuple(map(int, strip_view(view.src[0]).shape))
    flat = sum(i*s for i,s in zip(idx, strides(tuple(map(int, view.shape)))))
    return linear_offset(view.src[0], unravel(flat, old_shape))
  if view.op is Ops.PERMUTE:
    old_idx = [0] * len(idx)
    for new_axis, old_axis in enumerate(view.arg): old_idx[old_axis] = idx[new_axis]
    return linear_offset(view.src[0], tuple(old_idx))
  if view.op is Ops.SHRINK:
    return linear_offset(view.src[0], tuple(i+b for i,(b,_) in zip(idx, view.arg)))
  raise RuntimeError(f"unsupported view op {view.op}: {view}")

def index_offsets(idx_uop:UOp, env:dict[tuple[int, AxisType], int]) -> list[int]:
  if idx_uop.op is Ops.SHRINK:
    assert len(idx_uop.marg) == 1, idx_uop.marg
    st,en = idx_uop.marg[0]
    return index_offsets(idx_uop.src[0], env)[int(st):int(en)]
  assert idx_uop.op is Ops.INDEX
  base = strip_view(idx_uop.src[0])
  fixed = tuple(eval_int(x, env) for x in idx_uop.src[1:])
  rest_shape = tuple(map(int, base.shape[len(fixed):]))
  if not rest_shape: return [linear_offset(base, fixed)]
  return [linear_offset(base, fixed + unravel(i, rest_shape)) for i in range(prod(rest_shape))]

def bank(off_elems:int, itemsize:int) -> int:
  return ((off_elems * itemsize) // BANK_BYTES) % BANKS

def summarize(name:str, idx:UOp, itemsize:int) -> None:
  env_base = {(2, AxisType.LOCAL):0, (3, AxisType.LOCAL):0, (200, AxisType.LOOP):0, (201, AxisType.LOOP):0, (101, AxisType.REDUCE):0}
  by_lane = []
  for lane in range(32):
    env = {**env_base, (-1, AxisType.WARP):lane}
    offs = index_offsets(idx, env)
    banks = [bank(x, itemsize) for x in offs]
    by_lane.append((lane, offs[0], banks))
  first_unique:dict[int, set[int]] = {}
  for _,off,banks in by_lane: first_unique.setdefault(banks[0], set()).add(off)
  first = Counter({k:len(v) for k,v in first_unique.items()})
  all_banks = Counter(b for _,_,banks in by_lane for b in banks)
  conflicts = sum(v-1 for v in first.values() if v > 1)
  print(f"{name}: dtype_itemsize={itemsize} vector_elems={len(by_lane[0][2])}")
  print(f"  first-bank conflicts={conflicts} banks={dict(sorted(first.items()))}")
  print(f"  all-vector-bank pressure max={max(all_banks.values())} banks={dict(sorted(all_banks.items()))}")
  for lane, off, banks in by_lane:
    print(f"  lane={lane:2d} elem_off={off:4d} banks={banks}")

def find_wmmas(ast:UOp) -> list[UOp]:
  return [u for u in ast.toposort() if u.op in {Ops.SHAPED_WMMA, Ops.WMMA}]

def main() -> None:
  ast = None
  for line in sys.stdin:
    try: data = json.loads(line)
    except json.JSONDecodeError: continue
    if "ast = " in data.get("value", ""): ast = eval_pyrender(data["value"])
  if ast is None: raise RuntimeError("no pyrendered ast found on stdin")
  for i,wmma in enumerate(find_wmmas(ast)):
    for j,src in enumerate(wmma.src[:2]):
      idx = src.src[0] if src.op is Ops.SHRINK and src.src[0].op is Ops.INDEX else src
      if idx.op is not Ops.INDEX: continue
      base = strip_view(idx.src[0])
      if getattr(base.dtype, "addrspace", None) != AddrSpace.LOCAL: continue
      summarize(f"wmma{i}.src{j}", src, src.dtype.base.itemsize)

if __name__ == "__main__": main()
