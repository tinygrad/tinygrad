# for renderers with vector ALU ops (Renderer.vector_alu) the expander's shaped values are lowered as vectors, not split into lanes: a value of
# shape S is a vector of prod(S) lanes in row major order. an access of memory at a shaped index that's a scalar plus a const for each lane is
# vector accesses of the windows of the buffer its lanes are in, a movement op of a value is a gather of its lanes
import functools, operator
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, exec_alu
from tinygrad.dtype import dtypes, AddrSpace, Invalid
from tinygrad.helpers import prod, all_same, strides_for_shape, unwrap, dedup

def lane_of(u:UOp) -> tuple[UOp, int]|None:
  # u is lane i of the vector value x (not memory) if it's x.index(i)
  if u.op is not Ops.INDEX or len(u.src) != 2 or u.src[0].addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL, AddrSpace.REG) or \
     len(u.src[0].shape) != 1: return None
  i = u.src[1]
  while i.op is Ops.CAST: i = i.src[0]
  return (u.src[0], i.arg) if i.op is Ops.CONST else None

@functools.cache
def lane_map(u:UOp) -> list[int|None]:
  # the lane of u.src[0] each lane of the movement op u is (None in a pad)
  in_shape, st = u.src[0].shape, strides_for_shape(u.src[0].shape)
  if u.op is Ops.RESHAPE: return list(range(prod(u.shape)))
  # each dim of u is a dim of the src with a stride, its index offset by a const (and in a pad it's in the src if it's in its size)
  dims: list[tuple[int, int, int|None]]
  if u.op is Ops.EXPAND: dims = [(0, 0, None)]*(len(u.shape)-len(in_shape)) + [(0 if s == 1 else x, 0, None) for x,s in zip(st, in_shape)]
  elif u.op is Ops.PERMUTE: dims = [(st[p], 0, None) for p in u.marg]
  elif u.op is Ops.SHRINK: dims = [(x, b, None) for x,(b,_) in zip(st, u.marg)]
  else: dims = [(x, -b, int(n)) for x,(b,_),n in zip(st, u.marg, in_shape)] # PAD
  ret: list[int|None] = [0]
  for (x, off, n), sz in zip(dims, u.shape):
    ret = [None if r is None or (n is not None and not 0 <= j+off < n) else r + (j+off)*x for r in ret for j in range(int(sz))]
  return ret

def lanes(u:UOp, env:dict[UOp, int], cache:dict[UOp, tuple]) -> tuple|None:
  # the value of each lane of the int (or bool) expression u, its scalar leaves valued in env
  if u in cache: return cache[u]
  ret: tuple|None = None
  if u.op is Ops.CONST: ret = tuple(u.arg) if isinstance(u.arg, tuple) else (u.arg,)
  elif u.op is Ops.STACK:
    if all((x:=lanes(s, env, cache)) is not None and len(x) == 1 for s in u.src): ret = tuple(cache[s][0] for s in u.src)  # type: ignore[index]
  elif u.op is Ops.INDEX and u.src[0].addrspace is AddrSpace.ALU and all(i.op is Ops.CONST for i in u.src[1:]) and \
       (v:=lanes(u.src[0], env, cache)) is not None:
    ret = (v[sum(int(i.arg)*st for i,st in zip(u.src[1:], strides_for_shape(u.src[0].shape)))],)
  elif u.op in GroupOp.Movement:
    if (s:=lanes(u.src[0], env, cache)) is not None: ret = tuple(0 if i is None else s[i if len(s) > 1 else 0] for i in lane_map(u))
  elif u.op in GroupOp.ALU or u.op is Ops.CAST:
    if all((x:=lanes(s, env, cache)) is not None for s in u.src):
      srcs: list[tuple] = [cache[s] for s in u.src]
      if u.op is Ops.CAST: ret = srcs[0] if dtypes.is_int(u.dtype) or u.dtype == dtypes.bool else None
      else: ret = tuple(exec_alu(u.op, u.dtype, [v[bcast_lane(i, u.shape, s.shape)] for v,s in zip(srcs, u.src)]) for i in range(prod(u.shape)))
  elif u.shape == () and dtypes.is_int(u.dtype): ret = (env.setdefault(u, 0),)
  cache[u] = ret  # type: ignore[assignment]
  return ret

def const_lanes(u:UOp) -> tuple|None:
  # the lanes of u if it's only consts (no scalar of the kernel)
  if any(x.shape == () and x.op not in GroupOp.ALU|{Ops.CONST, Ops.CAST} for x in u.toposort()): return None
  return lanes(u, {}, {})

def split(u:UOp, cache:dict) -> tuple[UOp, tuple]|None:
  # u as a scalar plus a const for each lane (Invalid in an invalid lane): (the scalar, the consts), else None
  if u in cache: return cache[u]
  ret: tuple[UOp, tuple]|None = None
  if u.shape == (): ret = (u, (0,))
  elif (c:=const_lanes(u)) is not None: ret = (UOp.const(0, u.dtype), c)
  elif u.op is Ops.STACK and all(x.shape == () for x in u.src):
    # (lanes that are the same scalar plus consts)
    def peel(x:UOp) -> tuple[tuple[UOp, ...], int]:
      # (the terms that aren't consts in a canonical order, the sum of the consts)
      ts = list(x.split_uop(Ops.ADD))
      cs = [t.base.arg for t in ts if t.base.op is Ops.CONST and isinstance(t.base.arg, int)]
      return tuple(sorted([t for t in ts if not (t.base.op is Ops.CONST and isinstance(t.base.arg, int))], key=lambda t: t.tuplize)), sum(cs)
    ps = [peel(x) for x in u.src]
    if all_same([b for b,_ in ps]) and ps[0][0]: ret = (functools.reduce(operator.add, ps[0][0]), tuple(c for _,c in ps))
  elif u.op is Ops.CAST and dtypes.is_int(u.dtype) and (p:=split(u.src[0], cache)) is not None: ret = (p[0].cast(u.dtype), p[1])
  elif u.op in GroupOp.Movement and (p:=split(u.src[0], cache)) is not None: ret = (p[0], tuple(0 if j is None else p[1][j] for j in lane_map(u)))
  elif u.op is Ops.ADD and (a:=split(u.src[0], cache)) is not None and (b:=split(u.src[1], cache)) is not None:
    ret = (a[0] + b[0], tuple(a[1][bcast_lane(i, u.shape, u.src[0].shape)] + b[1][bcast_lane(i, u.shape, u.src[1].shape)]
                              if Invalid not in (a[1][bcast_lane(i, u.shape, u.src[0].shape)], b[1][bcast_lane(i, u.shape, u.src[1].shape)])
                              else Invalid for i in range(prod(u.shape))))
  elif u.op in (Ops.MUL, Ops.SHL) and (k:=const_lanes(u.src[1])) is not None and all_same(k) and (p:=split(u.src[0], cache)) is not None:
    m = k[0] if u.op is Ops.MUL else 1 << k[0]
    ret = (p[0] * m, tuple(x if x is Invalid else x * m for x in (p[1][bcast_lane(i, u.shape, u.src[0].shape)] for i in range(prod(u.shape)))))
  elif u.op is Ops.WHERE and u.src[2].base.is_invalid and (c:=const_lanes(u.src[0])) is not None and (p:=split(u.src[1], cache)) is not None:
    ret = (p[0], tuple(p[1][bcast_lane(i, u.shape, u.src[1].shape)] if c[bcast_lane(i, u.shape, u.src[0].shape)] else Invalid
                       for i in range(prod(u.shape))))
  cache[u] = ret
  return ret

def affine(idx:UOp, valid:UOp|None) -> tuple[UOp, list[int|None]]|None:
  # if idx is a scalar plus a const for each lane: (the index of a valid lane, the consts relative to it). an invalid lane is None
  # (a valid that's only consts is which lanes are)
  if (p:=split(idx, {})) is None or (valid is not None and (m:=const_lanes(valid)) is None): return None
  v = [Invalid if valid is not None and not m[bcast_lane(i, idx.shape, valid.shape)] else x for i,x in enumerate(p[1])]  # type: ignore[index]
  if (f:=next((i for i,x in enumerate(v) if x is not Invalid), None)) is None: return None
  return ((p[0] + v[f]).simplify(), [None if x is Invalid else x - v[f] for x in v])  # type: ignore[operator]

def bcast_lane(i:int, shape:tuple, src_shape:tuple) -> int:
  # lane i of a value of shape is this lane of its src of src_shape, broadcast to it
  if src_shape == shape: return i
  idx = [(i // st) % int(sz) for st,sz in zip(strides_for_shape(shape), shape)]
  src_idx = [0 if int(sz) == 1 else j for j,sz in zip(idx[len(idx)-len(src_shape):], src_shape)]
  return sum(j*st for j,st in zip(src_idx, strides_for_shape(src_shape)))

def lane_expr(u:UOp, i:int, cache:dict) -> UOp:
  # the scalar expression of lane i of u
  if u.shape == (): return u
  if (u, i) in cache: return cache[(u, i)]
  if u.op is Ops.CONST: ret = UOp.const(u.arg[i] if isinstance(u.arg, tuple) else u.arg, u.dtype)
  elif u.op is Ops.STACK: ret = lane_expr(u.src[i // (m:=int(prod(u.src[0].shape)))], i % m, cache)
  elif u.op in GroupOp.Movement: ret = lane_expr(u.src[0], 0 if (j:=lane_map(u)[i]) is None else j, cache)
  elif u.op in GroupOp.ALU or u.op is Ops.CAST: ret = u.replace(src=tuple(lane_expr(s, bcast_lane(i, u.shape, s.shape), cache) for s in u.src))
  else: ret = u.reshape((prod(u.shape),)).index(UOp.const(i))
  cache[(u, i)] = ret
  return ret

def sub(u:UOp, ls:tuple[int, ...], cache:dict) -> UOp:
  # the vector of the lanes ls of u, pushed through the elementwise and movement ops to the loads (a slice of a load is a load)
  if u.shape == (): return u.expand((len(ls),))
  if ls == tuple(range(prod(u.shape))): return u.reshape((len(ls),))
  if (u, ls) in cache: return cache[(u, ls)]
  if u.op is Ops.CONST or (u.op is Ops.STACK and all(x.shape == () for x in u.src)): ret = UOp.stack(*[lane_expr(u, i, {}) for i in ls])
  elif u.op in GroupOp.ALU or u.op is Ops.CAST:
    ret = u.replace(src=tuple(sub(x, tuple(bcast_lane(i, u.shape, x.shape) for i in ls), cache) for x in u.src))
  elif u.op in GroupOp.Movement and None not in (m:=lane_map(u)): ret = sub(u.src[0], tuple(m[i] for i in ls), cache)  # type: ignore[misc]
  elif u.op is Ops.LOAD and (ix:=u.src[0]).op is Ops.INDEX and len(ix.src) == 2 and ix.src[1].shape == ix.shape:
    ret = ix.src[0].index(sub(ix.src[1], ls, cache)).load(arg=u.arg)
  else: ret = UOp.stack(*[u.reshape((prod(u.shape),)).index(UOp.const(i)) for i in ls])
  cache[(u, ls)] = ret
  return ret

def gather(srcs:list[tuple[UOp, int|None]], shape) -> UOp:
  # the lanes i of the vectors v (or the scalars v, i is None) as a value of shape
  return UOp.stack(*[v if i is None else v.index(UOp.const(i)) for v,i in srcs]).reshape(shape)

def windows(offs:list[int], width:int) -> list[tuple[int, int]]:
  # (start, length) of the windows of up to width lanes covering the offsets
  ret: list[tuple[int, int]] = []
  for o in sorted(set(offs)):
    if ret and o < ret[-1][0] + width: ret[-1] = (ret[-1][0], o - ret[-1][0] + 1)
    else: ret.append((o, 1))
  return ret

def vector_access(ren, x:UOp) -> UOp|None:
  # a load or store at a shaped index of a buffer
  # (a load through a movement op of a scalar index is the movement op of the load, a store of one lane through it is a store at it)
  if (ix:=x.src[0]).op in GroupOp.Movement and ix.src[0].op is Ops.INDEX and ix.src[0].shape == ():
    if x.op is Ops.LOAD: return ix.replace(src=(ix.src[0].load(arg=x.arg),)+ix.src[1:])
    if prod(ix.shape) == 1: return ix.src[0].store(lane_expr(x.src[1], 0, {}))
  if (ix:=x.src[0]).op is not Ops.INDEX or len(ix.src) != 2 or (shape:=ix.shape) == () or ix.src[0].addrspace is AddrSpace.ALU or \
     ix.src[1].shape != shape: return None
  buf, idx = ix.src
  valid = None
  if idx.op is Ops.WHERE and idx.src[2].base.is_invalid: valid, idx = idx.src[0], idx.src[1]
  # (a valid of only consts marks the invalid lanes, else the access is gated by it)
  if (aff:=affine(idx, valid)) is not None and valid is not None: valid = None
  elif aff is None: aff = affine(idx, None)
  if aff is None:
    # a gather loads each lane
    if x.op is Ops.STORE: return None
    cache: dict = {}
    return UOp.stack(*[buf.index(lane_expr(ix.src[1], i, cache)).load(arg=x.arg) for i in range(prod(shape))]).reshape(shape)
  (base, offs), n, sz = aff, prod(shape), buf.dtype.itemsize
  width, zero = max(1, ren.vector_bytes // sz), UOp.const(0, x.dtype)
  wins = windows([o for o in offs if o is not None], width)
  # (a few bytes from more than two windows with gaps are loads of each, a gather of them costs more)
  if x.op is Ops.LOAD and n * sz <= 32 and len(wins) > 2 and sum(l for _,l in wins) > len(used:=sorted(set(o for o in offs if o is not None))):
    wins = [(o, 1) for o in used]
  def readable(start:int, length:int) -> bool:
    lo, hi = (base + start).vmin * sz, (base + start + length).vmax * sz
    return lo >= -ren.overread and hi <= buf.max_numel() * sz + ren.overread
  # (a window can be in the buffer's shape)
  if not all((base + s).vmax >= 0 and (base + s).vmin + l <= buf.max_numel() for s,l in wins): return None
  if x.op is Ops.LOAD:
    # with a valid the windows load ungated (the bytes past the buffer can be read) and the valid selects. a window that can be read only if
    # one of its lanes is valid (a valid lane is in the buffer, the rest of the window is in the bytes past it) is gated by that
    gates: list[UOp|None] = [None]*len(wins)
    if valid is not None and not all(readable(s, l) for s,l in wins):
      if ren.overread < ren.vector_bytes: return None
      vl = valid.reshape((n,)) if valid.shape != () else valid
      for w,(s,l) in enumerate(wins):
        if not readable(s, l): gates[w] = functools.reduce(operator.or_, dedup(lane_expr(vl, k, {}).simplify() for k,o in enumerate(offs)
                                                                               if o is not None and s <= o < s + l))
    # (a window of a lane is a scalar load)
    loads = [(UOp(Ops.SHRINK, src=(buf, at, UOp.const(l))) if l > 1 else buf.index(at)).load(arg=x.arg)
             for (s,l),g in zip(wins, gates) for at in [base + s if g is None else (base + s).valid(g)]]
    # (an invalid lane is 0)
    srcs = [(zero, None) if o is None else next((ld, o - s if l > 1 else None) for ld,(s,l) in zip(loads, wins) if s <= o < s + l)
            for o in offs]
    ret = loads[0].reshape(shape) if len(wins) == 1 and offs == list(range(wins[0][0], wins[0][0]+n)) and n > 1 else gather(srcs, shape)
    return valid.where(ret, ret.const_like(0)) if valid is not None else ret
  # a store writes each (valid) lane once, the lanes in a window are contiguous
  if len(set(o for o in offs if o is not None)) != len([o for o in offs if o is not None]): return None
  val, flat = x.src[1], x.src[1].reshape((n,)) if x.src[1].shape != () else None
  vflat = valid.reshape((n,)) if valid is not None and valid.shape != () else valid
  stores, cache = [], {}
  for s,l in wins:
    if sorted(o for o in offs if o is not None and s <= o < s + l) != list(range(s, s + l)): return None
    lanes_ = [offs.index(o) for o in range(s, s + l)]
    v = gather([(flat, j) for j in lanes_], (l,)) if flat is not None else val
    # (a window with lanes gated differently is a store with a gate for each lane)
    gs = [None if vflat is None else vflat if vflat.shape == () else lane_expr(vflat, j, cache) for j in lanes_]
    if not all_same(gs):
      stores.append(UOp(Ops.SHRINK, src=(buf, base + s, UOp.const(l))).store(v, gather([(unwrap(vflat), j) for j in lanes_], (l,))))
      continue
    at = (base + s) if gs[0] is None else (base + s).valid(gs[0])
    stores.append(buf.index(at).store(v.reshape(())) if l == 1 else UOp(Ops.SHRINK, src=(buf, at, UOp.const(l))).store(v))
  return UOp.group(*stores)

def vector_register(x:UOp) -> UOp|None:
  # a load or store of a whole register buffer (an accumulator) is a vector access of it (its alloc is flat, viewed in its shape)
  b = x.src[0]
  if b.addrspace is not AddrSpace.REG or b.op is Ops.SHRINK or (shape:=b.shape) == () or (x.op is Ops.STORE and x.src[1].shape != shape): return None
  if b.op is Ops.RESHAPE: b = b.src[0]
  sh = UOp(Ops.SHRINK, src=(b, UOp.const(0), UOp.const(prod(shape))))
  return sh.load().reshape(shape) if x.op is Ops.LOAD else sh.store(x.src[1].reshape((prod(shape),)))

def vector_movement(x:UOp) -> UOp|None:
  # a movement op of a value is a gather of its lanes (a reshape of a value is the same lanes), of consts it's a const
  if x.addrspace is not AddrSpace.ALU or x.src[0].shape == (): return None
  # (of index math, or a valid, it's the math of the moved operands: it stays vector math of consts and scalars)
  if (a:=x.src[0]).op in GroupOp.ALU|{Ops.CAST} and not is_data(a) and all(s.shape in ((), a.shape) for s in a.src):
    return a.replace(src=tuple(s if s.shape == () else x.replace(src=(s,)+x.src[1:]) for s in a.src))
  if x.op is Ops.RESHAPE: return None
  if x.src[0].op is Ops.CONST and (v:=lanes(x, {}, {})) is not None: return UOp.stack(*[UOp.const(c, x.dtype) for c in v]).reshape(x.shape)
  # (a gather of a gather picks from its srcs)
  if (stk:=x.src[0].src[0] if x.src[0].op is Ops.RESHAPE else x.src[0]).op is Ops.STACK and all(s.shape == () for s in stk.src):
    return UOp.stack(*[stk.src[i or 0] for i in lane_map(x)]).reshape(x.shape)
  src = x.src[0].reshape((prod(x.src[0].shape),))
  return gather([(src, i if i is not None else 0) for i in lane_map(x)], x.shape)

@functools.cache
def is_data(u:UOp) -> bool: return u.op in (Ops.LOAD, Ops.WMMA) or any(is_data(s) for s in u.src)

def flat_lane(x:UOp) -> UOp|None:
  # a lane of a value at a const index in each dim is the scalar expression of the lane (or a lane of its vector)
  if x.src[0].addrspace is not AddrSpace.ALU or len(x.src) < 2 or not all(i.op is Ops.CONST for i in x.src[1:]): return None
  # (a lane of a scalar, a value that simplified to a broadcast of one, is it)
  if x.src[0].shape == (): return x.src[0]
  # an index of the leading dims is a slice of the vector
  if len(x.src) - 1 < len(shape:=x.src[0].shape):
    rest, o = shape[len(x.src)-1:], sum(int(c.arg)*s for c,s in zip(x.src[1:], strides_for_shape(shape)))
    return sub(x.src[0], tuple(range(o, o + prod(rest))), {}).reshape(rest)
  i = sum(int(c.arg)*s for c,s in zip(x.src[1:], strides_for_shape(shape)))
  # (of index math it's the lane's expression, of data it's the lane of the vector)
  if x.src[0].op in GroupOp.ALU|GroupOp.Movement|{Ops.CAST, Ops.CONST, Ops.STACK} and not is_data(x.src[0]): return lane_expr(x.src[0], i, {})
  return x.src[0].reshape((prod(shape),)).index(UOp.const(i)) if len(x.src) > 2 else None

def select_lanes(x:UOp) -> UOp|None:
  # a stack of lanes of a value made by an elementwise op is the op on those lanes of its operands (less math, no wide intermediate value)
  if not x.src or (l0:=lane_of(x.src[0])) is None or any((l:=lane_of(s)) is None or l[0] is not l0[0] for s in x.src): return None
  # (the lanes of a reshape are the lanes of its src. a widening cast of the lanes is cheaper even if they're more than its lanes)
  if (v:=l0[0].src[0] if l0[0].op is Ops.RESHAPE else l0[0]).op not in GroupOp.ALU|{Ops.CAST} or \
     (len(x.src) >= prod(v.shape) and not (v.op is Ops.CAST and v.dtype.itemsize > v.src[0].dtype.itemsize)): return None
  return sub(v, tuple(int(lane_of(s)[1]) for s in x.src), {})  # type: ignore[index]

def memory_index(x:UOp) -> UOp|None:
  # a movement op of an access of memory at a shaped index is the access at the moved index, lanes of it are the access at those of the index
  if (ix:=x.src[0]).op is not Ops.INDEX or len(ix.src) != 2 or ix.src[0].addrspace is AddrSpace.ALU or ix.shape == () or \
     ix.src[1].shape != ix.shape: return None
  if x.op in GroupOp.Movement: return ix.src[0].index(x.replace(src=(ix.src[1],)+x.src[1:]))
  if x.op is Ops.INDEX and all(i.op is Ops.CONST for i in x.src[1:]): return ix.src[0].index(ix.src[1].index(*x.src[1:]))
  return None

pm_vector_lower = PatternMatcher([
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="x"), memory_index),
  (UPat((Ops.LOAD, Ops.STORE), name="x"), lambda ctx,x: ret if (ret:=vector_access(ctx, x)) is not None else vector_register(x)),
  (UPat((Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.SHRINK), name="x"), lambda x: vector_movement(x)),
  (UPat(Ops.INDEX, name="x"), lambda x: flat_lane(x)),
  (UPat(Ops.STACK, name="x"), lambda x: select_lanes(x)),
  # a scalar view of a buffer of one element is its index 0
  (UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0].index(UOp.const(0)) if x.marg == () and x.src[0].shape == (1,) else None),
])

# then the values are vectors: a reshape is the same lanes, a broadcast of a scalar is a stack of it, a stack of vectors is of their lanes
pm_vector_flatten = PatternMatcher([
  (UPat(Ops.STACK, name="x"), lambda x: UOp.stack(*[l for s in x.src for l in ([s] if s.shape == () else
   [s.reshape((prod(s.shape),)).index(UOp.const(i)) for i in range(prod(s.shape))])]) if x.addrspace is AddrSpace.ALU and
   any(s.shape != () for s in x.src) else None),
  (UPat(Ops.RESHAPE, name="x"), lambda x: None if x.addrspace is not AddrSpace.ALU else x.src[0] if len(x.src[0].shape) == 1 else
   UOp.stack(x.src[0]) if x.src[0].shape == () and x.shape != () else None),
  (UPat(Ops.EXPAND, name="x"), lambda x: UOp.stack(*[x.src[0]]*prod(x.shape)) if x.addrspace is AddrSpace.ALU and x.src[0].shape == () else None),
])

def vectorized(sink:UOp) -> bool:
  # every access of memory is a vector access (or of a scalar index) and no value has a movement op
  for u in sink.toposort():
    if u.op is Ops.INDEX and u.src[0].addrspace is not AddrSpace.ALU and u.shape != (): return False
    if u.op in (Ops.LOAD, Ops.STORE) and u.src[0].op not in (Ops.SHRINK, Ops.INDEX) and u.src[0].shape != (): return False
    if u.op in GroupOp.Movement and u.addrspace is AddrSpace.ALU: return False
    # (the simplifier divides index math in scalars)
    if u.op in (Ops.DIV, Ops.MOD) and u.dtype is dtypes.weakint and u.shape != (): return False
  return True

def vector_math(sink:UOp, vector_dtypes) -> bool:
  # all the math is in the vector dtypes (index math is weakint, a compare is in its operands' dtype)
  return all((u.src[0].dtype if u.op in GroupOp.Comparison else u.dtype) in vector_dtypes for u in sink.toposort()
             if u.op in GroupOp.ALU and (u.src[0].dtype if u.op in GroupOp.Comparison else u.dtype) is not dtypes.weakint)
