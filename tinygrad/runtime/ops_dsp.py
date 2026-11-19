from __future__ import annotations
import ctypes, functools, os, mmap, tempfile, pathlib, array, threading, contextlib, subprocess, struct, math, itertools, hashlib, operator
import types, time, traceback
from typing import cast
from tinygrad.device import BufferStorage, MMIOInterface, BufferSpec, Compiled, Allocator, HCQ_RUNTIME_DEV
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.dtype import dtypes, AddrSpace, DType
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp, AxisType, KernelInfo, uopfunc
from tinygrad.helpers import getenv, round_up, mv_address, to_mv, cpu_objdump, system, DEBUG, suppress_finalizing, Target, unwrap, dedup
from tinygrad.helpers import all_same, prod
from tinygrad.renderer.cstyle import ClangRenderer, base_rewrite, uops_to_dtypes, _wmma_name
from tinygrad.renderer import tc
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.codegen.late.vectorize import lane_of
from tinygrad.runtime.autogen import libc, qcom_dsp
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.hcq2 import HWQueue, ccall, cstruct, patch, unwrap_view, layout_args, pack_args, make_program
from tinygrad.engine.realize import get_call_kernel_args
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

# ***** vector rendering *****

def known_factor(x:UOp) -> int:
  # a factor of x, through the shifts the late decompositions make
  if x.op is Ops.CAST: return known_factor(x.src[0])
  if x.op is Ops.CONST: return abs(x.val) or 1 << 30
  if x.op is Ops.ADD: return math.gcd(known_factor(x.src[0]), known_factor(x.src[1]))
  if x.op is Ops.MUL: return known_factor(x.src[0]) * known_factor(x.src[1])
  if x.op is Ops.SHL and (sh:=known_factor(x.src[1])) == x.src[1].vmin == x.src[1].vmax: return known_factor(x.src[0]) << sh
  return 1

mask_dtype = {1: dtypes.int8, 2: dtypes.int16, 4: dtypes.int32, 8: dtypes.int64}

def is_vec(u:UOp) -> bool: return (u.addrspace is AddrSpace.ALU or u.op is Ops.CALL) and u._shape is not None and u.max_numel() > 1

def splat(ctx:DSPRenderer, s:UOp, n:int) -> str:
  # a scalar used in a vector op is broadcast
  return ctx[s] if is_vec(s) else f"(({ctx._render_dtype(s.dtype, n)})({ctx[s]}))"

def render_vec_alu(ctx:DSPRenderer, x:UOp) -> str|None:
  if not is_vec(x): return None
  srcs = [splat(ctx, s, x.max_numel()) for s in x.src]
  # the condition of a vector select is a mask of the result size
  if x.op is Ops.WHERE:
    # (a select of a vector register of bytes is vmux: clang miscompiles some)
    if x.dtype.itemsize == 1 and vpad(x.max_numel()) == 128:
      return f"(({ctx.render_type(x)})V6(vmux)(V6(vandvrt)((HVX_Vector)({srcs[0]}), 0x01010101), (HVX_Vector)({srcs[1]}), (HVX_Vector)({srcs[2]})))"
    return f"(-__builtin_convertvector({srcs[0]}, {ctx._render_dtype(mask_dtype[x.dtype.itemsize], x.max_numel())})?{srcs[1]}:{srcs[2]})"
  ret = ctx.code_for_op[x.op](*srcs, x.dtype)
  # vector compares are masks of the operand size (-1 is true), a vector of bools is bytes of 0 or 1
  return to_bool(ctx, ret, x.max_numel()) if x.op in GroupOp.Comparison else ret

# clang's vectors of bools are bits (and crash its HVX backend): a vector of bools renders as bytes of 0 or 1
def to_bool(ctx:DSPRenderer, mask:str, n:int) -> str: return f"(-__builtin_convertvector({mask}, {ctx._render_dtype(dtypes.uint8, n)}))"

# a vector of n lanes renders as the next power of two lanes, the lanes past n are undefined
def vpad(n:int) -> int: return 1 << (n-1).bit_length() if n > 1 else n
def shuffle(a:str, b:str, idxs, src_width:int|None=None, isz:int|None=None, b_splat:bool=False) -> str:
  idxs = list(idxs)
  idxs += [-1]*(vpad(len(idxs))-len(idxs))
  # clang miscompiles some shuffles that take a lane again after others (a broadcast in blocks is fine): they're permutations blended with
  # vmux (or for less than a vector register, the lanes go through memory it can't fold)
  # (the lanes of a broadcast b are all the same)
  runs = [i for j,i in enumerate(idxs) if i >= 0 and (j == 0 or idxs[j-1] != i) and not (b_splat and (i >= len(idxs) or a == b))]
  if len(runs) != len(set(runs)):
    # (more than a vector register is a shuffle for each register of it, concatenated)
    if isz is not None and len(idxs)*isz > 128 and (len(idxs)*isz) % 128 == 0:
      w = 128 // isz
      return concat([shuffle(a, b, idxs[c:c+w], isz=isz, b_splat=b_splat) for c in range(0, len(idxs), w)], w)
    if isz is None or len(idxs)*isz != 128:
      return (f"({{ __auto_type _a = ({a}); __auto_type _b = {'_a' if b == a else f'({b})'}; enum {{ _n = sizeof(_a)/sizeof(_a[0]) }}; "
              f"typedef __typeof__(_a[0]) _e; typedef _e _v __attribute__((ext_vector_type({len(idxs)}))); _e volatile _t[2*_n]; "
              f"for (int _i = 0; _i < _n; _i++) {{ _t[_i] = _a[_i]; _t[_n+_i] = _b[_i]; }} "
              f"(_v){{{','.join(f'_t[{max(i, 0)}]' for i in idxs)}}}; }})")
    # the k-th pass takes the k-th time each lane is taken
    passes: list[list[int]] = []
    seen: dict[int, int] = {}
    for j,i in enumerate(idxs):
      if i < 0: continue
      seen[i] = k = seen.get(i, -1) + 1
      while len(passes) <= k: passes.append([-1]*len(idxs))
      passes[k][j] = i
    def perm(p:list[int]) -> str:
      rest = iter(i for i in range(len(idxs)) if i not in set(p))
      return f"__builtin_shufflevector({a}, {b}, {','.join(str(i if i >= 0 else next(rest)) for i in p)})"
    ret = f"({{ __auto_type _s = {perm(passes[0])};"
    for p in passes[1:]:
      mask = "((hvx_ub){" + ",".join("1" if i >= 0 else "0" for i in p for _ in range(isz)) + "})"
      ret += f" _s = (__typeof__(_s))V6(vmux)(V6(vandvrt)((HVX_Vector){mask}, 0x01010101), (HVX_Vector)({perm(p)}), (HVX_Vector)_s);"
    return ret + " _s; })"
  # a shuffle of a vector to the same width with undefined lanes is a permutation of it: clang scalarizes it unless it's a permutation
  if src_width == len(idxs) and -1 in idxs and len(used:=[i for i in idxs if i >= 0]) == len(set(used)) and all(i < src_width for i in used):
    rest = iter(i for i in range(src_width) if i not in set(used))
    idxs = [i if i >= 0 else next(rest) for i in idxs]
  return f"__builtin_shufflevector({a}, {b}, {','.join(map(str, idxs))})"
def concat(parts:list[str], m:int) -> str:
  # (the lanes past the parts are undefined)
  parts = parts + [parts[-1]] * (vpad(len(parts)) - len(parts))
  while len(parts) > 1: parts, m = [shuffle(a, b, list(range(m)) + [vpad(m)+i for i in range(m)]) for a,b in zip(parts[::2], parts[1::2])], 2*m
  return parts[0]

def can_load_wide(u:UOp) -> bool:
  # (a whole vector register from where it loads is in the buffer or the bytes past it that can be read. what fits a scalar register doesn't)
  return u.op is Ops.LOAD and u.dtype in DSPRenderer.vector_dtypes and len(u.src) == 1 and (b:=u.src[0]).op is Ops.SHRINK and \
    b.src[0].addrspace is AddrSpace.GLOBAL and \
    8 < b.max_numel()*u.dtype.itemsize < 128 and b.src[1].get_valid() is UOp.const(True) and \
    b.src[1].vmax*u.dtype.itemsize + 128 <= b.src[0].max_numel()*u.dtype.itemsize + DSP_OVERREAD
# the lanes of the rendered vector u
def rwidth(ctx:DSPRenderer, u:UOp) -> int: return 128 // u.dtype.itemsize if u in ctx.wide_loads else vpad(u.max_numel())

def stack_lanes(ctx:DSPRenderer, srcs:tuple[UOp, ...]) -> str:
  if all(s is srcs[0] for s in srcs): return splat(ctx, srcs[0], len(srcs))
  n, isz, lanes = len(srcs), srcs[0].dtype.itemsize, [lane_of(s) for s in srcs]
  if all(l is not None for l in lanes):
    vecs = dedup([unwrap(l)[0] for l in lanes])
    # all the lanes of a power of two vectors of the same type in order are a concat
    if len(vecs) > 2 and len(vecs) & (len(vecs)-1) == 0 and all_same([(v.dtype, v.shape) for v in vecs]) and \
       all(rwidth(ctx, v) == vpad(v.max_numel()) for v in vecs) and lanes == [(v, i) for v in vecs for i in range(v.shape[0])]:
      return concat([ctx[v] for v in vecs], vecs[0].shape[0])
    # bytes of one or two vector registers in a few runs are rotations of them, blended: clang scalarizes the shuffle
    if len(vecs) <= 2 and isz == 1 and all(rwidth(ctx, v) == 128 for v in vecs) and vpad(n) == 128 and \
       2 <= len(runs:=[list(g) for _,g in itertools.groupby(enumerate(vecs.index(unwrap(l)[0])*128 + unwrap(l)[1] for l in lanes),
                                                            lambda x: x[1]-x[0])]) <= 4:
      a, b, ret = f"((HVX_Vector)({ctx[vecs[0]]}))", f"((HVX_Vector)({ctx[vecs[-1]]}))", ""
      for r in runs:
        # the run's lanes are lanes off+j of a:b
        off = r[0][1]-r[0][0]
        # (clang crashes on a rotation by 0)
        def rotate(hi:str, lo:str, o:int) -> str: return lo if o == 0 else f"V6(valignb)({hi}, {lo}, {o})"
        rot = rotate(b, a, off) if 0 <= off <= 127 and r[-1][1] < 256 else \
              rotate(a, a, off % 128) if r[-1][1] < 128 else rotate(b, b, (off-128) % 128)
        if not ret: ret = rot
        else:
          mask = "((hvx_ub){" + ",".join("1" if r[0][0] <= j <= r[-1][0] else "0" for j in range(128)) + "})"
          ret = f"V6(vmux)(V6(vandvrt)((HVX_Vector){mask}, 0x01010101), {rot}, {ret})"
      return f"(({ctx._render_dtype(srcs[0].dtype, 128)})({ret}))"
    # lanes from one or two vectors of the same type are a shuffle
    # bytes of a few vector registers are a permutation of each, blended: clang scalarizes a shuffle of two
    cat = [vecs.index(v)*rwidth(ctx, v) + i for v,i in map(unwrap, lanes)]
    if 2 <= len(vecs) <= 8 and isz == 1 and all(rwidth(ctx, v) == 128 for v in vecs) and vpad(n) == 128 and \
       not (len(vecs) == 2 and all_same([b-a for a,b in zip(cat, cat[1:])])):
      ret = ""
      for v in vecs:
        pv = f"(HVX_Vector)({shuffle(ctx[v], ctx[v], [i if u is v else -1 for u,i in map(unwrap, lanes)], 128, 1)})"
        mask = "((hvx_ub){" + ",".join("1" if j < n and unwrap(lanes[j])[0] is v else "0" for j in range(128)) + "})"
        ret = pv if not ret else f"V6(vmux)(V6(vandvrt)((HVX_Vector){mask}, 0x01010101), {pv}, {ret})"
      return f"(({ctx._render_dtype(dtypes.uint8, 128)})({ret}))"
    if len(vecs) <= 2 and all(v.dtype == vecs[0].dtype and rwidth(ctx, v) == rwidth(ctx, vecs[0]) for v in vecs):
      return shuffle(ctx[vecs[0]], ctx[vecs[-1]], [vecs.index(v)*rwidth(ctx, v) + i for v,i in map(unwrap, lanes)],
                     rwidth(ctx, vecs[0]) if len(vecs) == 1 else None, isz)
  # less than a vector register of scalars is a vector of them through memory (clang crashes on the shuffles of small vectors from them, and
  # on a vector of them that it makes vector math of)
  if all(l is None for l in lanes) and vpad(n)*isz < 128:
    return f"({{ {ctx.render_dtype(srcs[0].dtype) if srcs[0].dtype != dtypes.bool else 'unsigned char'} volatile _t[{vpad(n)}] = " + \
           f"{{{', '.join(ctx[s] for s in srcs)}}}; ({ctx._render_dtype(srcs[0].dtype, n)}){{{', '.join(f'_t[{i}]' for i in range(vpad(n)))}}}; }})"
  # more than a vector is concatenated vectors, clang crashes on some shuffles of them
  if n*isz > 128 and n % (c:=128//isz) == 0:
    return concat([stack_lanes(ctx, srcs[i:i+c]) for i in range(0, n, c)], c)
  # else the lanes of each vector and scalar are shuffled in one at a time, at the width of the widest (at least the HVX vector width,
  # clang crashes on shuffles of smaller vectors). a vector literal can crash clang
  # (bools are shuffled as bytes, clang scalarizes shuffles of bool vectors)
  w = vpad(max([n, 128 // isz] + [rwidth(ctx, unwrap(l)[0]) for l in lanes if l is not None]))
  parts_: dict[UOp, list[tuple[int, int]]] = {}
  for j,(s,l) in enumerate(zip(srcs, lanes)): parts_.setdefault(l[0] if l is not None else s, []).append((j, l[1] if l is not None else 0))
  def widen(p:UOp) -> str:
    if not is_vec(p): return splat(ctx, p, w)
    v = ctx[p]
    # (a shuffle to another width is a type of its own for older clangs)
    return v if (m:=rwidth(ctx, p)) == w else f"(({ctx._render_dtype(p.dtype, w)})({shuffle(v, v, list(range(m)) + [-1]*(w-m))}))"
  acc, done = "", set()
  for p,pl in sorted(parts_.items(), key=lambda kv: not is_vec(kv[0])):
    src = dict(pl)
    if not acc: acc = shuffle(widen(p), widen(p), [src.get(j, -1) for j in range(w)], isz=isz, b_splat=not is_vec(p))
    else: acc = shuffle(acc, widen(p), [j if j in done else w+src[j] if j in src else -1 for j in range(w)], isz=isz, b_splat=not is_vec(p))
    done |= set(src)
  if w != n: acc = shuffle(acc, acc, range(n))
  return acc

def render_vec_stack(ctx:DSPRenderer, x:UOp) -> str|None:
  if len(x.src) < 2 or any(s.shape != () for s in x.src): return None
  return stack_lanes(ctx, x.src)

# the high word of a 32x32 bit multiply, the requantize. HVX has no 64 bit multiply, vmpye+vmpyo make the 64 bit product
HVX_PRELUDE = """typedef long HVX_Vector __attribute__((__vector_size__(128))); typedef long HVX_VectorPair __attribute__((__vector_size__(256)));
#define V6(x) __builtin_HEXAGON_V6_##x##_128B
typedef int hvx_w __attribute__((ext_vector_type(32)));
typedef unsigned char hvx_ub __attribute__((aligned(128),ext_vector_type(128)));
typedef int hvx_w128 __attribute__((aligned(512),ext_vector_type(128)));
__attribute__((unused)) static inline __attribute__((always_inline))
hvx_w128 vrmpy4(hvx_w128 acc, hvx_ub x0, hvx_ub x1, hvx_ub x2, hvx_ub x3, unsigned w) {
  HVX_VectorPair s01 = V6(vshuffvdd)((HVX_Vector)x1, (HVX_Vector)x0, -1), s23 = V6(vshuffvdd)((HVX_Vector)x3, (HVX_Vector)x2, -1);
  HVX_VectorPair q0 = V6(vshuffvdd)(V6(lo)(s23), V6(lo)(s01), -2), q1 = V6(vshuffvdd)(V6(hi)(s23), V6(hi)(s01), -2);
  HVX_Vector a[4]; __builtin_memcpy(a, &acc, sizeof(a));
  a[0] = V6(vrmpyub_acc)(a[0], V6(lo)(q0), w); a[1] = V6(vrmpyub_acc)(a[1], V6(hi)(q0), w);
  a[2] = V6(vrmpyub_acc)(a[2], V6(lo)(q1), w); a[3] = V6(vrmpyub_acc)(a[3], V6(hi)(q1), w);
  __builtin_memcpy(&acc, a, sizeof(a)); return acc; }
__attribute__((unused)) static inline __attribute__((always_inline)) void interleave4(HVX_Vector *v, hvx_ub x0, hvx_ub x1, hvx_ub x2, hvx_ub x3) {
  HVX_VectorPair s01 = V6(vshuffvdd)((HVX_Vector)x1, (HVX_Vector)x0, -1), s23 = V6(vshuffvdd)((HVX_Vector)x3, (HVX_Vector)x2, -1);
  HVX_VectorPair q0 = V6(vshuffvdd)(V6(lo)(s23), V6(lo)(s01), -2), q1 = V6(vshuffvdd)(V6(hi)(s23), V6(hi)(s01), -2);
  v[0] = V6(lo)(q0); v[1] = V6(hi)(q0); v[2] = V6(lo)(q1); v[3] = V6(hi)(q1); }
__attribute__((unused)) static inline __attribute__((always_inline))
hvx_w128 vrmpy4v(hvx_w128 acc, hvx_ub x0, hvx_ub x1, hvx_ub x2, hvx_ub x3, hvx_ub w0, hvx_ub w1, hvx_ub w2, hvx_ub w3) {
  HVX_Vector a[4], x[4], w[4]; __builtin_memcpy(a, &acc, sizeof(a));
  interleave4(x, x0, x1, x2, x3); interleave4(w, w0, w1, w2, w3);
  for (int i = 0; i < 4; i++) a[i] = V6(vrmpyubv_acc)(a[i], x[i], w[i]);
  __builtin_memcpy(&acc, a, sizeof(a)); return acc; }
__attribute__((unused)) static inline __attribute__((always_inline)) void store_n(unsigned char *p, HVX_Vector v, int n) {
  // the first n bytes of v to p, any alignment: the parts in the two vector registers of memory with predicates
  unsigned off = (unsigned)p & 127; HVX_Vector *a = (HVX_Vector*)(p - off), z = V6(vd0)();
  HVX_Vector m = n >= 128 ? V6(lvsplatw)(0x01010101) : V6(vandqrt)(V6(pred_scalar2)(n), 0x01010101);
  V6(vS32b_qpred_ai)(V6(vandvrt)(V6(vlalignb)(m, z, off), 0x01010101), a, V6(vlalignb)(v, z, off));
  if (off + n > 128) V6(vS32b_qpred_ai)(V6(vandvrt)(V6(vlalignb)(z, m, off), 0x01010101), a+1, V6(vlalignb)(z, v, off)); }
// t[x] for each byte x, t is 256 bytes: vlut32 looks up 32 entries a time, entry 32s+j of a table vector is at byte 2j+64(s&1)+(s>>1)
// (the halves of each 128 entries interleaved)
__attribute__((unused)) static inline __attribute__((always_inline)) hvx_ub lut256(hvx_ub x, const unsigned char *t) {
  typedef unsigned char hvx_ub_u __attribute__((aligned(1),ext_vector_type(128)));
  hvx_ub a = *(const hvx_ub_u*)t, b = *(const hvx_ub_u*)(t+128);
  HVX_Vector ta = (HVX_Vector)__builtin_shufflevector(a, a, LUT_PERM), tb = (HVX_Vector)__builtin_shufflevector(b, b, LUT_PERM);
  HVX_Vector r = V6(vlutvvb)((HVX_Vector)x, ta, 0);
  for (int i = 1; i < 8; i++) r = V6(vlutvvb_oracc)(r, (HVX_Vector)x, i < 4 ? ta : tb, i);
  return (hvx_ub)r; }
__attribute__((unused)) static inline __attribute__((always_inline)) hvx_w mulhi_w(hvx_w a, hvx_w c) {
  HVX_VectorPair p = V6(vmpyewuh_64)((HVX_Vector)a, (HVX_Vector)c); p = V6(vmpyowh_64_acc)(p, (HVX_Vector)a, (HVX_Vector)c);
  return (hvx_w)V6(hi)(p); }""".replace("LUT_PERM", ",".join(str(p//2 + 64*(p%2)) for p in range(128)))

def const_val(u:UOp):
  while u.op is Ops.CAST: u = u.src[0]
  return u.arg if u.op is Ops.CONST else None

def mulhi(x:UOp, a:UOp, c:UOp, s:UOp) -> UOp|None:
  # (int)(((long long)a * c) >> 32) on a vector of int is mulhi_w 32 lanes at a time, c is an int: a const, a scalar or a vector like a
  if not is_vec(x) or x.dtype != dtypes.int32 or a.src[0].dtype != dtypes.int32 or const_val(s) != 32: return None
  n = x.max_numel()
  def lanes(v:str, i:int) -> str: return shuffle(v, v, [i+j if i+j < n else -1 for j in range(32)])
  if isinstance(cv:=const_val(c), int):
    if not -2**31 <= cv < 2**31: return None
  elif not (c.op is Ops.CAST and c.src[0].dtype == dtypes.int32 and (not is_vec(c.src[0]) or c.src[0].shape == a.src[0].shape)): return None
  def cpart(i:int) -> str: return f"((hvx_w)({cv}))" if isinstance(cv, int) else lanes("{1}", i) if is_vec(c.src[0]) else "((hvx_w)({1}))"
  parts = [f"mulhi_w({lanes('{0}', i)}, {cpart(i)})" for i in range(0, n, 32)]
  ret = concat(parts, 32)
  return UOp.native(ret if vpad(n) == 32*len(parts) else shuffle(ret, ret, range(n)), shape=x.shape, dtype=x.dtype, fmt=True) \
    .call(a.src[0], *([] if isinstance(cv, int) else [c.src[0]]))

def vrmpy_sum(x:UOp) -> UOp|None:
  # a sum of products of bytes is vrmpy: the sum of 4 products per lane, a vector of bytes times 4 scalar bytes
  if not (is_vec(x) and x.dtype == dtypes.int32 and vpad(x.max_numel()) == 128): return None
  def terms(u:UOp) -> list[UOp]: return terms(u.src[0]) + terms(u.src[1]) if u.op is Ops.ADD and u.shape == x.shape else [u]
  def u8vec(u:UOp) -> UOp|None: return u.src[0] if u.op is Ops.CAST and u.src[0].dtype == dtypes.uint8 and u.src[0].shape == x.shape else None
  def u8(u:UOp) -> UOp|None: return u.src[0] if u.op is Ops.CAST and u.src[0].dtype == dtypes.uint8 and u.src[0].shape == () else None
  prods: list[tuple[UOp, UOp]] = []
  sums: list[UOp] = []
  rest: list[UOp] = []
  vprods: list[tuple[UOp, UOp]] = []
  for t in terms(x):
    if t.op is Ops.MUL and (p:=next(((v, w) for a,b in (t.src, t.src[::-1]) if (v:=u8vec(a)) is not None and (w:=u8(b)) is not None), None)):
      prods.append(p)
    elif t.op is Ops.MUL and (v:=u8vec(t.src[0])) is not None and (w:=u8vec(t.src[1])) is not None: vprods.append((v, w))
    elif (v:=u8vec(t)) is not None: sums.append(v)
    else: rest.append(t)
  if len(prods) + len(sums) + len(vprods) < 3: return None
  groups = [prods[i:i+4] for i in range(0, len(prods), 4)] + [[(v, UOp.const(1, dtypes.uint8)) for v in sums[i:i+4]] for i in range(0, len(sums), 4)]
  acc = functools.reduce(UOp.__add__, rest) if rest else None
  for vg in [vprods[i:i+4] for i in range(0, len(vprods), 4)]:
    # a product of two vectors of bytes: both interleaved, vrmpy of the vectors (a missing product is the first with weights 0)
    o, zero = int(acc is not None), "((hvx_ub)(0))"
    args = [f"{{{o+i}}}" if i < len(vg) else zero for i in range(4)] + [f"{{{o+len(vg)+i}}}" if i < len(vg) else zero for i in range(4)]
    fn = UOp.native("vrmpy4v(" + ("{0}" if acc is not None else "((hvx_w128)(0))") + "".join(", "+a for a in args) + ")",
                             shape=x.shape, dtype=x.dtype, fmt=True)
    acc = fn.call(*([acc] if acc is not None else []), *[a for a,_ in vg], *[b for _,b in vg])
  for g in groups:
    g = g + [(g[0][0], UOp.const(0, dtypes.uint8))] * (4 - len(g))
    w = functools.reduce(UOp.__or__, [b.cast(dtypes.uint32) << (8*i) if i else b.cast(dtypes.uint32) for i,(_,b) in enumerate(g)])
    o = int(acc is not None)
    fn = UOp.native("vrmpy4(" + ("{0}" if o else "((hvx_w128)(0))") + "".join(f", {{{i+o}}}" for i in range(5)) + ")",
                             shape=x.shape, dtype=x.dtype, fmt=True)
    acc = fn.call(*([acc] if acc is not None else []), *[a for a,_ in g], w)
  return acc

# vrmpy tensor core: A is 4 rows of 128 bytes, interleaved to 4 vectors of [32][4] bytes. B is 4 bytes per column, C a row of 128 int per column
def render_vrmpy(name:str, dims:tuple[int, int, int], a:str, b:str, c:str, signed_b:bool) -> str:
  # A is K rows of M bytes, each vrmpy is 32 lanes (m) of 4 bytes (k): the rows are interleaved into vectors, M/32 for each 4 rows
  # (vrmpyub multiplies by unsigned bytes of B, vrmpybus by signed ones)
  n, m, k = dims
  if m == 128: inter = """HVX_VectorPair s01 = V6(vshuffvdd)(r[1], r[0], -1), s23 = V6(vshuffvdd)(r[3], r[2], -1);
  HVX_VectorPair q0 = V6(vshuffvdd)(V6(lo)(s23), V6(lo)(s01), -2), q1 = V6(vshuffvdd)(V6(hi)(s23), V6(hi)(s01), -2);
  HVX_Vector v[4] = {V6(lo)(q0), V6(hi)(q0), V6(lo)(q1), V6(hi)(q1)};"""
  else: inter = """HVX_VectorPair s0 = V6(vshuffvdd)(r[1], r[0], -1), q0 = V6(vshuffvdd)(V6(hi)(s0), V6(lo)(s0), -1);
  HVX_VectorPair s1 = V6(vshuffvdd)(r[3], r[2], -1), q1 = V6(vshuffvdd)(V6(hi)(s1), V6(lo)(s1), -1);
  HVX_Vector v[4] = {V6(lo)(q0), V6(hi)(q0), V6(lo)(q1), V6(hi)(q1)};"""
  # acc[n][j] for the M/32 vectors j, w[n][g] for the K/4 groups g of 4 rows
  # (B is words: the call passes the words of the scalar loads B is, see render_wmma)
  mv, kg = m//32, k//4
  return f"""static inline __attribute__((always_inline)) {c} __{name}_w({a} a, const unsigned int *w, {c} c) {{
  HVX_Vector r[4], acc[{n*mv}];
  __builtin_memcpy(r, &a, sizeof(r)); __builtin_memcpy(acc, &c, sizeof(acc));
  {inter}
  for (int n = 0; n < {n}; n++) for (int g = 0; g < {kg}; g++) for (int j = 0; j < {mv}; j++)
    acc[n*{mv}+j] = V6({'vrmpybus_acc' if signed_b else 'vrmpyub_acc'})(acc[n*{mv}+j], v[g*{mv}+j], w[n*{kg}+g]);
  __builtin_memcpy(&c, acc, sizeof(acc)); return c;
}}
static inline __attribute__((always_inline, unused)) {c} __{name}({a} a, {b} b, {c} c) {{
  unsigned int w[{n*kg}]; __builtin_memcpy(w, &b, sizeof(w)); return __{name}_w(a, w, c); }}"""

# (B's dtype can be different from A's)
def wmma_name(x:UOp) -> str: return _wmma_name(x) + (f"_{x.src[1].dtype.name}" if x.src[1].dtype != x.arg[1] else "")
def render_wmma(ctx:DSPRenderer, x:UOp) -> str:
  # a B that's whole scalar loads in order is their words
  vecs = dedup([unwrap(lane_of(s))[0] for s in b.src]) if (b:=x.src[1]).op is Ops.STACK and all(lane_of(s) is not None for s in b.src) else []
  if not vecs or [unwrap(lane_of(s)) for s in b.src] != [(v, i) for v in vecs for i in range(v.max_numel())] or \
     any(v.op is not Ops.LOAD or v.max_numel()*v.dtype.itemsize not in (4, 8) or v in ctx.wide_loads for v in vecs):
    return f"__{wmma_name(x)}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"
  words = [f"(unsigned int)(__builtin_bit_cast(unsigned long long, {ctx[v]}) >> {32*h})" if v.max_numel()*v.dtype.itemsize == 8 else
           f"__builtin_bit_cast(unsigned int, {ctx[v]})" for v in vecs for h in range(v.max_numel()*v.dtype.itemsize//4)]
  return f"__{wmma_name(x)}_w({ctx[x.src[0]]}, (const unsigned int[]){{{', '.join(words)}}}, {ctx[x.src[2]]})"

def render_range(ctx:DSPRenderer, x:UOp) -> str|None:
  # each core runs a contiguous chunk of the parallel loop
  if x is not ctx.parallel: return None
  i, n = ctx[x], ctx[x.src[0]]
  return f"for ({ctx.render_dtype(x.dtype)} {i} = (core*{n})/{NCORES}; {i} < ((core+1)*{n})/{NCORES}; {i}++) {{"

def render_wide_load(ctx:DSPRenderer, x:UOp, bidx:UOp) -> str|None:
  # a load of less than a vector register from a global buffer loads the whole register (the buffers can be read past their ends)
  return f"(*(({ctx._render_dtype(x.dtype, 128//x.dtype.itemsize)}_u*)({ctx[bidx]})))" if x in ctx.wide_loads else None

def render_store_pieces(ctx:DSPRenderer, bidx:UOp, var:UOp) -> str|None:
  if bidx.op is not Ops.SHRINK or (n:=bidx.max_numel()) < 2: return None
  return _store_pieces(ctx, ctx[bidx], ctx[var], var.dtype, n, bidx)

def _store_pieces(ctx:DSPRenderer, ptr:str, var:str, dtype:DType, n:int, bidx:UOp) -> str|None:
  # a store of up to a vector register that isn't a whole aligned one is masked stores of the vector register
  if 32 < (nb:=n*dtype.itemsize) <= 128 and (nb < 128 or known_factor(bidx.src[1]) % n) and bidx.src[0].addrspace is AddrSpace.GLOBAL:
    v = f"((HVX_Vector)({var if vpad(n)*dtype.itemsize == 128 else shuffle(var, var, list(range(n)) + [-1]*(128//dtype.itemsize-n))}))"
    return f"store_n((unsigned char*)({ptr}), {v}, {nb});"
  # a vector store that isn't a power of two lanes (rendered padded) stores pieces that are
  if vpad(n) == n: return None
  pieces, o = [], 0
  while o < n:
    l = 1 << ((n-o).bit_length()-1)
    val = f"({var})[{o}]" if l == 1 else shuffle(var, var, range(o, o+l))
    pieces.append(f"*(({ctx._render_dtype(dtype, l) + ('_u' if l > 1 else '')}*)({ptr}+{o})) = {val};")
    o += l
  return " ".join(pieces)

dsp_vec_rewrite = PatternMatcher([
  (UPat(Ops.WMMA, name="x"), render_wmma),
  (UPat(Ops.RANGE, name="x"), render_range),
  (UPat(Ops.STORE, src=(UPat.var("bidx"), UPat.var("var"))), render_store_pieces),
  (UPat(Ops.LOAD, src=(UPat.var("bidx"),), name="x"), render_wide_load),
  (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda ctx,x: to_bool(ctx, f"(({ctx[x.src[0]]}) != 0)", x.max_numel()) if is_vec(x) else None),
  (UPat(Ops.CAST, name="x"),
   lambda ctx,x: f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_type(x)})" if is_vec(x) or is_vec(x.src[0]) else None),
  (UPat(GroupOp.ALU, name="x"), render_vec_alu),
  (UPat(Ops.STACK, name="x"), render_vec_stack),
])

# ***** L2 prefetch *****
# there's no hardware prefetch into L2 and only one l2fetch in flight per thread. a load prefetches what it reads in the next iteration of the
# innermost loop it reads at least PREFETCH_MIN bytes in, as a box of rows. the prefetch goes right before the loop inside that one.
# a buffer of at most PREFETCH_SMALL bytes isn't prefetched, a prefetch of at least PREFETCH_CUR bytes is of the iteration itself
# what's read again in a loop (up to PREFETCH_REUSE bytes) is in L2 after its first iteration
PREFETCH_MIN, PREFETCH_SMALL, PREFETCH_CUR, PREFETCH_REUSE = 2048, 16384, 32768, 256 << 10

def footprint(idx:UOp, inner:list[UOp], consts:list[int], itemsize:int, width:int) -> list[tuple[int, int]]|None:
  # the byte intervals idx+c reads for all c in consts and all values of the inner ranges, relative to idx+min(consts) with them at 0
  def coef(r:UOp) -> int|None:
    # the stride of r in the affine idx
    ret = 0
    for t in idx.split_uop(Ops.ADD):
      if r not in t.toposort(): continue
      if t is r: ret += 1
      elif t.op is Ops.MUL and t.src[0] is r and isinstance(c:=const_val(t.src[1]), int): ret += c
      elif t.op is Ops.SHL and t.src[0] is r and isinstance(c:=const_val(t.src[1]), int): ret += 1 << c
      else: return None
    return ret
  strides = []
  for r in inner:
    if (s:=coef(r)) is None: return None
    strides.append((s * itemsize, int(r.vmax)+1))
  if prod(n for _,n in strides) * len(consts) > 4096: return None
  offs = sorted({(c-min(consts))*itemsize + sum(s*i for (s,_),i in zip(strides, x))
                 for c in consts for x in itertools.product(*[range(n) for _,n in strides])})
  ivs: list[tuple[int, int]] = []
  for o in offs:
    if ivs and o <= ivs[-1][1]: ivs[-1] = (ivs[-1][0], max(ivs[-1][1], o+width))
    else: ivs.append((o, o+width))
  return ivs

def l2fetch_rt(ivs:list[tuple[int, int]]) -> tuple[int, int, int]|None:
  # (stride, width, height) of an l2fetch of the intervals, each up to 16 bits. a contiguous interval is lines of 128 bytes
  if len(ivs) == 1 or all(a[1] == b[0] for a,b in zip(ivs, ivs[1:])): return (128, 128, min(0xffff, (ivs[-1][1]-ivs[0][0]+127)//128))
  if not all_same([b-a for a,b in ivs]) or not all_same([b[0]-a[0] for a,b in zip(ivs, ivs[1:])]):
    # irregular intervals that are mostly the hull prefetch the hull
    return l2fetch_rt([(ivs[0][0], ivs[-1][1])]) if (ivs[-1][1]-ivs[0][0]) <= 2*sum(b-a for a,b in ivs) else None
  if (stride:=ivs[1][0]-ivs[0][0]) > 0xffff: return None
  return (stride, min(0xffff, ivs[0][1]-ivs[0][0]), min(0xffff, len(ivs)))

def add_prefetch(sink:UOp) -> UOp|None:
  if any(u.op is Ops.CALL and u.body.op is Ops.NATIVE and "l2fetch" in str(u.body.arg.code) for u in sink.toposort()): return None
  # the loop tree: the parent of a range is the innermost range of its END
  ends = {u.src[1]:u for u in sink.toposort() if u.op is Ops.END and u.src[1].op is Ops.RANGE}
  def depth(r:UOp) -> int: return len(ends[r].ranges) if r in ends else -1
  parent = {r:max(e.ranges, key=depth) if e.ranges else None for r,e in ends.items()}
  # loads of a buffer that are the same up to a constant prefetch together
  groups: dict[tuple[UOp, UOp], list[tuple[UOp, int]]] = {}
  for u in sink.toposort():
    if u.op is not Ops.LOAD or (a:=u.src[0]).op not in (Ops.SHRINK, Ops.INDEX) or a.src[0].op is not Ops.PARAM or len(a.src) < 2: continue
    if (idx:=a.src[1].get_idx()).is_invalid or not all(r in ends for r in idx.ranges): continue
    # (the terms of the base in a canonical order)
    terms = [t.src[0] if t.op is Ops.CAST and t.src[0].op is Ops.CONST else t for t in idx.split_uop(Ops.ADD)]
    if not (rest:=[t for t in terms if t.op is not Ops.CONST]): continue
    c, base = sum(int(t.arg) for t in terms if t.op is Ops.CONST), functools.reduce(operator.add, sorted(rest, key=lambda t: t.tuplize))
    groups.setdefault((a.src[0], base), []).append((a, c))
  subs: dict[UOp, UOp] = {}
  plan: dict[UOp, list[tuple]] = {}
  for (buf, base), accs in groups.items():
    # a small buffer stays in L2 (and its prefetch would stop a bigger one in flight)
    if not base.ranges or buf.max_numel()*buf.dtype.itemsize <= PREFETCH_SMALL: continue
    # the loops around the loads, outermost first (all the loops of the index)
    chain = [max(base.ranges, key=depth)]
    while (p:=parent[chain[0]]) is not None: chain.insert(0, p)
    if not all(r in chain for r in base.ranges): continue
    consts, width, best = dedup([c for _,c in accs]), max(a.max_numel() for a,_ in accs) * buf.dtype.itemsize, None
    # inside a loop it doesn't index (but the outermost, the cores split it), what it reads is read again from L2 if it fits: it's only
    # prefetched in that loop's first iteration
    reuse = next((r for k,r in enumerate(chain) if (k > 0 or (r.vmax+1) >= 2*NCORES) and r not in base.ranges and (fp:=footprint(base,
                  [q for q in chain[k+1:] if q in base.ranges], consts, buf.dtype.itemsize, width)) is not None and
                  sum(b-a for a,b in fp) <= PREFETCH_REUSE), None)
    for i in reversed(range(len(chain))):
      if chain[i] not in base.ranges: continue
      if (ivs:=footprint(base, [r for r in chain[i+1:] if r in base.ranges], consts, buf.dtype.itemsize, width)) is None: break
      if (rt:=l2fetch_rt(ivs)) is None: break
      best = (i, rt, sum(b-a for a,b in ivs), chain, accs, base, consts, reuse if reuse is not None and chain.index(reuse) < i else None)
      if best[2] >= PREFETCH_MIN: break
    if best is not None: plan.setdefault(buf, []).append(best)
  # of the prefetches of a buffer, the big ones (or the biggest one)
  todo = [(buf, pp) for buf,ps in plan.items() for pp in ([q for q in ps if q[2] >= PREFETCH_MIN] or [max(ps, key=lambda q: q[2])])]
  # an l2fetch stops the one in flight: each loop prefetches only its biggest
  per_loop: dict[UOp, tuple[UOp, tuple]] = {}
  for buf,pp in todo:
    if (at:=pp[3][pp[0]]) not in per_loop or per_loop[at][1][2] < pp[2]: per_loop[at] = (buf, pp)
  for buf,(i, rt, size, chain, accs, base, consts, reuse) in per_loop.values():
    loop, inner = chain[i], [r for r in chain[i+1:] if r in base.ranges]
    # the loop's next iteration, the last one prefetches itself. a big one is prefetched in the iteration itself, it runs ahead of the loads
    tmp = UOp.variable("prefetch_loop", 0, int(loop.vmax), loop.dtype)
    nxt = base.substitute({loop:tmp, **{r:r.const_like(0) for r in inner}}).simplify().substitute(
      {tmp:loop if size >= PREFETCH_CUR else (loop < int(loop.vmax)).where(loop+1, loop)})
    # an l2fetch outside the mapped memory faults: the start is clamped into the buffer
    start = nxt + min(consts)
    extent = -(-((rt[2]-1)*rt[0] + rt[1]) // buf.dtype.itemsize)
    if extent > buf.max_numel(): continue
    if start.vmin < 0: start = (start < 0).where(start.const_like(0), start)
    if start.vmax > (last:=buf.max_numel()-extent): start = (start < last).where(start, start.const_like(last))
    # Rtt is (stride << 32) | (width << 16) | height
    # (the first iteration of the outermost loop is a core's first, see render_range)
    first = "1" if reuse is None else f"{{1}} == (core*{int(reuse.vmax)+1})/{NCORES}" if reuse is chain[0] else "{1} == 0"
    pf = UOp.native(f"if ({first}) __builtin_HEXAGON_Y5_l2fetch({{0}}, {(rt[0] << 32) | (rt[1] << 16) | rt[2]}ull);", fmt=True) \
      .call(buf.index(start), *([reuse] if reuse is not None else []))
    if i+1 < len(chain): subs[chain[i+1]] = (r:=subs.get(chain[i+1], chain[i+1])).replace(src=(r.src[0].after(pf),))
    else: subs.update({a:a.replace(src=(a.src[0].after(pf),)+a.src[1:]) for a,_ in accs})
  return sink.substitute(subs, name="prefetch") if subs else None

def minmax(w:UOp) -> UOp|None:
  # a vector select of x or a bound by a compare of them is a min or max (a clamp)
  if not is_vec(w) or not dtypes.is_int(w.dtype): return None
  c, a, b = w.src
  # (through the nots of the select's polarity)
  while c.op in (Ops.CMPNE, Ops.CMPEQ) and isinstance(cv:=const_val(c.src[1]), bool):
    c, a, b = (c.src[0], a, b) if (c.op is Ops.CMPEQ) == cv else (c.src[0], b, a)
  if c.op is not Ops.CMPLT: return None
  def same(x:UOp, y:UOp) -> bool: return x is y or (const_val(x) is not None and const_val(x) == const_val(y))
  # where(x<y, x, y) is min, where(x<y, y, x) is max
  if same(c.src[0], a) and same(c.src[1], b): fn = "__builtin_elementwise_min"
  elif same(c.src[0], b) and same(c.src[1], a): fn = "__builtin_elementwise_max"
  else: return None
  if not is_vec(a): a, b = b, a
  if not is_vec(a) or (is_vec(b) and b.shape != a.shape): return None
  # (a scalar is broadcast by the vector arithmetic)
  return UOp.native(f"{fn}({{0}}, {{1}})" if is_vec(b) else f"{fn}({{0}}, ({{0}})*0+({{1}}))", shape=w.shape, dtype=w.dtype,
                             fmt=True).call(a, b)

def lookup(x:UOp) -> UOp|None:
  # 128 bytes loaded from a table of 256 at the bytes of a vector (plus a const) are lut256
  if len(x.src) != 128 or x.dtype.itemsize != 1 or any(s.op is not Ops.LOAD or s.src[0].op is not Ops.INDEX for s in x.src): return None
  if (t:=x.src[0].src[0].src[0]).op is not Ops.PARAM or t.max_numel() > 256 or any(s.src[0].src[0] is not t for s in x.src): return None
  def byte_lane(u:UOp, c:int=0) -> tuple[UOp, int, int]|None:
    if u.op is Ops.CAST: return byte_lane(u.src[0], c)
    if u.op is Ops.ADD and isinstance(k:=const_val(u.src[1]), int): return byte_lane(u.src[0], c+k)
    return None if (l:=lane_of(u)) is None or l[0].dtype.itemsize != 1 else (*l, c)
  lanes = [l for s in x.src if 0 <= (i:=s.src[0].src[1]).vmin and i.vmax < 256 and (l:=byte_lane(i)) is not None]
  if len(lanes) != 128 or not all_same([(v, c) for v,_,c in lanes]) or [i for _,i,_ in lanes] != list(range(128)): return None
  v, _, c = lanes[0]
  return UOp.native(f"lut256((hvx_ub)({{0}}) + (hvx_ub)({c & 0xff}), {{1}})", shape=x.shape, dtype=x.dtype, fmt=True).call(v, t)

dsp_late = PatternMatcher([
  (UPat(Ops.STACK, name="x"), lookup),
  (UPat(Ops.SINK, name="sink"), add_prefetch),
  (UPat(Ops.WHERE, name="w"), minmax),
  (UPat(Ops.ADD, name="x"), vrmpy_sum),
  (UPat(Ops.CAST, name="x", src=(UPat(Ops.SHR, src=(UPat(Ops.MUL, src=[UPat(Ops.CAST, name="a"), UPat(name="c")]), UPat(name="s"))),)), mulhi),
])

# the DSP has 2 HVX contexts in 128 byte mode: a kernel runs on 2 cores, each a thread with a stack of DSP_STACK bytes
# the buffers can be read DSP_OVERREAD bytes before and after
NCORES, DSP_STACK, DSP_OVERREAD = 2, 64 << 10, 512
DSP_THREADS_PRELUDE = """typedef unsigned long qurt_thread_t;
typedef struct { char name[16]; unsigned char tcb_partition; unsigned char affinity; unsigned short priority; unsigned char asid;
  unsigned char bus_priority; unsigned short timetest_id; unsigned int stack_size; void *stack_addr; char padding[96]; } qurt_thread_attr_t;
int qurt_thread_create(qurt_thread_t *thread_id, qurt_thread_attr_t *attr, void (*entrypoint)(void *), void *arg);
int qurt_thread_join(qurt_thread_t tid, int *status); void qurt_thread_exit(int status);
void* malloc(unsigned int); void free(void*);"""

DSP_INLINE_BYTES = 32 << 10
class DSPRenderer(ClangRenderer):
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_typedef = "__attribute__((noinline)) void"
  packed_args = False
  vector_alu = True
  overread = DSP_OVERREAD
  # HVX v65 has no float
  vector_dtypes = dtypes.ints + (dtypes.bool,)
  type_map = { **ClangRenderer.type_map, dtypes.uint64: "unsigned long long", dtypes.int64: "long long" }
  code_for_op = {k:v for k,v in ClangRenderer.code_for_op.items() if k != Ops.SQRT}
  string_rewrite = dsp_vec_rewrite + base_rewrite
  tensor_cores = tc.hexagon
  late_matcher = dsp_late

  # every kernel runs on all the cores, the outermost loop is split between them
  extra_args = ["int core"]
  parallel: UOp|None = None
  inline: bool = False
  wide_loads: set[UOp] = set()

  def __init__(self, target:Target): self.target, self.compiler = target, DSPCompiler()

  def render(self, uops:list[UOp]) -> str:
    # the loop that's split between the cores: the only outermost loop, if its iterations may run in parallel and all the global stores are in it
    depth, top = 0, []
    for u in uops:
      if u.op is Ops.RANGE:
        if depth == 0: top.append(u)
        depth += 1
      if u.op is Ops.END: depth -= 1
    stores = [u for u in uops if u.op is Ops.STORE and u.src[0].addrspace is AddrSpace.GLOBAL]
    self.parallel = top[0] if len(top) == 1 and top[0].arg[0] in (AxisType.GLOBAL, AxisType.WEAK) and stores and \
      all(top[0] in st.ranges for st in stores) else None
    # a kernel with little memory or no parallel loop runs its cores one after the other on the calling thread
    self.inline = self.parallel is None or sum(u.max_numel()*u.dtype.itemsize for u in uops if u.op is Ops.PARAM and
                                               u.addrspace is AddrSpace.GLOBAL) < DSP_INLINE_BYTES
    # a small load whose lanes are only used one at a time loads the whole vector register
    users: dict[UOp, list[UOp]] = {}
    for u in uops:
      for s in u.src: users.setdefault(s, []).append(u)
    self.wide_loads = {u for u in uops if can_load_wide(u) and all(lane_of(x) is not None and x.src[0] is u for x in users.get(u, []))}
    try: return super().render(uops)
    finally: self.parallel, self.inline = None, False

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    # without a parallel loop the first core does all the work
    if self.parallel is None: kernel = ["  if (core != 0) return;"] + kernel
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  # register arrays are vectors
  # (with room for the padded vectors)
  def render_buffer(self, x:UOp):
    return f"{self._render_dtype(x.dtype)} {self[x]}[{vpad(x.max_numel()) + 128 // x.dtype.itemsize}] __attribute__((aligned(128)));"

  def _render_dtype(self, dtype:DType, sz:int=1, addrspace=AddrSpace.ALU, mutable=True, override_ptr=False, shape=None):
    return super()._render_dtype(dtypes.uint8 if dtype == dtypes.bool and sz > 1 else dtype, vpad(sz), addrspace, mutable, override_ptr, shape)

  def render_wmmas(self, uops) -> list[str]:
    def dt(s:UOp) -> str: return self._render_dtype(s.dtype, int(s.shape[-1]))
    return [render_vrmpy(wmma_name(u), u.arg[0], dt(u.src[0]), dt(u.src[1]), dt(u.src[2]), u.src[1].dtype == dtypes.int8)
            for u in {wmma_name(u):u for u in uops if u.op is Ops.WMMA}.values()]

  def render_type(self, u:UOp):
    return self._render_dtype(u.dtype, 128 // u.dtype.itemsize) if u in self.wide_loads else super().render_type(u)

  def render_ptr(self, u:UOp):
    # a vector access that isn't at a multiple of its size is unaligned
    if u.op is Ops.SHRINK and u.max_numel() > 1 and known_factor(u.src[1]) % vpad(u.max_numel()):
      return f"(({self._render_dtype(u.dtype, u.max_numel())}_u*)({self[u]}))"
    return super().render_ptr(u)

  def _vector_typedefs(self, uops) -> list[str]:
    # every dtype at every vector width: the vector ops splat scalars and select with masks
    counts = dedup([cnt for _,cnt in uops_to_dtypes(uops) if cnt > 1] + [32, 64, 128])
    # (bytes are always there: bools are bytes, shuffles of bytes blend)
    dts = dedup([u.dtype for u in uops if u.dtype in self.type_map and u.dtype not in (dtypes.void, dtypes.bool)] +
                [dtypes.uint8, *mask_dtype.values()])
    # and the HVX vector width of every dtype (shuffles of smaller vectors widen to it)
    # and all the smaller powers of two (the pieces of stores)
    pow2 = [1 << i for i in range(1, max(counts).bit_length()+1)]
    return dedup([self.render_vector_prefix(dt, cnt) for dt in dts for cnt in dedup(counts + pow2 + [128 // dt.itemsize]) if cnt > 1])

  def render_vector_prefix(self, dt, count:int) -> str:
    # (a vector is aligned to its size, a power of two: the type is the same as a shuffle's for older clangs)
    vec, count = self._render_dtype(dt, count := vpad(count)), vpad(count)
    return f"typedef {self.render_dtype(dt)} {vec} __attribute__((ext_vector_type({count})));\n" + \
           f"typedef {self.render_dtype(dt)} {vec}_u __attribute__((aligned(1),ext_vector_type({count})));"

  def _render_defines(self, uops) -> list[str]: return [HVX_PRELUDE] + self._vector_typedefs(uops) + self.render_wmmas(uops)

  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # the DSP runs kernel_run from a batch of commands (see DSP_UTIL_SRC) with the args laid out by TinyELF.iter_sig and the pool of workers
    def ty(b:UOp) -> str: return self._render_dtype(b.dtype) if b.addrspace == AddrSpace.ALU else "void*"
    offs = [o for o,_ in TinyELF.iter_sig(tuple((None, b.addrspace, b.dtype, ()) for _,(b,_) in bufs))]
    msrc = ['typedef struct { ' + ' '.join(f'{ty(b)} a{i};' for i,(_,(b,_)) in enumerate(bufs)) + ' int core; } kargs_t;',
            # the workers of the pool run cores of the kernel
            'typedef struct { void (*run)(void *pool, void (*fn)(void*), void *args, int argsize, int ncaller); } pool_t;',
            '__attribute__((unused)) static void kcore(void *p) { kargs_t *a = p; ' +
            f'{function_name}({", ".join([f"a->a{i}" for i in range(len(bufs))] + ["a->core"])}); }}',
            'void kernel_run(const unsigned char *a, pool_t *pool) {', f'kargs_t args[{NCORES}]; (void)pool;']
    # (a buffer is its address)
    msrc += [f'args[0].a{i} = ' + (f'*(const {ty(b)}*)(a+{o});' if b.addrspace == AddrSpace.ALU else f'(void*)*(const unsigned int*)(a+{o});')
             for i,((_,(b,_)),o) in enumerate(zip(bufs, offs))]
    msrc += [f"for (int i = 1; i < {NCORES}; i++) {{ args[i] = args[0]; args[i].core = i; }} args[0].core = 0;"]
    # the cores from ncaller on run on the workers, the ones before on the calling thread
    def run(ncaller:int) -> list[str]:
      return [f'{function_name}({", ".join([f"args[0].a{i}" for i in range(len(bufs))] + [str(c)])});' for c in range(ncaller)] if ncaller == NCORES \
        else [f"pool->run(pool, kcore, args, sizeof(kargs_t), {ncaller});"]
    # the calling thread's stack takes small frames of vectors only: a big frame runs on the workers (see DSPCompiler)
    msrc += ["#ifdef DSP_BIG_FRAME"] + run(0) + ["#else"] + run(NCORES if self.inline else 1) + ["#endif", "}"]
    return '\n'.join(msrc)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}

def rpc_sc(method=0, ins=0, outs=0, fds=0): return (method << 24) | (ins << 16) | (outs << 8) | fds
def rpc_prep_args(ins=None, outs=None, in_fds=None):
  ins, outs, in_fds = ins or list(), outs or list(), in_fds or list()

  pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs) + len(in_fds)))()
  fds = (ctypes.c_int32 * (len(ins) + len(outs) + len(in_fds)))(*([-1] * (len(ins) + len(outs))), *in_fds)
  attrs = (ctypes.c_uint32 * (len(ins) + len(outs) + len(in_fds)))(*([0] * (len(ins) + len(outs))), *([1] * (len(in_fds))))

  for i, mv in enumerate(ins + outs): pra[i].buf.pv, pra[i].buf.len = ctypes.c_void_p(mv_address(mv) if mv.nbytes > 0 else 0), mv.nbytes
  return pra, fds, attrs, (ins, outs)

# ***** the queue: a batch of commands the DSP runs in one call (see DSP_UTIL_SRC) *****

# a command is 3 u64s: op, a, b
DSP_EXEC, DSP_WAIT, DSP_WAIT_EQ, DSP_STORE, DSP_TIMESTAMP = range(5)

@uopfunc
def dsp_submit(req:UOp, ret:UOp, fd:UOp) -> UOp:
  idir, base, nr, struct_t = qcom_dsp.FASTRPC_IOCTL_INVOKE.args
  ioctl_cmd = (idir << 30) | (ctypes.sizeof(struct_t) << 16) | (base << 8) | nr
  return ret.index(0).store(ccall(libc.dll.ioctl, fd, UOp.const(ioctl_cmd, dtypes.uint32), req.index(0))).sink()

@uopfunc
def mockdsp_ring(cmds:UOp, addr:UOp, n:UOp) -> UOp: return ccall(MockDSP.lib.mockdsp_submit, addr, n).sink()

class DSPQueue(HWQueue):
  def cmd(self, op:int, a:UOp|int=0, b:UOp|int=0):
    self.q(*[x.cast(dtypes.uint64) if isinstance(x, UOp) else UOp.const(x, dtypes.uint64) for x in (op, a, b)])
  def exec(self, call:UOp, prg:UOp):
    args = get_call_kernel_args(call, prg, self.devs)
    kernargs = UOp(Ops.LINEAR, src=tuple(pack_args(rows:=layout_args(args), round_up(max([o + w.dtype.itemsize for o, w in rows], default=8), 8))),
                   arg="kernargs")
    # (the address of the program: the DSP reads its kernel_run there, see DSPDevice.function)
    self.cmd(DSP_EXEC, (p:=make_program(prg, 8, self.devs[0])).after(p.store(prg)).getaddr(self.devs), kernargs.getaddr(self.devs))
  def wait(self, signal:UOp, value:UOp, eq:bool=False): self.cmd(DSP_WAIT_EQ if eq else DSP_WAIT, signal.getaddr(self.devs), value)
  def signal(self, signal:UOp, value:UOp): self.cmd(DSP_STORE, signal.getaddr(self.devs), value)
  # (a timestamp slot is [signal][timestamp])
  def timestamp(self, signal:UOp): self.cmd(DSP_TIMESTAMP, signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64))
  def submit(self, cmdbuf:UOp) -> UOp:
    ib, off = unwrap_view(cmdbuf)
    addr, n = ib.getaddr(self.devs) + off, cmdbuf.max_numel()
    if cast(DSPDevice, self.dev).mock is not None: return mockdsp_ring(cmdbuf, addr, UOp.const(n, dtypes.uint64))
    # the util lib's RUN, its args are copied to the DSP
    args = patch(UOp.alloc((12,), dtypes.uint8, device=HCQ_RUNTIME_DEV.device, name="dsp_run"),
                 layout_args([DSP_UTIL_RUN, addr.cast(dtypes.uint32), n]), bytes(12))
    pra = cstruct(qcom_dsp.struct_remote_buf, pv=args.getaddr(HCQ_RUNTIME_DEV.device), len=12)
    req = cstruct(qcom_dsp.struct_fastrpc_ioctl_invoke, handle=UOp.variable("dsp_util", 0, 2**32-1, dtypes.uint32),
                  sc=rpc_sc(method=2, ins=1), pra=pra.getaddr(HCQ_RUNTIME_DEV.device))
    return dsp_submit(req.after(cmdbuf), UOp.alloc((1,), dtypes.int32, device=HCQ_RUNTIME_DEV.device, name="dsp_submit_ret"),
                      UOp.variable("dsp_fd", 0, 2**31-1, dtypes.int32))

Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("dsp_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), lambda s: DSPQueue(s).encode()),
])

# ***** memory *****

class DSPArena:
  # the DSP maps few buffers (~250): the buffers are parts of big ones, each mapped on the DSP once
  # they're uncached on the host, the DSP writes back and drops its caches when it starts and finishes a batch
  def __init__(self, dev:DSPDevice, size:int):
    self.dev, self.size = dev, size
    self.handle = qcom_dsp.ION_IOC_ALLOC(dev.ion_fd, len=size, align=0x1000, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=0).handle
    self.share_info = qcom_dsp.ION_IOC_SHARE(dev.ion_fd, handle=self.handle)
    self.va = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, self.share_info.fd, 0)
    self.dsp_addr, self.parts = dev.dsp_util(DSP_UTIL_MAP, size=size, fd=self.share_info.fd), TLSFAllocator(size)

DSP_ARENA = 32 << 20
class DSPAllocator(Allocator['DSPDevice']):
  # a buffer is its address on the DSP, the host accesses it through its own mapping
  def __init__(self, dev:DSPDevice):
    super().__init__(dev)
    self.arenas: list[DSPArena] = []

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    # the kernels read up to DSP_OVERREAD bytes before and after a buffer (vector loads with invalid lanes, clang widening small vector loads)
    asize = round_up(size + 2*DSP_OVERREAD, 128)
    # the mock's buffers are host memory, and the buffers before the DSP's session starts (its shell) are ION buffers of their own
    if (mock:=self.dev.mock is not None) or not hasattr(self.dev, "rpc_fd"):
      share_info = None if mock else \
        qcom_dsp.ION_IOC_SHARE(self.dev.ion_fd, handle=qcom_dsp.ION_IOC_ALLOC(self.dev.ion_fd, len=round_up(asize, 0x1000), align=0x1000,
                               heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=qcom_dsp.ION_FLAG_CACHED).handle)
      va = libc.mmap(0, round_up(asize, 0x1000), mmap.PROT_READ|mmap.PROT_WRITE,
                     mmap.MAP_SHARED|(mmap.MAP_ANONYMOUS if share_info is None else 0), share_info.fd if share_info is not None else -1, 0)
      return BufferStorage(va+DSP_OVERREAD, (None, va, share_info), MMIOInterface(va+DSP_OVERREAD, size))
    for arena in self.arenas:
      with contextlib.suppress(MemoryError):
        off = arena.parts.alloc(asize, 128)
        break
    else:
      self.arenas.append(arena:=DSPArena(self.dev, max(DSP_ARENA, round_up(asize, 0x1000))))
      off = arena.parts.alloc(asize, 128)
    return BufferStorage(arena.dsp_addr+off+DSP_OVERREAD, (arena, off, None), MMIOInterface(arena.va+off+DSP_OVERREAD, size))

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    # (the meta is the arena and the part's offset in it, or the mapping and its ION buffer)
    arena, at, share_info = storage.meta
    if arena is not None: return arena.parts.free(at)
    libc.munmap(at, round_up(unwrap(storage.host).nbytes + 2*DSP_OVERREAD, 0x1000))
    if share_info is not None:
      os.close(share_info.fd)
      qcom_dsp.ION_IOC_FREE(self.dev.ion_fd, handle=share_info.handle)

  def _copyin(self, dest:int, src:memoryview): raise RuntimeError("the host copies through the buffer's host mapping")
  def _copyout(self, dest:memoryview, src:int): raise RuntimeError("the host copies through the buffer's host mapping")
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

# ***** compiler *****

# the biggest stack frame the calling (fastrpc) thread runs: a bigger one of vectors crashes it
DSP_CALLER_FRAME = 4096
class DSPCompiler(Compiler):
  def __init__(self, mock:bool=False):
    self.mock, compiler_args = mock, "--target=hexagon -mcpu=hexagonv65 -fuse-ld=lld -nostdlib -mhvx=v65 -mhvx-length=128b -O2"
    # (qemu asserts translating some packets with a .cur load, the mock's code isn't scheduled to make them)
    if mock: self.args = f"-static {compiler_args} -mllvm -enable-cur-sched=false"
    else:
      # Generate link script to pass into clang. Aligning all used sections to 4k fixes invoke problem.
      # the sections are in three segments (read only, code, data), each a page aligned map on the DSP: it takes few maps
      groups = [['hash', 'dynsym', 'dynstr', 'rela.dyn', 'rela.plt', 'rodata'], ['text', 'plt'], ['dynamic', 'got', 'got.plt', 'data', 'bss']]
      sections_link = '\n'.join(f'.{n} : {"ALIGN(4096)" if i == 0 else ""} {{ *(.{n}) }}' for g in groups for i,n in enumerate(g))
      link_script = f"SECTIONS {{ . = 0x0; {sections_link}\n /DISCARD/ : {{ *(.note .note.* .gnu.hash .comment) }} }}"
      with tempfile.NamedTemporaryFile(delete=False) as self.link_ld:
        self.link_ld.write(link_script.encode())
        self.link_ld.flush()

      self.args = f"-shared {compiler_args} -Wl,-z,norelro -T{self.link_ld.name}"

    # the cache depends on how it's compiled and linked
    key = hashlib.sha256((compiler_args + link_script + str(DSP_CALLER_FRAME)).encode()).hexdigest()[:16] if not mock else ""
    super().__init__(None if mock else f"compile_dsp_{key}")

  def __del__(self):
    if not self.mock: os.unlink(self.link_ld.name)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as f:
      # (the lanes of a padded vector may load values that aren't used)
      def cc(flags:str): system(f"{getenv('CC','clang')} {self.args} {flags} -Wall -Werror -Wno-unused-variable -fno-stack-protector -x c -fPIC " +
                                f"-ffreestanding -nostdlib - -o {f.name}", input=src.encode())
      # a kernel with a big stack frame (vector spills) can't run on the calling thread, it's compiled to run on threads only
      if self.mock: cc("")
      else:
        try: cc(f"-Wframe-larger-than={DSP_CALLER_FRAME}")
        except RuntimeError as e:
          if "stack frame size" not in str(e): raise
          cc("-DDSP_BIG_FRAME")
      return pathlib.Path(f.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, "llvm-objdump")


class MockDSPRenderer(DSPRenderer):
  def __init__(self, target:Target): self.target, self.compiler = target, DSPCompiler(mock=True)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # https://gpages.juszkiewicz.com.pl/syscalls-table/syscalls.html
    # control register 21 is HEX_REG_QEMU_INSN_CNT, 0x6a15c000 loads it
    msrc = [mockdsp_boilerplate, 'void _start(void) {']
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL:
        sz = b[1][0].max_numel()*b[1][0].dtype.itemsize
        # unmapped guard pages (qemu's are 64KB) around the buffer catch out of bounds accesses. for loop for big reads
        rsz = round_up(sz+2*DSP_OVERREAD, 0x10000)
        msrc.append(f"char *g{i} = (char*)mmap2(0, {rsz+0x20000}, 3, 0x22, -1, 0)+0x10000; munmap(g{i}-0x10000, 0x10000); "
                    f"munmap(g{i}+{rsz}, 0x10000); char *buf{i} = g{i}+{DSP_OVERREAD}; "
                    f"for(int rd = 0; rd < {sz}; rd += read(0, buf{i}+rd, {sz}-rd));")
      else:
        msrc.append(f"{self._render_dtype(b[1][0].dtype)} val{i}; read(0, &val{i}, {b[1][0].dtype.itemsize});")
    msrc.append("unsigned int st = inscount();")
    params = [(f'(void*)buf{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'val{i}') for i,b in enumerate(bufs)]
    # the cores run one after the other, in reverse to catch races
    msrc += [f"{function_name}({', '.join(params + [str(c)])});" for c in reversed(range(NCORES))]
    msrc.append("unsigned int et = inscount() - st; write(1, &et, sizeof(et));")
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL: msrc.append(f"write(1, buf{i}, {b[1][0].max_numel()*b[1][0].dtype.itemsize});")
    msrc.append('exit(0); }')
    return '\n'.join(msrc)

# a lib that maps buffers on the DSP, makes the pool of workers (threads with their own stacks) the kernels run their cores on, loads the
# kernels and runs batches of commands
DSP_UTIL_MAP, DSP_UTIL_POOL, DSP_UTIL_LOAD, DSP_UTIL_RUN = range(4)
DSP_UTIL_SRC = DSP_THREADS_PRELUDE + f"""
typedef union {{ struct {{ void *pv; unsigned int len; }} buf; struct {{ int fd; unsigned int offset; }} dma; }} remote_arg;
void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);
struct dcvs_v2_req {{ int type; int _pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency; _Bool set_dcvs_params; short _pad2;
  char target_corner; char min_corner; char max_corner; int _pad3[3]; }};
int HAP_power_set(void*, void*); unsigned long long HAP_perf_get_qtimer_count(void);
int qurt_mem_cache_clean(unsigned int addr, unsigned int size, int opcode, int type);
void *dlopen(const char *name, int flags); void *dlsym(void *handle, const char *name);
typedef union {{ unsigned long long raw; struct {{ unsigned int signals, waiting, queue, attribute; }} X; }} qurt_signal_t;
void qurt_signal_init(qurt_signal_t *signal); void qurt_signal_set(qurt_signal_t *signal, unsigned int mask);
unsigned int qurt_signal_wait(qurt_signal_t *signal, unsigned int mask, unsigned int attribute);
void qurt_signal_clear(qurt_signal_t *signal, unsigned int mask);
typedef struct pool {{ void (*run)(struct pool *pool, void (*fn)(void*), void *args, int argsize, int ncaller);
  qurt_signal_t go[{NCORES}], done; void (*fn)(void*); char *args; int argsize; }} pool_t;
typedef struct {{ pool_t *pool; int core; }} worker_t;
static void worker(void *p) {{
  worker_t *w = p; pool_t *pool = w->pool;
  for (;;) {{
    qurt_signal_wait(&pool->go[w->core], 1, 0 /* QURT_SIGNAL_ATTR_WAIT_ANY */); qurt_signal_clear(&pool->go[w->core], 1);
    pool->fn(pool->args + w->core * pool->argsize);
    qurt_signal_set(&pool->done, 1 << w->core);
  }}
}}
static void run(pool_t *pool, void (*fn)(void*), void *args, int argsize, int ncaller) {{
  pool->fn = fn; pool->args = args; pool->argsize = argsize;
  for (int i = ncaller; i < {NCORES}; i++) qurt_signal_set(&pool->go[i], 1);
  for (int i = 0; i < ncaller; i++) fn((char*)args + i * argsize);
  unsigned int mask = ((1 << {NCORES}) - 1) & ~((1 << ncaller) - 1);
  if (mask) {{ qurt_signal_wait(&pool->done, mask, 1 /* QURT_SIGNAL_ATTR_WAIT_ALL */); qurt_signal_clear(&pool->done, mask); }}
}}
static pool_t *the_pool;
static pool_t *make_pool(void) {{
  if (the_pool) return the_pool;
  pool_t *pool = malloc(sizeof(pool_t)); pool->run = run; qurt_signal_init(&pool->done);
  for (int i = 0; i < {NCORES}; i++) {{
    worker_t *w = malloc(sizeof(worker_t)); w->pool = pool; w->core = i; qurt_signal_init(&pool->go[i]);
    qurt_thread_attr_t attr; qurt_thread_t th; __builtin_memset(&attr, 0, sizeof(attr)); attr.name[0] = 'w'; attr.priority = 100;
    attr.stack_size = {DSP_STACK}; attr.stack_addr = malloc({DSP_STACK});
    qurt_thread_create(&th, &attr, worker, w);
  }}
  return the_pool = pool;
}}
typedef struct {{ unsigned long long op, a, b; }} cmd_t;
static int run_cmds(const cmd_t *c, unsigned int n) {{
  // the host writes the commands and the kernel args, and reads what the commands and the kernels write: the DSP writes back and drops its
  // caches before and after
  qurt_mem_cache_clean(0, 0, 4 /* QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL */, 1 /* QURT_MEM_DCACHE */);
  int ret = 0;
  for (const cmd_t *e = c + n / sizeof(cmd_t); c < e && !ret; c++) {{
    unsigned long long *p = (unsigned long long*)(unsigned int)c->a;
    if (c->op == {DSP_EXEC}) (*(void (**)(const void*, pool_t*))p)((const void*)(unsigned int)c->b, the_pool);
    // (the commands run in order: a wait that isn't met now never is)
    else if (c->op == {DSP_WAIT}) ret = *p < c->b;
    else if (c->op == {DSP_WAIT_EQ}) ret = *p != c->b;
    else if (c->op == {DSP_STORE}) *p = c->b;
    else if (c->op == {DSP_TIMESTAMP}) *p = HAP_perf_get_qtimer_count();
  }}
  qurt_mem_cache_clean(0, 0, 4 /* QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL */, 1 /* QURT_MEM_DCACHE */);
  return ret;
}}
int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {{
  if ((sc>>24) != 2) return 0;
  unsigned int *in = pra[0].buf.pv, op = in[0], addr = in[1], size = in[2];
  if (op == {DSP_UTIL_MAP}) *(unsigned int*)pra[1].buf.pv = (unsigned int)HAP_mmap(0, size, 3, 0, pra[2].dma.fd, 0);
  if (op == {DSP_UTIL_POOL}) {{
    struct dcvs_v2_req req = {{.type=7, .dcvs_enable=0, .set_latency=1, .latency=100, .set_dcvs_params=1, .target_corner = 6 /* TURBO */}};
    HAP_power_set((void*)handle, (void*)&req);
    *(unsigned int*)pra[1].buf.pv = (unsigned int)make_pool();
  }}
  // a kernel's lib (named after the words, see RPCListener) and its kernel_run
  if (op == {DSP_UTIL_LOAD}) {{
    void *lib = dlopen((const char*)(in + 3), 2 /* RTLD_NOW */);
    *(unsigned int*)pra[1].buf.pv = lib ? (unsigned int)dlsym(lib, "kernel_run") : 0;
  }}
  if (op == {DSP_UTIL_RUN}) return run_cmds((const cmd_t*)addr, size);
  return 0;
}}"""

class DSPDevice(Compiled):
  renderers = [MockDSPRenderer] if getenv("MOCKDSP") else [DSPRenderer]
  timestamp_divider = 19.2

  @property
  def has_copy_queue(self) -> bool: return False

  def __init__(self, device:str=""):
    super().__init__(device, DSPAllocator(self), None)
    self.mock = MockDSP(self) if getenv("MOCKDSP") else None
    if self.mock is None:
      self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
      fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
      self.shell_buf = self.allocator.alloc(round_up(fastrpc_shell.nbytes, 0x1000), BufferSpec(nolru=True))
      ctypes.memmove(unwrap(self.shell_buf.host).addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

      self.init_dsp()
      RPCListener(self).start()
      self.util_handle = self.open_lib(DSPCompiler().compile_cached(DSP_UTIL_SRC))
      self.dsp_util(DSP_UTIL_POOL)
      self.var_vals = {"dsp_fd": self.rpc_fd, "dsp_util": self.util_handle}
    # a program is its kernel_run on the DSP (in the mock, the index of the program)
    Compiled.pm_link += PatternMatcher([(UPat(Ops.ALLOC, name="p").store(UPat(Ops.PROGRAM, name="prg")), lambda p, prg, d=self:
      p.store(UOp(Ops.BINARY, arg=struct.pack("<Q", d.function(prg)))) if p.device == d.device else None)])

  @functools.cache
  def function(self, prg:UOp) -> int:
    obj = TinyELF.from_program(prg)
    if self.mock is not None: return self.mock.load(obj)
    self.binded_lib, self.binded_lib_off = obj.lib, 0
    if not (fn:=self.dsp_util(DSP_UTIL_LOAD, name=f"tinylib{hashlib.sha256(obj.lib).hexdigest()[:16]}")):
      raise RuntimeError(f"DSP can't load {obj.name}")
    return fn

  def dsp_util(self, op:int, addr:int=0, size:int=0, fd:int|None=None, name:str="") -> int:
    pra, fds, attrs, _ = rpc_prep_args(ins=[memoryview(bytearray(array.array("I", [op, addr, size]).tobytes() + name.encode() + b"\0"))],
                                       outs=[out:=memoryview(bytearray(4)).cast('I')], in_fds=[fd] if fd is not None else [])
    qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.rpc_fd, fds=fds, attrs=attrs,
      inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=self.util_handle, sc=rpc_sc(method=2, ins=1, outs=1, fds=len(fds)-2), pra=pra))
    if op == DSP_UTIL_MAP and out[0] in (0, 0xffffffff): raise MemoryError(f"DSP can't map a buffer of {size} bytes")
    return out[0]

  def open_lib(self, lib:bytes) -> int:
    self.binded_lib, self.binded_lib_off = lib, 0
    # (a name of its own: the DSP's dlopen of an open name returns the open lib)
    fp = f"file:///tinylib{hashlib.sha256(lib).hexdigest()[:16]}?entry&_modver=1.0&_dom=cdsp\0"
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [len(fp), 0xff])), memoryview(bytearray(fp.encode()))],
                                 outs=[o1:=memoryview(bytearray(0x8)), o2:=memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=0, ins=2, outs=2), pra=pra)
    if o1.cast('i')[1] < 0: raise RuntimeError(f"Cannot open lib: {o2.tobytes().decode()}")
    return o1.cast('I')[0]

  def init_dsp(self):
    self.rpc_fd: int = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)
    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)
    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=(shell:=unwrap(self.shell_buf.host)).addr, filelen=shell.nbytes,
                                filefd=self.shell_buf.meta[2].fd)
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=3, sc=rpc_sc(method=3, ins=0, outs=0))

class RPCListener(threading.Thread):
  def __init__(self, device:DSPDevice):
    super().__init__()
    self.device, self.daemon = device, True

  def run(self):
    # Setup initial request arguments.
    context, status, TINYFD = 0, 0xffffffff, 0xffff
    req_args, _, _, _ = rpc_prep_args(ins=[msg_send:=memoryview(bytearray(0x10)).cast('I'), out_buf:=memoryview(bytearray(0x10000)).cast('I')],
                                      outs=[msg_recv:=memoryview(bytearray(0x10)).cast('I'), in_buf:=memoryview(bytearray(0x10000)).cast('I')])
    req_args[1].buf.len = 0

    while True:
      # Update message request and send it.
      msg_send[:] = array.array('I', [context, status, req_args[1].buf.len, in_buf.nbytes])

      try: qcom_dsp.FASTRPC_IOCTL_INVOKE(self.device.rpc_fd, handle=0x3, sc=0x04020200, pra=req_args)
      except OSError: continue # retry

      context, inbufs, outbufs = msg_recv[0], ((sc:=msg_recv[2]) >> 16) & 0xff, (msg_recv[2] >> 8) & 0xff

      in_ptr, out_ptr, objs = mv_address(in_buf), mv_address(out_buf), []
      for i in range(inbufs + outbufs):
        obj_ptr = round_up(in_ptr + 4, 8) if i < inbufs else round_up(out_ptr + 4, 8)
        objs.append(to_mv(obj_ptr, obj_size:=to_mv(in_ptr, 4).cast('I')[0]))
        if i < inbufs: in_ptr = obj_ptr + obj_size
        else:
          to_mv(out_ptr, 4).cast('I')[0] = obj_size
          out_ptr = obj_ptr + obj_size
          in_ptr += 4

      in_args, out_args = objs[:inbufs], objs[inbufs:]
      req_args[1].buf.len = out_ptr - mv_address(out_buf)

      status = 0 # reset status, will set if error
      if sc == 0x20200: pass # greating
      elif sc == 0x13050100: # open
        try: out_args[0].cast('I')[0] = TINYFD if (name:=in_args[3].tobytes()[:-1].decode()).startswith("tinylib") else os.open(name, os.O_RDONLY)
        except OSError: status = 1
      elif sc == 0x3010000:
        if (fd:=in_args[0].cast('I')[0]) != TINYFD: os.close(fd)
      elif sc == 0x9010000: # seek
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          assert in_args[0].cast('I')[2] == qcom_dsp.APPS_STD_SEEK_SET, "Supported only SEEK_SET"
          res, self.device.binded_lib_off = 0, in_args[0].cast('I')[1]
        else: res = os.lseek(fd, in_args[0].cast('I')[1], in_args[0].cast('I')[2])
        status = 0 if res >= 0 else res
      elif sc == 0x4010200: # read
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          buf = self.device.binded_lib[self.device.binded_lib_off:self.device.binded_lib_off+in_args[0].cast('I')[1]]
          self.device.binded_lib_off += len(buf)
        else: buf = os.read(fd, in_args[0].cast('I')[1])
        out_args[1][:len(buf)] = buf
        out_args[0].cast('I')[0:2] = array.array('I', [len(buf), int(len(buf) == 0)])
      elif sc == 0x1f020100: # stat
        stat = os.stat(in_args[1].tobytes()[:-1].decode())
        out_stat = qcom_dsp.struct_apps_std_STAT.from_address(mv_address(out_args[0]))
        for f in out_stat._real_fields_: out_stat.__setattr__(f[0], int(getattr(stat, f"st_{f[0]}", 0)))
      elif sc == 0x2010100: # mmap
        st = qcom_dsp.FASTRPC_IOCTL_MMAP(self.device.rpc_fd, fd=-1, flags=in_args[0].cast('I')[2], vaddrin=0, size=in_args[0].cast('Q')[3])
        out_args[0].cast('Q')[0:2] = array.array('Q', [0, st.vaddrout])
      else: raise RuntimeError(f"Unknown op: {sc=:X}")

# ***** mock DSP *****

mockdsp_boilerplate = '''/* DSP boilerplate */ static long syscall(long r0, long r1, long r2, long r3, long r4, long r5, long r6) {
long retval; __asm__ volatile("r0 = %1; r1 = %2; r2 = %3; r3 = %4; r4 = %5; r5 = %6; r6 = %7; trap0(#1); %0 = r0" : "=r" (retval)
  : "r" (r0), "r" (r1), "r" (r2), "r" (r3), "r" (r4), "r" (r5), "r" (r6) : "r0", "r1", "r2", "r3", "r4", "r5", "r6"); return retval; }
static int read(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 63); }}
static int write(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 64); }}
static int exit(int ret) {{ return syscall(ret, 0, 0, 0, 0, 0, 93); }}
static unsigned int inscount(void) {{ unsigned int ret; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r" (ret) : : "r0"); return ret; }}
static void *mmap2(void *addr, unsigned int length, int prot, int flags, int fd, unsigned long offset) {{
return (void*)syscall((long)addr, length, prot, flags, fd, offset, 222); }}
static int munmap(void *addr, unsigned int length) {{ return syscall((long)addr, length, 0, 0, 0, 0, 215); }}'''

class MockDSPProgram:
  def __init__(self, obj:TinyELF): self.name, self.lib, self.signature = obj.name, obj.lib, obj.signature
  def __call__(self, kernargs:int):
    # the args are laid out by TinyELF.iter_sig, a buffer is its address
    sig = list(zip(self.signature, TinyELF.iter_sig(self.signature)))
    vals = [struct.unpack_from(unwrap(dt.fmt), to_mv(kernargs + o, dt.itemsize))[0] for _,(o,dt) in sig]
    bufs = [(v, prod(shape)*dt.itemsize) for v,((_,a,dt,shape),_) in zip(vals, sig) if a is not AddrSpace.ALU]
    with tempfile.NamedTemporaryFile(suffix=".out") as dsp_lib:
      dsp_lib.write(self.lib)
      dsp_lib.flush()
      os.chmod(dsp_lib.name, 0o0777)
      proc = subprocess.run(["qemu-hexagon-static", *(['-strace'] if DEBUG >= 5 else []), dsp_lib.name],
        input=b''.join(struct.pack(unwrap(dt.fmt), v) if a is AddrSpace.ALU else bytes(to_mv(v, prod(shape)*dt.itemsize))
                       for v,((_,a,dt,shape),_) in zip(vals, sig)), stdout=subprocess.PIPE)
    if proc.returncode: raise RuntimeError(f"{self.name} failed in qemu with {proc.returncode}")
    offset = 4
    for addr, size in bufs:
      to_mv(addr, size)[:] = proc.stdout[offset:offset+size]
      offset += size
    assert offset == len(proc.stdout)

class MockDSP:
  # the mock's batches run in python (the host program calls mockdsp_submit, see DSPQueue), each kernel in qemu
  lib = types.SimpleNamespace()
  def __init__(self, dev:DSPDevice):
    self.dev, self.programs = dev, list[MockDSPProgram]()
    MockDSP.lib.mockdsp_submit = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_uint64)(self.submit)
    MockDSP.lib.mockdsp_submit.__name__ = "mockdsp_submit"
    DLL._loaded_["mockdsp"] = cast(ctypes.CDLL, MockDSP.lib)

  def load(self, obj:TinyELF) -> int:
    self.programs.append(MockDSPProgram(obj))
    return len(self.programs) - 1

  def submit(self, addr:int, n:int):
    # (an exception can't leave the callback, the device's error state reports it)
    try:
      for op, a, b in itertools.batched(to_mv(addr, n).cast('Q'), 3):
        if op == DSP_EXEC: self.programs[to_mv(a, 8).cast('Q')[0]](b)
        elif op in (DSP_WAIT, DSP_WAIT_EQ) and not (to_mv(a, 8).cast('Q')[0] >= b if op == DSP_WAIT else to_mv(a, 8).cast('Q')[0] == b):
          raise RuntimeError(f"a wait for {b} isn't met")
        elif op == DSP_STORE: to_mv(a, 8).cast('Q')[0] = b
        elif op == DSP_TIMESTAMP: to_mv(a, 8).cast('Q')[0] = int(time.perf_counter_ns() * self.dev.timestamp_divider / 1000)
    except Exception:
      traceback.print_exc()
      self.dev.error_state.host.view(fmt='q')[0] = 1
