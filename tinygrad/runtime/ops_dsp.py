from __future__ import annotations
import ctypes, functools, os, mmap, tempfile, pathlib, array, threading, contextlib, subprocess, struct, math, itertools, hashlib, operator
from tinygrad.device import BufferStorage, MMIOInterface, BufferSpec, Compiled, Allocator, Program
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.dtype import dtypes, AddrSpace, DType
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.helpers import getenv, round_up, round_down, mv_address, to_mv, cpu_objdump, system, DEBUG, suppress_finalizing, Target, unwrap, dedup
from tinygrad.helpers import all_same, prod
from tinygrad.renderer.cstyle import ClangRenderer, base_rewrite, wmma_args, uops_to_dtypes
from tinygrad.renderer import tc
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.codegen.late.vectorize import lane_of
from tinygrad.runtime.autogen import libc, qcom_dsp
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
    return f"(-__builtin_convertvector({srcs[0]}, {ctx._render_dtype(mask_dtype[x.dtype.itemsize], x.max_numel())})?{srcs[1]}:{srcs[2]})"
  ret = ctx.code_for_op[x.op](*srcs, x.dtype)
  # vector compares are masks of the operand size
  return f"__builtin_convertvector({ret}, {ctx.render_type(x)})" if x.op in GroupOp.Comparison else ret

# a vector of n lanes renders as the next power of two lanes, the lanes past n are undefined
def vpad(n:int) -> int: return 1 << (n-1).bit_length() if n > 1 else n
def shuffle(a:str, b:str, idxs, src_width:int|None=None) -> str:
  idxs = list(idxs)
  idxs += [-1]*(vpad(len(idxs))-len(idxs))
  # a shuffle of a vector to the same width with undefined lanes is a permutation of it: clang scalarizes it unless it's a permutation
  if src_width == len(idxs) and -1 in idxs and len(used:=[i for i in idxs if i >= 0]) == len(set(used)) and all(i < src_width for i in used):
    rest = iter(i for i in range(src_width) if i not in set(used))
    idxs = [i if i >= 0 else next(rest) for i in idxs]
  return f"__builtin_shufflevector({a}, {b}, {','.join(map(str, idxs))})"
def concat(parts:list[str], m:int) -> str:
  while len(parts) > 1: parts, m = [shuffle(a, b, list(range(m)) + [vpad(m)+i for i in range(m)]) for a,b in zip(parts[::2], parts[1::2])], 2*m
  return parts[0]

def can_load_wide(u:UOp) -> bool:
  # (a whole vector register from where it loads is in the buffer or the bytes past it that can be read)
  return u.op is Ops.LOAD and len(u.src) == 1 and (b:=u.src[0]).op is Ops.SHRINK and b.src[0].addrspace is AddrSpace.GLOBAL and \
    1 < b.max_numel() and b.max_numel()*u.dtype.itemsize < 128 and b.src[1].get_valid() is UOp.const(True) and \
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
        rot = f"V6(valignb)({b}, {a}, {off})" if 0 <= off <= 127 and r[-1][1] < 256 else \
              f"V6(valignb)({a}, {a}, {off % 128})" if r[-1][1] < 128 else f"V6(valignb)({b}, {b}, {(off-128) % 128})"
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
        pv = f"(HVX_Vector)({shuffle(ctx[v], ctx[v], [i if u is v else -1 for u,i in map(unwrap, lanes)], 128)})"
        mask = "((hvx_ub){" + ",".join("1" if j < n and unwrap(lanes[j])[0] is v else "0" for j in range(128)) + "})"
        ret = pv if not ret else f"V6(vmux)(V6(vandvrt)((HVX_Vector){mask}, 0x01010101), {pv}, {ret})"
      return f"(({ctx._render_dtype(dtypes.uint8, 128)})({ret}))"
    if len(vecs) <= 2 and all(v.dtype == vecs[0].dtype and rwidth(ctx, v) == rwidth(ctx, vecs[0]) for v in vecs):
      return shuffle(ctx[vecs[0]], ctx[vecs[-1]], [vecs.index(v)*rwidth(ctx, v) + i for v,i in map(unwrap, lanes)],
                     rwidth(ctx, vecs[0]) if len(vecs) == 1 else None)
  # more than a vector is concatenated vectors, clang crashes on some shuffles of them
  if n*isz > 128 and n % (c:=128//isz) == 0:
    return concat([stack_lanes(ctx, srcs[i:i+c]) for i in range(0, n, c)], c)
  # else the lanes of each vector and scalar are shuffled in one at a time, at the width of the widest (at least the HVX vector width,
  # clang crashes on shuffles of smaller vectors). a vector literal can crash clang
  # (bools are shuffled as bytes, clang scalarizes shuffles of bool vectors)
  w, is_bool = vpad(max([n, 128 // isz] + [rwidth(ctx, unwrap(l)[0]) for l in lanes if l is not None])), srcs[0].dtype == dtypes.bool
  parts_: dict[UOp, list[tuple[int, int]]] = {}
  for j,(s,l) in enumerate(zip(srcs, lanes)): parts_.setdefault(l[0] if l is not None else s, []).append((j, l[1] if l is not None else 0))
  def widen(p:UOp) -> str:
    if not is_vec(p): return f"(({ctx._render_dtype(dtypes.uint8, w)})({ctx[p]}))" if is_bool else splat(ctx, p, w)
    v = f"__builtin_convertvector({ctx[p]}, {ctx._render_dtype(dtypes.uint8, rwidth(ctx, p))})" if is_bool else ctx[p]
    return v if (m:=rwidth(ctx, p)) == w else shuffle(v, v, list(range(m)) + [-1]*(w-m))
  acc, done = "", set()
  for p,pl in sorted(parts_.items(), key=lambda kv: not is_vec(kv[0])):
    src = dict(pl)
    if not acc: acc = shuffle(widen(p), widen(p), [src.get(j, -1) for j in range(w)])
    else: acc = shuffle(acc, widen(p), [j if j in done else w+src[j] if j in src else -1 for j in range(w)])
    done |= set(src)
  if w != n: acc = shuffle(acc, acc, range(n))
  return f"__builtin_convertvector(({acc}) != 0, {ctx._render_dtype(dtypes.bool, n)})" if is_bool else acc

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
__attribute__((unused)) static inline __attribute__((always_inline)) hvx_w mulhi_w(hvx_w a, int c) {
  HVX_VectorPair p = V6(vmpyewuh_64)((HVX_Vector)a, V6(lvsplatw)(c)); p = V6(vmpyowh_64_acc)(p, (HVX_Vector)a, V6(lvsplatw)(c));
  return (hvx_w)V6(hi)(p); }"""

def const_val(u:UOp):
  while u.op is Ops.CAST: u = u.src[0]
  return u.arg if u.op is Ops.CONST else None

def mulhi(x:UOp, a:UOp, c:UOp, s:UOp) -> UOp|None:
  # (int)(((long long)a * c) >> 32) on a vector of int is mulhi_w 32 lanes at a time
  if not is_vec(x) or x.dtype != dtypes.int32 or a.src[0].dtype != dtypes.int32 or const_val(s) != 32: return None
  if not isinstance(cv:=const_val(c), int) or not -2**31 <= cv < 2**31: return None
  n = x.max_numel()
  parts = [f"mulhi_w({shuffle('{0}', '{0}', [i+j if i+j < n else -1 for j in range(32)])}, {cv})" for i in range(0, n, 32)]
  if len(parts) & (len(parts)-1): parts += [parts[-1]] * (vpad(len(parts)) - len(parts))
  ret = concat(parts, 32)
  return UOp.native(ret if vpad(n) == 32*len(parts) else shuffle(ret, ret, range(n)), shape=x.shape, dtype=x.dtype, fmt=True).call(a.src[0])

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
def render_vrmpy(name:str, dims:tuple[int, int, int], a:str, b:str, c:str) -> str:
  # A is K rows of M bytes, each vrmpy is 32 lanes (m) of 4 bytes (k): the rows are interleaved into vectors, M/32 for each 4 rows
  n, m, k = dims
  if m == 128: inter = """HVX_VectorPair s01 = V6(vshuffvdd)(r[1], r[0], -1), s23 = V6(vshuffvdd)(r[3], r[2], -1);
  HVX_VectorPair q0 = V6(vshuffvdd)(V6(lo)(s23), V6(lo)(s01), -2), q1 = V6(vshuffvdd)(V6(hi)(s23), V6(hi)(s01), -2);
  HVX_Vector v[4] = {V6(lo)(q0), V6(hi)(q0), V6(lo)(q1), V6(hi)(q1)};"""
  else: inter = """HVX_VectorPair s0 = V6(vshuffvdd)(r[1], r[0], -1), q0 = V6(vshuffvdd)(V6(hi)(s0), V6(lo)(s0), -1);
  HVX_VectorPair s1 = V6(vshuffvdd)(r[3], r[2], -1), q1 = V6(vshuffvdd)(V6(hi)(s1), V6(lo)(s1), -1);
  HVX_Vector v[4] = {V6(lo)(q0), V6(hi)(q0), V6(lo)(q1), V6(hi)(q1)};"""
  # acc[n][j] for the M/32 vectors j, w[n][g] for the K/4 groups g of 4 rows
  mv, kg = m//32, k//4
  return f"""static inline __attribute__((always_inline)) {c} __{name}({a} a, {b} b, {c} c) {{
  HVX_Vector r[4], acc[{n*mv}]; int w[{n*kg}];
  __builtin_memcpy(r, &a, sizeof(r)); __builtin_memcpy(acc, &c, sizeof(acc)); __builtin_memcpy(w, &b, sizeof(w));
  {inter}
  for (int n = 0; n < {n}; n++) for (int g = 0; g < {kg}; g++) for (int j = 0; j < {mv}; j++)
    acc[n*{mv}+j] = V6(vrmpyub_acc)(acc[n*{mv}+j], v[g*{mv}+j], w[n*{kg}+g]);
  __builtin_memcpy(&c, acc, sizeof(acc)); return c;
}}"""

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
  # a store of up to a vector register that isn't a whole aligned one is masked stores of the vector register
  if 32 < (nb:=n*var.dtype.itemsize) <= 128 and (nb < 128 or known_factor(bidx.src[1]) % n) and bidx.src[0].addrspace is AddrSpace.GLOBAL:
    v = f"((HVX_Vector)({ctx[var]}))" if vpad(n)*var.dtype.itemsize == 128 else \
      f"((HVX_Vector)({shuffle(ctx[var], ctx[var], list(range(n)) + [-1]*(128//var.dtype.itemsize-n))}))"
    return f"store_n((unsigned char*)({ctx[bidx]}), {v}, {nb});"
  # a vector store that isn't a power of two lanes (rendered padded) stores pieces that are
  if vpad(n) == n: return None
  pieces, o = [], 0
  while o < n:
    l = 1 << ((n-o).bit_length()-1)
    val = f"{ctx[var]}[{o}]" if l == 1 else shuffle(ctx[var], ctx[var], range(o, o+l))
    pieces.append(f"*(({ctx._render_dtype(var.dtype, l) + ('_u' if l > 1 else '')}*)({ctx[bidx]}+{o})) = {val};")
    o += l
  return " ".join(pieces)

dsp_vec_rewrite = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), render_range),
  (UPat(Ops.STORE, src=(UPat.var("bidx"), UPat.var("var"))), render_store_pieces),
  (UPat(Ops.LOAD, src=(UPat.var("bidx"),), name="x"), render_wide_load),
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
    # the loops around the loads, outermost first
    chain = [max(base.ranges, key=depth)]
    while (p:=parent[chain[0]]) is not None: chain.insert(0, p)
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

dsp_late = PatternMatcher([
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
    # the loop that's split between the cores: the only outermost loop, if all the global stores are in it
    depth, top = 0, []
    for u in uops:
      if u.op is Ops.RANGE:
        if depth == 0: top.append(u)
        depth += 1
      if u.op is Ops.END: depth -= 1
    stores = [u for u in uops if u.op is Ops.STORE and u.src[0].addrspace is AddrSpace.GLOBAL]
    self.parallel = top[0] if len(top) == 1 and stores and all(top[0] in st.ranges for st in stores) else None
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
    return super()._render_dtype(dtype, vpad(sz), addrspace, mutable, override_ptr, shape)

  def render_wmmas(self, uops) -> list[str]:
    wmmas = [render_vrmpy(name, dims, *[self._render_dtype(dt, sz) for dt,sz in zip((din, din, dout), szs)])
             for name,dims,din,dout,szs in wmma_args(uops)]
    return wmmas

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
    dts = dedup([u.dtype for u in uops if u.dtype in self.type_map and u.dtype is not dtypes.void] +
                [dtypes.bool, *mask_dtype.values()])
    # and the HVX vector width of every dtype (shuffles of smaller vectors widen to it)
    # and all the smaller powers of two (the pieces of stores)
    pow2 = [1 << i for i in range(1, max(counts).bit_length()+1)]
    return dedup([self.render_vector_prefix(dt, cnt) for dt in dts for cnt in dedup(counts + pow2 + [128 // dt.itemsize]) if cnt > 1])

  def render_vector_prefix(self, dt, count:int) -> str:
    count = vpad(count)
    return super().render_vector_prefix(dt, count) + f"\ntypedef {self.render_dtype(dt)} {self._render_dtype(dt, count)}_u " + \
      f"__attribute__((aligned(1),ext_vector_type({count})));"

  def _render_defines(self, uops) -> list[str]:
    return ['''/* DSP boilerplate */ struct dcvs_v2_req { int type; int _pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency;
      _Bool set_dcvs_params; short _pad2; char target_corner; char min_corner; char max_corner; int _pad3[3];};''','int HAP_power_set(void*, void*);',
      'typedef union { struct { void *pv; unsigned int len; } buf; struct { int fd; unsigned int offset; } dma; } remote_arg;',
      'unsigned long long HAP_perf_get_time_us(void);', HVX_PRELUDE] + self._vector_typedefs(uops) + self.render_wmmas(uops)

  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    def ty(b:UOp) -> str: return self._render_dtype(b.dtype) if b.addrspace == AddrSpace.ALU else "void*"
    gbufs = [i for i,(_,(b,_)) in enumerate(bufs) if b.addrspace == AddrSpace.GLOBAL]
    msrc = ['typedef struct { ' + ' '.join(f'{ty(b)} a{i};' for i,(_,(b,_)) in enumerate(bufs)) + ' int core; } kargs_t;',
            # the workers of the pool (see DSP_UTIL_SRC) run cores of the kernel
            'typedef struct { void (*run)(void *pool, void (*fn)(void*), void *args, int argsize, int ncaller); } pool_t;',
            '__attribute__((unused)) static void kcore(void *p) { kargs_t *a = p; ' +
            f'{function_name}({", ".join([f"a->a{i}" for i in range(len(bufs))] + ["a->core"])}); }}',
            'int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {',
            'struct dcvs_v2_req req = {.type=7, .dcvs_enable=0, .set_latency=1, .latency=100, .set_dcvs_params=1, .target_corner = 6 /* TURBO */};',
            'HAP_power_set((void*)handle, (void*)&req);', 'if ((sc>>24) != 2) return 0;',
            f'kargs_t args[{NCORES}]; pool_t *pool = (pool_t*)*(unsigned int*)((char*)pra[0].buf.pv+{len(bufs)*8}); (void)pool;']
    # the buffers are mapped on the DSP for their lifetime (see DSPAllocator), the args are their DSP addresses
    msrc += [f'args[0].a{i} = ({ty(b)})*({"unsigned int" if i in gbufs else ty(b)}*)((char*)pra[0].buf.pv+{i*8});' for i,(_,(b,_)) in enumerate(bufs)]
    msrc += [f"for (int i = 1; i < {NCORES}; i++) {{ args[i] = args[0]; args[i].core = i; }} args[0].core = 0;"]
    msrc += ["unsigned long long start = HAP_perf_get_time_us();"]
    # the cores from ncaller on run on the workers, the ones before on the calling thread
    def run(ncaller:int) -> list[str]:
      return [f'{function_name}({", ".join([f"args[0].a{i}" for i in range(len(bufs))] + [str(c)])});' for c in range(ncaller)] if ncaller == NCORES \
        else [f"pool->run(pool, kcore, args, sizeof(kargs_t), {ncaller});"]
    # the calling thread's stack takes small frames of vectors only: a big frame runs on the workers (see DSPCompiler)
    msrc += ["#ifdef DSP_BIG_FRAME"] + run(0) + ["#else"] + run(NCORES if self.inline else 1) + ["#endif"]
    msrc += [             "*(unsigned long long *)(pra[1].buf.pv) = HAP_perf_get_time_us() - start;"]
    msrc += ["return 0; }"]
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

class DSPProgram(Program['DSPDevice']):
  def __init__(self, dev:DSPDevice, obj:TinyELF, links=None): self.dev, self.lib, self.signature = dev, obj.lib, obj.signature

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait=False, **kw):
    bufs = [b for b,(_,a,_,_) in zip(args, self.signature) if a is not AddrSpace.ALU]
    if len(bufs) >= 16: raise RuntimeError(f"Too many buffers to execute: {len(bufs)}")

    pra, fds, attrs, _ = rpc_prep_args(ins=[var_vals_mv:=memoryview(bytearray(len(args)*8+8))], outs=[timer:=memoryview(bytearray(8)).cast('Q')])
    for i,(b,(_,a,dt,_)) in enumerate(zip(args, self.signature)):
      struct.pack_into(unwrap(dt.fmt) if a is AddrSpace.ALU else 'I', var_vals_mv, i*8, b if a is AddrSpace.ALU else b.dsp_addr)
    # (and the pool the kernel runs its cores on)
    struct.pack_into('I', var_vals_mv, len(args)*8, self.dev.pool)
    self.dev.exec_lib(self.lib, rpc_sc(method=2, ins=1, outs=1, fds=0), pra, fds, attrs)
    return timer[0] / 1e6

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info, offset:int=0, dsp_addr:int=0, arena:DSPArena|None=None):
    self.va_addr, self.size, self.share_info, self.offset, self.dsp_addr, self.arena = va_addr, size, share_info, offset, dsp_addr, arena

class DSPArena:
  # the DSP maps few buffers (~250, the libs count too): the buffers are parts of big ones, each mapped on the DSP once
  # they're uncached on the host (it only copies in and out of them), its caches don't need syncs
  def __init__(self, dev:DSPDevice, size:int):
    self.dev, self.size = dev, size
    self.handle = qcom_dsp.ION_IOC_ALLOC(dev.ion_fd, len=size, align=0x1000, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=0).handle
    self.share_info = qcom_dsp.ION_IOC_SHARE(dev.ion_fd, handle=self.handle)
    self.va = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, self.share_info.fd, 0)
    self.dsp_addr, self.parts = dev.dsp_util(DSP_UTIL_MAP, size=size, fd=self.share_info.fd), TLSFAllocator(size)

DSP_ARENA = 32 << 20
class DSPAllocator(Allocator['DSPDevice']):
  def __init__(self, dev:DSPDevice):
    super().__init__(dev)
    self.arenas: list[DSPArena] = []

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    # the kernels read up to DSP_OVERREAD bytes before and after a buffer (vector loads with invalid lanes, clang widening small vector loads)
    asize = round_up(size + 2*DSP_OVERREAD, 128)
    # a buffer only the host accesses (like the timeline) is host memory, and the buffers before the DSP's session starts (its shell)
    if getenv("MOCKDSP") or options.host or not hasattr(self.dev, "rpc_fd"):
      share_info = None if getenv("MOCKDSP") or options.host else \
        qcom_dsp.ION_IOC_SHARE(self.dev.ion_fd, handle=qcom_dsp.ION_IOC_ALLOC(self.dev.ion_fd, len=round_up(asize, 0x1000), align=0x1000,
                               heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=qcom_dsp.ION_FLAG_CACHED).handle)
      va = libc.mmap(0, round_up(asize, 0x1000), mmap.PROT_READ|mmap.PROT_WRITE,
                     mmap.MAP_SHARED|(mmap.MAP_ANONYMOUS if share_info is None else 0), share_info.fd if share_info is not None else -1, 0)
      return BufferStorage(opaque:=DSPBuffer(va+DSP_OVERREAD, size, share_info, offset=DSP_OVERREAD), share_info, MMIOInterface(opaque.va_addr, size))
    for arena in self.arenas:
      with contextlib.suppress(MemoryError):
        off = arena.parts.alloc(asize, 128)
        break
    else:
      self.arenas.append(arena:=DSPArena(self.dev, max(DSP_ARENA, round_up(asize, 0x1000))))
      off = arena.parts.alloc(asize, 128)
    # (the host accesses it through copyin and copyout only, they sync the DSP's caches)
    return BufferStorage(DSPBuffer(arena.va+off+DSP_OVERREAD, size, None, off+DSP_OVERREAD, arena.dsp_addr+off+DSP_OVERREAD, arena))

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if (arena:=storage.buf.arena) is not None: return arena.parts.free(storage.buf.offset-DSP_OVERREAD)
    libc.munmap(storage.buf.va_addr-DSP_OVERREAD, round_up(storage.buf.size + 2*DSP_OVERREAD, 0x1000))
    if storage.buf.share_info is not None:
      os.close(storage.buf.share_info.fd)
      qcom_dsp.ION_IOC_FREE(self.dev.ion_fd, handle=storage.buf.share_info.handle)

  # the DSP's caches aren't coherent with the host: before the host touches a buffer the DSP writes back and drops its lines of it
  def _sync(self, buf:DSPBuffer, size:int):
    if buf.dsp_addr: self.dev.dsp_util(DSP_UTIL_FLUSH_INVALIDATE, start:=round_down(buf.dsp_addr, 64), round_up(buf.dsp_addr+size, 64)-start)
  def _copyin(self, dest:DSPBuffer, src:memoryview):
    self._sync(dest, src.nbytes)
    ctypes.memmove(dest.va_addr, mv_address(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:DSPBuffer):
    self._sync(src, dest.nbytes)
    ctypes.memmove(mv_address(dest), src.va_addr, dest.nbytes)
  def _offset(self, buf, size:int, offset:int):
    return DSPBuffer(buf.va_addr+offset, size, buf.share_info, buf.offset+offset, buf.dsp_addr+offset if buf.dsp_addr else 0, buf.arena)

# the biggest stack frame the calling (fastrpc) thread runs: a bigger one of vectors crashes it
DSP_CALLER_FRAME = 4096
class DSPCompiler(Compiler):
  def __init__(self, mock:bool=False):
    self.mock, compiler_args = mock, "--target=hexagon -mcpu=hexagonv65 -fuse-ld=lld -nostdlib -mhvx=v65 -mhvx-length=128b -O2"
    if mock: self.args = f"-static {compiler_args}"
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
      def cc(flags:str): system(f"{getenv('CC','clang')} {self.args} {flags} -Wall -Werror -fno-stack-protector -x c -fPIC " +
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
  def _render_defines(self, uops) -> list[str]: return [HVX_PRELUDE] + self._vector_typedefs(uops) + self.render_wmmas(uops)
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

# the libs open at once: the DSP maps ~250 things (a lib is one)
DSP_OPEN_LIBS = 128
# a lib that maps buffers on the DSP and writes back and drops the DSP's cache lines of them
# and that runs the pool of workers (threads with their own stacks) the kernels run their cores on
DSP_UTIL_MAP, DSP_UTIL_FLUSH_INVALIDATE, DSP_UTIL_POOL = 0, 1, 2
DSP_UTIL_SRC = DSP_THREADS_PRELUDE + f"""
typedef union {{ struct {{ void *pv; unsigned int len; }} buf; struct {{ int fd; unsigned int offset; }} dma; }} remote_arg;
void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);
int qurt_mem_cache_clean(unsigned int addr, unsigned int size, int opcode, int type);
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
int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {{
  if ((sc>>24) != 2) return 0;
  unsigned int *in = pra[0].buf.pv, op = in[0], addr = in[1], size = in[2];
  if (op == 0) *(unsigned int*)pra[1].buf.pv = (unsigned int)HAP_mmap(0, size, 3, 0, pra[2].dma.fd, 0);
  if (op == 1) qurt_mem_cache_clean(addr, size, 2 /* QURT_MEM_CACHE_FLUSH_INVALIDATE */, 1 /* QURT_MEM_DCACHE */);
  if (op == 2) *(unsigned int*)pra[1].buf.pv = (unsigned int)make_pool();
  return 0;
}}"""
class DSPDevice(Compiled):
  renderers = [MockDSPRenderer] if getenv("MOCKDSP") else [DSPRenderer]
  def __init__(self, device:str=""):
    if getenv("MOCKDSP"): super().__init__(device, DSPAllocator(self), MockDSPProgram)
    else:
      self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
      super().__init__(device, DSPAllocator(self), DSPProgram)
      fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
      self.shell_buf = self.allocator.alloc(round_up(fastrpc_shell.nbytes, 0x1000), BufferSpec(nolru=True)).buf
      ctypes.memmove(self.shell_buf.va_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

      self.init_dsp()
      RPCListener(self).start()
      self.util_lib = DSPCompiler().compile_cached(DSP_UTIL_SRC)
      self.pool = self.dsp_util(DSP_UTIL_POOL)

  def dsp_util(self, op:int, addr:int=0, size:int=0, fd:int|None=None) -> int:
    pra, fds, attrs, _ = rpc_prep_args(ins=[memoryview(array.array('I', [op, addr, size]))], outs=[out:=memoryview(bytearray(4)).cast('I')],
                                       in_fds=[fd] if fd is not None else [])
    self.exec_lib(self.util_lib, rpc_sc(method=2, ins=1, outs=1, fds=len(fds)-2), pra, fds, attrs)
    if op == DSP_UTIL_MAP and out[0] in (0, 0xffffffff): raise MemoryError(f"DSP can't map a buffer of {size} bytes")
    return out[0]

  def open_lib(self, lib):
    self.binded_lib, self.binded_lib_off = lib, 0
    # (a name of its own: the DSP's dlopen of an open name returns the open lib)
    fp = f"file:///tinylib{hashlib.sha256(lib).hexdigest()[:16]}?entry&_modver=1.0&_dom=cdsp\0"
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [len(fp), 0xff])), memoryview(bytearray(fp.encode()))],
                                 outs=[o1:=memoryview(bytearray(0x8)), o2:=memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=0, ins=2, outs=2), pra=pra)
    if o1.cast('i')[1] < 0: raise RuntimeError(f"Cannot open lib: {o2.tobytes().decode()}")
    return o1.cast('I')[0]

  def close_lib(self, handle):
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [handle, 0xff]))], outs=[memoryview(bytearray(0x8)), memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=1, ins=1, outs=2), pra=pra)

  def exec_lib(self, lib, sc, args, fds, attrs):
    # a lib stays open: opening and closing one for every call leaks on the DSP, it dies after ~1500 calls
    def _exec_lib():
      if (handle:=self.lib_handles.get(lib)) is None:
        # (up to DSP_OPEN_LIBS of them, the oldest kernel is closed. the util lib stays, its workers run its code)
        if len(self.lib_handles) >= DSP_OPEN_LIBS:
          self.close_lib(self.lib_handles.pop(next(l for l in self.lib_handles if l is not getattr(self, "util_lib", None))))
        handle = self.lib_handles[lib] = self.open_lib(lib)
      qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.rpc_fd, fds=fds, attrs=attrs, inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=handle, sc=sc, pra=args))
    try: _exec_lib()
    except (OSError, PermissionError):
      # DSP might ask for a connection reset or just fail with operation not permitted, try to reset connection.
      self.init_dsp()
      try: _exec_lib()
      except (OSError, PermissionError) as e: raise RuntimeError(e)

  def init_dsp(self):
    self.lib_handles: dict[bytes, int] = {}
    if hasattr(self, 'rpc_fd'):
      with contextlib.suppress(OSError):
        qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=4, sc=rpc_sc(method=2, ins=0, outs=0)) # pylint: disable=access-member-before-definition
      os.close(self.rpc_fd) # pylint: disable=access-member-before-definition

    self.rpc_fd: int = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)
    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)
    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=self.shell_buf.va_addr, filelen=self.shell_buf.size, filefd=self.shell_buf.share_info.fd)
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

class MockDSPProgram(Program[DSPDevice]):
  def __init__(self, dev:DSPDevice, obj:TinyELF, links=None): self.lib, self.signature = obj.lib, obj.signature
  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait=False, **kw):
    with tempfile.NamedTemporaryFile(suffix=".out") as dsp_lib:
      dsp_lib.write(self.lib)
      dsp_lib.flush()
      os.chmod(dsp_lib.name, 0o0777)
      proc = subprocess.run(["qemu-hexagon-static", *(['-strace'] if DEBUG >= 5 else []), dsp_lib.name],
        input=b''.join(struct.pack(unwrap(dt.fmt), x) if a is AddrSpace.ALU else bytes(to_mv(x.va_addr, x.size))
                       for x,(_,a,dt,_) in zip(args, self.signature)),
        stdout=subprocess.PIPE, check=True)
    offset = 4
    for x in (x for x,(_,a,_,_) in zip(args, self.signature) if a is not AddrSpace.ALU):
      to_mv(x.va_addr, x.size)[:] = proc.stdout[offset:offset+x.size]
      offset += x.size
    assert offset == len(proc.stdout)
    return struct.unpack("I", proc.stdout[0:4])[0] / 1e9  # pretend it's 1 Ghz, but this is an inscount, not a time
