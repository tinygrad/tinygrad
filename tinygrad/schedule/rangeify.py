from itertools import product
from typing import cast
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, PtrDType, ImageDType, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, _substitute, ssimplify, KernelInfo
from tinygrad.uop.ops import track_rewrites, graph_rewrite, identity_element, sint, AxisType
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.helpers import argsort, prod, all_same, pluralize, getenv, flatten, dedup, unwrap, all_int, DEBUG, SPLIT_REDUCEOP, Metadata, WINO
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_unparented
from tinygrad.codegen.opt import Opt
from tinygrad.schedule.indexing import run_rangeify, BufferizeOpts, ALWAYS_CONTIGUOUS, IndexingContext, apply_movement_op

# creation can recurse a lot
import sys
sys.setrecursionlimit(10000)

# *****************
# 0. do some cleanup rewrites, mostly copied from the old stuff

def find_permutes(a:UOp, b:UOp, assign:UOp):
  if not (permutes:=[s for s in b.toposort(gate=lambda s:s.op not in ALWAYS_CONTIGUOUS)
                     if s.op in GroupOp.Movement and s.op not in {Ops.RESHAPE, Ops.EXPAND, Ops.PAD, Ops.SHRINK}]): return
  target = a.base
  for p in permutes:
    if any(s is target for s in p.toposort(gate=lambda s:s.op not in ALWAYS_CONTIGUOUS-{Ops.BUFFER})): return assign.replace(src=(a, b.contiguous()))

def split_reduceop(reduce:UOp, x:UOp):
  if prod(reduce.shape) == 0: return None
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.
  real_strides = unwrap(x.st).real_strides(ignore_valid=True)
  if not (split_candidates:=[(i,d) for i in reduce.arg[1] for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and real_strides[i]!=0]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted.r(*reduce.arg).contiguous().r(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape).replace(tag=reduce.tag)

earliest_rewrites = PatternMatcher([
  # just removing it works...
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.FUSE), name="x"), lambda x: x.src[0]),

  # merge adjacent RESHAPES, safe because they are not tagged
  (UPat(Ops.RESHAPE, name="x2").f(Ops.RESHAPE, name="x"), lambda x,x2: x.replace(src=(x2.src[0],)) if x.tag is None and x2.tag is None else None),

  # remove CONTIGUOUS if the BUFFER is already contiguous
  (UPat(Ops.BUFFER).f(Ops.RESHAPE, name="r").f(Ops.CONTIGUOUS, name="c"), lambda r,c: r.replace(tag=c.tag)),

  # split_reduceop
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), split_reduceop),

  # preserve tags?
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),

  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x.st is not None and x.size == 0 else None),

  # remove contiguous on movement ops before a copy on disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.CONTIGUOUS).f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: copy.replace(src=(x,)+copy.src[1:]) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # push copy past movement ops to disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: x.replace(src=(copy.replace(src=(x.src[0],)+copy.src[1:], tag=None),)+x.src[1:], tag=copy.tag) \
      if isinstance(x.device, str) and x.device.startswith("DISK") else None),

  # ** copy rules **

  # early fixup const copy
  (UPat(Ops.COPY, src=(UPat.var("s"), UPat()), name="c"), lambda c,s: c.const_like(ss.arg) if (ss:=s.base).op is Ops.CONST else None),

  # COPY and source size need to match
  # TODO: expand after copy creates issues with tagging
  (UPat(Ops.COPY, src=(UPat(GroupOp.Movement, name="r"), UPat(name="d")), name="c"),
   lambda c,r,d: c.replace(src=(r.contiguous(), d)) if r.size != r.base.size else None),

  # copy only to different device
  (UPat(Ops.COPY, src=(UPat.var("x"), UPat()), name="copy"), lambda x,copy: x.f(Ops.NOOP, tag=copy.tag) if x.device == copy.device else None),

  # ** assign rules **

  # assign only to buffer, otherwise make it a CONTIGUOUS
  (UPat(Ops.ASSIGN, src=(UPat(GroupOp.All-{Ops.BUFFER}, name="target"), UPat(name="x")), name="assign"),
   lambda x,target,assign: x.f(Ops.CONTIGUOUS, tag=assign.tag) if ((t:=target.base).op is not Ops.BUFFER and \
       not (t.op is Ops.MSTACK and all(s.op is Ops.BUFFER for s in t.src))) else None),

   # realize before assign if input permutes the target buffer
   (UPat(Ops.ASSIGN, src=(UPat.var("a"), UPat.var("b")), name="assign"), find_permutes),

  # contiguous buffer is buffer, this is for *correctness* of assign, not just speed
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.BUFFER),)), lambda root: root.src[0].forced_reshape(root.shape).rtag(root.tag)),
])

# *****************
# 3a. rangeify (movement)

# movement op on INDEX as a PatternMatcher
pm_mops = PatternMatcher([
  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"),
   lambda r,idx: r.src[0].index(*apply_movement_op(r, idx.src[1:]), dtype=idx.dtype, arg=idx.arg)),
])


# *****************
# 3.2 Winograd

winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]]

# def synth_mat(mat: list[list[int|float]], o: UOp, r: UOp, dtype=dtypes.float) -> UOp:
#   rows = len(mat)
#   cols = len(mat[0]) if rows else 0
#   assert rows > 0 and cols > 0, "synth_mat: expected non-empty matrix"

#   # make a (rows x cols) constant table (values don't matter for plumbing tests)
#   ones = UOp.const(dtype, 1.0).reshape((1, 1)).expand((rows, cols)).contiguous()

#   # CRITICAL: bufferize on the SAME axes you'll use to index
#   # We store with (r, o) so buf.src == (value, r, o)
#   buf = ones.bufferize(r, o, arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))

#   # Index with the SAME (r, o) pair → idx.src == (buf, r, o)
#   return 3
#   return buf.index(r, o, dtype=dtype)

def synth_mat(mat: list[list[int|float]], o: UOp, r: UOp, dtype=dtypes.float) -> UOp:
  z    = UOp.const(dtype, 0)
  cells = [((r*(len(mat[0]))+o)).eq(i).where(UOp.const(dtype, v), z) for i, v in enumerate(flatten(mat)) if v]
  while len(cells) > 1:
    cells = [cells[i] + cells[i+1] if i+1 < len(cells) else cells[i] for i in range(0, len(cells), 2)]
  return (o+r).cast(dtype)
  return cells[0] if cells else z

def _ci(i: int) -> UOp:  return UOp.const(dtypes.index, i)
def _cf(x: float) -> UOp: return UOp.const(dtypes.float, float(x))

def one_hot(axis: UOp, i: int) -> UOp:
  # WHERE(CMEQ(axis,i), 1.0, 0.0)
  return UOp(Ops.WHERE, dtypes.float,
             src=(UOp(Ops.CMPEQ, dtypes.bool, src=(axis, _ci(i))), _cf(1.0), _cf(0.0)))

def nmode_kron(buf: UOp, axes: list[UOp], B: list[list[float]], outer_axes: list[UOp]) -> UOp:
  """
  Y[axes] = sum_{p0..p_{k-1}} ( Π_m B[axes[m], p_m] ) * D[p0..p_{k-1}]
  """
  N, k = len(B), len(axes)
  # 1) all constant reads of D (N^k)
  if outer_axes != []:
    grid = [ (pt, buf.index(*map(_ci, pt), *outer_axes)) for pt in product(range(N), repeat=k) ]
  else:
    grid = [ (pt, buf.index(*map(_ci, pt))) for pt in product(range(N), repeat=k) ]
  # 2) per-axis masked coeff tables α^{(m)}_p(axis_m) = sum_i 1_{axis_m==i} * B[i,p]
  alpha = [
    [ sum( one_hot(axes[m], i) * _cf(B[i][p]) for i in range(N) )
      for p in range(N) ]
    for m in range(k)
  ]
  # 3) term for each tuple p: D[p] * Π_m α^{(m)}_{p_m}(axis_m)
  terms = [ D * _prod(alpha[m][pt[m]] for m in range(k)) for pt, D in grid ]
  # 4) fold the sum (pairwise to keep trees shallow)
  while len(terms) > 1:
    terms = [terms[i] if i+1>=len(terms) else (terms[i] + terms[i+1]) for i in range(0,len(terms),2)]
  return terms[0] if terms else _cf(0.0)

def _prod(xs):
  it = iter(xs)
  acc = next(it, None)
  if acc is None: return _cf(1.0)
  for x in it: acc = acc * x
  return acc

def winoguard(redu):
 #TODO - Can we make this simpler?
 if redu.arg is not Ops.ADD or redu.src[0].op is not Ops.MUL: return None
 three_axes = {ax for ax in redu.src[1:] if ax.op is Ops.RANGE and int(ax.vmax+1) == 3}
 def collect_pairs(node, axes, out):
  if not axes or node.op is Ops.BUFFERIZE: return
  if node.op is Ops.ADD and len(node.src) == 2:
   for rng, loop in ((node.src[0], node.src[1]), (node.src[1], node.src[0])):
    if rng in axes and loop.op is Ops.RANGE and loop.arg[1] is AxisType.LOOP:
     axes.remove(rng); out.append((loop, rng, node))
  for s in node.src: collect_pairs(s, axes, out)
 oa, ob = [], []
 collect_pairs(redu.src[0].src[0], set(three_axes), oa)
 collect_pairs(redu.src[0].src[1], set(three_axes), ob)
 if len(oa) >= 2 and not ob: return (0, oa)
 if len(ob) >= 2 and not oa: return (1, ob)
 return None

def winowrite(ctx: IndexingContext, redu: UOp):
  #TODO - Add where filters so index does not read garbage - or do we?
  #TODO - Generalize to n-dimensions
  #TODO - Merge synth_mat with reduce and mul to create kron product
  #TODO - Fix test errors
  # 1) Use winoguard to find the activation branch and candidate (loop, reduce) axis pairs.
  #TODO - Breaks with 3D and Cin
  guard = winoguard(redu)
  if guard is None:
    return None
  act_branch, axis_pairs = guard
  act_like, w_like = redu.src[0].src
  if act_branch == 1:
    act_like, w_like = w_like, act_like
  if act_like.op is not Ops.INDEX or w_like.op is not Ops.INDEX: #will extend to not index when I am not fucking sick
    print("Not an index bailing!"); return None
  act_buf = act_like.src[0]
  (_, ky, oy_add), (_, kx, ox_add), *_ = axis_pairs

  # --- 1) Split oy/ox into tiles + in-tile -------------------------------
  zero_map = {ax: ax.const_like(0) for ax in (ky, kx)}
  oy_base = oy_add.substitute(zero_map).simplify()   # the outer oy loop
  ox_base = ox_add.substitute(zero_map).simplify()

  # replace "symbolic" ty/tx with *real ranges* so we can bufferize on them
  # tile extents: ceil_div(oy_extent, 4) and ceil_div(ox_extent, 4)
  oy_extent = int(oy_base.vmax + 1)
  ox_extent = int(ox_base.vmax + 1)
  # split oy/ox
    # --- Split oy/ox, but only use iy/ix at the end ---
  oy_base = oy_add.substitute({ky: ky.const_like(0), kx: kx.const_like(0)}).simplify()
  ox_base = ox_add.substitute({ky: ky.const_like(0), kx: kx.const_like(0)}).simplify()
  iy = (oy_base % 4).simplify()
  ix = (ox_base % 4).simplify()
  tx = ((ox_base + 3)// 4).simplify()
  ty = ((oy_base + 3)// 4).simplify()
  #print(f"oyb: {oy_add}, oxb: {ox_add}")

  # Tile loops (outer)
  # TY = ctx.new_range((int(oy_base.vmax+1)+3)//4, AxisType.LOOP)
  # TX = ctx.new_range((int(ox_base.vmax+1)+3)//4, AxisType.LOOP)

  # # ---- X̂ branch axes (disjoint) ----
  # c6x = ctx.new_range(6, AxisType.LOOP)
  # r6x = ctx.new_range(6, AxisType.LOOP)
  # u6x = ctx.new_range(6, AxisType.REDUCE)
  # v6x = ctx.new_range(6, AxisType.REDUCE)
  # RU = ctx.new_range(6, AxisType.LOOP)
  # UX = ctx.new_range(6, AxisType.LOOP)
  
  # # Build mats for X̂ using ONLY these axes
  # B_ur_x = synth_mat(list(zip(*winograd_Bt)), r6x, u6x)
  # Bt_cv_x = synth_mat(winograd_Bt,           v6x, c6x)      # reduce v6x, index c6x

  # X_vu = act_like.substitute({oy_add: TY*4 + RU, ox_add: TX*4 + UX}).cast(dtypes.float).bufferize(TY, TX, RU, UX, arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))
  # # Xhat_expr = ((X_vu * B_ur_x).reduce(u6x, arg=Ops.ADD, dtype=dtypes.float) * Bt_cv_x)\
  # #             .reduce(v6x, arg=Ops.ADD, dtype=dtypes.float)
  # #Xhat_expr = nmode_kron(X_vu, [c6x, r6x], Bt)
  # # TY1 = ctx.new_range((int(oy_base.vmax+1)+3)//4, AxisType.LOOP)
  # # TX1 = ctx.new_range((int(ox_base.vmax+1)+3)//4, AxisType.LOOP)
  # Xhat_expr = X_vu.index(TY, TX, c6x, r6x) + 1
  # Xhat_expr = nmode_kron(X_vu, [c6x, r6x], Bt, [TY, TX])
  

  # # Bufferize X̂ on (TY, TX, c6x, r6x). No oy/ox/ix/iy here.
  # XHAT = Xhat_expr.bufferize(TY, TX, c6x, r6x,
  #         arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))

  TYx = ctx.new_range((int(oy_base.vmax+1)+3)//4, AxisType.LOOP)
  TXx = ctx.new_range((int(ox_base.vmax+1)+3)//4, AxisType.LOOP)
  RU  = ctx.new_range(6, AxisType.LOOP)
  UX  = ctx.new_range(6, AxisType.LOOP)

  # bufferize the “data” branch in 2×2×6×6 tile space
  X_vu = act_like.substitute({oy_add: TYx*4 + RU, ox_add: TXx*4 + UX}) \
                .cast(dtypes.float) \
                .bufferize(TYx, TXx, RU, UX, arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))
  TYh = ctx.new_range((int(oy_base.vmax+1)+3)//4, AxisType.LOOP)
  TXh = ctx.new_range((int(ox_base.vmax+1)+3)//4, AxisType.LOOP)
  # build the Kron *reading from X_vu* with the SAME tile ranges (TYx,TXx); only (u,v) vary as consts
  def kron_from_Xvu(B, c6x, r6x):
      N = len(B) #TXh, TYh
      reads = [ ((u,v), X_vu.index(TXh, TYh, _ci(u), _ci(v))) for u in range(N) for v in range(N) ]
      alpha = [ sum(one_hot(c6x,i)*_cf(B[i][u]) for i in range(N)) for u in range(N) ]
      beta  = [ sum(one_hot(r6x,j)*_cf(B[j][v]) for j in range(N)) for v in range(N) ]
      terms = [ Duv * alpha[u] * beta[v] for (u,v), Duv in reads ]
      while len(terms) > 1:
          terms = [terms[i] if i+1>=len(terms) else (terms[i] + terms[i+1]) for i in range(0, len(terms), 2)]
      return terms[0]

  # free eval axes for the transformed 6×6 tile
  c6x = ctx.new_range(6, AxisType.LOOP)
  r6x = ctx.new_range(6, AxisType.LOOP)

  Xhat_expr = kron_from_Xvu(Bt, c6x, r6x)

  # bufferize X̂ into its OWN tile namespace (don’t reuse TYx/TXx here!)
  
  XHAT = Xhat_expr.bufferize(TYh, TXh, c6x, r6x,
            arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))

  # ---- Ĝ branch axes (separate free indices!) ----
  c6g = ctx.new_range(6, AxisType.LOOP)
  r6g = ctx.new_range(6, AxisType.LOOP)

  # ---- Ĝ branch axes (free indices for GHAT) ----
  c6g = ctx.new_range(6, AxisType.LOOP)
  r6g = ctx.new_range(6, AxisType.LOOP)

  # discover outer axes of the weight INDEX (everything except ky,kx)
  w_buf = w_like.src[0]
  w_axes = tuple(ax for ax in w_like.src[1:] if ax not in {ky, kx})

  KYO = ctx.new_range(3, AxisType.LOOP)
  KXO = ctx.new_range(3, AxisType.LOOP)
  w_like = w_like.substitute({ky:KYO, kx:KXO})
  print("THE RESULT OF SUBSTITUTION", w_like)
  w_f32 = w_like.bufferize(KYO, KXO, arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))
  # cast once (fractions in G)
 #w_f32 = w_buf.cast(dtypes.float) if w_buf.dtype != dtypes.float else w_buf

  def ghat_kron(kernel_buf: UOp, ug: UOp, vg: UOp,
              G: list[list[float]],
              outer_axes: tuple[UOp, ...] = ()) -> UOp:
    """
    Build GHAT[ug,vg] = sum_{p,q} G[ug,p] * g[p,q] * G[vg,q]
    as a REDUCE-free expression using one-hot masks (Kronecker factoring).

    - kernel_buf: weight buffer (… , 3, 3) with any outer axes before (p,q)
    - ug, vg: 6-point LOOP ranges (free axes of GHAT)
    - G: Winograd G matrix (6x3)
    - outer_axes: any ranges to index into kernel_buf before (p,q)
    """
    N = len(G)      # 6
    K = len(G[0])   # 3

    # 1) read g[p,q] with constant indices (and pass-through any outer axes)
    if outer_axes:
      reads = [ ((p,q), kernel_buf.index(*outer_axes, _ci(p), _ci(q)))
                for p in range(K) for q in range(K) ]
    else:
      reads = [ ((p,q), kernel_buf.index(_ci(p), _ci(q)))
                for p in range(K) for q in range(K) ]

    # 2) α_p(ug) = sum_i 1_{ug==i} * G[i,p], β_q(vg) = sum_j 1_{vg==j} * G[j,q]
    alpha = [ sum(one_hot(ug, i) * _cf(G[i][p]) for i in range(N)) for p in range(K) ]
    beta  = [ sum(one_hot(vg, j) * _cf(G[j][q]) for j in range(N)) for q in range(K) ]

    # 3) GHAT term-wise (no reduce):  g[p,q] * α_p(ug) * β_q(vg)
    terms = [ g_pq * alpha[p] * beta[q] for (p,q), g_pq in reads ]

    # 4) pairwise-sum fold to keep the add tree shallow
    while len(terms) > 1:
      terms = [terms[i] if i+1>=len(terms) else (terms[i] + terms[i+1])
              for i in range(0, len(terms), 2)]
    return terms[0] if terms else _cf(0.0)
  print(w_axes)
  # pure Kronecker build: GHAT[c6g, r6g] = sum_{p,q} G[c6g,p]*g[p,q]*G[r6g,q]
  Ghat_expr = ghat_kron(w_f32, c6g, r6g, winograd_G) #, outer_axes=w_axes)

  # bufferize GHAT only on its two free axes (outer axes are *not* free here)
  GHAT = Ghat_expr.bufferize(c6g, r6g, arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))

  # G_r   = (w_like.cast(dtypes.float) * 3).reduce(kx, arg=Ops.ADD, dtype=dtypes.float)
  # Ghat_expr = (G_r * 3).reduce(ky, arg=Ops.ADD, dtype=dtypes.float)

  # # Bufferize Ĝ on (c6g, r6g). Still no oy/ox/ix/iy here.
  # GHAT = Ghat_expr.bufferize(c6g, r6g,
  #         arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL))

  # ---- Final stage: fresh reducers + A/At with ix/iy only here ----
  c6r = ctx.new_range(6, AxisType.REDUCE)
  r6r = ctx.new_range(6, AxisType.REDUCE)

  Mhat = XHAT.index(ty, tx, ix, iy) * GHAT.index(ix, iy)#* GHAT.index(ix, iy)

  A_ix  = synth_mat(list(zip(*winograd_At)), ix,  r6r)  # ix appears ONLY here
  At_iy = synth_mat(winograd_At,             c6r, iy)   # iy appears ONLY here
  A_ix = 3
  At_iy = 3
  out = ((Mhat).reduce(r6r, arg=Ops.ADD, dtype=dtypes.float))\
        .reduce(c6r, arg=Ops.ADD, dtype=dtypes.float)

  return Mhat

winograd_rewrite = PatternMatcher([
 (UPat(Ops.REDUCE, name="redu"), lambda ctx, redu: winowrite(ctx, redu))
 ])

B_data = [
  [ 1,  1, -1],
  [ 1,  1, -1],
  [0, 0, 0]
]
# def pratice_rewrite(ctx: IndexingContext, redu: UOp):
#   if redu.src[0].op is not Ops.MUL: return None
#   mul = redu.src[0]
#   #print("Choosing first")
#   idx = mul.src[0]
#   idx2 = mul.src[1]
#   ky = idx.src[1]
#   kx = idx.src[2]
#   buf = idx.src[0]
#   print(idx2.src[2])
#   #print(f"ky: {ky}, kx: {kx}, buf: {buf}")
#   # The where(invalid) pattern gets optimized in symbolic.py line 477 based on validity analysis
#   cond = kx.eq(UOp.const(dtypes.index, 3))

#   rp1, rp2, rp3 = [buf.index(UOp.const(dtypes.index, i), kx) for i in range(3)]
#   # Guard the first index parameter: when kx==3 use 0, otherwise use invalid (gets optimized)
#   guarded_first_idx = cond.where(UOp.const(dtypes.index, 0), UOp.invalid())
#   return (buf.index(guarded_first_idx, ky) * idx2.replace(src=(idx2.src[0], UOp(Ops.CONST, dtypes.index, arg=0), idx2.src[2])))



Bt =[                          # (6,6)
  [4,  0, -5,  0, 1, 0],
  [0, -4, -4,  1, 1, 0],
  [0,  4, -4, -1, 1, 0],
  [0, -2, -1,  2, 1, 0],
  [0,  2, -1, -2, 1, 0],
  [0,  4,  0, -5, 0, 1],
]
def pratice_rewrite(ctx: IndexingContext, redu: UOp):
  # Expect MUL with activation-like INDEX on the left
  if redu.src[0].op is not Ops.MUL: return None
  mul = redu.src[0]
  act_like = mul.src[1]
  w_like = mul.src[0]
  if act_like.op is not Ops.INDEX or len(act_like.src) < 3: return None

  #print(f"act_like: {act_like}, w_like: {w_like}")
  buf = w_like.src[0]
  ky  = w_like.src[1]   # output row loop (iy)
  kx  = act_like.src[2]   # output col loop (ix)
  #print(f"ky: {ky}, kx: {kx}, buf: {buf}")
  # print(f"ky: {ky}, kx: {kx}, buf: {buf}")
  # # --- helpers ---
  # def consti(i: int) -> UOp: return UOp.const(dtypes.index, i)
  # def constf(x: float) -> UOp: return UOp.const(dtypes.float, x)

  # one, zero, neg1 = constf(1.0), constf(0.0), constf(-1.0)

  # # ***** 1) READ D WITH ONLY CONST INDICES (no SUBSTITUTE, no NOOP) *****
  # # 3x3 toy -> nine scalar INDEX nodes
  # D00 = buf.index(consti(0), consti(0))
  # D01 = buf.index(consti(0), consti(1))
  # D02 = buf.index(consti(0), consti(2))
  # D10 = buf.index(consti(1), consti(0))
  # D11 = buf.index(consti(1), consti(1))
  # D12 = buf.index(consti(1), consti(2))
  # D20 = buf.index(consti(2), consti(0))
  # D21 = buf.index(consti(2), consti(1))
  # D22 = buf.index(consti(2), consti(2))

  # # ***** 2) B COEFFICIENTS AS LITERALS (no indexing B) *****
  # # Your B_data (from the file) for rows is:
  # #   rows 0,1: [1, 1, -1], row 2: [0,0,0]
  # # We'll express these per-i (ky) using tiny masks (no tables).
  # is_i01 = UOp(Ops.CMPLT, dtypes.bool, src=(ky, consti(2)))     # ky in {0,1}
  # row_mask = UOp(Ops.WHERE, dtypes.float, src=(is_i01, one, zero))
  # # row coeffs: for i∈{0,1} → [1,1,-1], for i=2 → [0,0,0]
  # b_ip0 = row_mask * one
  # b_ip1 = row_mask * one
  # b_ip2 = row_mask * neg1

  # # For columns, reuse the same pattern: j∈{0,1} → [1,1,-1], j=2 → [0,0,0]
  # is_j01 = UOp(Ops.CMPLT, dtypes.bool, src=(kx, consti(2)))     # kx in {0,1}
  # col_mask = UOp(Ops.WHERE, dtypes.float, src=(is_j01, one, zero))
  # b_jq0 = col_mask * one
  # b_jq1 = col_mask * one
  # b_jq2 = col_mask * neg1

  # # ***** 3) FORM C_q(i) = Σ_p B[i,p]*D[p,q] *****
  # C0 = b_ip0*D00 + b_ip1*D10 + b_ip2*D20
  # C1 = b_ip0*D01 + b_ip1*D11 + b_ip2*D21
  # C2 = b_ip0*D02 + b_ip1*D12 + b_ip2*D22

  # # ***** 4) FORM Y(i,j) = Σ_q C_q(i) * B[j,q] *****
  # out = C0*b_jq0 + C1*b_jq1 + C2*b_jq2

  return nmode_kron(buf, [ky, kx], Bt, [])

practice = PatternMatcher([
 (UPat(Ops.REDUCE, name="redu"), lambda ctx, redu: pratice_rewrite(ctx, redu))
 ])
# *****************
# 3.5 cleanups

# Ops.NOOP happens when we have a COPY to the device the Tensor is already on. We treat it like COPY here for MSTACK.
ALWAYS_RUN_OPS = {Ops.CONTIGUOUS, Ops.COPY, Ops.ASSIGN, Ops.NOOP}

# you don't know in the first pass if axes are going to die, this happens if there's an EXPAND to the left
def cleanup_dead_axes(b:UOp):
  # don't optimize ALWAYS_RUN_OPS
  if b.src[0].op in ALWAYS_RUN_OPS: return None

  new_rng = []
  hit = False
  reshape: list[sint] = []
  for s,rng in zip(b.shape, b.src[1:]):
    # skip for symbolic. TODO: fix this
    if rng.op is Ops.RANGE and rng.src[0].op is not Ops.CONST: return None
    # CONSTs are already dead axes
    if rng.op is Ops.CONST or (rng.op is Ops.RANGE and rng not in b.src[0].ranges):
      reshape.append(1)
      hit = True
    else:
      reshape.append(s)
      new_rng.append(rng)
  if hit:
    # move the tag to the expand. NOTE: this expand tag might not survive
    return b.replace(src=b.src[0:1]+tuple(new_rng), tag=None).reshape(tuple(reshape)).expand(b.shape).replace(tag=b.tag)

# if a buffer is being stored just for permutes or something, remove it
# we want to reexpress the indexes of idx2 in terms of the implied b1
def remove_bufferize(src:UOp, buf:UOp, idx:UOp):
  # see if we can't do it, should this ever hit?
  assert len(buf.src) == len(idx.src), f"index on wrong bufferize, {len(buf.src)} != {len(idx.src)}"
  assert all(x.op in {Ops.RANGE, Ops.CONST} for x in buf.src[1:])

  # if it's user contiguous, we never remove it
  if src.op in ALWAYS_RUN_OPS: return None

  # we don't want to bufferize threefry, also causes problems because not all platforms support long
  if src.op is not Ops.THREEFRY:
    # *** here is where we compute the cost ***
    # if we return None, the bufferize is kept

    accessed_buffers: list[UOp] = []
    reduces: list[UOp] = []
    def red_gate(x:UOp):
      if x.op is Ops.INDEX:
        accessed_buffers.append(x)
        return False
      if x.op is Ops.REDUCE: reduces.append(x)
      return True
    src.toposort(gate=red_gate)
    del red_gate

    # if this is generated from multiple buffers, don't remove this buffer
    if len(dedup([x.src[0] for x in accessed_buffers])) > 2: return None

    # if any reduces access a buffer, don't remove this buffer
    buffer_in_reduce = False
    def buf_gate(x:UOp):
      nonlocal buffer_in_reduce
      if x.op in {Ops.BUFFER, Ops.BUFFERIZE}: buffer_in_reduce = True
      return not buffer_in_reduce
    UOp.sink(*[x.src[0] for x in reduces]).toposort(gate=buf_gate)
    del buf_gate
    if buffer_in_reduce: return None
  # if it makes it here, the bufferize is removed
  # this is the ranges replaced
  # NOTE: if buf src is a const, we don't replace it
  replaces = flatten([(k,v) for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST])
  return UOp(Ops.SUBSTITUTE, dtype=src.dtype, src=(src, UOp(Ops.NOOP, src=tuple(replaces[0::2])), UOp(Ops.NOOP, src=tuple(replaces[1::2]))))

def pre_bufferize(b:UOp, x:UOp, copy:UOp):
  nb = b.replace(src=(b.src[0].contiguous(),)+b.src[1:])
  return copy.replace(src=(x.replace(src=(nb,)+x.src[1:]), copy.src[1]))

pm_cleanups = pm_mops+PatternMatcher([
  (UPat(Ops.BUFFERIZE, name="b"), cleanup_dead_axes),
  (UPat(GroupOp.All-{Ops.BUFFERIZE, Ops.BUFFER}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  (UPat((Ops.BUFFERIZE), name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType)
    and (resolve(prod(x.dtype.shape)!=prod(x.shape)) or x.shape[-1]%4!=0) else None),
  # remove noop buffers. if we look at the next index we can remove even more of these
  # NOTE: this is mostly the same case as below, but if there's no INDEX this gets more
  (UPat(Ops.INDEX, name="idx").f(Ops.BUFFERIZE, allow_any_len=True, name="b2"),
   lambda idx,b2: idx.src[0].replace(tag=nt if len(nt:=(idx.src[0].tag or ()) + (b2.tag or ())) else None) if idx.src[1:] == b2.src[1:] \
       and idx.src[0].op is not Ops.BUFFER_VIEW else None),
  # remove reindexing with cost function
  (UPat.var("src").f(Ops.BUFFERIZE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_bufferize),
  # no buffers for const
  (UPat(Ops.CONST, name='c').f(Ops.BUFFERIZE, allow_any_len=True, name="b"), lambda c,b: b.const_like(c.arg).rtag(b.tag)),
  # copy on CONST is CONST
  (UPat(Ops.COPY, src=(UPat.cvar("x"), UPat()), name="copy"), lambda copy,x: copy.const_like(x.arg)),
  (UPat(Ops.COPY, src=(UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.COPY}).f(Ops.BUFFERIZE, allow_any_len=True, name="b")
                       .f(Ops.INDEX, allow_any_len=True, name="x"), UPat()), name="copy"), pre_bufferize),
  # mstack on CONST is CONST
  (UPat(Ops.MSTACK, src=(UPat.var("s"),), allow_any_len=True).f(Ops.INDEX, allow_any_len=True),
   lambda s: UOp.const(c.dtype, c.arg) if (c:=s.base).op is Ops.CONST else None),
])

def late_buffer_view(t:UOp, b:UOp):
  if isinstance(b.device, str) and b.device.startswith("DISK"):
    rngs = b.src[1:]
    size = prod(shape := [int(r.vmax+1) for r in rngs])

    # walk up for the INDEX
    x = t
    while not any(u.op is Ops.INDEX for u in x.src): x = x.src[0]
    x = next(u for u in x.src if u.op is Ops.INDEX)

    if len(shape) == 0: offset = x.src[1].arg
    else: offset = max(sum(idx.vmin for idx in x.src[1:]), 0)

    return b.replace(src=(UOp(Ops.BUFFER_VIEW, t.dtype, (x.base,), (size, offset), tag=t.tag),) + b.src[1:])
  return b
to_bufferview = PatternMatcher([
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), name="t").f(Ops.BUFFERIZE, allow_any_len=True, name="b"), late_buffer_view),
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS)).f(Ops.BUFFER_VIEW, name="b"), lambda b: b.replace(src=b.src[0].src)),
])

DEVICE_MAX_BUFS = {"METAL": 31, "WEBGPU": 8} # TODO: get from device?
def limit_bufs(ctx:IndexingContext, root:UOp):
  if (device:=root._device) is None: return None # no device, index related calculations
  device = device if isinstance(device, str) else device[0].split(":")[0]
  if not (MAX_BUFS:=getenv("MAX_KERNEL_BUFFERS", DEVICE_MAX_BUFS.get(device, 0))): return None

  bufs: set[UOp] = set()
  def gate_input(u:UOp):
    # TODO: add cache to fix n^2
    if is_load:=(u.op in {Ops.BUFFERIZE, Ops.ASSIGN, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK, Ops.DEFINE_VAR}): bufs.add(u)
    return not is_load
  root.toposort(gate=gate_input)

  if len(bufs) > MAX_BUFS - 1: # NOTE: this -1 is for the output buffer
    srcs = []
    for s in root.src:
      if s.op in GroupOp.Elementwise:
        # Insert bufferize: all AxisType.REDUCE before bufferize are AxisType.LOOP
        orig_ranges, end_ranges = s.ranges, [x.replace(arg=(next(ctx.range_idx), AxisType.LOOP)) if x.op is Ops.RANGE else x for x in s.ranges]
        s = s.substitute(dict(zip(orig_ranges, end_ranges))).bufferize(*end_ranges, arg=BufferizeOpts(device=s.device)).index(*orig_ranges)
      srcs.append(s)
    return root.replace(src=tuple(srcs))
pm_limit_bufs = PatternMatcher([(UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), limit_bufs)])

# *****************
# 4. put in buffers for bufferize
# TODO: should BUFFERIZE look a lot more like STORE
# BUFFERIZE has device in arg
# BUFFERIZE doesn't have indexing, that's implied by the ranges it closes
# BUFFERIZE returns the BUFFER ready for INDEXing (doing this will make splitting a lot easier)
# NOTE: this has been fixed up a bit

def bufferize_to_store(x:UOp):
  rngs = x.src[1:]
  shape = tuple([int(r.vmax+1) for r in rngs])
  size = prod(shape)
  assert size > 0, f"no zero sized buffers {shape}"

  sdtype = x.dtype.ptr(size=size, addrspace=x.arg.addrspace)
  if x.src[0].op is Ops.ASSIGN:
    assign_target, assign_src, assign_mops = x.src[0].src
    assert assign_target.op is Ops.INDEX, f"{assign_target.op} is not index"
    # in assign, this is the buffer size, not the bufferize size
    # TODO: assign_mops here
    ret = assign_target.replace(dtype=sdtype).store(assign_src, *rngs, dtype=x.dtype)
    mops = []
    walk = assign_mops
    while walk is not assign_mops.base:
      mops.append((walk.op, walk.arg))
      walk = walk.src[0]
    for m in mops[::-1]: ret = ret._mop(*m)
    return ret.forced_reshape(shape).replace(tag=x.tag)

  # NOTE: the DEFINE_LOCAL needs to be disambiguated here
  if sdtype.addrspace == AddrSpace.GLOBAL:
    buf = UOp.new_buffer(x.arg.device, size, x.dtype)
    ret = buf.reshape(shape).index(*rngs, dtype=sdtype).store(x.src[0], *rngs, dtype=x.dtype)
    ret = ret.forced_reshape(shape)
    # TODO: is this right? what if it's offset
    if any(r.op is Ops.RANGE and r.src[0].op is not Ops.CONST for r in rngs):
      sym_shape = tuple([ssimplify(r.src[0]) if r.op is not Ops.CONST else 1 for r in rngs])
      ret = ret.shrink(tuple([(0,x) for x in sym_shape]))
    return ret.replace(tag=x.tag)

  # handle locals
  tag = x.arg.device
  if tag is None: tag = UOp.unique().arg # TODO: hack
  buf = UOp(Ops.DEFINE_LOCAL, sdtype, arg=tag)
  # store has the other dtype here
  # TODO: how is this unified?
  return buf.reshape(shape).index(*rngs, dtype=sdtype).store(x.src[0], *rngs, dtype=sdtype).forced_reshape(shape, dtype=x.dtype)

pm_add_buffers = pm_mops+to_bufferview+PatternMatcher([
  (UPat(Ops.BUFFERIZE, name="x"), bufferize_to_store),

  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src]), tag=None).reshape(m.src[0].arg).rtag(m.tag)),
])

# *****************
# 5. split into kernels

@dataclass
class LocalAddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)
  vars:dict = field(default_factory=dict)
  range:int = 0
  parent_tags:list = field(default_factory=list)
  opts:tuple|None = None

def debuf(ctx:LocalAddBufferContext, buf:UOp):
  ret = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(buf.arg), arg=ctx.dg)
  if buf not in ctx.map: ctx.map[buf] = buf
  ctx.dg += 1
  return ret

def unbind_kernel(ctx:LocalAddBufferContext, b:UOp):
  ctx.vars[b] = None
  return b.src[0]

def handle_assign(ctx:LocalAddBufferContext, assign:UOp):
  buf = assign.as_buf()
  # HACK to put the buffer in the MAP instead of MSTACK/MSELECT
  if buf.op in {Ops.MSTACK, Ops.MSELECT}: buf = buf.src[0]
  assert buf not in ctx.map
  ctx.map[buf] = assign
  return buf

def renumber_range(ctx:LocalAddBufferContext, r:UOp):
  if r.tag is not None: return None
  ret = r.replace(arg=(ctx.range,)+r.arg[1:], tag=())
  ctx.range += 1
  return ret

def find_bufs(x:UOp):
  idxs = [s for s in x.toposort(gate=lambda x: x.op is not Ops.ASSIGN) if s.op is Ops.INDEX]
  read_from: dict[UOp, Ops] = {}
  if any((buf:=idx.as_buf()).op is Ops.BUFFER and read_from.setdefault(buf, op:=idx.src[0].op) is not op for idx in idxs):
    raise RuntimeError(f"cycle detected while indexing {buf}")

to_define_global = PatternMatcher([
  (UPat(Ops.STORE, name="x"), find_bufs),
  (UPat(Ops.BUFFER, name="buf"), debuf),
  (UPat(Ops.BIND, name="b"), unbind_kernel),
  (UPat((Ops.ASSIGN, Ops.MSTACK, Ops.MSELECT), name="assign"), handle_assign),

  # HACK in case any CONSTs were replaced
  # this is only needed if you are using symbolic
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"), lambda c: c.replace(src=()) if len(c.src) else None),

  # remove RANGE with 0 size
  (UPat(Ops.RANGE, name="r"), lambda r: UOp.const(dtypes.index, 0) if r.vmax == 0 else None),

  # renumber the ranges starting with 0 so that kernel deduping works
  (UPat(Ops.RANGE, name="r"), renumber_range),
])

def get_contiguous(ctx:LocalAddBufferContext, x:UOp):
  if isinstance(x.arg, tuple) and all(isinstance(y, Opt) for y in x.arg): ctx.opts = x.arg
  return x.src[0]

rangeify_codegen = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="x"), get_contiguous),

  # no NOOP in the kernel graph
  # TODO: this can be moved into codegen?
  (UPat(Ops.NOOP, name="x"), lambda x: x.src[0]),

  # strip the arg from store
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(arg=None) if x.arg is not None else None),

  # add loads to non ptr indexes
  # TODO: this can be moved into codegen?
  (UPat((Ops.DEFINE_GLOBAL, Ops.STORE), name="dg").f(Ops.INDEX, name="idx", allow_any_len=True),
   lambda dg,idx: None if isinstance(idx.dtype, (PtrDType, ImageDType)) else idx.replace(dtype=dg.dtype, arg=None).load()),

  # TODO: this can be moved into codegen
  (UPat(Ops.STORE, name="store").f(Ops.INDEX, allow_any_len=True, name="idx").f(Ops.LOAD),
    lambda store,idx: idx.replace(src=(store.as_buf(),)+idx.src[1:]).load(store if idx.dtype.addrspace != AddrSpace.LOCAL else store.barrier())),

  # TODO: hack for group for reduce
  (UPat(Ops.IF, src=(UPat.var("gate"), UPat(Ops.LOAD, src=(UPat.var("src"), UPat.var("barrier"))),)),
   lambda src, barrier, gate: src.load(UOp(Ops.IF, src=(gate, barrier)))),
])

def remove_metadata_tags(ctx:LocalAddBufferContext, x:UOp):
  if x.tag is None or x.tag == (): return None
  ctx.parent_tags += list(x.tag)
  return x.replace(tag=None)

pm_remove_tags = PatternMatcher([
  # remove all the tags
  (UPat(GroupOp.All, name="x"), remove_metadata_tags),
])

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  metadata: tuple[Metadata, ...] = ()
  def __repr__(self):
    ast_rep = f"SINK{tuple(s.op for s in self.ast.src)}" if self.ast.op is Ops.SINK else repr(self.ast.op)
    return f"<Kernel {len(list(self.ast.toposort()))} {ast_rep} {self.metadata}>"

def split_store(ctx:list[UOp], x:UOp):
  if len(x.ranges): return None
  if x.src[0].ptrdtype.addrspace is AddrSpace.LOCAL: return None

  # local kernel rewrite
  lctx = LocalAddBufferContext()
  ret = graph_rewrite(x, to_define_global+pm_flatten_range+rangeify_codegen+pm_remove_tags, ctx=lctx, name="kernel split", bottom_up=True)

  # gather the metadata
  metadatas = [ctx[y].metadata for y in lctx.parent_tags]

  # NOTE: the hack for COPY is here
  ret = ret.sink(arg=KernelInfo(opts_to_apply=lctx.opts) if lctx.opts is not None else None) \
    if ret.src[1].op not in {Ops.COPY, Ops.BUFFER_VIEW} else ret.src[1]
  kernel_arg = Kernel(ret,tuple(dedup(flatten([x for x in metadatas if x is not None])))[::-1])
  kernel = UOp(Ops.KERNEL, src=tuple(lctx.map.values())+tuple(lctx.vars.keys()), arg=kernel_arg)
  if ret.op is Ops.SINK and not all_same([x.device for x in kernel.src if x.op is not Ops.BIND]):
    raise RuntimeError(f"all buffers must be on the same device: {tuple(b.buf_uop.buffer for b in kernel.src)}")
  return x.as_buf().assign(kernel)

split_kernels = PatternMatcher([
  (UPat(Ops.STORE, name="x"), split_store),
])

def tag_uop(ctx:list[UOp], x:UOp):
  if x.tag is not None: return None
  ctx.append(x)
  return x.replace(tag=(len(ctx)-1,))
add_tags = PatternMatcher([
  # don't tag BUFFERs, they are global
  (UPat(GroupOp.All-{Ops.BUFFER, Ops.CONST, Ops.DEVICE, Ops.UNIQUE, Ops.DEFINE_VAR, Ops.BIND,
                     Ops.MSTACK, Ops.MSELECT}.union(GroupOp.Movement), name="x"), tag_uop),
  (UPat({Ops.MSTACK, Ops.MSELECT}, name="x"), lambda ctx,x: None if all(s.op is Ops.BUFFER for s in x.src) else tag_uop(ctx, x)),
])

# support for using a contiguous permuted view instead of the parent view if one exists
# modified from kernelize.py to not use ShapeTracker

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  x = src
  while x is not src.base:
    if x.op is Ops.PERMUTE: contig = contig.permute(argsort(x.arg))
    elif x.op is Ops.RESHAPE: contig = contig.reshape(x.src[0].shape)
    else: return None
    x = x.src[0]
  ctx[src.base] = contig
replace_contiguous = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, src=(UPat(GroupOp.Movement, name="src"),), name="contig"), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
])

def do_sub_recurse(s:UOp):
  x,keys,values = s.src[0], s.src[1].src, s.src[2].src
  # SUBSTITUTE applied to SUBSTITUTE runs the child SUB on the parents. though this is probably wrong in the generic case
  if x.op is Ops.SUBSTITUTE:
    sub_k = UOp(Ops.SUBSTITUTE, src=(x.src[1],)+s.src[1:])
    sub_v = UOp(Ops.SUBSTITUTE, src=(x.src[2],)+s.src[1:])
    return UOp(Ops.SUBSTITUTE, dtype=x.dtype, src=(x.src[0], sub_k, sub_v))
  # here we actually do the SUBSTITUTE
  if x in keys: return values[keys.index(x)]
  # we filter any keys where the ranges don't overlap. this keeps the algorithm O(output graph size)
  x_ranges = x.ranges
  new_kv = {k:v for k,v in zip(keys,values) if any(r in x_ranges for r in k.ranges)}
  # if there's no SUBSTITUTEs left, we can just return x
  if len(new_kv) == 0: return x
  # then we add SUBSTITUTE to all parents
  uop_keys, uop_values = UOp(Ops.NOOP, src=tuple(new_kv.keys())), UOp(Ops.NOOP, src=tuple(new_kv.values()))
  return x.replace(src=tuple([UOp(Ops.SUBSTITUTE, dtype=y.dtype, src=(y,uop_keys,uop_values)) for y in x.src]))
pm_substitute_recurse = PatternMatcher([(UPat(Ops.SUBSTITUTE, src=(UPat(), UPat(Ops.NOOP), UPat(Ops.NOOP)), name="s"), do_sub_recurse)])

@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len([u for u in UOp.sink(*ret.values()).toposort() if u.op is Ops.KERNEL]))}", True)
def get_rangeify_map(sink:UOp) -> dict[UOp, UOp]:
  uop_list: list[UOp] = []
  tsink = graph_rewrite(sink, add_tags, ctx=uop_list, bottom_up=True, name="number the uops")

  tsink = graph_rewrite(tsink, earliest_rewrites+replace_contiguous, ctx={}, name="earliest rewrites")

  # convert movement ops to ranges
  tsink, rctx = run_rangeify(tsink, getenv("DEBUG_RANGEIFY", 0))

  #if WINO: tsink = graph_rewrite(tsink, practice, ctx=rctx, name="practice")
  if WINO: tsink = graph_rewrite(tsink, winograd_rewrite, ctx=rctx, name="winograd")

  # NOTE: sym (vs symbolic_simple) breaks things here because ranges with len 1 aren't handled right
  tsink = graph_rewrite(tsink, symbolic_simple+pm_reduce_unparented, name="symbolic")  # this supports const folding
  tsink = graph_rewrite(tsink, pm_cleanups, bottom_up=True, name="remove costly buffers")
  # TODO: can you substitute and remove costly buffers at the same time?
  tsink = graph_rewrite(tsink, pm_substitute_recurse, bottom_up=True, name="run substitutes")
  tsink = graph_rewrite(tsink, pm_limit_bufs, ctx=rctx, name="limit buffers")

  # rebuild the sink with all the BUFFERIZEs with tags, this is what's ending up in the tensor graph
  # MSTACK stacks multiple BUFFERIZEs in one tagged tensor
  # if it's not tagged by here, it's out
  tsink = UOp.sink(*[x for x in tsink.backward_slice if x.base.op in {Ops.BUFFERIZE, Ops.MSTACK, Ops.CONST, Ops.BUFFER} and \
                     x.tag is not None and len(x.tag)])

  if getenv("VIZ"): graph_rewrite(tsink, PatternMatcher([]), name="View Tagged Rangeify")

  # bufferize -> store
  tsink = graph_rewrite(tsink, pm_add_buffers, bottom_up=True, name="bufferize to store")
  tsink = graph_rewrite(tsink, split_kernels, ctx=uop_list, name="split kernels")

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in tsink.toposort():
    if u.op is not Ops.ASSIGN: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      # TODO: this is probably broken for MSELECT/MSTACK
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.ASSIGN and x.buf_uop is s for x in u.toposort()):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on ASSIGN or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep: tsink = graph_rewrite(tsink, _substitute, ctx=assign_rep, bottom_up=True, name="fix_assign")

  if getenv("VIZ"): graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")

  becomes_map: dict[UOp, UOp] = {}
  for s in tsink.src:
    assert s.tag is not None
    for a in s.tag:
      if a is None: continue
      becomes_map[uop_list[cast(int, a)]] = s.replace(tag=None)
  return becomes_map
