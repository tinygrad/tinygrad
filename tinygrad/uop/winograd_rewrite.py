"""
Winograd F(4x4, 3x3) as a generic UOp graph rewrite rule.
"""
from __future__ import annotations

from tinygrad.uop.ops import UOp, Ops, resolve
from tinygrad.helpers import prod, all_int, DEBUG, WINO_GRAPH, ceildiv
from tinygrad.dtype import dtypes

WINO_KERNEL = 3   # r — filter spatial size
WINO_TILE   = 4   # m — output tile size
WINO_SIZE   = WINO_TILE + WINO_KERNEL - 1   # 6 — transform domain size

_G: list[list[float]]  = [[1/4,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],[1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]]
_Bt: list[list[float]] = [[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],[0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]]
_At: list[list[float]] = [[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]]

def _apply_winograd_matrix_uop(mat: list[list[float]], t: UOp) -> UOp:
  M, N = len(mat), len(mat[0])
  for d in range(2):
    out_pieces = []
    for i in range(M):
      piece: UOp | None = None
      for k in range(N):
        coeff = float(mat[i][k])
        if coeff == 0.0: continue
        sl = tuple((0, s) if j != d else (k, k+1) for j, s in enumerate(t.shape))
        extracted = t.shrink(sl)
        if coeff != 1.0:
          if coeff == -1.0:
            extracted = -extracted
          else:
            extracted = extracted * extracted.const_like(coeff)
        piece = extracted if piece is None else piece + extracted
      if piece is None:
        sl = tuple((0, s) if j != d else (0, 1) for j, s in enumerate(t.shape))
        piece = t.const_like(0.0).shrink(sl)
      out_pieces.append(piece)
    t = out_pieces[0].cat(*out_pieces[1:], dim=d)
  return t

def _extract_affine(expr: UOp) -> dict[UOp, int] | None:
  if expr.op is Ops.CONST: return {}
  if expr.op is Ops.RANGE: return {expr: 1}
  if expr.op is Ops.CAST: return _extract_affine(expr.src[0])
  if expr.op is Ops.ADD:
    a, b = _extract_affine(expr.src[0]), _extract_affine(expr.src[1])
    if a is None or b is None: return None
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
  if expr.op is Ops.MUL:
    c, v = (expr.src[0], expr.src[1]) if expr.src[0].op is Ops.CONST else (expr.src[1], expr.src[0])
    if c.op is Ops.CONST and (sub := _extract_affine(v)) is not None:
      return {k: val * int(c.arg) for k, val in sub.items()}
  return None

def _find_logical_base(uop: UOp, target_shape: tuple[int, ...]) -> UOp | None:
  _MOVEMENT = {Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK, Ops.PAD, Ops.CAST}
  last_match = uop if uop.shape == target_shape else None
  while uop.op in _MOVEMENT:
    uop = uop.src[0]
    if uop.shape == target_shape:
      last_match = uop
  return last_match

def _probe_input(body: UOp, operand: UOp, kernel_axes: list[int], reduce_axes: tuple, pm_mops) -> tuple[int, int] | None:
  test_rngs = [UOp.range(body.shape[ax], ax) if resolve(body.shape[ax] > 1) else UOp.const(dtypes.weakint, 0) for ax in range(len(body.shape))]
  try:
    simplified = operand.index(*test_rngs).substitute({operand.base: UOp(Ops.NOOP)}, extra_pm=pm_mops)
  except Exception: return None
  idx_nodes = [n for n in simplified.toposort() if n.op is Ops.INDEX]
  if not idx_nodes: return None
  idx_node = idx_nodes[-1]
  spatial_out_axes = []
  claimed = set()
  idx_srcs = [idx_node.src[1]] if len(idx_node.src) == 2 else idx_node.src[2:]
  for i, idx_expr in enumerate(idx_srcs):
    if not (terms := _extract_affine(idx_expr)): continue
    for k_ax in kernel_axes:
      r_k = test_rngs[k_ax]
      if r_k.op is Ops.RANGE and (ck := terms.get(r_k, 0)) != 0:
        for ax in range(len(body.shape)):
          r_out = test_rngs[ax]
          if ax not in reduce_axes and ax not in claimed and r_out.op is Ops.RANGE and terms.get(r_out, 0) == ck:
            claimed.add(ax)
            spatial_out_axes.append(ax)
            break
  if len(spatial_out_axes) != 2: return None
  return (spatial_out_axes[0], spatial_out_axes[1])

def _probe_filter(body: UOp, operand: UOp, kernel_axes: list[int], pm_mops) -> tuple[tuple[int, int], set[int]] | None:
  test_rngs = [UOp.range(body.shape[ax], ax) if resolve(body.shape[ax] > 1) else UOp.const(dtypes.weakint, 0) for ax in range(len(body.shape))]
  try:
    simplified = operand.index(*test_rngs).substitute({operand.base: UOp(Ops.NOOP)}, extra_pm=pm_mops)
  except Exception: return None
  active_ids = {r.arg[0] for r in simplified.toposort() if r.op is Ops.RANGE}
  idx_nodes = [n for n in simplified.toposort() if n.op is Ops.INDEX]
  if not idx_nodes: return None
  idx_node = idx_nodes[-1]
  pos = {}
  idx_srcs = [idx_node.src[1]] if len(idx_node.src) == 2 else idx_node.src[2:]
  for i, idx_expr in enumerate(idx_srcs):
    if not (terms := _extract_affine(idx_expr)): continue
    for k_ax in kernel_axes:
      r_k = test_rngs[k_ax]
      if r_k.op is Ops.RANGE and terms.get(r_k, 0) != 0:
        pos[k_ax] = abs(terms[r_k]) if len(idx_node.src) == 2 else -i
  if len(pos) != 2 or pos[kernel_axes[0]] == pos[kernel_axes[1]]: return None
  canon_k = (kernel_axes[0], kernel_axes[1]) if pos[kernel_axes[0]] > pos[kernel_axes[1]] else (kernel_axes[1], kernel_axes[0])
  return canon_k, active_ids

def try_winograd_rewrite(reduce: UOp, x: UOp, pm_mops) -> UOp | None:
  if not WINO_GRAPH: return None
  if not all_int(x.shape) or not all_int(reduce.shape): return None
  if reduce.arg[0] is not Ops.ADD: return None
  reduce_axes: tuple[int, ...] = reduce.arg[1]
  if len(reduce_axes) < 3: return None
  body = x
  while body.op is Ops.CAST: body = body.src[0]
  if body.op is not Ops.MUL: return None
  a, b = body.src[0], body.src[1]
  kernel_axes = [ax for ax in reduce_axes if body.shape[ax] == WINO_KERNEL]
  if len(kernel_axes) != 2: return None
  channel_axes = [ax for ax in reduce_axes if ax not in kernel_axes]
  if len(channel_axes) != 1: return None
  cin_ax = channel_axes[0]
  cin = body.shape[cin_ax]

  probe_inp_a = _probe_input(body, a, kernel_axes, reduce_axes, pm_mops)
  probe_filt_b = _probe_filter(body, b, kernel_axes, pm_mops)
  probe_inp_b = _probe_input(body, b, kernel_axes, reduce_axes, pm_mops)
  probe_filt_a = _probe_filter(body, a, kernel_axes, pm_mops)

  if probe_inp_b and probe_filt_a and not any(ax in probe_filt_a[1] for ax in probe_inp_b):
    canon_k = probe_filt_a[0]
    # Note: _probe_input discovers spatial axes in iteration order of kernel_axes.
    # Therefore, swapping them here based on canon_k correctly restores Y/X physical order.
    spatial_out_axes = (probe_inp_b[1], probe_inp_b[0]) if canon_k != tuple(kernel_axes) else probe_inp_b
    kernel_axes = list(canon_k)
    inp_uop, filt_uop = b, a
    filt_active = probe_filt_a[1]
  elif probe_inp_a and probe_filt_b and not any(ax in probe_filt_b[1] for ax in probe_inp_a):
    canon_k = probe_filt_b[0]
    # Note: _probe_input discovers spatial axes in iteration order of kernel_axes.
    # Therefore, swapping them here based on canon_k correctly restores Y/X physical order.
    spatial_out_axes = (probe_inp_a[1], probe_inp_a[0]) if canon_k != tuple(kernel_axes) else probe_inp_a
    kernel_axes = list(canon_k)
    inp_uop, filt_uop = a, b
    filt_active = probe_filt_b[1]
  else:
    return None

  cout_axes = [ax for ax in range(len(body.shape)) if ax in filt_active and ax not in kernel_axes and ax != cin_ax]
  if len(cout_axes) != 1:
    if DEBUG >= 2: print(f"[winograd_rewrite] failed cout_axes ({cout_axes}) constraint, likely grouped conv or unsupported broadcast")
    return None

  _cout = body.shape[cout_axes[0]]
  _cin = body.shape[cin_ax]
  assert isinstance(_cout, int) and isinstance(_cin, int)
  cout, cin = _cout, _cin

  bs_axes = [ax for ax in range(len(body.shape)) if ax not in reduce_axes and ax not in spatial_out_axes and ax not in cout_axes]
  # Note: This assumes a single unified batch dimension in the base tensor.
  # If the base graph uses multiple batch dims (e.g. bs1, bs2), this prod() correctly sizes the volume,
  # but _find_logical_base will fail to match the flattened target shape.
  _bs = prod(body.shape[ax] for ax in bs_axes)
  assert isinstance(_bs, int)
  bs = _bs

  o_spatial = tuple(reduce.shape[ax] for ax in spatial_out_axes)
  if not all(isinstance(s, int) for s in o_spatial) or len(o_spatial) != 2: return None
  os_y, os_x = int(o_spatial[0]), int(o_spatial[1])
  o_spatial_t = (os_y, os_x)
  t_spatial = (ceildiv(os_y, WINO_TILE), ceildiv(os_x, WINO_TILE))

  if cout < 8 or prod(t_spatial) < 4: return None

  inp_target_shape = (bs, cin, os_y + WINO_KERNEL - 1, os_x + WINO_KERNEL - 1)
  filt_target_shape = (cout, cin, 3, 3)

  inp_base = _find_logical_base(inp_uop, inp_target_shape)
  filt_base = _find_logical_base(filt_uop, filt_target_shape)
  if inp_base is None or filt_base is None: return None

  try:
    result = _build_winograd(inp_base, filt_base, bs, cin, cout, o_spatial_t, t_spatial)
    if result.shape != reduce.shape:
      result = result.reshape(reduce.shape)
    return result
  except Exception as e:
    if DEBUG >= 1: print(f"[winograd_rewrite] replacement failed: {e}")
    return None

def _build_winograd(inp: UOp, filt: UOp, bs: int, cin: int, cout: int, o_spatial: tuple[int, int], t_spatial: tuple[int, int]) -> UOp:
  S = WINO_SIZE   # 6
  M = WINO_TILE   # 4
  needed = tuple(t * M + WINO_KERNEL - 1 for t in t_spatial)
  cur    = tuple(inp.shape[2 + i] for i in range(2))
  extra  = tuple(n - c for n, c in zip(needed, cur))
  if any(e < 0 for e in extra):
    raise ValueError(f"input spatial {cur} too small for needed {needed}")
  if any(e > 0 for e in extra):
    pad_spec = ((0, 0), (0, 0)) + tuple((0, e) for e in extra)
    inp = inp.pad(pad_spec)

  from tinygrad.uop.math import pool_uop
  d = pool_uop(inp, (S, S), (M, M))
  d = d.permute(4, 5, 0, 1, 2, 3)

  g = filt.permute(2, 3, 0, 1)
  gfactors = _apply_winograd_matrix_uop(_G, g)
  gfactors = gfactors.reshape((S, S, 1, cout, cin, 1, 1))

  dfactors = _apply_winograd_matrix_uop(_Bt, d)
  dfactors = dfactors.reshape((S, S, bs, 1, cin) + tuple(t_spatial))

  m = gfactors * dfactors
  m = m.sum(axis=4)

  out = _apply_winograd_matrix_uop(_At, m)
  out = out.permute(2, 3, 4, 0, 5, 1)
  out = out.reshape((bs, cout) + tuple(t * M for t in t_spatial))

  shrink_spec = ((0, bs), (0, cout)) + tuple((0, s) for s in o_spatial)
  return out.shrink(shrink_spec)
