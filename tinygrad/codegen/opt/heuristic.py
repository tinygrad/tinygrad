import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS, TC_OPT, TC_SELECT, USE_TC, AMX
from tinygrad.dtype import ImageDType
from tinygrad.uop.ops import Ops, resolve, AxisType
from tinygrad.codegen.opt.postrange import Scheduler

def _apply_tc_opts(k:Scheduler) -> Scheduler|None:
  """Try tensor core optimization. Returns optimized kernel or None if TC doesn't apply."""
  if not (USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or TC_OPT.value >= 1)): return None
  tk = k.copy()
  try:
    rngs = tk.apply_opt(Opt(OptOps.TC, 0, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))
  except KernelOptError:
    return None
  # skip hand-coded TC opts if AMX, upcasting will make kernel slower
  if AMX: return tk
  max_upcast = getattr(tk.ren, 'max_upcast_size', 64)
  if rngs is not None:
    for tc_dim in [1, 0]:  # attempt to upcast M and N
      szs = [sz for sz in [5,4,3,2] if rngs[tc_dim].src[0].divides(sz) is not None and tk.upcast_size() * sz < max_upcast]
      if szs: rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
    if tk.upcast_size() < max_upcast and (szs := [sz for sz in [4,2] if rngs[0].src[0].divides(sz) is not None]):
      tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), szs[0]))
  return tk

def _apply_image_upcast(k:Scheduler) -> None:
  """Upcast float4 images early so we don't accidentally add locals before the upcast."""
  for buf_index, buf in enumerate(k.bufs):
    if not isinstance(buf.src[0].dtype, ImageDType): continue
    unit_stride_axes_mul_4 = [k.rngs.index(c) for c in k.bufs[buf_index].src[1].get_idx().split_uop(Ops.ADD)
                              if c.op is Ops.RANGE and (c.vmax+1) % 4 == 0]
    if not unit_stride_axes_mul_4: continue
    axis = unit_stride_axes_mul_4[0]
    if axis in k.upcastable_dims: k.apply_opt(Opt(OptOps.UPCAST, axis, 4))
    elif axis in k.unrollable_dims: k.apply_opt(Opt(OptOps.UNROLL, k.unrollable_dims.index(axis), 4))

def _apply_matvec(k:Scheduler) -> bool:
  """Try matvec optimization. Returns True if applied."""
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  if not (k.ren.has_local and getenv("MV", 1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1)): return False
  if not (k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.ren.has_shared): return False
  mulop = k.reduceop.src[0]
  if not (mulop.op is Ops.MUL and mulop.src[0].op is Ops.INDEX and mulop.src[1].op is Ops.INDEX): return False
  idx0, idx1 = mulop.src[0].src[1].get_idx(), mulop.src[1].src[1].get_idx()
  reduce_rngs = k.ranges_of(AxisType.REDUCE)
  if not reduce_rngs: return False
  first_reduce_rng = reduce_rngs[0]
  if not (any(u is first_reduce_rng for u in idx0.split_uop(Ops.ADD)) and all(r in idx1.ranges for r in idx0.ranges)): return False
  for global_idx in k.axes_of(AxisType.GLOBAL):
    if first_reduce_rng.src[0].divides(MV_THREADS_PER_ROW) is None: continue
    if k.full_shape[global_idx] % (MV_BLOCKSIZE * MV_ROWS_PER_THREAD) != 0: continue
    if DEBUG >= 3: print(f"MATVEC: {k.full_shape=} {first_reduce_rng.render()} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
    try:
      if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
    except KernelOptError: pass
    if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
    if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
    return True
  return False

def _apply_upcast_heuristics(k:Scheduler) -> None:
  """Apply various upcast heuristics."""
  # upcast small masked dims (e.g. from Tensor.stack)
  to_upcast: list[int] = []
  for axis in k.upcastable_dims:
    is_masked = any(any(o is k.rngs[axis] for o in u.src[0].backward_slice) for u in k.ast.backward_slice if u.op is Ops.WHERE)
    if k.full_shape[axis] <= 7 and is_masked and prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= 49:
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))

  # upcast non-reduce axes based on stride heuristic
  is_dsp = k.ren is not None and k.ren.device == "DSP"
  upcasted_axis: set[int] = set()
  while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and k.upcast_size() < 32:
    xb_choices = []
    for axis, upcast_amount in itertools.product(k.upcastable_dims, ([128] if is_dsp and not upcasted_axis else []) if is_dsp else [3, 4]):
      if axis in upcasted_axis or k.full_shape[axis] % upcast_amount != 0: continue
      rng = k.rngs[axis]
      if not any(rng not in b.src[1].get_idx().backward_slice and
                 all(r2 in b.src[1].get_idx().backward_slice for r2 in k.ranges_of(AxisType.UPCAST, AxisType.UNROLL)) for b in k.bufs): continue
      num_strides, sum_strides = 0, 0
      for b in k.bufs:
        idx = b.src[1].get_idx()
        if rng in idx.backward_slice: num_strides += 1
        for c in idx.split_uop(Ops.ADD):
          if c is rng: sum_strides += 1
          elif c.op is Ops.MUL and c.src[0] is rng and c.src[1].op is Ops.CONST: sum_strides += c.src[1].arg
          elif c.op is Ops.MUL and c.src[1] is rng and c.src[0].op is Ops.CONST: sum_strides += c.src[0].arg
      xb_choices.append((num_strides, sum_strides, axis, upcast_amount))
    if not xb_choices: break
    xb_choices = sorted(xb_choices)
    if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
    k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
    upcasted_axis.add(xb_choices[0][2])

def _apply_unroll(k:Scheduler) -> None:
  """Unroll small reduce dimensions."""
  max_upcast = getattr(k.ren, 'max_upcast_size', 64)
  if not k.unrollable_dims: return
  if not ((k.upcast_size() <= 4 or not k.axes_of(AxisType.UNROLL)) and k.upcast_size() < max_upcast): return
  try:
    s = k.full_shape[k.unrollable_dims[-1]]
    if s <= 32:
      k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
      # upcast second reduce dim too if small (but respect max_upcast_size for tight register budgets)
      if k.unrollable_dims and s <= 3 and k.full_shape[k.unrollable_dims[-1]] <= 3 and (max_upcast >= 64 or k.upcast_size() < max_upcast):
        k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
    elif k.full_shape[k.unrollable_dims[-1]] % 4 == 0:
      k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 4))
  except KernelOptError: pass

def _apply_local(k:Scheduler) -> None:
  """Apply local memory optimizations."""
  if not k.ren.has_local: return
  if NOLOCALS:
    k.apply_opt(Opt(OptOps.NOLOCALS))
    return
  # prioritize making expand axes local
  local_axis_ranking = [(any(k.rngs[axis] not in b.src[1].get_idx().backward_slice for b in k.bufs), axis)
                        for axis in k.axes_of(AxisType.GLOBAL, AxisType.LOOP) if k.rngs[axis].src[0].op is Ops.CONST]
  to_local: list[tuple[int, int]] = []
  for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
    local_size = prod(sz for _, sz in to_local)
    local_sz = next((x for x in ([32] * (axis == 0) + [16, 8, 4, 3, 2]) if k.full_shape[axis] % x == 0 and local_size * x <= 128), None)
    if local_sz is not None: to_local.append((axis, local_sz))
  deleted_shape = 0
  for axis, local_sz in sorted(to_local[:3]):
    axis -= deleted_shape
    will_delete_shape = local_sz == k.full_shape[axis]
    k.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
    if will_delete_shape: deleted_shape += 1

def _apply_threading(k:Scheduler) -> None:
  """Apply threading optimizations for CPU-like devices."""
  if not (k.ren.has_threads and k.ren.global_max is not None): return
  for threads in [32, 16, 12, 8, 6, 5, 4, 3, 2]:
    if threads > k.ren.global_max[0] or resolve(prod(k.full_shape) // (128 << 10) < threads): continue
    for axis in k.axes_of(AxisType.LOOP):
      if k.full_shape[axis] % threads == 0:
        try: k.apply_opt(Opt(OptOps.THREAD, axis, threads))
        except KernelOptError: pass
        break
    if k.applied_opts and k.applied_opts[-1].op is OptOps.THREAD: break

def hand_coded_optimizations(k:Scheduler) -> Scheduler:
  """
  Apply hand-coded kernel optimizations based on heuristics.

  Tries optimizations in order: tensor cores, image upcast, matvec, grouping, upcasts, unrolls, locals, threading.
  """
  # try tensor cores first
  if (tk := _apply_tc_opts(k)) is not None: return tk

  k = k.copy()
  _apply_image_upcast(k)
  if _apply_matvec(k): return k

  # try grouping for small output shapes
  if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 2048), False):
    for axis, sz in itertools.product((0, 1, 2), (16,)):
      try:
        k.apply_opt(Opt(OptOps.GROUPTOP, axis, sz))
        break
      except KernelOptError: pass

  if k.group_for_reduces: return k

  _apply_upcast_heuristics(k)
  _apply_unroll(k)

  # fallback upcast if nothing else worked
  if not k.upcasted and k.upcastable_dims and k.full_shape[k.upcastable_dims[-1]] % 4 == 0:
    k.apply_opt(Opt(OptOps.UPCAST, k.upcastable_dims[-1], 4))

  _apply_local(k)
  _apply_threading(k)
  return k
