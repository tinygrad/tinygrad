import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS, TC_OPT, TC_SELECT, USE_TC, IMAGE
from tinygrad.uop.ops import Ops, resolve, AxisType
from tinygrad.codegen.late.coalese import image_valid_dims
from tinygrad.codegen.opt.postrange import Scheduler

def hand_coded_optimizations(k:Scheduler) -> Scheduler:
  # first try the tensor cores
  """ Attempts to apply a tensor core optimization to the kernel. If one exists and applies properly, return true, otherwise return false.
  Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

  Keyword arguments:
  use_tensor_cores -- controls how tensor cores are applied (default 1)
    0: will disable any tensor core matching
    1: enable tensor cores
    2: apply tensor core shape but don't use UOp.WMMA
  extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default None)
  tc_select -- specifies which tensor core(s) to use for optimization (default -1)
    -1: iterates through all available tensor cores in order and uses the first one that matches the requirements (dims and dtypes)
    [0-N]: uses only the n'th tensor core available; useful for search
  tc_opt -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
    0: applies to only kernels with a single reduce axis and direct Ops.LOAD into Ops.MUL
    1: allows kernels with multiple reduce axes and also multiplication of Ops.CAST'd buffers
    2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
  """
  # NOTE: unless TC_OPT is > 0, we only trigger tensor cores if there's only one reduce axis
  if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):
    good_tc_opt = False
    tk = k.copy()
    try: # check TC first and apply hand-coded opts if successful
      rngs = tk.apply_opt(Opt(OptOps.TC, 0, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))
      good_tc_opt = True
    except KernelOptError:
      pass
    if good_tc_opt:
      if rngs is not None:
        tc_sizes = [r.src[0] for r in rngs]
        skinny_output = any(resolve(sz < 2, False) for sz in tc_sizes[:2])
        very_long_reduce = resolve(tc_sizes[2] >= 4096, False)
        long_reduce = resolve(tc_sizes[2] >= 1024, False)
        small_m = resolve(tc_sizes[0] <= 32, False)
        tiny_m = resolve(tc_sizes[0] <= 16, False)
        if resolve(tc_sizes[0] >= 32, False) and resolve(tc_sizes[0] < 33, False) and resolve(tc_sizes[1] >= 1536, False) and \
           resolve(tc_sizes[2] >= 128, False) and resolve(tc_sizes[2] <= 512, False):
          upcast_n = 8 if resolve(tc_sizes[1] >= 4096, False) else 6
          rngs[1] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[1]), upcast_n))[0]
          rngs[0] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[0]), 2))[0]
          rngs[0] = tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), 8))[0]
          rngs[1] = tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[1]), 2))[0]
          return tk
        for tc_dim in [1,0]: # attempt to upcast M and N
          if skinny_output or resolve(tc_sizes[tc_dim] >= 32768, False): continue
          short_conv_n = tc_dim == 1 and resolve(tc_sizes[0] >= 65536, False) and resolve(tc_sizes[1] == 18, False) and \
                         resolve(tc_sizes[2] <= 4, False)
          upcast_sizes = [2] if short_conv_n or (very_long_reduce and (tiny_m or (tc_dim == 0 and small_m))) else [5,4,3,2]
          szs = [sz for sz in upcast_sizes if rngs[tc_dim].src[0].divides(sz) is not None]
          if szs:
            # set it to the replaced range
            rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
        if skinny_output:
          outer_rngs = [r for r in tk.rngs if r.arg[-1] is AxisType.GLOBAL and r not in rngs[:2]]
          if outer_rngs and outer_rngs[0].src[0].divides(2) is not None:
            tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(outer_rngs[0]), 2))
          if very_long_reduce:
            outer_rngs = [r for r in tk.rngs if r.arg[-1] is AxisType.GLOBAL and r not in rngs[:2]]
            if outer_rngs and (outer_local:=next((x for x in (16, 4, 2) if outer_rngs[0].src[0].divides(x) is not None), None)):
              tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(outer_rngs[0]), outer_local))
          return tk
        local_sizes = [2] if long_reduce and small_m else [4,2]
        if (szs := [sz for sz in local_sizes if rngs[0].src[0].divides(sz) is not None]):
          rngs[0] = tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), szs[0]))[0]
        if long_reduce and small_m and resolve(tc_sizes[1] >= 128, False) and \
           rngs[1].arg[-1] is AxisType.GLOBAL and rngs[1].src[0].divides(3) is not None:
          rngs[1] = tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[1]), 3))[0]
        if tk.applied_opts[-1] == Opt(OptOps.LOCAL, 1, 4):
          outer_rngs = [r for r in tk.rngs if r.arg[-1] is AxisType.GLOBAL]
          if outer_rngs and resolve(outer_rngs[0].src[0] >= 384, False) and \
             (outer_local:=next((x for x in (4, 3, 2) if outer_rngs[0].src[0].divides(x) is not None), None)):
            tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(outer_rngs[0]), outer_local))
        elif resolve(tc_sizes[0] <= 4, False) and resolve(tc_sizes[2] >= 256, False):
          outer_rngs = [r for r in tk.rngs if r.arg[-1] is AxisType.GLOBAL]
          outer_local = 16 if resolve(tc_sizes[2] >= 512, False) else 2
          if (outer_rng:=next((r for r in reversed(outer_rngs) if r.src[0].divides(outer_local) is not None), None)) is not None:
            tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(outer_rng), outer_local))
      return tk

  # make a copy so it does not mutate the input
  k = k.copy()

  # upcast float4 images, this must be early so we don't accidentally add locals before the upcast
  if IMAGE:
    for buf_index,buf in enumerate(k.bufs):
      if image_valid_dims(buf.src[0].dtype, buf.src[0].max_numel(), k.ren.target.arch):
        # part of is_expanded
        unit_stride_axes_mul_4 = [k.rngs.index(c) for c in k.bufs[buf_index].src[1].get_idx().split_uop(Ops.ADD) if
          c.op is Ops.RANGE and (c.vmax+1)%4 == 0]
        if len(unit_stride_axes_mul_4):
          if (axis:=unit_stride_axes_mul_4[0]) in k.upcastable_dims:
            k.apply_opt(Opt(OptOps.UPCAST, axis, 4))
          elif axis in k.unrollable_dims:
            k.apply_opt(Opt(OptOps.UNROLL, k.unrollable_dims.index(axis), 4))

  # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  if k.ren.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
    k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.ren.has_shared and \
    (mulop:=k.reduceop.src[0]).op is Ops.MUL and mulop.src[0].op is Ops.INDEX and mulop.src[1].op is Ops.INDEX:
    idx0, idx1 = mulop.src[0].src[1].get_idx(), mulop.src[1].src[1].get_idx()
    if k.ranges_of(AxisType.REDUCE):
      first_reduce_rng = k.ranges_of(AxisType.REDUCE)[0]
      if any(u is first_reduce_rng for u in idx0.split_uop(Ops.ADD)) and all(r in idx1.ranges for r in idx0.ranges):
        for global_idx in k.axes_of(AxisType.GLOBAL):
          if first_reduce_rng.src[0].divides(MV_THREADS_PER_ROW) is not None and k.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3:
              print(f"MATVEC: {k.full_shape=} {first_reduce_rng.render()} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
            try:
              if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            except KernelOptError: pass
            if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
            return k

  # are we grouping? (requires local shape support)
  if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 2048), False):
    for axis, sz in itertools.product((0, 1, 2), (16,)):
      try:
        k.apply_opt(Opt(OptOps.GROUPTOP, axis, sz))
        break
      except KernelOptError: pass

  # no more opt if we are grouping
  if k.group_for_reduces: return k

  # **** below this line need to be optional and benchmarked ****

  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  to_upcast: list[int] = []
  where_gate_rngs = {r for u in k.ast.backward_slice if u.op is Ops.WHERE for r in u.src[0].ranges}
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in k.upcastable_dims:
    # for Schedule, we check if the range is used in INDEX gates or WHERE gates
    is_masked = k.rngs[axis] in where_gate_rngs
    if k.full_shape[axis] <= 7 and is_masked and prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= 7 * 7:
      # upcasting a masked global axis moves that range out of the launch grid into each work-item
      # under IMAGE, skip the upcast unless enough global work-items remain after it to hide memory latency
      if IMAGE and k.axis_types[axis] is AxisType.GLOBAL:
        global_upcast = prod(k.full_shape[i] for i in to_upcast if k.axis_types[i] is AxisType.GLOBAL) * k.full_shape[axis]
        global_items_after = prod(k.full_shape[i] for i in k.axes_of(AxisType.GLOBAL)) // global_upcast
        if resolve(global_items_after < getenv("OCCUPANCY_FLOOR", 4096), False): continue
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))

  # potentially do more upcasts of non reduce axes based on a heuristic
  is_dsp = k.ren is not None and k.ren.target.device == "DSP"
  upcasted_axis: set[int] = set()
  while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and (k.upcast_size() < 2):
    xb_choices = []
    # consider all upcastable axes with 3 or 4 upcast (128 on the DSP)
    upcast_amounts = ([128] if not len(upcasted_axis) else []) if is_dsp else ([3,4] if k.reduceop is not None else [2,3,4])
    for axis, upcast_amount in itertools.product(k.upcastable_dims, upcast_amounts):
      # if we haven't upcasted it, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis in upcasted_axis or k.full_shape[axis]%upcast_amount != 0: continue
      rng = k.rngs[axis]
      if any(rng not in b.src[1].get_idx().backward_slice and all(r2 in b.src[1].get_idx().backward_slice
          for r2 in k.ranges_of(AxisType.UPCAST, AxisType.UNROLL)) for b in k.bufs):
        num_strides, sum_strides = 0, 0
        for b in k.bufs:
          idx = b.src[1].get_idx()
          if rng in idx.backward_slice: num_strides += 1
          for c in idx.split_uop(Ops.ADD):
            if c is rng: sum_strides += 1
            if c.op is Ops.MUL and c.src[0] is rng and c.src[1].op is Ops.CONST: sum_strides += c.src[1].arg
            if c.op is Ops.MUL and c.src[1] is rng and c.src[0].op is Ops.CONST: sum_strides += c.src[0].arg
        xb_choices.append((num_strides, sum_strides, axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
      k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
      upcasted_axis.add(xb_choices[0][2])
    else: break

  # if last reduce dim is small(ish), loop unroll the reduce
  # NOTE: this can fail on multireduce with mismatching dimensions, this is okay
  four_by_four_reduce = len(k.unrollable_dims) >= 2 and all(resolve(k.full_shape[x] == 4, False) for x in k.unrollable_dims[-2:])
  try:
    if k.unrollable_dims and (k.upcast_size() <= 4 or not k.axes_of(AxisType.UNROLL)) and (k.upcast_size() < 64):
      if (s:=k.full_shape[k.unrollable_dims[-1]]) <= 4:
        k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
        # if it's small, upcast a second reduce dimension too
        if k.unrollable_dims and s <= 3 and k.full_shape[k.unrollable_dims[-1]] <= 3:
          k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
      else:
        for splits in [4]:
          if k.full_shape[axis:=k.unrollable_dims[-1]]%splits == 0:
            k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, splits))
            break
  except KernelOptError: pass

  # if nothing at all is upcasted and it's easy to, do an upcast
  for splits in [4]:
    if not k.upcasted and k.upcastable_dims and k.full_shape[k.upcastable_dims[-1]] % splits == 0:
      k.apply_opt(Opt(OptOps.UPCAST, k.upcastable_dims[-1], splits))

  # **** local groups ****

  if k.ren.has_local:
    if NOLOCALS:
      k.apply_opt(Opt(OptOps.NOLOCALS))
    else:
      special_local = False
      if four_by_four_reduce and len(global_axes:=k.axes_of(AxisType.GLOBAL, AxisType.LOOP)) >= 3:
        lk, local_rngs = k.copy(), [k.rngs[x] for x in global_axes[-3:]]
        try:
          for rng, sz in zip(local_rngs, (8, 8, 4)):
            lk.apply_opt(Opt(OptOps.LOCAL, lk.rngs.index(rng), sz))
          k, special_local = lk, True
        except KernelOptError: pass
      # prioritize making expand axes local
      if not special_local:
        local_axis_ranking = [(any(k.rngs[axis] not in b.src[1].get_idx().backward_slice for b in k.bufs), axis) \
                                for axis in k.axes_of(AxisType.GLOBAL, AxisType.LOOP) if k.rngs[axis].src[0].op is Ops.CONST]
        to_local: list[tuple[int, int]] = []
        for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
          local_size = prod(sz for _, sz in to_local)
          local_sz: int|None = next((x for x in ([32] * (axis == 0) + [16,8,4,3,2]) if k.full_shape[axis] % x == 0 and local_size * x <= 128), None)
          if local_sz is not None: to_local.append((axis, local_sz))
        deleted_shape = 0
        for axis, local_sz in sorted(to_local[:3]):
          axis = axis - deleted_shape
          will_delete_shape = local_sz == k.full_shape[axis]
          k.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
          if will_delete_shape: deleted_shape += 1

      # Both 3x3 reduce axes are already fully unrolled above. Tile the exposed
      # spatial axes without changing reduction order.
      unroll_sizes = [k.full_shape[x] for x in k.axes_of(AxisType.UNROLL)]
      global_axes = k.axes_of(AxisType.GLOBAL, AxisType.LOOP)
      if unroll_sizes == [3, 3] and len(global_axes) >= 2:
        axis_size = k.full_shape[global_axes[1]]
        if axis_size <= 16: k.apply_opt(Opt(OptOps.UPCAST, 1, 0))
        elif axis_size % 4 == 0: k.apply_opt(Opt(OptOps.UPCAST, 1, 4))
        if len(global_axes) >= 4 and k.full_shape[global_axes[2]] == 4: k.apply_opt(Opt(OptOps.LOCAL, 2, 4))

      remaining_reduces = [k.full_shape[x] for x in k.unrollable_dims]
      if len(remaining_reduces) == 2 and remaining_reduces[0] == 6 and remaining_reduces[1] >= 16 and \
         remaining_reduces[1] % 4 == 0 and k.upcast_size() <= 16:
        k.apply_opt(Opt(OptOps.UNROLL, 1, 4))

  # **** threading ****

  if k.ren.has_threads and k.ren.global_max is not None:
    for threads in [32,16,12,8,6,5,4,3,2]:
      # Skip if too many threads. Heuristic: use about 128K ops per thread
      if threads > k.ren.global_max[0] or resolve(prod(k.full_shape) // (128 << 10) < threads): continue
      for axis in k.axes_of(AxisType.LOOP):
        if k.full_shape[axis] % threads == 0:
          try: k.apply_opt(Opt(OptOps.THREAD, axis, threads))
          except KernelOptError: pass
          break
      if k.applied_opts and k.applied_opts[-1].op is OptOps.THREAD: break

  return k
