import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS, TC_OPT, TC_SELECT, USE_TC, AMX, AVX512
from tinygrad.dtype import ImageDType
from tinygrad.uop.ops import Ops, resolve, AxisType
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
    # skip hand-coded TC opts if AMX, upcasting will make kernel slower
    if good_tc_opt and not AMX:
      if rngs is not None:
        for tc_dim in [1,0]: # attempt to upcast M and N
          szs = [sz for sz in [5,4,3,2] if rngs[tc_dim].src[0].divides(sz) is not None]
          if szs:
            # set it to the replaced range
            rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
        if (szs := [sz for sz in [4,2] if rngs[0].src[0].divides(sz) is not None]): # attempt to local N
          tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), szs[0]))
      return tk

  # make a copy so it does not mutate the input
  k = k.copy()

  # upcast float4 images, this must be early so we don't accidentally add locals before the upcast
  for buf_index,buf in enumerate(k.bufs):
    if isinstance(buf.src[0].dtype, ImageDType):
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

  # CPU-specific: larger upcast tiles are better than K-unrolling for cache efficiency
  # BEAM search found that 64 accumulators (e.g., 4x16 tile) beats 16 accumulators + UNROLL=4
  is_cpu_with_threads = k.ren.has_threads and not k.ren.has_local
  cpu_matmul = is_cpu_with_threads and k.reduceop is not None  # scope CPU opts to matmul only
  # CPU matvec: when output is essentially 1D (only one non-trivial dim), it's memory-bound
  # Note: output_shape may have trailing 1s (batch dims), so count non-one dimensions
  cpu_matvec = cpu_matmul and sum(s != 1 for s in k.output_shape) == 1

  cpu_upcast_limit = 8 if cpu_matvec else (64 if cpu_matmul else 32)

  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  to_upcast: list[int] = []
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in k.upcastable_dims:
    # for Schedule, we check if the range is used in INDEX gates or WHERE gates
    is_masked = any(any(o is k.rngs[axis] for o in u.src[0].backward_slice) for u in k.ast.backward_slice if u.op is Ops.WHERE)
    if k.full_shape[axis] <= 7 and is_masked and prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= 7 * 7:
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))

  # potentially do more upcasts of non reduce axes based on a heuristic
  is_dsp = k.ren is not None and k.ren.device == "DSP"
  upcasted_axis: set[int] = set()
  # CPU: try larger upcast amounts first (8,4) to create bigger tiles for better cache efficiency
  # BEAM search found that 4x16 tiles beat 4x4 tiles + UNROLL for CPU matmul
  # CPU matvec: use more accumulators [16,12,8] for ILP to hide memory latency (like BLAS/MKL)
  # CPU matvec: lower threshold (16) to ensure small outputs still get ILP - memory-bound ops need latency hiding
  upcast_amounts = ([128] if not len(upcasted_axis) else []) if is_dsp else ([8,4] if cpu_matvec else ([8,4,3] if cpu_matmul else [3,4]))
  upcast_threshold = 16 if cpu_matvec else 1024
  while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= upcast_threshold) and (k.upcast_size() < cpu_upcast_limit):
    xb_choices = []
    # consider all upcastable axes with given upcast amounts
    for axis, upcast_amount in itertools.product(k.upcastable_dims, upcast_amounts):
      # check if axis can be upcasted (mods evenly)
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
        # For CPU matmul, negate upcast_amount to prefer larger tiles
        sort_key = (num_strides, sum_strides, axis, -upcast_amount if cpu_matmul else upcast_amount)
        xb_choices.append((sort_key, axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices, key=lambda x: x[0])
      if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
      k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][1], xb_choices[0][2]))
      upcasted_axis.add(xb_choices[0][1])
    else: break

  # if last reduce dim is small(ish), loop unroll the reduce
  # NOTE: this can fail on multireduce with mismatching dimensions, this is okay
  # CPU with large upcast tiles: skip UNROLL as it doesn't help with cache efficiency
  try:
    skip_unroll = cpu_matmul and k.upcast_size() >= 32
    if not skip_unroll and k.unrollable_dims and (k.upcast_size() <= 4 or not k.axes_of(AxisType.UNROLL)) and (k.upcast_size() < 64):
      if (s:=k.full_shape[k.unrollable_dims[-1]]) <= 32:
        k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
        # if it's small, upcast a second reduce dimension too
        if k.unrollable_dims and s <= 3 and k.full_shape[k.unrollable_dims[-1]] <= 3:
          k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
      else:
        # CPU matvec: try larger UNROLL for more ILP in the reduce loop
        unroll_splits = [16, 8, 4] if cpu_matvec else [4]
        for splits in unroll_splits:
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
      # prioritize making expand axes local
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

  # **** threading ****

  if k.ren.has_threads and k.ren.global_max is not None:
    for threads in [32,16,12,8,6,5,4,3,2]:
      # Skip if too many threads. Heuristic: use about 128K ops per thread
      if threads > k.ren.global_max[0] or resolve(prod(k.full_shape) // (128 << 10) < threads): continue
      # For CPU matmul, try higher axes first for better cache locality
      loop_axes = list(reversed(k.axes_of(AxisType.LOOP))) if cpu_matmul else k.axes_of(AxisType.LOOP)
      for axis in loop_axes:
        if k.full_shape[axis] % threads == 0:
          try: k.apply_opt(Opt(OptOps.THREAD, axis, threads))
          except KernelOptError: pass
          break
      if k.applied_opts and k.applied_opts[-1].op is OptOps.THREAD: break

  return k
