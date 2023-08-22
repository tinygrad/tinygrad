import itertools
from tinygrad.helpers import DEBUG, prod, getenv, ImageDType, dtypes
from tinygrad.ops import ReduceOps, BinaryOps, UnaryOps, LazyOp
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.lazy import LazyBuffer

# auto opt is disabled
import time
from typing import Callable
def apply_opt(k, x):
  for axis, amt, typ in x:
    if axis is None or amt == 1: continue
    if typ == "R":
      typ = "U"
      axis += k.first_reduce
    assert k.full_shape[axis] % amt == 0, "no longer valid shift"
    if typ == "U":
      k.shift_to(axis, amt)
      k.upcast()
    elif typ == "L":
      k.shift_to(axis, amt, insert_before=k.first_reduce)
      k.local_dims += 1
  k.simplify_ones()

UPCASTS = [1,2,3,4,5,6,7,8]
LOCALS = [1,2,3,4,5,6,7,8,16,24,32]
def kernel_optimize_search(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, baseline):
  import nevergrad as ng
  def opt(x):
    try:
      k = create_k()
      k.process()
      apply_opt(k, x)
      prg = to_prg(k)
      first_tm = prg.exec(k.bufs, force_wait=True, optimizing=True)
      if baseline*5 < first_tm*1000: return first_tm*1000  # very slow
      tm = min([first_tm]+[prg.exec(k.bufs, force_wait=True, optimizing=True) for _ in range(2)])*1000
      return tm
    except Exception:
      if DEBUG >= 3:
        import traceback
        traceback.print_exc()
      return 10000_000   # 10000 seconds is infinity
  opts = []
  for i in range(k.first_reduce):
    # TODO: the upcast always happen first, you might want to reverse this?
    # TODO: the order of the locals might improve things too
    opts.append(ng.p.TransitionChoice([(i,s,"U") for s in UPCASTS if k.full_shape[i]%s == 0]))
    opts.append(ng.p.TransitionChoice([(i,s,"L") for s in LOCALS if k.full_shape[i]%s == 0]))
  for i in range(k.shape_len-k.first_reduce):
    opts.append(ng.p.TransitionChoice([(i,s,"R") for s in UPCASTS if k.full_shape[k.first_reduce+i]%s == 0]))
  if not opts: return "BASELINE"
  search_space = prod([len(x.choices) for x in opts])
  st = time.perf_counter()
  optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Tuple(*opts), budget=min(search_space, 200))
  recommendation = optimizer.minimize(opt)
  et = time.perf_counter() - st
  if DEBUG >= 1: print(f"optimizer({et:6.2f} s to search) space {search_space:8d} with tm {recommendation.loss:5.2f} ms vs baseline {baseline:5.2f} ms, a {baseline/recommendation.loss:5.2f}x gain : {k.colored_shape()}")
  return recommendation.value if recommendation.loss < baseline else "BASELINE"

# optimization
global_db = None
def kernel_optimize(k:Linearizer, create_k:Callable[[], Linearizer], to_prg):
  global global_db

  k.process()
  skey = str(k.key)

  if getenv("KOPT") == 2 and global_db is None:
    import shelve
    global_db = shelve.open("/tmp/kopt_cache")

  if global_db is not None and skey in global_db:
    choice = global_db[skey]
  elif any(not isinstance(x, int) for x in k.full_shape):
    # don't optimize variable shapes
    choice = "BASELINE"
  else:
    # get baseline
    def get_baseline():
      k = create_k()
      hand_coded_optimizations(k)
      prg = to_prg(k)
      return min([prg.exec(k.bufs, force_wait=True, optimizing=True) for _ in range(5)])*1000
    choice = kernel_optimize_search(k, create_k, to_prg, get_baseline())
    if global_db is not None:
      global_db[skey] = choice
      global_db.sync()

  if choice == "BASELINE": hand_coded_optimizations(k)
  else: apply_opt(k, choice)

def required_optimizations(k:Linearizer, early_only=False):
  for buf_index,buf in enumerate(k.bufs):
    unit_stride_axes_mul_4 = [i for i in k.sts[buf_index].unit_stride_axes(ignore_valid=True) if k.sts[buf_index].shape[i]%4 == 0]
    if (not early_only or buf in k.earlybufs) and k.bufs[buf_index].dtype.__class__ is ImageDType:
      assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {k.bufs[buf_index]}"
      if all(x < (k.shape_len-k.upcasted) for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in k.upcast_in_mid_reduce_axes:
        k.shift_to(unit_stride_axes_mul_4[0], 4)
        k.upcast()

def hand_coded_optimizations(k:Linearizer):
  k.process()

  # if there's images in the earlybufs, we have to make an axis the 4 loading one
  required_optimizations(k, early_only=True)

  # simplify
  k.simplify_ones()

  # should use HIP tensor cores?
  if getenv("TC", 1) != 0 and k.bufs[0].device == "HIP" and k.reduceop and k.reduceop.op == ReduceOps.SUM and \
      isinstance(k.reduceop.src[0], LazyOp) and k.reduceop.src[0].op == UnaryOps.CAST and \
      isinstance(k.reduceop.src[0].src[0], LazyOp) and k.reduceop.src[0].src[0].op == BinaryOps.MUL and \
      isinstance(k.reduceop.src[0].src[0].src[0], LazyBuffer) and isinstance(k.reduceop.src[0].src[0].src[1], LazyBuffer) and k.opts.has_local and \
      k.reduceop.src[0].src[0].src[0].dtype == dtypes.half and k.reduceop.src[0].src[0].src[1].dtype == dtypes.half:
    # HIP tensor cores are 16x16x16
    buf0 = k.bufs.index(k.reduceop.src[0].src[0].src[0])
    buf1 = k.bufs.index(k.reduceop.src[0].src[0].src[1])
    buf0_strides = k.sts[buf0].real_strides()
    buf1_strides = k.sts[buf1].real_strides()
    axis_buf0 = [(i,k.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides) if s == 0 and k.full_shape[i]%16 == 0 and i < k.first_reduce]
    axis_buf1 = [(i,k.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides) if s == 0 and k.full_shape[i]%16 == 0 and i < k.first_reduce]
    if axis_buf0 and axis_buf1 and k.full_shape[k.first_reduce]%8 == 0 and (k.shape_len-k.first_reduce) == 1:
      if DEBUG >= 3: print("HIP TENSOR CORES", axis_buf0, axis_buf1)
      k.use_tensor_cores = getenv("TC", 1) == 1  # TC=2 will do the shape ops without the WMMA
      k.reverse_upcast_dir = True

      # TODO: select axis in smart way
      s0, s1 = axis_buf0[-1][0], axis_buf1[-1][0]
      global_count = k.first_reduce

      # upcast first
      if k.full_shape[k.first_reduce] > 16: k.shift_to(k.first_reduce, 16)
      k.upcast()

      # 2 locals
      k.shift_to(s1, 16, insert_before=k.first_reduce)  # axis 2
      k.shift_to(s0, 16, insert_before=k.first_reduce)  # axis 3
      k.local_dims += 1

      # output shape
      k.shift_to(k.first_reduce-2, 8)
      k.upcast()

      # split local dim
      k.shift_to(k.first_reduce-1, 8, insert_before=k.first_reduce)  # axis 3

      # final global upcast
      for ax in [s1, s0]:
        for upc in [4,3,2]:
          if k.full_shape[ax]%upc == 0:
            k.shift_to(ax, upc)
            k.upcast()
            break

      # alias buffer
      alias_pattern = [0]*global_count + [0,0,1] + [0] * (k.shape_len-k.upcasted-k.first_reduce) + [2,3] + [0]*(k.upcasted-2)
      k.alias_buffer(buf0, alias_pattern)
      k.alias_buffer(buf1, alias_pattern)

      # two fake locals
      if k.use_tensor_cores:
        k.local_dims += 2
        k.exclude_local_upcast += 2

      # early exit
      return

  # should use METAL tensor cores?
  # first, confirm it's a straightforward mulacc on a device with real locals
  tensor_cores_allowed = getenv("TC", 1) != 0 and (getenv("TC", 1) == 2 or (k.bufs[0].device == "METAL" and getenv("CI", "") != "true"))
  if tensor_cores_allowed and k.reduceop and k.reduceop.op == ReduceOps.SUM and \
      isinstance(k.reduceop.src[0], LazyOp) and k.reduceop.src[0].op == BinaryOps.MUL and \
      isinstance(k.reduceop.src[0].src[0], LazyBuffer) and isinstance(k.reduceop.src[0].src[1], LazyBuffer) and k.opts.has_local:
    # METAL tensor cores are 8x8x8, with 2 elements per thread in the 32 thread warp
    buf0 = k.bufs.index(k.reduceop.src[0].src[0])
    buf1 = k.bufs.index(k.reduceop.src[0].src[1])
    buf0_strides = k.sts[buf0].real_strides()
    buf1_strides = k.sts[buf1].real_strides()
    axis_buf0 = [(i,k.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides) if s == 0 and k.full_shape[i]%8 == 0 and i < k.first_reduce]
    axis_buf1 = [(i,k.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides) if s == 0 and k.full_shape[i]%8 == 0 and i < k.first_reduce]
    if axis_buf0 and axis_buf1 and k.full_shape[k.first_reduce]%8 == 0 and (k.shape_len-k.first_reduce) == 1:
      if DEBUG >= 3: print("METAL TENSOR CORES", axis_buf0, axis_buf1)
      k.use_tensor_cores = getenv("TC", 1) == 1  # TC=2 will do the shape ops without the WMMA

      # TODO: select axis in smart way
      s0, s1 = axis_buf0[-1][0], axis_buf1[-1][0]
      global_count = k.first_reduce

      # upcast first
      if k.full_shape[k.first_reduce] > 8: k.shift_to(k.first_reduce, 8)
      k.upcast()

      # 2 locals
      k.shift_to(s1, 8, insert_before=k.first_reduce)  # axis 2
      k.shift_to(s0, 8, insert_before=k.first_reduce)  # axis 3

      # permuted+upcast for tensor cores
      k.shift_to(global_count, 4, insert_before=k.first_reduce)
      k.shift_to(global_count+1, 4, insert_before=k.first_reduce)
      k.shift_to(k.first_reduce-1, 2)
      k.upcast()

      # final global upcast
      for ax in [s1, s0]:
        for upc in [4,3,2]:
          if k.full_shape[ax]%upc == 0:
            k.shift_to(ax, upc)
            k.upcast()
            break

      # alias buffer
      k.local_dims = k.first_reduce - global_count
      alias_pattern = [0]*global_count + [2] * k.local_dims + [0] * (k.shape_len-k.upcasted-k.first_reduce) + [1,1] + [3] * (k.upcasted-2)
      k.alias_buffer(buf0, alias_pattern)
      k.alias_buffer(buf1, alias_pattern)

      # very late upcast to run group at the same time. only if actually using real tensor cores, otherwise local isn't a simdgroup
      if k.use_tensor_cores and k.full_shape[s0] % 2 == 0:
        k.shift_to(s0, 2, insert_before=k.first_reduce-k.local_dims)
        k.local_dims += 1
        k.exclude_local_upcast += 1

      # early exit
      return

  if k.opts.has_local and all(isinstance(s, int) for s in k.sts[0].shape[:k.first_reduce]):
    # are we grouping? (requires local shape support)
    if not k.float4_axis(0) and k.first_reduce <= 2 and k.first_reduce + 1 <= k.shape_len and prod(k.sts[0].shape[:k.first_reduce]) <= 2048:
      # TODO: use 1024 if it's allowed in a smarter way
      for sz in (([256, 16]) if prod(k.sts[0].shape[:k.first_reduce]) <= 32 else [16]):
        if all(st.shape[k.first_reduce] % sz == 0 or st.shape[k.first_reduce] == 1 for st in k.sts):
          k.shift_to(k.first_reduce, sz, top=True, insert_before=k.first_reduce + len(k.group_for_reduce))
          k.group_for_reduce.append(sz)
          break

    # are we upcasting in mid reduce? (only for images)
    if k.bufs[0].dtype.name.startswith('image') and not k.float4_axis(0) and k.group_for_reduce and k.first_reduce <= 2 and prod(k.sts[0].shape) > 1:
      axes = k.sts[0].unit_stride_axes()
      assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
      if k.sts[0].shape[axes[0]]%4 == 0:
        k.shift_to(axes[0], 4, insert_before=k.first_reduce + len(k.group_for_reduce))   # insert at the end of the grouped axis
        k.group_for_reduce.append(4)

  # now do everything required
  required_optimizations(k)

  # simplify (sets first_reduce)
  k.simplify_ones()

  # use more opencl indexing if the output buffer is an image and we have room
  if k.bufs[0].dtype.name.startswith('image') and k.first_reduce+len(k.group_for_reduce) < 3:
    base_shape = k.bufs[0].dtype.shape
    if (base_shape[0]*base_shape[1]) % k.sts[0].shape[0] == 0 and k.sts[0].shape[0]//base_shape[0] != 0:
      if DEBUG >= 4: print("split opencl", base_shape, k.sts[0].shape)
      k.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
      k.simplify_ones()

  # no more opt if we are grouping
  if k.group_for_reduce: return

  # **** below this line need to be optional and benchmarked ****

  # potentially do more upcasts of non reduce axes based on a heuristic
  upcasted_axis = set()
  while prod(k.sts[0].shape[:k.first_reduce]) >= 1024:
    xb_choices = []
    for axis, upcast_amount in itertools.product(range(k.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
      # if we haven't upcasted it, it's not symbolic, it mods, and some buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis not in upcasted_axis and isinstance(k.full_shape[axis], int) and k.full_shape[axis]%upcast_amount == 0 and any(k.sts[buf_index].views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in k.upcasted_axis(buf_index)) for buf_index in range(len(k.sts))):
        xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in k.sts), sum(st.views[-1].strides[axis] for st in k.sts), axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
      k.shift_to(xb_choices[0][2], amount=xb_choices[0][3])
      k.upcast()
      k.simplify_ones()
      upcasted_axis.add(xb_choices[0][2])
    else:
      break

  # if last dim is small(ish) and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast. NOTE: careful, this has broken VALIDHACKS
  if k.first_reduce < (k.shape_len-k.upcasted) and (len(list(k.shape_offsets(k.full_buf_index))) <= 4 or not any(r for _,_,r in k.upcasted_axis(k.full_buf_index))):
    if (s:=k.full_unupcasted_shape[-1]) <= 32 and isinstance(s, int):  # NOTE: cannot loop unroll symbolic axis
      k.upcast()
      # if it's small, upcast a second reduce dimension too
      if k.first_reduce < (k.shape_len-k.upcasted) and s <= 3 and k.full_unupcasted_shape[-1] <= 3: k.upcast()
    else:
      for splits in [4]:
        if k.full_unupcasted_shape[-1]%splits == 0:
          k.shift_to(len(k.full_unupcasted_shape)-1, splits, insert_before=len(k.full_unupcasted_shape))
          k.upcast()
          break

  # if nothing at all is upcasted and it's easy to, do an upcast
  # TODO: this is breaking the tests
  for splits in [4]:
    if k.upcasted == 0 and k.full_unupcasted_shape and k.full_unupcasted_shape[-1] % splits == 0:
      k.shift_to(len(k.full_unupcasted_shape)-1, splits, insert_before=len(k.full_unupcasted_shape))
      k.upcast()

  # **** local groups ****

  if k.opts.has_local:
    for axis in range(k.first_reduce - k.local_dims - 1, -1, -1):
      local_size = prod(k.full_shape[k.first_reduce-k.local_dims:k.first_reduce])
      if k.full_shape[axis] == 1: continue
      last_try = k.local_dims == 0 and axis == 0
      if any(k.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(k.sts))) or last_try:
        for sz in [x for x in (([32] if last_try else []) + [16,8,4,3]) if k.full_shape[axis] % x == 0 and local_size*x <= 128]:
          k.shift_to(axis, sz, insert_before=k.first_reduce-k.local_dims)
          k.local_dims += 1
          break
      if k.local_dims >= 3: break
  k.simplify_ones()
