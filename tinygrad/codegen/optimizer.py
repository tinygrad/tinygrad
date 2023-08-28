from typing import Tuple, List, cast
import itertools, math
from tinygrad.helpers import DEBUG, prod, getenv, ImageDType, dtypes
from tinygrad.ops import ReduceOps, BinaryOps, UnaryOps, LazyOp
from tinygrad.codegen.kernel import Kernel, LocalBuffer
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker, View

class OptimizedKernel(Kernel):
  def process(self) -> None:
    if hasattr(self, "sts"): return   # already processed
    super().process()

    # move all reduce axes to the end
    reduce = list(enumerate(zip(self.full_shape, self.sts[0].shape)))
    permute = tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    self.reshape_and_permute(None, permute)

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

  # ******************** base simplifiers ********************

  # apply reshape and permute to all shapetrackers
  def reshape_and_permute(self, new_shape_fxn, axis):
    for st in self.sts:
      if new_shape_fxn is not None: st.reshape(tuple(new_shape_fxn(st.shape)))
      if axis is not None: st.permute(tuple(axis))

  # drops the final dimension
  def upcast(self):
    assert self.full_shape[-1] != 1, "can't upcast a dimension with size 1"
    self.upcasted += 1

  # axis : the axis to pull from
  # amount : the amount to take
  # top : if you want to pull that amount from the top
  # insert_before : place to insert the new stuff
  def shift_to(self, axis, amount, top=False, insert_before=None):
    if insert_before is None: insert_before = self.shape_len
    move_axis = axis if top else axis+1
    if move_axis < insert_before: insert_before += 1
    self.reshape_and_permute(
      lambda x: list(x[0:axis]) + (([amount, x[axis]//amount] if top else [x[axis]//amount, amount]) if x[axis] > 1 else [1,1]) + list(x[axis+1:]),
      [i for i in range(insert_before) if i != move_axis] + [move_axis] + [i for i in range(insert_before, self.shape_len+1) if i != move_axis])

  # ******************** complex simplifiers ********************

  def simplify_ones(self):
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    if self.shape_len == 0: return
    all_ones = [s==1 for s in self.full_shape]
    self.local_dims -= sum(all_ones[self.first_reduce-self.local_dims:self.first_reduce])
    self.upcasted -= sum(all_ones[self.shape_len-self.upcasted:])
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)

  def simplify_merge_adjacent(self):
    if self.shape_len == 0: return
    shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]

    # merge dimensions if we can, multi get_shape_strides
    # TODO: does this always preserve the reduce dimension, NO
    # TODO: move this into shapetracker, with tests!
    rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for j in range(len(shapes)):
        # TODO: added the always mergeability of 1s, is this right? if so, add to shapetracker in the 1 case
        can_merge.append(strides[j][i] is not None and ((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*cast(int, strides[j][i])) or (strides[j][i] == 0 and rets[j][-1][1] == 0)))
      # more can merge than this
      mergeable = all(can_merge) and i != self.first_reduce
      for j in range(len(shapes)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
        else: rets[j].append((shapes[j][i], strides[j][i]))

    # do the reshapes
    for i,x in enumerate(rets): self.sts[i].reshape(tuple([y[0] for y in x]))

  # ******************** GPU simplifiers ********************
  def _limit_size(self, x: Tuple[int], max_size: List) -> Tuple[int, ...]:
    new_shape,dims = list(x), len(x)
    for i in range(dims):
      next_idx = (i + 1) % dims
      while new_shape[i] > max_size[i]:
        new_shape[i] = new_shape[i] // 2
        if (new_shape[next_idx] <= max_size[next_idx]):
          new_shape[next_idx] = new_shape[next_idx] * 2
        else:
          next_idx = (next_idx + 1) % dims
          new_shape[next_idx] = new_shape[next_idx] * 2
    return tuple(new_shape)

  def limit_global_dims(self, limit: int, global_max: List[int], local_max: List[int]):
    # sometimes, there's more dimensions than len(self.lang.gid).
    # compact all the dimensions into the first
    # NOTE: this might make multiview shapetrackers
    if (self.first_reduce-self.local_dims) > limit:
      num_to_merge = ((self.first_reduce-self.local_dims) - limit)+1
      self.reshape_and_permute(lambda x: (prod(x[0:num_to_merge]),)+x[num_to_merge:], None)
      if DEBUG >= 3: print("reshaped to", self.full_shape, "due to too many global dimensions")
    # Check the global allocation limit, current the global_size will be flipped during codegen
    # and then padded right with 1s if its length < 3 which makes this part a bit awkward to write
    global_dims = self.first_reduce-self.local_dims
    if global_dims > 0:
      if global_max:
        tmp = global_max[:global_dims] + (local_max[:self.local_dims] if local_max else [])
        if max(global_max) < max(self.full_shape[:global_dims]): self.reshape_and_permute(lambda x: self._limit_size(x, tmp + [math.inf] * (len(self.full_shape)-len(tmp))), None)
        assert max(global_max) >= max(self.full_shape[:global_dims]), f"device max allocation {max(self.full_shape[:global_dims])} exceeds global dim maximum {max(global_max)}"
      for i in range(global_dims-1):
        if self.full_shape[i] > global_max[i]:
          order = list(range(len(self.full_shape)))
          order[i], order[global_dims-1] = order[global_dims-1], order[i]
          self.reshape_and_permute(None, order)
          if DEBUG >= 3: print("permuted global dim", order, "due to allocation exceeds global limit")

  def alias_buffer(self, i, pattern):
    assert len(pattern) == len(self.sts[i].shape), f"must include a pattern for each shape {pattern} {self.sts[i].shape}"

    bst = 1
    real_strides = self.sts[i].real_strides()
    shp, stride = [(s if p != 0 else 1) for s,p in zip(self.sts[i].shape, pattern)], [0]*len(pattern)
    for priority in range(1, max(pattern)+1):  # priority. 0 is non local and ignored
      for j,p in enumerate(pattern):
        if priority == p and real_strides[j] != 0:
          stride[j] = bst
          bst *= shp[j]

    self.sts.append(ShapeTracker(tuple(shp), [View(tuple(shp), tuple(stride))]))
    self.bufs.append(LocalBuffer(name=f"ldata{i}", size=self.sts[-1].size()))
    if DEBUG >= 4: print("aliasing buffer", self.sts[i])
    self.local_alias[i] = self.bufs[-1]

  # ******************** high level optimizers ********************

  def apply_auto_opt(self, x):
    for axis, amt, typ in x:
      if axis is None or amt == 1: continue
      if typ == "R":
        typ = "U"
        axis += self.first_reduce
      assert self.full_shape[axis] % amt == 0, "no longer valid shift"
      if typ == "U":
        self.shift_to(axis, amt)
        self.upcast()
      elif typ == "L":
        self.shift_to(axis, amt, insert_before=self.first_reduce)
        self.local_dims += 1
    self.simplify_ones()

  def required_optimizations(self, early_only=False):
    for buf_index,buf in enumerate(self.bufs):
      unit_stride_axes_mul_4 = [i for i in self.sts[buf_index].unit_stride_axes(ignore_valid=True) if self.sts[buf_index].shape[i]%4 == 0]
      if (not early_only or buf in self.earlybufs) and self.bufs[buf_index].dtype.__class__ is ImageDType:
        assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {self.bufs[buf_index]}"
        if all(x < (self.shape_len-self.upcasted) for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:
          self.shift_to(unit_stride_axes_mul_4[0], 4)
          self.upcast()

  def hand_coded_optimizations(self):
    self.process()

    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    self.required_optimizations(early_only=True)

    # simplify
    self.simplify_ones()

    # should use HIP tensor cores?
    if getenv("TC", 1) != 0 and self.bufs[0].device == "HIP" and self.reduceop and self.reduceop.op == ReduceOps.SUM and \
        isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == UnaryOps.CAST and \
        isinstance(self.reduceop.src[0].src[0], LazyOp) and self.reduceop.src[0].src[0].op == BinaryOps.MUL and \
        isinstance(self.reduceop.src[0].src[0].src[0], LazyBuffer) and isinstance(self.reduceop.src[0].src[0].src[1], LazyBuffer) and self.opts.has_local and \
        self.reduceop.src[0].src[0].src[0].dtype == dtypes.half and self.reduceop.src[0].src[0].src[1].dtype == dtypes.half:
      # HIP tensor cores are 16x16x16
      buf0 = self.bufs.index(self.reduceop.src[0].src[0].src[0])
      buf1 = self.bufs.index(self.reduceop.src[0].src[0].src[1])
      buf0_strides = self.sts[buf0].real_strides()
      buf1_strides = self.sts[buf1].real_strides()
      axis_buf0 = [(i,self.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides) if s == 0 and self.full_shape[i]%16 == 0 and i < self.first_reduce]
      axis_buf1 = [(i,self.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides) if s == 0 and self.full_shape[i]%16 == 0 and i < self.first_reduce]
      if axis_buf0 and axis_buf1 and self.full_shape[self.first_reduce]%8 == 0 and (self.shape_len-self.first_reduce) == 1:
        if DEBUG >= 3: print("HIP TENSOR CORES", axis_buf0, axis_buf1)
        self.use_tensor_cores = getenv("TC", 1) == 1  # TC=2 will do the shape ops without the WMMA
        self.reverse_upcast_dir = True

        # TODO: select axis in smart way
        s0, s1 = axis_buf0[-1][0], axis_buf1[-1][0]
        global_count = self.first_reduce

        # upcast first
        if self.full_shape[self.first_reduce] > 16: self.shift_to(self.first_reduce, 16)
        self.upcast()

        # 2 locals
        self.shift_to(s1, 16, insert_before=self.first_reduce)  # axis 2
        self.shift_to(s0, 16, insert_before=self.first_reduce)  # axis 3
        self.local_dims += 1

        # output shape
        self.shift_to(self.first_reduce-2, 8)
        self.upcast()

        # split local dim
        self.shift_to(self.first_reduce-1, 8, insert_before=self.first_reduce)  # axis 3

        # final global upcast
        for ax in [s1, s0]:
          for upc in [4,3,2]:
            if self.full_shape[ax]%upc == 0:
              self.shift_to(ax, upc)
              self.upcast()
              break

        # alias buffer
        alias_pattern = [0]*global_count + [0,0,1] + [0] * (self.shape_len-self.upcasted-self.first_reduce) + [2,3] + [0]*(self.upcasted-2)
        self.alias_buffer(buf0, alias_pattern)
        self.alias_buffer(buf1, alias_pattern)

        # two fake locals
        if self.use_tensor_cores:
          self.local_dims += 2
          self.exclude_local_upcast += 2

        # early exit
        return

    # should use METAL tensor cores?
    # first, confirm it's a straightforward mulacc on a device with real locals
    tensor_cores_allowed = getenv("TC", 1) != 0 and (getenv("TC", 1) == 2 or (self.bufs[0].device == "METAL" and getenv("CI", "") != "true"))
    if tensor_cores_allowed and self.reduceop and self.reduceop.op == ReduceOps.SUM and \
        isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == BinaryOps.MUL and \
        isinstance(self.reduceop.src[0].src[0], LazyBuffer) and isinstance(self.reduceop.src[0].src[1], LazyBuffer) and self.opts.has_local:
      # METAL tensor cores are 8x8x8, with 2 elements per thread in the 32 thread warp
      buf0 = self.bufs.index(self.reduceop.src[0].src[0])
      buf1 = self.bufs.index(self.reduceop.src[0].src[1])
      buf0_strides = self.sts[buf0].real_strides()
      buf1_strides = self.sts[buf1].real_strides()
      axis_buf0 = [(i,self.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides) if s == 0 and self.full_shape[i]%8 == 0 and i < self.first_reduce]
      axis_buf1 = [(i,self.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides) if s == 0 and self.full_shape[i]%8 == 0 and i < self.first_reduce]
      if axis_buf0 and axis_buf1 and self.full_shape[self.first_reduce]%8 == 0 and (self.shape_len-self.first_reduce) == 1:
        if DEBUG >= 3: print("METAL TENSOR CORES", axis_buf0, axis_buf1)
        self.use_tensor_cores = getenv("TC", 1) == 1  # TC=2 will do the shape ops without the WMMA

        # TODO: select axis in smart way
        s0, s1 = axis_buf0[-1][0], axis_buf1[-1][0]
        global_count = self.first_reduce

        # upcast first
        if self.full_shape[self.first_reduce] > 8: self.shift_to(self.first_reduce, 8)
        self.upcast()

        # 2 locals
        self.shift_to(s1, 8, insert_before=self.first_reduce)  # axis 2
        self.shift_to(s0, 8, insert_before=self.first_reduce)  # axis 3

        # permuted+upcast for tensor cores
        self.shift_to(global_count, 4, insert_before=self.first_reduce)
        self.shift_to(global_count+1, 4, insert_before=self.first_reduce)
        self.shift_to(self.first_reduce-1, 2)
        self.upcast()

        # final global upcast
        for ax in [s1, s0]:
          for upc in [4,3,2]:
            if self.full_shape[ax]%upc == 0:
              self.shift_to(ax, upc)
              self.upcast()
              break

        # alias buffer
        self.local_dims = self.first_reduce - global_count
        alias_pattern = [0]*global_count + [2] * self.local_dims + [0] * (self.shape_len-self.upcasted-self.first_reduce) + [1,1] + [3] * (self.upcasted-2)
        self.alias_buffer(buf0, alias_pattern)
        self.alias_buffer(buf1, alias_pattern)

        # very late upcast to run group at the same time. only if actually using real tensor cores, otherwise local isn't a simdgroup
        if self.use_tensor_cores and self.full_shape[s0] % 2 == 0:
          self.shift_to(s0, 2, insert_before=self.first_reduce-self.local_dims)
          self.local_dims += 1
          self.exclude_local_upcast += 1

        # early exit
        return

    if self.opts.has_local and all(isinstance(s, int) for s in self.sts[0].shape[:self.first_reduce]):
      # are we grouping? (requires local shape support)
      if not self.float4_axis(0) and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
        # TODO: use 1024 if it's allowed in a smarter way
        for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
          if all(st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts):
            self.shift_to(self.first_reduce, sz, top=True, insert_before=self.first_reduce + len(self.group_for_reduce))
            self.group_for_reduce.append(sz)
            break

      # are we upcasting in mid reduce? (only for images)
      if self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:
        axes = self.sts[0].unit_stride_axes()
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
        if self.sts[0].shape[axes[0]]%4 == 0:
          self.shift_to(axes[0], 4, insert_before=self.first_reduce + len(self.group_for_reduce))   # insert at the end of the grouped axis
          self.group_for_reduce.append(4)

    # now do everything required
    self.required_optimizations()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # use more opencl indexing if the output buffer is an image and we have room
    if self.bufs[0].dtype.name.startswith('image') and self.first_reduce+len(self.group_for_reduce) < 3:
      base_shape = self.bufs[0].dtype.shape
      if (base_shape[0]*base_shape[1]) % self.sts[0].shape[0] == 0 and self.sts[0].shape[0]//base_shape[0] != 0:
        if DEBUG >= 4: print("split opencl", base_shape, self.sts[0].shape)
        self.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
        self.simplify_ones()

    # no more opt if we are grouping
    if self.group_for_reduce: return

    # no more opt if there's non ints in any shapes
    # TODO: this is due to a bug. repro by commenting this one while running GPT-2 with the JIT
    if self.has_variable_shape(): return

    # **** below this line need to be optional and benchmarked ****

    # potentially do more upcasts of non reduce axes based on a heuristic
    upcasted_axis = set()
    while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if we haven't upcasted it, it's not symbolic, it mods, and some buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
        if axis not in upcasted_axis and isinstance(self.full_shape[axis], int) and self.full_shape[axis]%upcast_amount == 0 and any(self.sts[buf_index].views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index in range(len(self.sts))):
          xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in self.sts), sum(st.views[-1].strides[axis] for st in self.sts), axis, upcast_amount))
      if xb_choices:
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.shift_to(xb_choices[0][2], amount=xb_choices[0][3])
        self.upcast()
        self.simplify_ones()
        upcasted_axis.add(xb_choices[0][2])
      else:
        break

    # if last dim is small(ish) and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast. NOTE: careful, this has broken VALIDHACKS
    if self.first_reduce < (self.shape_len-self.upcasted) and (len(list(self.shape_offsets(self.full_buf_index))) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))):
      if (s:=self.full_unupcasted_shape[-1]) <= 32 and isinstance(s, int):  # NOTE: cannot loop unroll symbolic axis
        self.upcast()
        # if it's small, upcast a second reduce dimension too
        if self.first_reduce < (self.shape_len-self.upcasted) and s <= 3 and self.full_unupcasted_shape[-1] <= 3: self.upcast()
      else:
        for splits in [4]:
          if self.full_unupcasted_shape[-1]%splits == 0:
            self.shift_to(len(self.full_unupcasted_shape)-1, splits, insert_before=len(self.full_unupcasted_shape))
            self.upcast()
            break

    # if nothing at all is upcasted and it's easy to, do an upcast
    # TODO: this is breaking the tests
    for splits in [4]:
      if self.upcasted == 0 and self.full_unupcasted_shape and self.full_unupcasted_shape[-1] % splits == 0:
        self.shift_to(len(self.full_unupcasted_shape)-1, splits, insert_before=len(self.full_unupcasted_shape))
        self.upcast()

    # **** local groups ****

    if self.opts.has_local:
      for axis in range(self.first_reduce - self.local_dims - 1, -1, -1):
        local_size = prod(self.full_shape[self.first_reduce-self.local_dims:self.first_reduce])
        if self.full_shape[axis] == 1: continue
        last_try = self.local_dims == 0 and axis == 0
        if any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))) or last_try:
          for sz in [x for x in (([32] if last_try else []) + [16,8,4,3]) if self.full_shape[axis] % x == 0 and local_size*x <= 128]:
            self.shift_to(axis, sz, insert_before=self.first_reduce-self.local_dims)
            self.local_dims += 1
            break
        if self.local_dims >= 3: break
    self.simplify_ones()
