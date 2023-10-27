from __future__ import annotations
from typing import Tuple, List, cast, Optional
from dataclasses import dataclass
import itertools, math, os
from tinygrad.helpers import DEBUG, prod, getenv, ImageDType
from tinygrad.ops import ReduceOps, BinaryOps, UnaryOps, LazyOp, BufferOps
from tinygrad.codegen.kernel import Kernel, LocalBuffer, LinearizerOptions, tensor_cores
from tinygrad.shape.shapetracker import ShapeTracker, get_contraction
from tinygrad.shape.view import View, strides_for_shape
from enum import Enum, auto

class OptOps(Enum):
  UPCAST = auto(); UPCASTMID = auto(); UNROLL = auto(); LOCAL = auto(); LASTLOCAL = auto(); GROUP = auto(); GROUPTOP = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

@dataclass(frozen=True, order=True)
class Opt:
  op: OptOps
  axis: int
  amt: int
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, amt={self.amt})"

class OptimizedKernel(Kernel):
  def __init__(self, ast:LazyOp, opts:Optional[LinearizerOptions]=None):
    super().__init__(ast, opts)

    # move all reduce axes to the end
    reduce = list(enumerate(zip(self.full_shape, self.sts[0].shape)))
    permute = tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    self.reshape_and_permute(None, permute)

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

    self.applied_opts: List[Opt] = []

  # ******************** base simplifiers ********************

  # apply reshape and permute to all shapetrackers
  def reshape_and_permute(self, new_shape_fxn, axis):
    new_sts = []
    for st in self.sts:
      if new_shape_fxn is not None: st = st.reshape(tuple(new_shape_fxn(st.shape)))
      if axis is not None: st = st.permute(tuple(axis))
      new_sts.append(st)
    self.sts = new_sts

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

  def simplify_ones(self) -> bool:
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    if self.shape_len == 0: return False
    all_ones = [s==1 for s in self.full_shape]
    self.local_dims -= sum(all_ones[self.first_reduce-self.local_dims:self.first_reduce])
    self.upcasted -= sum(all_ones[self.shape_len-self.upcasted:])
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)
    return any(all_ones)

  def simplify_merge_adjacent(self):
    if self.shape_len == 0: return
    shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]

    # if it's an image, insert fake strides such that this fusion doesn't happen across image axes
    if self.bufs[0].dtype.name.startswith('image'):
      base_shape = self.bufs[0].dtype.shape
      if shape_idx_groups := get_contraction(self.output_shape, base_shape):
        special_strides: Tuple[int, ...] = tuple()
        for i,g in enumerate(shape_idx_groups):
          shape_piece = tuple(self.output_shape[x] for x in g)
          assert prod(shape_piece) == base_shape[i], f"get_contraction was wrong? {shape_piece} != {base_shape[i]}"
          special_strides += strides_for_shape(shape_piece)
        # adding the fake image shape
        shapes.append(self.output_shape)
        strides.append(special_strides)

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
    for i,x in enumerate(rets[:len(self.sts)]): self.sts[i] = self.sts[i].reshape(tuple([y[0] for y in x]))

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

  def limit_dims_to_max(self, global_max: List[int], local_max: List[int]):
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

    self.sts.append(ShapeTracker((View.create(tuple(shp), tuple(stride)),)))
    self.bufs.append(LocalBuffer(name=f"ldata{i}", size=self.sts[-1].size()))
    if DEBUG >= 4: print("aliasing buffer", self.sts[i])
    self.local_alias[i] = self.bufs[-1]

  # ******************** high level optimizers ********************

  def apply_tensor_cores(self, use_tensor_cores=1, extra_opts:Optional[List[Opt]]=None):
    if use_tensor_cores and self.opts.has_local and self.reduceop and self.reduceop.op == ReduceOps.SUM and self.opts.device in tensor_cores:
      for tc in tensor_cores[self.opts.device]:
        if not((tc.arch is None or tc.arch == os.uname().machine) and isinstance(self.reduceop.src[0], LazyOp)): continue
        has_cast = tc.dtype_in != tc.dtype_out

        if has_cast and not(isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == UnaryOps.CAST and self.reduceop.src[0].arg[0] == tc.dtype_out): continue
        mul_op = self.reduceop.src[0].src[0] if has_cast else self.reduceop.src[0]

        if not(isinstance(mul_op, LazyOp) and mul_op.op == BinaryOps.MUL): continue
        if not(isinstance(mul_op.src[0], LazyOp) and mul_op.src[0].op == BufferOps.MEM and mul_op.src[0].arg.dtype == tc.dtype_in): continue
        if not(isinstance(mul_op.src[1], LazyOp) and mul_op.src[1].op == BufferOps.MEM and mul_op.src[1].arg.dtype == tc.dtype_in): continue
        buf0, buf1 = self.bufs.index(cast(LazyOp, mul_op.src[0].arg)), self.bufs.index(cast(LazyOp, mul_op.src[1].arg))
        buf0_strides, buf1_strides = self.sts[buf0].real_strides(), self.sts[buf1].real_strides()
        axis_buf0 = [(i,self.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides[:self.first_reduce]) if s == 0 and self.full_shape[i]%tc.dims[0] == 0]
        axis_buf1 = [(i,self.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides[:self.first_reduce]) if s == 0 and self.full_shape[i]%tc.dims[1] == 0]

        if not(axis_buf0 and axis_buf1 and self.full_shape[self.first_reduce]%tc.dims[2] == 0 and self.full_shape[self.first_reduce] >= tc.dims[2] and (self.shape_len-self.first_reduce) == 1): continue

        if DEBUG >= 3: print("TENSOR CORES", axis_buf0, axis_buf1, tc)

        s0, s1 = axis_buf0[-1][0], axis_buf1[-1][0] # TODO: select axis in smart way
        s0_exists, s1_exists = True, True
        assert s0 != s1 and self.full_shape[s0]%tc.dims[0] == 0 and self.full_shape[s1]%tc.dims[1] == 0
        def fix(needed, ax):
          nonlocal s0, s1, s0_exists, s1_exists
          if not needed: return
          if s0_exists and ax == s0:
            if s1_exists and s0 < s1: s1 -= 1
            s0_exists = False
          elif s1_exists and ax == s1:
            if s0_exists and s1 < s0: s0 -= 1
            s1_exists = False

        # tensor core -- unroll the reduce dim, upcast input, then create the correct thread pattern
        self.apply_opt(Opt(OptOps.UNROLL, 0, tc.dims[2]))
        self.apply_opt(Opt(OptOps.UPCAST, s0 if tc.upcast_dim == 0 else s1, (tc.dims[0]*tc.dims[2])//prod([a[1] for a in tc.threads])))
        for (tc_dim, tc_amt) in tc.threads:
          fix(self.apply_opt(Opt(OptOps.LASTLOCAL, s0 if tc_dim == 0 else s1, tc_amt)), s0 if tc_dim == 0 else s1)

        # assert tensor core and prevent extra_opts from altering the key shape structure
        if use_tensor_cores == 1: self.tensor_core = tc # TC=2 will do the shape ops without the WMMA

        if extra_opts is not None:
          for opt in extra_opts:
            self.apply_opt(opt)
        else:
          # hand-coded TC opts
          if s1_exists:
            s1_div = [upc for upc in [5,4,3,2,1] if self.full_shape[s1]%upc == 0][0]
            if s1_div != 1: fix(self.apply_opt(Opt(OptOps.UPCAST, s1, s1_div)), s1)
          if s0_exists:
            s0_div = [upc for upc in [5,4,3,2,1] if self.full_shape[s0]%upc == 0][0]
            if s0_div != 1: fix(self.apply_opt(Opt(OptOps.UPCAST, s0, s0_div)), s0)
          if self.tensor_core and s0_exists:
            for upc in [4,2]:
              if self.full_shape[s0] % upc == 0:
                self.apply_opt(Opt(OptOps.LASTLOCAL, s0, upc))
                break

        # alias buffer
        alias_pattern = [0]*(self.global_dims+(self.local_dims-len(tc.threads))) + [2]*(len(tc.threads)) + [0]*(self.shape_len-self.upcasted-self.first_reduce) + [1,1] + [3]*(self.upcasted-2)
        self.alias_buffer(buf0, alias_pattern)
        self.alias_buffer(buf1, alias_pattern)
        return True
    return False

  def apply_opt(self, opt:Opt):
    self.applied_opts.append(opt)
    axis = opt.axis + (self.first_reduce if opt.op == OptOps.UNROLL else (self.first_reduce+len(self.group_for_reduce) if opt.op == OptOps.GROUP or opt.op == OptOps.GROUPTOP else 0))
    amt = opt.amt if opt.amt != 0 else self.full_shape[axis]
    assert self.full_shape[axis] % amt == 0, "no longer valid shift"
    assert isinstance(amt, int) and amt != 1, "shift of amt 1 or Node is meaningless"
    if opt.op == OptOps.LOCAL:        # cyan
      assert axis < self.first_reduce, "can't local a reduce"
      assert not(self.tensor_core), "can't local with tensor cores"
      self.shift_to(axis, amt, insert_before=self.first_reduce)
      self.local_dims += 1
    elif opt.op == OptOps.LASTLOCAL:  # cyan
      assert axis < self.first_reduce, "can't local a reduce"
      self.shift_to(axis, amt, insert_before=self.first_reduce-self.local_dims)
      self.local_dims += 1
    elif opt.op == OptOps.GROUP:      # green
      assert axis >= self.first_reduce + len(self.group_for_reduce) and axis < self.shape_len-self.upcasted, "must be reduce axis to group"
      assert not(self.tensor_core), "can't group with tensor cores"
      self.shift_to(axis, amt, insert_before=self.first_reduce + len(self.group_for_reduce))
      self.group_for_reduce.append(amt)
    elif opt.op == OptOps.GROUPTOP:   # green
      assert axis >= self.first_reduce + len(self.group_for_reduce) and axis < self.shape_len-self.upcasted, "must be reduce axis to group"
      assert not(self.tensor_core), "can't group with tensor cores"
      self.shift_to(axis, amt, top=True, insert_before=self.first_reduce + len(self.group_for_reduce))
      self.group_for_reduce.append(amt)
    elif opt.op == OptOps.UNROLL:     # purple
      assert axis < self.shape_len-self.upcasted, "can't upcasted already upcasted"
      assert amt <= 32, "don't unroll more than 32"
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op == OptOps.UPCAST:     # yellow
      assert axis < self.first_reduce, "upcast is for non-reduce"
      assert amt <= 8, "don't upcast more than 8"
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op == OptOps.UPCASTMID:  # white
      assert self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1, "invalid upcast mid reduce"
      axes = self.sts[0].unit_stride_axes()
      assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
      assert axes[0] == axis, "wrong axis"
      assert amt == 4, "don't upcast mid anything but 4"
      self.shift_to(axis, amt, insert_before=self.first_reduce + len(self.group_for_reduce))
      self.group_for_reduce.append(amt)
    return self.simplify_ones()

  def required_optimizations(self, early_only=False):
    for buf_index,buf in enumerate(self.bufs):
      unit_stride_axes_mul_4 = [i for i in self.sts[buf_index].unit_stride_axes(ignore_valid=True) if self.sts[buf_index].shape[i]%4 == 0]
      if (not early_only or buf in self.earlybufs) and self.bufs[buf_index].dtype.__class__ is ImageDType:
        assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {self.bufs[buf_index]}"
        if all(x < (self.shape_len-self.upcasted) for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:
          if unit_stride_axes_mul_4[0] < self.first_reduce:
            self.apply_opt(Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))
          else:
            self.apply_opt(Opt(OptOps.UNROLL, unit_stride_axes_mul_4[0]-self.first_reduce, 4))

  def hand_coded_optimizations(self):
    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    self.required_optimizations(early_only=True)

    # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
    MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
    if self.opts.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
        self.reduceop and self.reduceop.op == ReduceOps.SUM and len(self.full_shape) >= 2 and self.opts.has_shared and \
        isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == BinaryOps.MUL and \
        self.reduceop.src[0].src[0].op == BufferOps.MEM and self.reduceop.src[0].src[1].op == BufferOps.MEM:
      buf0 = self.bufs.index(cast(LazyOp, self.reduceop.src[0].src[0]).arg)
      buf1 = self.bufs.index(cast(LazyOp, self.reduceop.src[0].src[1]).arg)
      buf0_strides = self.sts[buf0].real_strides()
      buf1_strides = self.sts[buf1].real_strides()
      def has_expanded_axis(s, st): return any(x > 1 and y == 0 for x,y in zip(s,st))
      if buf0_strides[self.first_reduce] == 1 and not (has_expanded_axis(self.sts[buf0].shape, buf0_strides) and has_expanded_axis(self.sts[buf1].shape, buf1_strides)):
        for global_idx in range(self.global_dims):
          if self.full_shape[self.first_reduce]%MV_THREADS_PER_ROW == 0 and self.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3: print(f"MATVEC: full_shape={self.full_shape} first_reduce={self.first_reduce} buf0_strides={buf0_strides} blocksize={MV_BLOCKSIZE} threads_per_row={MV_THREADS_PER_ROW} rows_per_thread={MV_ROWS_PER_THREAD}")
            if MV_THREADS_PER_ROW > 1:
              self.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            if MV_BLOCKSIZE > 1:
              self.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if MV_ROWS_PER_THREAD > 1:
              self.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
            return

    if self.opts.has_local and self.opts.has_shared and all(isinstance(s, int) for s in self.sts[0].shape[:self.first_reduce]):
      # are we grouping? (requires local shape support)
      if not self.float4_axis(0) and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
        # TODO: use 1024 if it's allowed in a smarter way
        for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
          if all(st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts):
            self.apply_opt(Opt(OptOps.GROUPTOP, 0, sz))
            break

      # are we upcasting in mid reduce? (only for images)
      if self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:
        axes = self.sts[0].unit_stride_axes()
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
        if self.sts[0].shape[axes[0]]%4 == 0:
          self.apply_opt(Opt(OptOps.UPCASTMID, axes[0], 4))

    # now do everything required
    self.required_optimizations()

    # no more opt if we are grouping
    if self.group_for_reduce: return

    # **** below this line need to be optional and benchmarked ****

    # TODO: doing extra upcasts with images doesn't work for some reason (maybe has to do with to_image_idx)
    # to trigger the above bug, remove prod(self.full_shape[self.shape_len - self.upcasted:]) from the below
    # expression and run test/test_ops.py with IMAGE=2
    # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
    # this can be made much smarter
    to_upcast: List[int] = []
    # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
    for axis in range(self.first_reduce):
      # we might want to be able to split axes that are masked, or refuse to merge them in simplify_merge_adjacent
      # for now skip upcasting here if there is a symbolic axis
      if isinstance(self.full_shape[axis], int) and self.full_shape[axis] <= 7 and any(st.axis_is_masked(axis) for st in self.sts) and \
        prod(self.full_shape[self.shape_len - self.upcasted:]) * prod(self.full_shape[j] for j in to_upcast) * self.full_shape[axis] <= 7 * 7:
        if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
        to_upcast.append(axis)
    for axis in to_upcast[::-1]:
      self.apply_opt(Opt(OptOps.UPCAST, axis, 0))

    # potentially do more upcasts of non reduce axes based on a heuristic
    upcasted_axis = set()
    while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if we haven't upcasted it, it's not symbolic, it mods, and some buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
        if axis not in upcasted_axis and isinstance(self.full_shape[axis], int) and self.full_shape[axis]%upcast_amount == 0 and any(st.views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index, st in enumerate(self.sts)):
          xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in self.sts), sum(st.views[-1].strides[axis] for st in self.sts), axis, upcast_amount))
      if xb_choices:
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
        upcasted_axis.add(xb_choices[0][2])
      else:
        break

    # if last dim is small(ish) and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast. NOTE: careful, this has broken VALIDHACKS
    if self.first_reduce < (self.shape_len-self.upcasted) and (len(list(self.shape_offsets(self.full_buf_index))) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))):
      if (s:=self.full_unupcasted_shape[-1]) <= 32 and isinstance(s, int):  # NOTE: cannot loop unroll symbolic axis
        self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
        # if it's small, upcast a second reduce dimension too
        if self.first_reduce < (self.shape_len-self.upcasted) and s <= 3 and (s2:=self.full_unupcasted_shape[-1]) <= 3 and isinstance(s2, int):
          self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
      else:
        for splits in [4]:
          if self.full_unupcasted_shape[-1]%splits == 0:
            self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, splits))
            break

    # if nothing at all is upcasted and it's easy to, do an upcast
    # TODO: this is breaking the tests
    for splits in [4]:
      if self.upcasted == 0 and self.full_unupcasted_shape and self.full_unupcasted_shape[-1] % splits == 0:
        self.apply_opt(Opt(OptOps.UPCAST, len(self.full_unupcasted_shape)-1, splits))

    # **** local groups ****

    if self.opts.has_local:
      if getenv("NOLOCALS") and self.local_dims == 0 and not self.group_for_reduce:
        self.dont_use_locals = True
      else:
        # prioritize making expand axes local
        local_axis_ranking = [(any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))), axis) for axis in range(len(self.full_shape[:self.first_reduce]))]
        to_local: List[Tuple[int, int]] = []
        for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
          local_size = prod(sz for _, sz in to_local)
          local_sz: Optional[int] = next((x for x in ([32] * (axis == 0) + [16, 8, 4, 3, 2]) if self.full_shape[axis] % x == 0 and local_size * x <= 128), None)
          if local_sz is not None: to_local.append((axis, local_sz))
        deleted_shape = 0
        for axis, local_sz in sorted(to_local[:3]):
          axis = axis - deleted_shape
          will_delete_shape = local_sz == self.full_shape[axis]
          self.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
          if will_delete_shape: deleted_shape += 1
