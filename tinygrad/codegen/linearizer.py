from typing import List, Tuple, Any, Optional
import itertools, math
from collections import defaultdict
from enum import Enum, auto

from tinygrad.helpers import dedup, colored, all_same, ImageDType, DEBUG, prod
from tinygrad.ops import LazyOp, get_lazyops, get_buffers, FlopCounter, get_lazyop_info, map_buffers, UnaryOps
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import MovementOps, ReduceOps, BinaryOps
from tinygrad.shape.shapetracker import ShapeTracker, View, strides_for_shape
from tinygrad.shape.symbolic import Variable

class UOps(Enum): LOOP = auto(); DEFINE_LOCAL = auto(); LOAD = auto(); ALU = auto(); CONST = auto(); ENDLOOP = auto(); STORE = auto(); LOOP_GLOBAL = auto(); ENDLOOP_GLOBAL = auto(); LOOP_LOCAL = auto(); ENDLOOP_LOCAL = auto();

def get_first_reduce(shapes):
  for i in range(len(shapes[0])):
    if not all_same([x[i] for x in shapes]): return i
  return len(shapes[0])  # off the end

class Register:
  def __init__(self, name:str):
    self.name = name
    self.axis: List[Tuple[int, int, bool]] = []
  def array(self, length, stride, reduce): self.axis.append((length, stride, reduce))
  def size(self): return prod([x[0] for x in self.axis])
  def offsets(self): return [sum(t) for t in itertools.product(*[[y*x[1] for y in range(x[0])] for x in self.axis[::-1]])] if len(self.axis) else [0]
  def can_float4(self): return any(a[0:2] == (4,1) for a in self.axis)
  # TODO: this is sort of a hack, it gets the accumulator indices
  def acc_offsets(self):
    if len(self.axis) == 0: return [0]
    acc_strides = [x*(1-self.axis[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in self.axis[::-1])))]
    return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(self.axis[::-1])])]
  def __repr__(self): return f"<{self.name}{f'{self.axis}'}>"

class Linearizer:
  def __init__(self, ast:LazyOp, output_buffer:LazyBuffer):
    # NOTE: if there's a RESHAPE, we skip it. the output shape is set from the reduce op or a latebuf
    self.ast = ast.src[0] if ast.op == MovementOps.RESHAPE else ast

    # get the output buffers
    self.bufs = [output_buffer] + dedup(get_buffers(ast))

    # key for lookup in cache (can change, str might not be right)
    # bufs are needed because kernels like f(x) = x + x and f(x, y) = x + y have the same str(ast), but are different kernels.
    # mapping the buffers to integers is required because a-b != b-a (and how would you tell a and b apart?)
    self.key = f"ASTKernelKey ast={str(map_buffers({x:i for i,x in enumerate(self.bufs)}, ast))} bufs={self.bufs}"

  def process(self) -> None:
    if hasattr(self, "sts"): return   # already processed

    # fetch lazyop info
    self.info: FlopCounter = get_lazyop_info(self.ast)

    # there's only allowed to be one reduceop
    reduceops = [x for x in get_lazyops(self.ast) if x.op in ReduceOps]
    assert len(dedup(reduceops)) <= 1, "max one reduce op in an ast"
    self.reduceop = reduceops[0] if reduceops else None

    # get earlybufs, before the one reduce op
    self.earlybufs = dedup(get_buffers(self.reduceop)) if self.reduceop else []

    # create new shapetrackers inside this kernel, we will permute them
    self.sts: List[ShapeTracker] = [x.st.copy() for x in self.bufs]
    for st in self.sts: st.simplify()

    # make the output buffer shape correct in here
    self.sts[0].reshape(self.info.shape)
    self.full_buf_index: int = self.bufs.index(self.earlybufs[0]) if len(self.earlybufs) > 0 else 0

    # move all reduce axes to the end
    reduce = list(enumerate(zip(self.full_shape, self.sts[0].shape)))
    permute = tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    self.reshape_and_permute(None, permute)

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

    # is this generic?
    self.registers = [Register(f"data{i}") for i in range(len(self.bufs))]
    self.group_for_reduce: List[int] = []

  def linearize(self):
    # add a local buffer for multistage reduce
    if len(self.group_for_reduce):
      self.bufs.append(None)
      # TODO: the strides of this can be controlled
      st = ShapeTracker(tuple([1] * self.first_reduce + self.group_for_reduce + [1] * (self.shape_len - len(self.group_for_reduce) - self.first_reduce) + [x[0] for x in self.registers[0].axis]))
      buftoken = Register("temp")
      # manual upcast of the local
      for _,_,r in self.registers[0].axis[::-1]:
        buftoken.array(st.shape[-1], st.views[-1].strides[-1], r)
        st.views[-1] = View(st.shape[0:-1], st.views[-1].strides[0:-1], st.views[-1].offset)
      self.sts.append(st)
      self.registers.append(buftoken)

    # uops
    self.uops = []

    # TODO: add upcasting to float4 here
    def global_buf(i, store=None):
      if store:
        return [self.uop(UOps.STORE, (v, self.registers[i].name, *self.sts[i].expr_idxs(o, idxs))) for v,o in zip(store, self.registers[i].offsets())]
      else:
        return [self.uop(UOps.LOAD, (self.registers[i].name, *self.sts[i].expr_idxs(o, idxs)), self.registers[i].name+"_") for o in self.registers[i].offsets()]

    # parse AST
    loaded_buffers = {}
    acc = []
    self.ssa = defaultdict(int)

    # indexes
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.full_shape)]

    # global loop
    global_idxs = idxs[:self.first_reduce]
    self.uop(UOps.LOOP_GLOBAL, global_idxs)

    # local loop
    if self.group_for_reduce:
      # NOTE: this is assuming the global size = the local size in these dims. in general, this doesn't have to be true
      local_idxs = idxs[self.first_reduce:self.first_reduce+len(self.group_for_reduce)]
      self.uop(UOps.DEFINE_LOCAL, (self.registers[-1].name, self.sts[-1].size()*self.registers[-1].size()))
      self.uop(UOps.LOOP_LOCAL, local_idxs)

    # reduce op
    if self.reduceop is not None:
      # define accumulator
      acc = [self.uop(UOps.CONST, ({ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[self.reduceop.op],), 'acc') for _ in self.registers[0].offsets()]

      # reduce loop
      reduce_idxs = idxs[self.first_reduce+len(self.group_for_reduce):]
      self.uop(UOps.LOOP, reduce_idxs)

      # load earlybufs
      loaded_buffers.update({b:global_buf(i) for i,b in enumerate(self.bufs) if b in self.earlybufs and i != 0})

      # run early AST
      red = self.ast_parse(self.reduceop.src[0], None, loaded_buffers)

      # accumulate
      for o,r in zip(self.registers[self.full_buf_index].acc_offsets(), red):
        self.uop(UOps.ALU, (acc[o], {ReduceOps.SUM: BinaryOps.ADD, ReduceOps.MAX: BinaryOps.MAX}[self.reduceop.op], (acc[o],r)))

      # end the reduce loop
      self.uop(UOps.ENDLOOP, reduce_idxs)

    # end the local loop, do the local reduce
    if self.group_for_reduce:
      global_buf(-1, acc)  # store accumulators
      self.uop(UOps.ENDLOOP_LOCAL, local_idxs)   # this is a barrier on GPUs

      # if any group_for_reduce items aren't reduces, upcast them here
      for j in self.upcast_in_mid_reduce_axes:
        self.reshape_and_permute(None, [i for i in range(self.shape_len) if i != j] + [j])
        self.upcast()
        self.group_for_reduce.pop()

      # define late accumulator
      acc = [self.uop(UOps.CONST, ({ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[self.reduceop.op],), 'lacc') for _ in self.registers[-1].offsets()]

      end_local_idxs = idxs[self.first_reduce:self.first_reduce+len(self.group_for_reduce)]
      self.uop(UOps.LOOP, end_local_idxs)
      red = global_buf(-1)
      # accumulate
      for o,r in zip(self.registers[-1].acc_offsets(), red):
        self.uop(UOps.ALU, (acc[o], {ReduceOps.SUM: BinaryOps.ADD, ReduceOps.MAX: BinaryOps.MAX}[self.reduceop.op], (acc[o],r)))
      self.uop(UOps.ENDLOOP, end_local_idxs)

    # load latebufs
    loaded_buffers.update({b:global_buf(i) for i,b in enumerate(self.bufs) if b not in self.earlybufs and i != 0 and b is not None})

    # run late AST
    val = self.ast_parse(self.ast, acc, loaded_buffers)

    # store
    global_buf(0, val)

    # end the global loop
    self.uop(UOps.ENDLOOP_GLOBAL, global_idxs)

    # print
    self.printbufs()
    for x in self.uops:
      print(x)

  def uop(self, uop:UOps, arg:Any, name:Optional[str]=None):
    if name is not None:
      ret = f"{name}{self.ssa[name]}"
      self.ssa[name] += 1
      self.uops.append((uop, (ret,)+arg))
      return ret
    else:
      self.uops.append((uop, arg))

  def ast_parse(self, x, acc, loaded_buffers) -> List[str]:
    if not isinstance(x, LazyOp): return loaded_buffers[x]
    if x.op == UnaryOps.CAST: return self.ast_parse(x.src[0], acc, loaded_buffers)  # cast isn't an ALU op
    if x.op in ReduceOps: return acc
    values = [self.ast_parse(v, acc, loaded_buffers) for v in x.src]
    return [self.uop(UOps.ALU, (x.op, val), 'alu') for val in zip(*values)]

  @property
  def first_reduce(self) -> int: return get_first_reduce([x.shape for i,x in enumerate(self.sts) if self.bufs[i] is not None])

  @property
  def full_shape(self) -> Tuple[int, ...]: return self.sts[self.full_buf_index].shape

  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def upcast_in_mid_reduce_axes(self) -> List[int]: return [j for j in range(self.first_reduce, self.first_reduce+len(self.group_for_reduce)) if self.full_shape[j] == self.sts[0].shape[j]]

  def colorshape(self, pad=50) -> str:
    axis = [(f"{rs:4d}", (("green" if i in self.upcast_in_mid_reduce_axes else "cyan") if i < self.first_reduce + len(self.group_for_reduce) else "red") if i >= self.first_reduce else "blue") for i, rs in enumerate(self.full_shape)]
    axis += [(f"{s:4d}", 'magenta' if reduce else 'yellow') for s, _, reduce in self.registers[self.full_buf_index].axis[::-1]]
    return ' '.join([colored(*x) for x in axis])+(" "*(pad-len(' '.join([x[0] for x in axis]))))

  def printbufs(self, prefix=""):
    for i in range(len(self.sts)):
      print(prefix, f"{i:3d} {str(self.bufs[i].realized) if self.bufs[i] is not None else 'FAKE':47s} {str(self.registers[i]):38s}", self.sts[i].views)
    print(self.colorshape())

  # ******************** base simplifiers ********************

  def simplify_merge_adjacent(self):
    shapes, strides = [x.shape for x in self.sts], [x.views[-1].strides for x in self.sts]

    # merge dimensions if we can, multi get_shape_strides
    # TODO: does this always preserve the reduce dimension, NO
    # TODO: move this into shapetracker, with tests!
    rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for j in range(len(shapes)):
        # TODO: added the always mergeability of 1s, is this right? if so, add to shapetracker in the 1 case
        can_merge.append((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*strides[j][i]) or (strides[j][i] == 0 and rets[j][-1][1] == 0))
      # more can merge than this
      mergeable = all(can_merge) and i != self.first_reduce
      for j in range(len(shapes)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
        else: rets[j].append((shapes[j][i], strides[j][i]))

    # do the reshapes
    for i,x in enumerate(rets): self.sts[i].reshape(tuple(y[0] for y in x))

  def simplify_ones(self):
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    all_ones = [all(st.shape[i]==1 for st in self.sts) for i in range(self.shape_len)]
    # keep at least 1 one
    if all(all_ones): all_ones[-1] = False
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)

  # apply reshape and permute to all shapetrackers
  def reshape_and_permute(self, new_shape_fxn, axis):
    for st in self.sts:
      if new_shape_fxn is not None: st.reshape(tuple(new_shape_fxn(st.shape)))
      if axis is not None: st.permute(tuple(axis))

  # ******************** complex simplifiers ********************

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

  # drops the final dimension
  def upcast(self):
    upcasted = [x.shape[-1] for x in self.sts if x.shape[-1] != 1]
    assert len(upcasted) >= 1 and all_same(upcasted), f"can't upcast mismatch {upcasted}"
    for st,buftoken in zip(self.sts, self.registers):
      # add last axis to the buftoken (if it's not a 1)
      if st.shape[-1] == upcasted[0]: buftoken.array(st.shape[-1], st.views[-1].strides[-1], len(upcasted) != len(self.sts))
      # remove the last axis (unless it's the only dimension, then make it a 1)
      st.views[-1] = View(st.shape[0:-1], st.views[-1].strides[0:-1], st.views[-1].offset) if len(st.shape) > 1 else View((1,), (0,), st.views[-1].offset)

  # ******************** GPU simplifiers ********************

  def required_optimizations(self, early_only=False):
    for buf_index,buf in enumerate(self.bufs):
      upcast_strides = [self.sts[buf_index].strides[i] for i in self.upcast_in_mid_reduce_axes]
      if (not early_only or buf in self.earlybufs) and isinstance(self.bufs[buf_index].dtype, ImageDType) and not (self.registers[buf_index].can_float4() or (buf not in self.earlybufs and (1 in upcast_strides))):
        axes = [i for i,x in enumerate(self.sts[buf_index].strides) if x == 1]
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes} on buf_index {buf_index}, {self.sts[buf_index]}"
        assert self.sts[buf_index].shape[axes[0]]%4 == 0, f"axis:{axes[0]} in buffer {buf_index} is not a multiple of 4, {self.sts[buf_index].shape}"
        self.shift_to(axes[0], 4)
        self.upcast()
        assert self.registers[buf_index].can_float4()

  def hand_coded_optimizations(self):
    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    self.required_optimizations(early_only=True)

    # simplify (sets first_reduce)
    self.simplify_ones()

    # are we grouping? (requires local shape support)
    if not self.registers[0].can_float4() and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
      # TODO: use 1024 if it's allowed in a smarter way
      for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
        if all([st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts]):
          self.shift_to(self.first_reduce, sz, top=True, insert_before=self.first_reduce)
          self.group_for_reduce.append(sz)
          break

    # are we upcasting in mid reduce? (only for images)
    if self.bufs[0].dtype.name.startswith('image') and not self.registers[0].can_float4() and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:
      axes = [i for i,x in enumerate(self.sts[0].strides) if x == 1]
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

    # **** below this line need to be optional and benchmarked ****

    # potentially do more upcasts of non reduce axes based on a heuristic
    while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if it mods, and some buffer has stride 0 on axis while having no stride 0 in the buftoken
        if self.full_shape[axis]%upcast_amount == 0 and any(self.sts[buf_index].strides[axis] == 0 and not any(x[1] == 0 for x in self.registers[buf_index].axis) for buf_index in range(len(self.sts))):
          xb_choices.append((sum(st.strides[axis]>0 for st in self.sts), sum(st.strides[axis] for st in self.sts), axis, upcast_amount))
      if len(xb_choices):
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.shift_to(xb_choices[0][2], amount=xb_choices[0][3])
        self.upcast()
        self.simplify_ones()
      else:
        break

    # if last dim <= 5 and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast. NOTE: careful, this has broken VALIDHACKS
    if self.first_reduce < self.shape_len and self.full_shape[-1] <= 5 and (max([x.size() for i,x in enumerate(self.registers) if self.bufs[i] in self.earlybufs]) <= 4 or not any(r for _,_,r in self.registers[self.full_buf_index].axis)):
      self.upcast()
