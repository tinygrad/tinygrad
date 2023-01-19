from tinygrad.helpers import dedup, all_same
from tinygrad.ops import LazyOp, MovementOps, get_lazyop_info, get_buffers, ReduceOps, get_lazyops
from tinygrad.shape import ShapeTracker

def get_first_reduce(shapes):
  for i in range(len(shapes[0])):
    if not all_same([x[i] for x in shapes]):
      return i
  return len(shapes[0])  # off the end

# ast kernel can contain one ReduceOp with arbitrary Binary/Unary ops
class ASTKernel:
  def __init__(self, ast:LazyOp):
    # key for lookup in cache (can change, str might not be right)
    self.input_ast = ast
    self.key = str(ast)

    # if the AST ends with a RESHAPE, we remove it and create the buffer accordingly
    if ast.op == MovementOps.RESHAPE:
      output_shape = ast.arg
      ast = ast.src[0]
    else:
      output_shape = None

    self.info = get_lazyop_info(ast)
    self.bufs = dedup(get_buffers(ast))
    reduceops = [x for x in get_lazyops(ast) if x.op in ReduceOps]
    assert len(dedup(reduceops)) <= 1, "max one reduce op in an ast"
    self.reduceop = reduceops[0] if reduceops else None
    self.earlybufs = dedup(get_buffers(self.reduceop)) if self.reduceop else []
    self.ast = ast

    # create the buffer we are returning (as the same type as the input buffers) and add it as the first buffer
    self.ret = type(self.bufs[0])(output_shape if output_shape else self.info.shape)
    if hasattr(self.ret, "cl"): self.ret.cl  # does the allocation of unbacked buffer, pylint: disable=W0104
    self.bufs = [type(self.ret)(self.info.shape, hostbuf=self.ret)] + self.bufs

    # check valid AST kernel
    assert all_same([x.shape for x in self.earlybufs]), "all earlybufs must have the same shape"
    assert all_same([x.shape for x in self.bufs if x not in self.earlybufs]), "all latebufs must have the same shape"
    assert all_same([len(x.shape) for x in self.bufs]), "all bufs must have the same shape size"

  def print(self):
    buf_count = -1
    op_count = -1
    cache = {}
    def print_ast(x, name=None):
      nonlocal buf_count, op_count
      if x not in cache:
        if not isinstance(x, LazyOp):
          if name is None:
            buf_count += 1
            name = f"buf{buf_count}"
          print(f"buf{buf_count} = {x}")
          cache[x] = name
        else:
          srcs = [print_ast(y) for y in x.src]
          if name is None:
            op_count += 1
            name = f"op{op_count}"
          print(f"{name} = LazyOp({str(x.op)}, ({','.join(srcs)},), {x.arg})")
          cache[x] = name
      return cache[x]
    print_ast(self.input_ast, "ast")

  def process(self):
    # get shape, strides, and offset
    # if it's a multiview buffer we take the final view
    self.shapes = [x.shape for x in self.bufs]
    self.strides = [x.st.views[-1].strides for x in self.bufs]
    self.offsets = [x.st.views[-1].offset for x in self.bufs]  # include the offsets (as is)
    self.last_reduce = len(self.shapes[0])
    self.simplify_ones()
    self.simplify_merge_adjacent()

  def simplify_ones(self):
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    all_ones = [all(s[i]==1 for s in self.shapes) for i in range(len(self.shapes[0]))]
    # keep at least 1 one
    if all(all_ones):
      all_ones[-1] = False
    self.shapes = [[s[i] for i in range(len(s)) if not all_ones[i]] for s in self.shapes]
    self.strides = [[s[i] for i in range(len(s)) if not all_ones[i]] for s in self.strides]
    self.last_reduce -= sum(all_ones)
    # find first mismatch, don't reduce this
    self.first_reduce = get_first_reduce(self.shapes)

  def simplify_merge_adjacent(self):
    shapes, strides = self.shapes, self.strides

    # merge dimensions if we can, multi get_shape_strides
    # TODO: does this always preserve the reduce dimension, NO
    # TODO: move this into shapetracker, with tests!
    rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for j in range(len(shapes)):
        # TODO: added the always mergability of 1s, is this right? if so, add to shapetracker in the 1 case
        can_merge.append((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*strides[j][i]) or (strides[j][i] == 0 and rets[j][-1][1] == 0))
      # more can merge than this
      can_merge = all(can_merge) and i != self.first_reduce
      if can_merge:
        self.last_reduce -= 1
      for j in range(len(shapes)):
        if can_merge:
          rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
        else:
          rets[j].append((shapes[j][i], strides[j][i]))
    self.shapes, self.strides = [[y[0] for y in x] for x in rets], [[y[1] for y in x] for x in rets]
    self.first_reduce = get_first_reduce(self.shapes)

  @property
  def shape_len(self): return len(self.shapes[0])

  # this should be aware of the three parts to the shape
  #  * the input/output dimensions
  #  * the reduce dimensions
  #  * the size outputted by each kernel
  def reshape_and_permute(self, new_shape_fxn, axis):
    new_shapes, new_strides = [], []
    for shape, stride in zip(self.shapes, self.strides):
      st = ShapeTracker(tuple(shape))
      st.strided(*zip(shape, stride))
      # TODO: handle reduced shape here
      st.reshape(*new_shape_fxn(shape))
      if axis is not None: st.permute(*axis)
      assert len(st.views) == 1
      new_shapes.append(st.shape)
      new_strides.append(st.strides)
    self.shapes, self.strides = new_shapes, new_strides