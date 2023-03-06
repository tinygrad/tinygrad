import functools, math
from typing import ClassVar, List
from llvmlite import ir  # type: ignore
from tinygrad.codegen.ast import ASTKernel
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, LazyOp, ASTRunner
from tinygrad.helpers import DEBUG, prod

from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, GeNode, LtNode, SumNode, AndNode
def int_const(x): return ir.Constant(ir.IntType(64), x)
render_llvm = {
  Variable: lambda self,ops,ctx: self.expr,
  NumNode: lambda self,ops,ctx: int_const(self.b),
  MulNode: lambda self,ops,ctx: ctx.mul(self.a.render(ops,ctx), int_const(self.b)),
  DivNode: lambda self,ops,ctx: ctx.sdiv(self.a.render(ops,ctx), int_const(self.b)),
  ModNode: lambda self,ops,ctx: ctx.srem(self.a.render(ops,ctx), int_const(self.b)),
  GeNode: lambda self,ops,ctx: ctx.icmp_signed(">=", self.a.render(ops,ctx), int_const(self.b)),
  LtNode: lambda self,ops,ctx: ctx.icmp_signed("<", self.a.render(ops,ctx), int_const(self.b)),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.add(a,b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.and_(a,b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx))
}

class LLVMCodegen(ASTKernel):
  op_lookup : ClassVar = {
    UnaryOps.NOOP: lambda builder,x: x,
    UnaryOps.NEG: lambda builder,x: builder.fneg(x, flags=('fast',)),
    UnaryOps.EXP: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.exp', [ir.FloatType()]), [x], fastmath=('fast',)),
    UnaryOps.LOG: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.log', [ir.FloatType()]), [x], fastmath=('fast',)),
    UnaryOps.NOT: lambda builder,x: builder.fsub(ir.Constant(ir.FloatType(), 1), x, flags=('fast',)),
    BinaryOps.ADD: lambda builder,x,y: builder.fadd(x,y, flags=('fast',)),
    BinaryOps.SUB: lambda builder,x,y: builder.fsub(x,y, flags=('fast',)),
    BinaryOps.MUL: lambda builder,x,y: builder.fmul(x,y, flags=('fast',)),
    BinaryOps.DIV: lambda builder,x,y: builder.fdiv(x,y, flags=('fast',)),
    BinaryOps.POW: lambda builder,x,y: builder.call(builder._block.module.declare_intrinsic('llvm.pow', [ir.FloatType()]), [x,y], fastmath=('fast',)),
    BinaryOps.CMPEQ: lambda builder,x,y: builder.uitofp(builder.fcmp_ordered("==", x, y, flags=('fast',)), ir.FloatType()),
    BinaryOps.MAX: lambda builder,x,y: builder.select(builder.fcmp_unordered(">", x, y, flags=('fast',)), x, y, flags=('fast',))
  }
  start_for_op : ClassVar = {
    ReduceOps.SUM: ir.Constant(ir.FloatType(), 0),
    ReduceOps.MAX: ir.Constant(ir.FloatType(), -math.inf)
  }

  def codegen(self):
    self.process()
    if DEBUG >= 3: self.printbufs("old:", DEBUG>=4)

    # this stuff can't be hand coded
    kernel_output_axis : List[int] = []
    """
    CACHE_DIM = 32
    if len(k.shapes[0]) == 2:
      # cache tiling, makes permute fast
      k.reshape_and_permute(
        lambda shape: (shape[0]//CACHE_DIM, CACHE_DIM, shape[1]//CACHE_DIM, CACHE_DIM),
        (0,2,1,3))
    elif len(k.shapes[0]) == 3:
      if k.reduceop:
        if k.strides[1][-1] == 1 and k.strides[2][-1] == 1:
          DY, DX = 8, 8
        elif k.strides[1][-1] in [1,0] and k.strides[1][-2] in [1,0]:
          DY, DX = 4, 16
        else:
          DY, DX = 16, 4
        # matmul: YyXxK -> YXKyx
        k.reshape_and_permute(
          lambda shape: (shape[0]//DY, DY, shape[1]//DX, DX, shape[2]),
          (0,2,4,1,3))
        kernel_output_axis = [-2, -1]
      else:
        CACHE_L2_DIM = 256
        k.reshape_and_permute(
          lambda shape: (shape[0], shape[1]//CACHE_L2_DIM, CACHE_L2_DIM, shape[2]),
          (1,0,2,3))
        kernel_output_axis = [-1]
    elif len(k.shapes[0]) == 7:
      # conv: split chans and X
      DY, DX = 4, 16
      k.reshape_and_permute(
        lambda shape: (shape[0], shape[1]//DY, DY, shape[2], shape[3]//DX, DX, shape[4], shape[5], shape[6]),
        (0,1,3,4,6,7,8,2,5))
      kernel_output_axis = [-2, -1]
    """

    # the 4x4 need to go all the way at the end, even after reduce
    output_shape = self.sts[0].shape
    full_shape_options = [x.shape for x in self.sts if x.shape != output_shape]
    full_shape = output_shape if len(full_shape_options) == 0 else full_shape_options[0]

    full_shape = full_shape if not kernel_output_axis else full_shape[:-len(kernel_output_axis)]
    kernel_output_dim = prod([self.sts[0].shape[a] for a in kernel_output_axis])
    kernel_output_type = ir.FloatType() if kernel_output_dim == 1 else ir.VectorType(ir.FloatType(), kernel_output_dim)

    def get_idxs(builder, idx, buf_index):
      idx_offsets = [0]
      """
      for axis in kernel_output_axis:
        new_idx_offsets = []
        for s in range(k.shapes[buf_index][axis]):
          for i in idx_offsets:
            new_idx_offsets.append(i + s * k.strides[buf_index][axis])
        idx_offsets = new_idx_offsets
      """
      return [builder.add(idx, int_const(i)) for i in idx_offsets]

    # *** llvm specific below this line ***

    # create llvm function
    module = ir.Module(name=__file__)
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [ir.FloatType().as_pointer()]*(len(self.bufs))), name='exec')

    # force llvmlite to allow us to add function attribute then add the attribute
    func.attributes._known = func.attributes._known.union(frozenset(['"no-nans-fp-math"="true"']))
    func.attributes.add('"no-nans-fp-math"="true"')

    # construct the structure of the loops
    loop_entry, loop_exit = [ir.IRBuilder(func.append_basic_block(name="entry"))], []
    for i,_ in enumerate(full_shape): loop_entry.append(ir.IRBuilder(func.append_basic_block(name=f"loop_{i}")))
    for i,_ in enumerate(full_shape): loop_exit.append(ir.IRBuilder(func.append_basic_block(name=f"loopexit_{len(full_shape)-1-i}")))
    loop_exit.append(ir.IRBuilder(func.append_basic_block(name="exit")))
    loop_exit = loop_exit[::-1]

    # add the buffer indexing
    idx_level = [[int_const(st.offset)] for st in self.sts]
    for i in range(len(full_shape)):
      for j in range(len(self.bufs)):
        # stride
        si = loop_entry[i+1].phi(ir.IntType(64), name=f"idx_{j}_{i}")
        si.add_incoming(idx_level[j][-1], loop_entry[i]._block)
        si_ps = loop_exit[i+1].add(si, int_const(self.sts[j].views[-1].strides[i]))
        si.add_incoming(si_ps, loop_exit[i+1]._block)
        idx_level[j].append(si)

    # the ast parser
    def ast_parse(builder, x, level, reduce_result=None):
      if not isinstance(x, LazyOp):
        m = kernel_output_type(ir.Undefined)
        buf_index = self.bufs.index(x)
        for i, idx in enumerate(get_idxs(builder, idx_level[buf_index][level], buf_index)):
          # first view is already implictly handled
          idx, valid = x.st._expr_idx(Variable(idx, 0, prod(x.st.shape)))
          idx = idx.render(render_llvm, builder)
          if valid.min == 0:
            valid = valid.render(render_llvm, builder)
            # this always does the load, so we have it load *0 if the arg won't be used
            # TODO: would control flow be faster?
            aug_idx = builder.select(valid, idx, int_const(0))
            element = builder.select(valid, builder.load(builder.gep(func.args[buf_index], [aug_idx], inbounds=True)), ir.Constant(ir.FloatType(), 0))
          else:
            element = builder.load(builder.gep(func.args[buf_index], [idx], inbounds=True))
          m = element if kernel_output_dim == 1 else builder.insert_element(m, element, int_const(i))
        return m
      if isinstance(x.op, ReduceOps):
        if reduce_result is None:
          raise RuntimeError("no reduce")
        return reduce_result
      values = [ast_parse(builder, v, level, reduce_result) for v in x.src]

      m = kernel_output_type(ir.Undefined)
      if kernel_output_dim == 1:
        return LLVMCodegen.op_lookup[x.op](builder, *values)
      else:
        # TODO: this only has to be done for certain ops
        for i in range(kernel_output_dim):
          value = [builder.extract_element(v, int_const(i)) for v in values]
          element = LLVMCodegen.op_lookup[x.op](builder, *value)
          m = builder.insert_element(m, element, int_const(i))
        return m

    # add the ast + final store
    store_loop = output_shape.index(1) if 1 in output_shape else -1

    # do the early ast
    reduce_result = None
    if self.reduceop:
      reduce_input = ast_parse(loop_exit[-1], self.reduceop.src[0], -1)
      phis = [LLVMCodegen.start_for_op[self.reduceop.op]]  # type: ignore
      if kernel_output_dim > 1: phis = [kernel_output_type(phis * kernel_output_dim)]
      for i in range(store_loop+1, len(loop_entry)):
        val = loop_entry[i].phi(kernel_output_type, f"reduce_phi_{i}")
        val.add_incoming(phis[-1], loop_entry[i-1]._block)
        phis.append(val)

      if self.reduceop.op == ReduceOps.SUM:
        reduce_result = loop_exit[-1].fadd(reduce_input, val, flags=('fast',))
      elif self.reduceop.op == ReduceOps.MAX:
        reduce_result = loop_exit[-1].select(loop_exit[-1].fcmp_unordered(">", val, reduce_input, flags=('fast',)), val, reduce_input, flags=('fast',))

      for i,phi in enumerate(phis[1:]):
        phi.add_incoming(reduce_result, loop_exit[store_loop+1+i]._block)

    # do the late ast
    result = ast_parse(loop_exit[store_loop], self.ast, store_loop, reduce_result=reduce_result)

    # store result
    builder = loop_exit[store_loop]
    for i, idx in enumerate(get_idxs(builder, idx_level[0][store_loop], 0)):
      element = result if kernel_output_dim == 1 else builder.extract_element(result, int_const(i))
      builder.store(element, builder.gep(func.args[0], [idx], inbounds=True))

    # add the looping
    for i,s in enumerate(full_shape):
      loop_entry[i].branch(loop_entry[i+1]._block)
      idx = loop_entry[i+1].phi(ir.IntType(64), name=f"loopvar_{i}")
      idx.add_incoming(int_const(0), loop_entry[i]._block)
      idx_p1 = loop_exit[i+1].add(idx, int_const(1))
      idx.add_incoming(idx_p1, loop_exit[i+1]._block)
      loop_exit[i+1].cbranch(loop_exit[i+1].icmp_unsigned("==", idx_p1, int_const(s)), loop_exit[i]._block, loop_entry[i+1]._block)

    loop_entry[-1].branch(loop_exit[-1]._block)
    loop_exit[0].ret_void()

    return ASTRunner('exec', str(module), op_estimate=self.info.flops, mem_estimate=sum(prod(x._base_shape) for x in self.bufs))
