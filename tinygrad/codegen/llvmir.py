from typing import Final, Dict, Callable, Any, List, Optional
import functools
from llvmlite import ir  # type: ignore
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.helpers import dtypes
from tinygrad.ops import Op, ASTRunner, UnaryOps, BinaryOps, FusedOps

from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, GeNode, LtNode, SumNode, AndNode
def int_const(x): return ir.Constant(ir.IntType(64), x)
render_llvm = {
  NumNode: lambda self,ops,ctx: int_const(self.b),
  MulNode: lambda self,ops,ctx: ctx.mul(self.a.render(ops,ctx), int_const(self.b)),
  DivNode: lambda self,ops,ctx: ctx.sdiv(self.a.render(ops,ctx), int_const(self.b)),
  ModNode: lambda self,ops,ctx: ctx.srem(self.a.render(ops,ctx), int_const(self.b)),
  GeNode: lambda self,ops,ctx: ctx.icmp_signed(">=", self.a.render(ops,ctx), int_const(self.b)),
  LtNode: lambda self,ops,ctx: ctx.icmp_signed("<", self.a.render(ops,ctx), int_const(self.b)),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.add(a,b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.and_(a,b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx))
}

class LLVMIRCodegen(Linearizer):
  code_for_op: Final[Dict[Op, Callable]] = {
    UnaryOps.EXP: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.exp', [ir.FloatType()]), [x], fastmath=('fast',)),
    UnaryOps.LOG: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.log', [ir.FloatType()]), [x], fastmath=('fast',)),
    BinaryOps.ADD: lambda builder,x,y: builder.fadd(x,y, flags=('fast',)),
    BinaryOps.SUB: lambda builder,x,y: builder.fsub(x,y, flags=('fast',)),
    BinaryOps.MUL: lambda builder,x,y: builder.fmul(x,y, flags=('fast',)),
    BinaryOps.DIV: lambda builder,x,y: builder.fdiv(x,y, flags=('fast',)),
    BinaryOps.POW: lambda builder,x,y: builder.call(builder._block.module.declare_intrinsic('llvm.pow', [ir.FloatType()]), [x,y], fastmath=('fast',)),
    BinaryOps.CMPEQ: lambda builder,x,y: builder.uitofp(builder.fcmp_ordered("==", x, y, flags=('fast',)), ir.FloatType()),
    BinaryOps.MAX: lambda builder,x,y: builder.select(builder.fcmp_unordered(">", x, y, flags=('fast',)), x, y, flags=('fast',)),
    FusedOps.MULACC: lambda builder,x,y,z: builder.fadd(builder.fmul(y,z, flags=('fast',)), x, flags=('fast',)),
  }
  def codegen(self):
    self.process()
    # no optimize, this doesn't support local
    self.linearize()

    # create llvm function
    module = ir.Module(name=__file__)
    func_dtypes = [{dtypes.float16:ir.HalfType(), dtypes.float32:ir.FloatType()}[buf.dtype] for buf in self.bufs]
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [x.as_pointer() for x in func_dtypes]), name='exec')

    # force llvmlite to allow us to add function attribute then add the attribute
    func.attributes._known = func.attributes._known.union(frozenset(['"no-nans-fp-math"="true"']))
    func.attributes.add('"no-nans-fp-math"="true"')

    bb = [ir.IRBuilder(func.append_basic_block("entry"))]
    loop_blocks = []
    reduce_phis: List = []
    # TODO: newvar probably shouldn't be optional
    lvars: Dict[Optional[str], Any] = {}  # this Any is an llvm type
    render_llvm[Variable] = lambda self,ops,ctx: lvars[self.expr]

    for uop,newvar,args in self.uops:
      if uop == UOps.CONST:
        lvars[newvar] = ir.Constant(ir.FloatType(), args[0])
        reduce_phis.append(newvar)
      if uop == UOps.LOOP:
        for var in args[0]:
          if isinstance(var, NumNode): continue
          bb.append(ir.IRBuilder(func.append_basic_block(f"loop_body_{var.expr}")))
          bb[-2].branch(bb[-1]._block)

          phis = []
          for rp in reduce_phis:
            incoming = lvars[rp]
            lvars[rp] = bb[-1].phi(ir.FloatType())
            lvars[rp].add_incoming(incoming, bb[-2]._block)
            phis.append((rp, lvars[rp]))
          loop_blocks.append((bb[-1], phis))

          lvars[var.expr] = bb[-1].phi(ir.IntType(64), name=var.expr)
          lvars[var.expr].add_incoming(int_const(var.min), bb[-2]._block)
      if uop == UOps.ENDLOOP:
        for var in args[0][::-1]:
          if isinstance(var, NumNode): continue
          block, phis = loop_blocks.pop()
          idx_p1 = bb[-1].add(lvars[var.expr], int_const(1))
          lvars[var.expr].add_incoming(idx_p1, bb[-1]._block)
          for n,phi in phis: phi.add_incoming(lvars[n], bb[-1]._block)
          bb.append(ir.IRBuilder(func.append_basic_block(f"loop_exit_{var.expr}")))
          bb[-2].cbranch(bb[-2].icmp_unsigned("==", idx_p1, int_const(var.max+1)), bb[-1]._block, block._block)
      if uop == UOps.LOAD:
        idx, valid = args[1].render(render_llvm, bb[-1]), args[2].render(render_llvm, bb[-1])
        if args[2].min == 0:
          aug_idx = bb[-1].select(valid, idx, int_const(0))
          lvars[newvar] = bb[-1].select(valid, bb[-1].load(bb[-1].gep(func.args[args[0]], [aug_idx], inbounds=True)), ir.Constant(func_dtypes[args[0]], 0))
        else:
          lvars[newvar] = bb[-1].load(bb[-1].gep(func.args[args[0]], [idx], inbounds=True))
      if uop == UOps.STORE:
        assert args[2].min == 1, "store must be valid"
        idx = args[1].render(render_llvm, bb[-1])
        bb[-1].store(lvars[args[3]], bb[-1].gep(func.args[args[0]], [idx], inbounds=True))
      if uop == UOps.ALU:
        lvars[newvar if newvar is not None else args[2]] = self.code_for_op[args[0]](bb[-1], *[lvars[x] for x in args[1]])

    bb[-1].ret_void()
    return ASTRunner('exec', str(module), op_estimate=self.info.flops, mem_estimate=self.mem_estimate)
