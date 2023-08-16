from typing import Final, Dict, Callable, Any, List, Optional, Tuple
import functools
from llvmlite import ir  # type: ignore
from tinygrad.codegen.linearizer import UOps, UOp, Token, MemOp, ConstOp
from tinygrad.helpers import dtypes
from tinygrad.ops import Op, UnaryOps, BinaryOps, TernaryOps

from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, LtNode, SumNode, AndNode
def int_const(x): return ir.Constant(ir.IntType(64), x)
render_llvm = {
  NumNode: lambda self,ops,ctx: int_const(self.b),
  MulNode: lambda self,ops,ctx: ctx.mul(self.a.render(ops,ctx), int_const(self.b)),
  DivNode: lambda self,ops,ctx: ctx.sdiv(self.a.render(ops,ctx), int_const(self.b)),
  ModNode: lambda self,ops,ctx: ctx.srem(self.a.render(ops,ctx), int_const(self.b)),
  LtNode: lambda self,ops,ctx: ctx.icmp_signed("<", self.a.render(ops,ctx), int_const(self.b)),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.add(a,b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.and_(a,b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx))
}

code_for_op: Final[Dict[Op, Callable]] = {
  UnaryOps.EXP2: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.exp2', [ir.FloatType()]), [x], fastmath=('fast',)),
  UnaryOps.LOG2: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.log2', [ir.FloatType()]), [x], fastmath=('fast',)),
  UnaryOps.SIN: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.sin', [ir.FloatType()]), [x], fastmath=('fast',)),
  UnaryOps.SQRT: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.sqrt', [ir.FloatType()]), [x], fastmath=('fast',)),
  BinaryOps.ADD: lambda builder,x,y: builder.fadd(x,y, flags=('fast',)),
  BinaryOps.SUB: lambda builder,x,y: builder.fsub(x,y, flags=('fast',)),
  BinaryOps.MUL: lambda builder,x,y: builder.fmul(x,y, flags=('fast',)),
  BinaryOps.DIV: lambda builder,x,y: builder.fdiv(x,y, flags=('fast',)),
  BinaryOps.CMPLT: lambda builder,x,y: builder.uitofp(builder.fcmp_ordered("<", x, y, flags=('fast',)), ir.FloatType()),
  BinaryOps.MAX: lambda builder,x,y: builder.select(builder.fcmp_unordered(">", x, y, flags=('fast',)), x, y, flags=('fast',)),
  TernaryOps.MULACC: lambda builder,x,y,z: builder.fadd(builder.fmul(x,y, flags=('fast',)), z, flags=('fast',)),
  TernaryOps.WHERE: lambda builder,x,y,z: builder.select(builder.fcmp_unordered("!=", x, ir.Constant(ir.FloatType(), 0), flags=('fast',)), y, z, flags=('fast',)),
}

dtype_to_llvm_dtype = {dtypes.float16:ir.HalfType(), dtypes.bfloat16:ir.IntType(16), dtypes.float32:ir.FloatType(), dtypes.int8:ir.IntType(8), dtypes.uint8:ir.IntType(8), dtypes.bool: ir.IntType(1), dtypes.int64: ir.IntType(64), dtypes.int32: ir.IntType(32)}

def cast(bb, val, input_type, output_type):
  if input_type == output_type: return val

  if output_type == dtypes.float32:
    if dtypes.is_int(input_type) or input_type == dtypes.bool:
      val = bb[-1].uitofp(val, ir.FloatType()) if dtypes.is_unsigned(input_type) or input_type == dtypes.bool else bb[-1].sitofp(val, ir.FloatType())
    elif input_type == dtypes.bfloat16:
      val = bb[-1].sext(val, ir.IntType(32))
      val = bb[-1].shl(val, ir.Constant(ir.IntType(32), 16))
      val = bb[-1].bitcast(val, ir.FloatType())
    else:
      val = bb[-1].fpext(val, ir.FloatType())
    return val

  if input_type == dtypes.float32:
    if dtypes.is_int(output_type) or output_type == dtypes.bool:
      val = bb[-1].fptoui(val, dtype_to_llvm_dtype[output_type]) if dtypes.is_unsigned(output_type) or output_type == dtypes.bool else bb[-1].fptosi(val, dtype_to_llvm_dtype[output_type])
    elif output_type == dtypes.bfloat16:
      val = bb[-1].bitcast(val, ir.IntType(32))
      val = bb[-1].lshr(val, ir.Constant(ir.IntType(32), 16))
      val = bb[-1].trunc(val, ir.IntType(16))
    else:
      val = bb[-1].fptrunc(val, dtype_to_llvm_dtype[output_type])
    return val

  raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

def uops_to_llvm_ir(function_name:str, uops:List[UOp]) -> Tuple[str, Optional[List[int]], Optional[List[int]]]:
  # all llvm stuff goes into a module
  module = ir.Module(name=__file__)

  # extract global buffers
  buf_to_dtype = {args[0]:args[1] for uop,_,_,args in uops if uop == UOps.DEFINE_GLOBAL}
  buf_index = {x:i for i,x in enumerate(buf_to_dtype.keys())}

  # create llvm function
  func_dtypes = [dtype_to_llvm_dtype[dtype] for dtype in buf_to_dtype.values()]
  func = ir.Function(module, ir.FunctionType(ir.VoidType(), [x.as_pointer() for x in func_dtypes]), name=function_name)

  # force llvmlite to allow us to add function attribute then add the attribute
  func.attributes._known = func.attributes._known.union(frozenset(['"no-nans-fp-math"="true"']))
  func.attributes.add('"no-nans-fp-math"="true"')

  bb = [ir.IRBuilder(func.append_basic_block("entry"))]
  loop_blocks = []
  reduce_phis: List = []
  # TODO: newvar probably shouldn't be optional
  lvars: Dict[Optional[Token], Any] = {}  # this Any is an llvm type
  render_llvm[Variable] = lambda self,ops,ctx: lvars[self.expr]

  for uop,newvar,vin,args in uops:
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
      assert newvar is not None and isinstance(args, (MemOp, ConstOp))
      valid = args.valid.render(render_llvm, bb[-1])
      if isinstance(args, ConstOp):
        assert newvar.dtype == dtypes.float, "newvar must be float"
        if args.valid.min == 0 and args.valid.max == 1:
          val = bb[-1].select(valid, ir.Constant(ir.FloatType(), args.value), ir.Constant(ir.FloatType(), args.invalid_value))
        else:
          val = ir.Constant(ir.FloatType(), args.value if args.valid.min == 1 else args.invalid_value)
        # TODO: this is a hack. it shouldn't be const that signals this
        reduce_phis.append(newvar)
      else:
        idx = args.idx.render(render_llvm, bb[-1])
        if args.valid.min == 0:
          aug_idx = bb[-1].select(valid, idx, int_const(0))
          val = bb[-1].select(valid, bb[-1].load(bb[-1].gep(func.args[buf_index[args.name]], [aug_idx], inbounds=True)), ir.Constant(dtype_to_llvm_dtype[args.memory_dtype], args.invalid_value))
        else:
          val = bb[-1].load(bb[-1].gep(func.args[buf_index[args.name]], [idx], inbounds=True))
        val = cast(bb, val, args.memory_dtype, newvar.dtype)
      lvars[newvar] = val
    if uop == UOps.STORE:
      assert args.valid.min == 1 and isinstance(args, MemOp), "store must be valid and to memory"
      idx = args.idx.render(render_llvm, bb[-1])
      element = cast(bb, lvars[vin[0]], vin[0].dtype, args.memory_dtype)
      bb[-1].store(element, bb[-1].gep(func.args[buf_index[args.name]], [idx], inbounds=True))
    if uop == UOps.ALU:
      lvars[newvar] = code_for_op[args](bb[-1], *[lvars[x] for x in vin])

  bb[-1].ret_void()
  return str(module), None, None
