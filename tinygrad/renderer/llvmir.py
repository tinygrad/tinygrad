from typing import Final, Dict, Callable, Any, List, Optional, Tuple
from llvmlite import ir
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.helpers import dtypes
from tinygrad.ops import Op, UnaryOps, BinaryOps, TernaryOps

LLVM_FAST_MATH_FLAGS = ('nsz', 'arcp', 'contract', 'afn', 'reassoc') # All from fast math, but nnan and ninf

def is_bool(t:ir.Type): return isinstance(t, ir.IntType) and t.width == 1
code_for_op: Final[Dict[Op, Callable]] = {
  UnaryOps.NEG: lambda builder,x: builder.xor(x, ir.Constant(ir.IntType(1), 1)) if is_bool(x.type) else builder.neg(x) if isinstance(x.type, ir.IntType) else builder.fneg(x, flags=LLVM_FAST_MATH_FLAGS),
  UnaryOps.EXP2: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.exp2', [ir.FloatType()]), [x], fastmath=LLVM_FAST_MATH_FLAGS),
  UnaryOps.LOG2: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.log2', [ir.FloatType()]), [x], fastmath=LLVM_FAST_MATH_FLAGS),
  UnaryOps.SIN: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.sin', [ir.FloatType()]), [x], fastmath=LLVM_FAST_MATH_FLAGS),
  UnaryOps.SQRT: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.sqrt', [ir.FloatType()]), [x], fastmath=LLVM_FAST_MATH_FLAGS),
  BinaryOps.ADD: lambda builder,x,y: builder.add(x,y) if isinstance(x.type, ir.IntType) else builder.fadd(x,y, flags=LLVM_FAST_MATH_FLAGS),
  BinaryOps.SUB: lambda builder,x,y: builder.sub(x,y) if isinstance(x.type, ir.IntType) else builder.fsub(x,y, flags=LLVM_FAST_MATH_FLAGS),
  BinaryOps.MUL: lambda builder,x,y: builder.mul(x,y) if isinstance(x.type, ir.IntType) else builder.fmul(x,y, flags=LLVM_FAST_MATH_FLAGS),
  BinaryOps.DIV: lambda builder,x,y: builder.sdiv(x,y) if isinstance(x.type, ir.IntType) else builder.fdiv(x,y, flags=LLVM_FAST_MATH_FLAGS),
  BinaryOps.CMPLT: lambda builder,x,y: builder.icmp_unsigned("<", x, y) if is_bool(x.type) else builder.icmp_signed("<", x, y) if isinstance(x.type, ir.IntType) else builder.fcmp_unordered("<", x, y, flags=LLVM_FAST_MATH_FLAGS),
  BinaryOps.MAX: lambda builder,x,y: builder.select(builder.icmp_unsigned(">", x, y) if is_bool(x.type) else builder.icmp_signed(">", x, y) if isinstance(x.type, ir.IntType) else builder.fcmp_unordered(">", x, y, flags=LLVM_FAST_MATH_FLAGS), x, y),
  BinaryOps.MOD: lambda builder,x,y: builder.urem(x,y) if is_bool(x.type) else builder.srem(x,y) if isinstance(x.type, ir.IntType) else builder.frem(x,y),
  TernaryOps.MULACC: lambda builder,x,y,z: builder.fadd(builder.fmul(x,y, flags=LLVM_FAST_MATH_FLAGS), z, flags=LLVM_FAST_MATH_FLAGS),
  TernaryOps.WHERE: lambda builder,x,y,z: builder.select(builder.trunc(x, ir.IntType(1)) if isinstance(x.type, ir.IntType) else builder.fcmp_unordered("!=", x, ir.Constant(ir.FloatType(), 0), flags=LLVM_FAST_MATH_FLAGS), y, z
  ),
}

dtype_to_llvm_dtype = {dtypes.float64:ir.DoubleType(), dtypes.float16:ir.HalfType(), dtypes.bfloat16:ir.IntType(16), dtypes.float32:ir.FloatType(), dtypes.int8:ir.IntType(8), dtypes.uint8:ir.IntType(8), dtypes.bool: ir.IntType(1), dtypes.int64: ir.IntType(64), dtypes.int32: ir.IntType(32), dtypes._arg_int32: ir.IntType(32), dtypes.int16:ir.IntType(16), dtypes.uint16:ir.IntType(16), dtypes.uint32:ir.IntType(32), dtypes.uint64:ir.IntType(64)}

def cast(bb, val, input_type, output_type):
  if input_type == output_type: return val

  if dtypes.is_float(input_type):
    if dtypes.is_float(output_type):
      if output_type.itemsize > input_type.itemsize: return bb[-1].fpext(val, dtype_to_llvm_dtype[output_type])
      return bb[-1].fptrunc(val, dtype_to_llvm_dtype[output_type])
    if dtypes.is_int(output_type):
      if dtypes.is_unsigned(output_type): return bb[-1].fptoui(val, dtype_to_llvm_dtype[output_type])
      return bb[-1].fptosi(val, dtype_to_llvm_dtype[output_type])
    if output_type == dtypes.bool: return bb[-1].fcmp_unordered('!=', cast(bb, val, input_type, dtypes.float32), ir.Constant(ir.FloatType(), 0))

  if dtypes.is_unsigned(input_type) or input_type == dtypes.bool:
    if output_type == dtypes.float16:
      val = bb[-1].uitofp(val, ir.FloatType())
      return bb[-1].fptrunc(val, ir.HalfType())
    if dtypes.is_float(output_type): return bb[-1].uitofp(val, dtype_to_llvm_dtype[output_type])
    if dtypes.is_int(output_type):
      if input_type.itemsize > output_type.itemsize: return bb[-1].trunc(val, dtype_to_llvm_dtype[output_type])
      return bb[-1].zext(val, dtype_to_llvm_dtype[output_type])
    if output_type == dtypes.bool: return bb[-1].icmp_unsigned('!=', val, ir.Constant(val.type, 0))

  if dtypes.is_int(input_type):
    if output_type == dtypes.float16:
      val = bb[-1].sitofp(val, ir.FloatType())
      return bb[-1].fptrunc(val, ir.HalfType())
    if dtypes.is_float(output_type): return bb[-1].sitofp(val, dtype_to_llvm_dtype[output_type])
    if dtypes.is_int(output_type):
      if input_type.itemsize > output_type.itemsize: return bb[-1].trunc(val, dtype_to_llvm_dtype[output_type])
      return bb[-1].sext(val, dtype_to_llvm_dtype[output_type])
    if output_type == dtypes.bool: return bb[-1].icmp_signed('!=', val, ir.Constant(val.type, 0))

  raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

def const(args, dtype): return ir.Constant(dtype_to_llvm_dtype[dtype], int(args) if dtypes.is_int(dtype) else bool(args) if dtype == dtypes.bool else args)

def uops_to_llvm_ir(function_name:str, uops:List[UOp]) -> Tuple[str, Dict]:
  # all llvm stuff goes into a module
  module = ir.Module(name=__file__)

  # extract global buffers
  buf_to_dtype = {u.arg[0]:u.arg[1] for u in uops if u.uop == UOps.DEFINE_GLOBAL}
  buf_index = {x:i for i,x in enumerate(buf_to_dtype.keys())}

  # create llvm function
  func_dtypes = [(dtype_to_llvm_dtype[dtype],dtype) for dtype in buf_to_dtype.values()]
  func = ir.Function(module, ir.FunctionType(ir.VoidType(), [x.as_pointer() if dt!=dtypes._arg_int32 else x for x,dt in func_dtypes]), name=function_name)
  for a in func.args:
    if a.type.is_pointer: a.add_attribute("noalias")

  # add the function attribute "no-nans-fp-math"="true", which informs llvm that it allowed to use vectorization optimizations
  func.attributes._known = func.attributes._known.union(frozenset(['"no-nans-fp-math"="true"']))
  func.attributes.add('"no-nans-fp-math"="true"')

  bb = [ir.IRBuilder(func.append_basic_block("entry"))]
  loop_blocks: List = []
  reduce_phis: List = []
  # TODO: newvar probably shouldn't be optional
  lvars: Dict[Optional[UOp], Any] = {}  # this Any is an llvm type

  for bufname,dtype in buf_to_dtype.items():
    if dtype == dtypes._arg_int32: lvars[bufname] = bb[-1].sext(func.args[buf_index[bufname]], ir.IntType(32))

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    if uop == UOps.LOOP:
      bb.append(ir.IRBuilder(func.append_basic_block(f"loop_body_{len(loop_blocks)}")))
      bb[-2].branch(bb[-1]._block)

      phis = []
      for rp in reduce_phis:
        incoming = lvars[rp]
        lvars[rp] = bb[-1].phi(dtype_to_llvm_dtype[rp.dtype])
        lvars[rp].add_incoming(incoming, bb[-2]._block)
        phis.append((rp, lvars[rp]))

      lvars[u] = bb[-1].phi(ir.IntType(32), name=f"loop{len(loop_blocks)}")
      lvars[u].add_incoming(lvars[vin[0]], bb[-2]._block)
      loop_blocks.append((bb[-1], phis))
    if uop == UOps.END:
      block, phis = loop_blocks.pop()
      idx_p1 = bb[-1].add(lvars[vin[0]], ir.Constant(ir.IntType(32), 1))
      lvars[vin[0]].add_incoming(idx_p1, bb[-1]._block)
      for n,phi in phis: phi.add_incoming(lvars[n], bb[-1]._block)
      bb.append(ir.IRBuilder(func.append_basic_block(f"loop_exit_{len(loop_blocks)}")))
      bb[-2].cbranch(bb[-2].icmp_unsigned("<", idx_p1, lvars[vin[0].vin[1]]), block._block, bb[-1]._block)
    if uop == UOps.DEFINE_GLOBAL:
      lvars[u] = func.args[buf_index[args[0]]]
    if uop == UOps.DEFINE_ACC:
      lvars[u] = const(args, dtype)
      reduce_phis.append(u)
    if uop == UOps.SPECIAL:
      lvars[u] = lvars[args.expr]
    if uop == UOps.CONST:
      lvars[u] = const(args, dtype)
    if uop == UOps.LOAD:
      assert dtype is not None
      if len(vin) > 2:
        gate = bb[-1].trunc(lvars[vin[2]], ir.IntType(1))
        aug_idx = bb[-1].select(gate, lvars[vin[1]], ir.Constant(ir.IntType(32), 0))
        val = bb[-1].load(bb[-1].gep(lvars[vin[0]], [aug_idx], inbounds=True))
        val = cast(bb, val, vin[0].dtype, dtype)
        val = bb[-1].select(gate, val, lvars[vin[3]])
      else:
        val = bb[-1].load(bb[-1].gep(lvars[vin[0]], [lvars[vin[1]]], inbounds=True))
        val = cast(bb, val, vin[0].dtype, dtype)
      lvars[u] = val
    if uop == UOps.PHI:
      lvars[u] = lvars[vin[1]]
      # PHI UOps can link to other PHI Uops, backtrace this to DEFINE_ACC
      backward = vin[0]
      while backward.uop == UOps.PHI: backward = backward.vin[0]
      lvars[backward] = lvars[u]
    if uop == UOps.STORE:
      element = cast(bb, lvars[vin[2]], vin[2].dtype, vin[0].dtype)
      def store_op(): bb[-1].store(element, bb[-1].gep(lvars[vin[0]], [lvars[vin[1]]], inbounds=True))
      if len(vin) > 3:
        with bb[-1].if_then(bb[-1].trunc(lvars[vin[3]], ir.IntType(1))): store_op()
      else: store_op()
    if uop == UOps.ALU:
      lvars[u] = code_for_op[args](bb[-1], *[lvars[x] for x in vin])
    if uop == UOps.CAST: lvars[u] = cast(bb, lvars[vin[0]], vin[0].dtype, dtype)

  bb[-1].ret_void()
  return str(module), {}
