from typing import Final, Dict, Callable, Any, List, Optional
from llvmlite import ir
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import Op, UnaryOps, BinaryOps, TernaryOps

MFLAGS = ('nsz', 'arcp', 'contract', 'afn', 'reassoc') # All from fast math, but nnan and ninf

def is_bool_or_unsigned(dtype: DType): return dtype == dtypes.bool or dtypes.is_unsigned(dtype)

code_for_op: Final[Dict[Op, Callable]] = {
  UnaryOps.NEG: lambda builder, x, var_dtype: builder.neg(x) if dtypes.is_int(var_dtype) else builder.fneg(x, flags=MFLAGS),
  UnaryOps.EXP2: lambda builder, x, var_dtype: builder.call(builder._block.module.declare_intrinsic('llvm.exp2', [x.type]), [x], fastmath=MFLAGS),
  UnaryOps.LOG2: lambda builder, x, var_dtype: builder.call(builder._block.module.declare_intrinsic('llvm.log2', [x.type]), [x], fastmath=MFLAGS),
  UnaryOps.SIN: lambda builder, x, var_dtype: builder.call(builder._block.module.declare_intrinsic('llvm.sin', [x.type]), [x], fastmath=MFLAGS),
  UnaryOps.SQRT: lambda builder, x, var_dtype: builder.call(builder._block.module.declare_intrinsic('llvm.sqrt', [x.type]), [x], fastmath=MFLAGS),
  BinaryOps.ADD: lambda builder, x, y, var_dtype: builder.or_(x, y) if var_dtype == dtypes.bool else builder.add(x, y) if dtypes.is_int(var_dtype) else builder.fadd(x, y, flags=MFLAGS),  # noqa: E501
  BinaryOps.SUB: lambda builder, x, y, var_dtype: builder.sub(x, y) if dtypes.is_int(var_dtype) else builder.fsub(x, y, flags=MFLAGS),
  BinaryOps.MUL: lambda builder, x, y, var_dtype: builder.mul(x, y) if is_bool_or_unsigned(var_dtype) or dtypes.is_int(var_dtype) else builder.fmul(x, y, flags=MFLAGS),  # noqa: E501
  BinaryOps.DIV: lambda builder, x, y, var_dtype: builder.udiv(x, y) if is_bool_or_unsigned(var_dtype) else builder.sdiv(x, y) if dtypes.is_int(var_dtype) else builder.fdiv(x, y, flags=MFLAGS),  # noqa: E501
  BinaryOps.CMPLT: lambda builder, x, y, var_dtype: builder.icmp_unsigned("<", x, y) if is_bool_or_unsigned(var_dtype) else builder.icmp_signed("<", x, y) if dtypes.is_int(var_dtype) else builder.fcmp_unordered("<", x, y, flags=MFLAGS),  # noqa: E501
  BinaryOps.CMPEQ: lambda builder, x, y, var_dtype: builder.icmp_unsigned("==", x, y) if is_bool_or_unsigned(var_dtype) else builder.icmp_signed("==", x, y) if dtypes.is_int(var_dtype) else builder.fcmp_unordered("==", x, y, flags=MFLAGS),  # noqa: E501
  BinaryOps.MAX: lambda builder, x, y, var_dtype: builder.select(builder.icmp_unsigned(">", x, y) if is_bool_or_unsigned(var_dtype) else builder.icmp_signed(">", x, y) if dtypes.is_int(var_dtype) else builder.fcmp_unordered(">", x, y, flags=MFLAGS), x, y),  # noqa: E501
  BinaryOps.MOD: lambda builder, x, y, var_dtype: builder.urem(x, y) if is_bool_or_unsigned(var_dtype) else builder.srem(x, y) if dtypes.is_int(var_dtype) else builder.frem(x, y),  # noqa: E501
  BinaryOps.XOR: lambda builder, x, y, var_dtype: builder.xor(x, y),
  TernaryOps.MULACC: lambda builder, x, y, z, var_dtype: builder.fadd(builder.fmul(x, y, flags=MFLAGS), z, flags=MFLAGS) if dtypes.is_float(var_dtype) else builder.add(builder.mul(x, y), z),  # noqa: E501
  TernaryOps.WHERE: lambda builder, x, y, z, var_dtype: builder.select(x, y, z),
}

dtype_to_llvm_dtype = { dtypes.bool:ir.IntType(1), dtypes.int8:ir.IntType(8), dtypes.uint8:ir.IntType(8), dtypes.int16:ir.IntType(16),
  dtypes.uint16:ir.IntType(16), dtypes.int32:ir.IntType(32), dtypes.uint32:ir.IntType(32), dtypes.int64:ir.IntType(64), dtypes.uint64:ir.IntType(64),
  dtypes.float16:ir.HalfType(), dtypes.bfloat16:ir.IntType(16), dtypes.float32:ir.FloatType(), dtypes.float64:ir.DoubleType() }

def cast(bb, val, input_type, output_type, bitcast=False):
  if input_type == output_type: return val
  llvm_type = dtype_to_llvm_dtype[output_type]
  if bitcast: return bb[-1].bitcast(val, llvm_type)

  if input_type == dtypes.bfloat16:
    val = bb[-1].bitcast(bb[-1].shl(bb[-1].sext(val, ir.IntType(32)), ir.Constant(ir.IntType(32), 16)),val, ir.FloatType())
    input_type = dtypes.float32
  if output_type == dtypes.bfloat16:
    val = cast(bb, val, input_type, dtypes.float32)
    return bb[-1].trunc(bb[-1].lshr(bb[-1].bitcast(val, ir.IntType(32)), ir.Constant(ir.IntType(32), 16)), ir.IntType(16))

  if dtypes.is_float(input_type):
    if dtypes.is_float(output_type):
      return bb[-1].fpext(val, llvm_type) if output_type.itemsize > input_type.itemsize else bb[-1].fptrunc(val, llvm_type)
    if dtypes.is_int(output_type): return bb[-1].fptoui(val, llvm_type) if dtypes.is_unsigned(output_type) else bb[-1].fptosi(val, llvm_type)
    if output_type == dtypes.bool: return bb[-1].fcmp_unordered('!=', cast(bb, val, input_type, dtypes.float32), ir.Constant(ir.FloatType(), 0))

  if dtypes.is_unsigned(input_type) or input_type == dtypes.bool:
    if output_type == dtypes.float16: return bb[-1].fptrunc(bb[-1].uitofp(val, ir.FloatType()), ir.HalfType())
    if dtypes.is_float(output_type): return bb[-1].uitofp(val, dtype_to_llvm_dtype[output_type])
    if dtypes.is_int(output_type): return bb[-1].trunc(val, llvm_type) if input_type.itemsize > output_type.itemsize else bb[-1].zext(val, llvm_type)
    if output_type == dtypes.bool: return bb[-1].icmp_unsigned('!=', val, ir.Constant(val.type, 0))

  if dtypes.is_int(input_type):
    if output_type == dtypes.float16: return bb[-1].fptrunc(bb[-1].sitofp(val, ir.FloatType()), ir.HalfType())
    if dtypes.is_float(output_type): return bb[-1].sitofp(val, llvm_type)
    if dtypes.is_int(output_type): return bb[-1].trunc(val, llvm_type) if input_type.itemsize > output_type.itemsize else bb[-1].sext(val, llvm_type)
    if output_type == dtypes.bool: return bb[-1].icmp_signed('!=', val, ir.Constant(val.type, 0))

  raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

def const(args, dtype):
  # TODO: remove int from int(args) once const args conform with dtype
  return ir.Constant(dtype_to_llvm_dtype[dtype], int(args) if dtypes.is_int(dtype) else bool(args) if dtype == dtypes.bool else args)

def uops_to_llvm_ir(function_name:str, uops:List[UOp]) -> str:
  # all llvm stuff goes into a module
  module = ir.Module(name=__file__)

  # extract global buffers
  buf_to_dtype = {u.arg:u.dtype for u in uops if u.uop == UOps.DEFINE_GLOBAL}
  buf_index = {x:i for i,x in enumerate(buf_to_dtype.keys())}

  # create llvm function
  func_dtypes = [(dtype_to_llvm_dtype[dtype],dtype) for dtype in buf_to_dtype.values() if dtype is not None]
  func = ir.Function(module, ir.FunctionType(ir.VoidType(), [x.as_pointer() if isinstance(dt, PtrDType) else x for x,dt in func_dtypes]), name=function_name)  # noqa: E501
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
    if not isinstance(dtype, PtrDType) and dtype == dtypes.int32: lvars[bufname] = bb[-1].sext(func.args[buf_index[bufname]], ir.IntType(32))

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    if uop is UOps.STORE:
      element = cast(bb, lvars[vin[2]], vin[2].dtype, vin[0].dtype)
      def store_op(): bb[-1].store(element, bb[-1].gep(lvars[vin[0]], [lvars[vin[1]]], inbounds=True))
      if len(vin) > 3:
        with bb[-1].if_then(lvars[vin[3]]): store_op()
      else: store_op()
    elif uop is UOps.ENDLOOP:
      block, phis = loop_blocks.pop()
      idx_p1 = bb[-1].add(lvars[vin[0]], ir.Constant(ir.IntType(32), 1))
      lvars[vin[0]].add_incoming(idx_p1, bb[-1]._block)
      for n,phi in phis: phi.add_incoming(lvars[n], bb[-1]._block)
      bb.append(ir.IRBuilder(func.append_basic_block(f"loop_exit_{len(loop_blocks)}")))
      bb[-2].cbranch(bb[-2].icmp_unsigned("<", idx_p1, lvars[vin[0].vin[1]]), block._block, bb[-1]._block)
    else:
      assert dtype is not None, f"None dtype for uop {uop}"
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
      elif uop is UOps.DEFINE_ACC:
        lvars[u] = const(args, dtype)
        reduce_phis.append(u)
      elif uop is UOps.LOAD:
        if len(vin) > 2:
          aug_idx = bb[-1].select(lvars[vin[2]], lvars[vin[1]], ir.Constant(ir.IntType(32), 0))
          val = bb[-1].load(bb[-1].gep(lvars[vin[0]], [aug_idx], inbounds=True))
          val = bb[-1].select(lvars[vin[2]], val, lvars[vin[3]])
        else:
          val = bb[-1].load(bb[-1].gep(lvars[vin[0]], [lvars[vin[1]]], inbounds=True))
        lvars[u] = val
      elif uop is UOps.PHI:
        lvars[u] = lvars[vin[1]]
        # PHI UOps can link to other PHI Uops, backtrace this to DEFINE_ACC
        backward = vin[0]
        while backward.uop == UOps.PHI: backward = backward.vin[0]
        lvars[backward] = lvars[u]
      elif uop is UOps.ALU:
        lvars[u] = code_for_op[args](bb[-1], *[lvars[x] for x in vin], dtype if args not in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else vin[0].dtype)
      elif uop is UOps.CAST: lvars[u] = cast(bb, lvars[vin[0]], vin[0].dtype, dtype, bitcast=isinstance(args, tuple) and args[1])
      elif uop is UOps.DEFINE_GLOBAL: lvars[u] = func.args[buf_index[args]]
      elif uop is UOps.SPECIAL: lvars[u] = lvars[args.expr]
      elif uop is UOps.CONST: lvars[u] = const(args, dtype)
      else: raise RuntimeError(f"failed to render {uop}")

  bb[-1].ret_void()
  return str(module)
