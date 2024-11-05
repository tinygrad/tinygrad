from typing import Dict, Callable, List, Optional
from llvmlite import ir
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer

MFLAGS = ('nsz', 'arcp', 'contract', 'afn') # All from fast math, but nnan and ninf and reassoc

def is_bool_or_unsigned(dtype: DType): return dtype == dtypes.bool or dtypes.is_unsigned(dtype)

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

def const(args, dtype): return ir.Constant(dtype_to_llvm_dtype[dtype], args)

class LLVMRenderer(Renderer):
  device = "LLVM"
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None
  code_for_op: Dict[Ops, Callable] = {
    UnaryOps.RECIP: lambda builder, x, dtype: builder.fdiv(const(1, dtype), x, flags=MFLAGS),
    UnaryOps.SQRT: lambda builder, x, dtype: builder.call(builder.module.declare_intrinsic('llvm.sqrt', [x.type]), [x], fastmath=MFLAGS),
    BinaryOps.ADD: lambda builder, x, y, dtype: builder.or_(x, y) if dtype == dtypes.bool else builder.add(x, y) if dtypes.is_int(dtype) else builder.fadd(x, y, flags=MFLAGS),  # noqa: E501
    BinaryOps.MUL: lambda builder, x, y, dtype: builder.mul(x, y) if is_bool_or_unsigned(dtype) or dtypes.is_int(dtype) else builder.fmul(x, y, flags=MFLAGS),  # noqa: E501
    BinaryOps.IDIV: lambda builder, x, y, dtype: builder.udiv(x, y) if is_bool_or_unsigned(dtype) else builder.sdiv(x, y),
    BinaryOps.CMPLT: lambda builder, x, y, dtype: builder.icmp_unsigned("<", x, y) if is_bool_or_unsigned(dtype) else builder.icmp_signed("<", x, y) if dtypes.is_int(dtype) else builder.fcmp_unordered("<", x, y, flags=MFLAGS),  # noqa: E501
    BinaryOps.CMPNE: lambda builder, x, y, dtype: builder.icmp_unsigned("!=", x, y) if is_bool_or_unsigned(dtype) else builder.icmp_signed("!=", x, y) if dtypes.is_int(dtype) else builder.fcmp_unordered("!=", x, y, flags=MFLAGS),  # noqa: E501
    BinaryOps.MAX: lambda builder, x, y, dtype: builder.select(builder.icmp_unsigned(">", x, y) if is_bool_or_unsigned(dtype) else builder.icmp_signed(">", x, y) if dtypes.is_int(dtype) else builder.fcmp_unordered(">", x, y, flags=MFLAGS), x, y),  # noqa: E501
    BinaryOps.MOD: lambda builder, x, y, dtype: builder.urem(x, y) if is_bool_or_unsigned(dtype) else builder.srem(x, y) if dtypes.is_int(dtype) else builder.frem(x, y),  # noqa: E501
    BinaryOps.XOR: lambda builder, x, y, dtype: builder.xor(x, y), BinaryOps.AND: lambda builder, x, y, dtype: builder.and_(x, y), BinaryOps.OR: lambda builder, x, y, dtype: builder.or_(x, y), # noqa: E501
    BinaryOps.SHL: lambda builder, x, y, dtype: builder.shl(x, y), BinaryOps.SHR: lambda builder, x, y, dtype: builder.lshr(x, y) if dtypes.is_unsigned(dtype) else builder.ashr(x, y), # noqa: E501
    TernaryOps.WHERE: lambda builder, x, y, z, dtype: builder.select(x, y, z)}

  def render(self, name:str, uops:List[UOp]) -> str:
    # all llvm stuff goes into a module
    module = ir.Module(name=__file__)

    # extract global buffers (NOTE: this isn't right if DEFINE_GLOBAL is out of order)
    buf_to_dtype = {u.arg:u.dtype for u in uops if u.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR}}
    buf_index = {x:i for i,x in enumerate(buf_to_dtype.keys())}

    # create llvm function
    func_dtypes = [(dtype_to_llvm_dtype[dtype.base if isinstance(dtype, PtrDType) else dtype],dtype) for dtype in buf_to_dtype.values()]
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [x.as_pointer() if isinstance(dt, PtrDType) else x for x,dt in func_dtypes]), name=name)
    for a in func.args:
      if a.type.is_pointer: a.add_attribute("noalias")

    bb = [ir.IRBuilder(func.append_basic_block("entry"))]
    loop_blocks: List = []
    reduce_phis: List = []
    lvars: Dict[Optional[UOp], ir.Instruction] = {}

    for bufname,dtype in buf_to_dtype.items():
      if not isinstance(dtype, PtrDType) and dtype == dtypes.int32: lvars[bufname] = bb[-1].sext(func.args[buf_index[bufname]], ir.IntType(32))

    for u in uops:
      uop,dtype,src,args = u.op,u.dtype,u.src,u.arg
      if uop is Ops.INDEX:
        lvars[u] = bb[-1].gep(lvars[src[0]], [lvars[src[1]]], inbounds=True)
      elif uop is Ops.STORE:
        if len(src) > 2:
          with bb[-1].if_then(lvars[src[2]]): bb[-1].store(lvars[src[1]], lvars[src[0]])
        else:
          bb[-1].store(lvars[src[1]], lvars[src[0]])
      elif uop is Ops.ENDRANGE:
        loop_entry_bb, phis = loop_blocks.pop()
        idx_p1 = bb[-1].add(lvars[src[0]], ir.Constant(ir.IntType(32), 1))
        lvars[src[0]].add_incoming(idx_p1, bb[-1].block)
        for n,phi in phis: phi.add_incoming(lvars[n], bb[-1].block)
        bb.append(ir.IRBuilder(func.append_basic_block(f"loop_exit_{len(loop_blocks)}")))
        bb[-2].cbranch(bb[-2].icmp_unsigned("<", idx_p1, lvars[src[0].src[1]]), loop_entry_bb, bb[-1].block)
      else:
        if uop is Ops.RANGE:
          bb.append(ir.IRBuilder(func.append_basic_block(f"loop_body_{len(loop_blocks)}")))
          bb[-2].branch(bb[-1].block)

          phis = []
          for rp in reduce_phis:
            incoming = lvars[rp]
            lvars[rp] = bb[-1].phi(dtype_to_llvm_dtype[rp.dtype])
            lvars[rp].add_incoming(incoming, bb[-2].block)
            phis.append((rp, lvars[rp]))

          lvars[u] = bb[-1].phi(ir.IntType(32), name=f"loop{len(loop_blocks)}")
          lvars[u].add_incoming(lvars[src[0]], bb[-2].block)
          loop_blocks.append((bb[-1].block, phis))
        elif uop is Ops.DEFINE_ACC:
          lvars[u] = const(src[0].arg, dtype)
          reduce_phis.append(u)
        elif uop is Ops.LOAD:
          if len(src) > 1:
            with bb[-1].if_else(lvars[src[2]]) as (then, otherwise):
              with then:
                val1 = bb[-1].load(lvars[src[0]])
                then_blk = bb[-1].block
              with otherwise: otherwise_blk = bb[-1].block
            val = bb[-1].phi(val1.type)
            val.add_incoming(val1, then_blk)
            val.add_incoming(lvars[src[1]], otherwise_blk)
          else:
            val = bb[-1].load(lvars[src[0]])
          lvars[u] = val
        elif uop is Ops.ASSIGN:
          lvars[u] = lvars[src[1]]
          # ASSIGN UOps can link to other ASSIGN Uops, backtrace this to DEFINE_ACC
          backward = src[0]
          while backward.op is Ops.ASSIGN: backward = backward.src[0]
          lvars[backward] = lvars[u]
        elif uop in GroupOp.ALU:
          lvars[u] = self.code_for_op[uop](bb[-1], *[lvars[x] for x in src], src[0].dtype if uop in {BinaryOps.CMPLT, BinaryOps.CMPNE} else dtype)
        elif uop in {Ops.CAST, Ops.BITCAST}: lvars[u] = cast(bb, lvars[src[0]], src[0].dtype, dtype, bitcast=uop is Ops.BITCAST)
        elif uop in {Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR}: lvars[u] = func.args[buf_index[args]]
        elif uop is Ops.CONST: lvars[u] = const(args, dtype)
        else: raise RuntimeError(f"failed to render {uop}")

    bb[-1].ret_void()
    return str(module)
