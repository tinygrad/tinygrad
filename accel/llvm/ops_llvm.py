# https://github.com/numba/llvmlite/blob/main/llvmlite/ir/builder.py
from __future__ import annotations
import hashlib
import math
from typing import Tuple, Union, Dict, List
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker, ZeroView
from tinygrad.ops import LazyOp
import ctypes
import numpy as np
from llvmlite import ir
from ctypes import CFUNCTYPE
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps, get_buffers, get_lazyops
import llvmlite.binding as llvm

int_const = lambda x: ir.Constant(ir.IntType(64), x)
def idx_deref(builder, buf, ptr, idx):
  if DEBUG >= 1:
    print(buf.st.expr(), ptr)
  # TODO: unify this with expr in ShapeTracker
  valid = None
  for v in buf.st.views[::-1]:
    if isinstance(v, ZeroView):
      if valid is None:
        valid = ir.Constant(ir.IntType(1), 1)
      acc = 1
      for s,(x,y) in list(zip(v.old_shape, v.arg))[::-1]:
        if x < 0 or y > s:
          lr = idx
          if acc != 1:
            lr = builder.sdiv(lr, int_const(acc))
          lr = builder.srem(lr, int_const(y-x))
          if x != 0:
            lr = builder.add(lr, int_const(x))
          if x < 0:
            valid = builder.and_(valid, builder.icmp_signed(">=", lr, int_const(0)))
          if y > s:
            valid = builder.and_(valid, builder.icmp_signed("<", lr, int_const(s)))
        acc *= y-x
    else:
      acc = 1
      ret = int_const(v.offset)
      for i,(d,s) in enumerate(v.shape_strides[::-1]):
        if d != 1 and s != 0:
          lr = idx
          if acc != 1:
            lr = builder.sdiv(lr, int_const(acc))
          lr = builder.srem(lr, int_const(d))
          if s != 1:
            lr = builder.mul(lr, int_const(s))
          ret = builder.add(ret, lr)
        acc *= d
      idx = ret
  if valid is not None:
    return builder.select(valid, builder.load(builder.gep(ptr, [idx])), ir.Constant(ir.FloatType(), 0))
  else:
    return builder.load(builder.gep(ptr, [idx]))

target_machine, engine = None, None
def init_llvm():
  global target_machine, engine
  llvm.initialize()
  llvm.initialize_native_target()
  llvm.initialize_native_asmprinter()  # yes, even this one
  target = llvm.Target.from_default_triple()
  target_machine = target.create_target_machine()
  engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), target_machine)

  # cache
  def notify_func(module, buffer):
    #print("notify", module.name)
    with open(f"/tmp/llvmcache/{module.name}", "wb") as f:
      f.write(buffer)
  def getbuffer_func(module):
    #print("getbuffer", module.name)
    try:
      with open(f"/tmp/llvmcache/{module.name}", "rb") as f:
        return f.read()
    except FileNotFoundError:
      return None
  # enable cache
  #engine.set_object_cache(notify_func, getbuffer_func)

# TODO: write this
# TODO: Refactor LLVMBuffer and GPUBuffer into ShapeTrackedBuffer
class LLVMBuffer:
  op_lookup = {
    UnaryOps.NOOP: lambda builder,x: x,
    UnaryOps.NEG: lambda builder,x: builder.fneg(x),
    UnaryOps.RELU: lambda builder,x: builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x), x, ir.Constant(ir.FloatType(), 0)),
    UnaryOps.EXP: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.exp', [ir.FloatType()]), [x]),
    UnaryOps.LOG: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.log', [ir.FloatType()]), [x]),
    UnaryOps.SIGN: lambda builder,x: builder.select(builder.fcmp_ordered("==", x, ir.Constant(ir.FloatType(), 0)), ir.Constant(ir.FloatType(), 0),
                                                    builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x), ir.Constant(ir.FloatType(), 1), ir.Constant(ir.FloatType(), -1))),
    UnaryOps.RECIPROCAL: lambda builder,x: builder.fdiv(ir.Constant(ir.FloatType(), 1), x),
    BinaryOps.ADD: lambda builder,x,y: builder.fadd(x,y),
    BinaryOps.SUB: lambda builder,x,y: builder.fsub(x,y),
    BinaryOps.MUL: lambda builder,x,y: builder.fmul(x,y),
    BinaryOps.DIV: lambda builder,x,y: builder.fdiv(x,y),
    BinaryOps.POW: lambda builder,x,y: builder.call(builder._block.module.declare_intrinsic('llvm.pow', [ir.FloatType()]), [x,y]),
    BinaryOps.CMPEQ: lambda builder,x,y: builder.uitofp(builder.fcmp_ordered("==", x, y), ir.FloatType()),
  }
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self._buf = (ctypes.c_float * (prod(self.shape)))() if hostbuf is None else hostbuf._buf

  # universal for shape tracked
  def movement_op(x, op:MovementOps, arg): return type(x)(ShapeTracker(x.st).movement_op(op, arg), x)

  # universal
  def unary_op(x, op:UnaryOps): return type(x)(x.shape).exec_ast(LazyOp(op=op, src=(x,)))
  def binary_op(x, op:BinaryOps, y): return type(x)(x.shape).exec_ast(LazyOp(op=op, src=(x, y)))
  def reduce_op(x, op:ReduceOps, new_shape:Tuple[int, ...]): return type(x)(new_shape).exec_ast(LazyOp(op=op, src=(x,)))
  def contiguous_op(x): return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)

  @staticmethod
  def fromCPU(x):
    ret = LLVMBuffer(x.shape)
    ctypes.memmove(ret._buf, x.astype(np.float32).ctypes.data, prod(ret.shape)*4)
    return ret
  
  def toCPU(x): return np.ctypeslib.as_array(x.contiguous_op()._buf)[:prod(x.shape)].reshape(x.shape).copy()

  # ast can contain one ReduceOp with arbitrary Binary/Unary ops
  def exec_ast(ret, ast:Union[LLVMBuffer, LazyOp]):
    # get the real buffers from the ast
    bufs = get_buffers(ast)
    reduceops = [x for x in get_lazyops(ast) if isinstance(x.op, ReduceOps)]
    assert len(reduceops) <= 1, "max one reduce op in an ast"
    earlybufs = get_buffers(reduceops[0]) if len(reduceops) > 0 else []

    if engine is None:
      init_llvm()
    module = ir.Module(name=__file__)
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [ir.PointerType(ir.FloatType())]*(1+len(bufs))), name='exec')

    # enter
    start_builder = ir.IRBuilder(func.append_basic_block(name="entry"))
    body_builder = ir.IRBuilder(func.append_basic_block(name="inner_loop"))
    start_builder.branch(body_builder._block)

    idx = body_builder.phi(ir.IntType(64))
    idx.add_incoming(ir.Constant(ir.IntType(64), 0), start_builder._block)

    reduce_builder = ir.IRBuilder(func.append_basic_block(name="reduce_loop"))
    store_builder = ir.IRBuilder(func.append_basic_block(name="store_block"))

    def ast_parse(builder, x, idx, reduce_result=None, depth=0):
      if DEBUG >= 1:
        print("  "*depth+"ast:", reduce_result, x)
      if not isinstance(x, LazyOp):
        return idx_deref(builder, x, func.args[bufs.index(x)+1], idx)
      if isinstance(x.op, ReduceOps):
        if reduce_result is None:
          raise Exception("no reduce")
        return reduce_result
      values = [ast_parse(builder, v, idx, reduce_result, depth=depth+1) for v in x.src]
      return LLVMBuffer.op_lookup[x.op](builder, *values)

    if len(reduceops) > 0:
      assert len(earlybufs[0].shape) == len(ret.shape), "reduce only possible on matching shapes"
      red = prod([s for s,n in zip(earlybufs[0].shape, ret.shape) if n == 1])
      red_idx_start = body_builder.mul(idx, int_const(red))
      red_idx_end = body_builder.add(red_idx_start, int_const(red-1))
      red_idx = reduce_builder.phi(ir.IntType(64))
      val = reduce_builder.phi(ir.FloatType())
      red_idx.add_incoming(red_idx_start, body_builder._block)
      reduce_input = ast_parse(reduce_builder, reduceops[0].src[0], red_idx)

      if reduceops[0].op == ReduceOps.SUM:
        val.add_incoming(ir.Constant(ir.FloatType(), 0), body_builder._block)
        reduce_result = reduce_builder.fadd(reduce_input, val)
      elif reduceops[0].op == ReduceOps.MAX:
        val.add_incoming(ir.Constant(ir.FloatType(), -math.inf), body_builder._block)
        reduce_result = reduce_builder.call(ir.Function(module, ir.FunctionType(ir.FloatType(), [ir.FloatType(), ir.FloatType()]), name="llvm.maxnum.f32"), [reduce_input, val])
      else:
        raise Exception(f"unknown ReduceOps {ast.op}")
      val.add_incoming(reduce_result, reduce_builder._block)

      red_idx_p1 = reduce_builder.add(red_idx, int_const(1))
      red_idx.add_incoming(red_idx_p1, reduce_builder._block)
      reduce_builder.cbranch(reduce_builder.icmp_unsigned("==", red_idx, red_idx_end), store_builder._block, reduce_builder._block)
    else:
      reduce_result = None
      reduce_builder.branch(store_builder._block)

    body_builder.branch(reduce_builder._block)
    result = ast_parse(store_builder, ast, idx, reduce_result)
    store_builder.store(result, store_builder.gep(func.args[0], [idx]))
    idx_new = store_builder.add(idx, ir.Constant(ir.IntType(64), 1))
    idx.add_incoming(idx_new, store_builder._block)

    exit_builder = ir.IRBuilder(func.append_basic_block(name="exit"))
    exit_builder.ret_void()

    store_builder.cbranch(store_builder.icmp_unsigned("==", idx, ir.Constant(ir.IntType(64), prod(ret.shape)-1)), exit_builder._block, body_builder._block)

    # **** llvm running ****
    llvm_ir = str(module)
    if DEBUG >= 2:
      print(llvm_ir)

    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    mod.name = hashlib.sha1(llvm_ir.encode('utf-8')).hexdigest()
    if DEBUG >= 3:
      print(target_machine.emit_assembly(mod))
    engine.add_module(mod)
    engine.finalize_object()

    # call function
    bufs = [ret] + bufs
    cfunc = CFUNCTYPE(ctypes.c_int, *[ctypes.POINTER(ctypes.c_float) for _ in bufs])(engine.get_function_address('exec'))
    cfunc(*[x._buf for x in bufs])

    # we are done
    engine.remove_module(mod)

    return ret
