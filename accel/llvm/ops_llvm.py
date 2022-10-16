# https://github.com/numba/llvmlite/blob/main/llvmlite/ir/builder.py
from __future__ import annotations
import os
import math
from typing import Tuple, Union
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker, ZeroView
import ctypes
import numpy as np
from llvmlite import ir
from ctypes import CFUNCTYPE
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps
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

# TODO: write this
# TODO: Refactor LLVMBuffer and GPUBuffer into ShapeTrackedBuffer
class LLVMBuffer:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self._buf = (ctypes.c_float * (prod(self.shape)))() if hostbuf is None else hostbuf._buf

  # copied from GPUBuffer
  def movement_op(x, op:MovementOps, arg): return type(x)(ShapeTracker(x.st).movement_op(op, arg), x)
  def contiguous_op(x): return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)

  def unary_op(x, op:UnaryOps): return type(x)(x.shape)._dumb_processing_op([x], op)
  def binary_op(x, op:BinaryOps, y): return type(x)(x.shape)._dumb_processing_op([x, y], op)
  def reduce_op(x, op:ReduceOps, new_shape:Tuple[int, ...]):
    return type(x)(new_shape)._dumb_processing_op([x], op)
    #._processing_op([("A", x)], code="acc", earlycode=GPUBuffer.code_for_op[op], earlybufs=set("A"), op=op)

  @staticmethod
  def fromCPU(x):
    ret = LLVMBuffer(x.shape)
    ctypes.memmove(ret._buf, x.ctypes.data, prod(ret.shape)*4)
    return ret
  
  def toCPU(x): return np.ctypeslib.as_array(x.contiguous_op()._buf)[:prod(x.shape)].reshape(x.shape).copy()

  function_idx = 0
  def _dumb_processing_op(ret, bufs, op):
    global nogc
    name = f"fpadd_{LLVMBuffer.function_idx}"
    LLVMBuffer.function_idx += 1

    if engine is None:
      init_llvm()

    module = ir.Module(name=__file__)
    llvm_pow = module.declare_intrinsic('llvm.pow', [ir.FloatType()])
    llvm_log = module.declare_intrinsic('llvm.log', [ir.FloatType()])
    llvm_exp = module.declare_intrinsic('llvm.exp', [ir.FloatType()])
    llvm_maxnum = ir.Function(module, ir.FunctionType(ir.FloatType(), [ir.FloatType(), ir.FloatType()]), name="llvm.maxnum.f32")

    typ = ir.PointerType(ir.FloatType())
    fnty = ir.FunctionType(ir.VoidType(), [typ]*(1+len(bufs)))
    func = ir.Function(module, fnty, name=name)

    start_block = func.append_basic_block(name="entry")
    block = func.append_basic_block(name="inner_loop")
    builder = ir.IRBuilder(block)
    start_builder = ir.IRBuilder(start_block)
    start_builder.branch(block)

    start = ir.Constant(ir.IntType(64), 0)
    end = ir.Constant(ir.IntType(64), prod(ret.shape)-1)
    idx = builder.phi(ir.IntType(64))
    idx.add_incoming(start, start_block)

    op_lookup = {
      BinaryOps.ADD: builder.fadd,
      BinaryOps.SUB: builder.fsub,
      BinaryOps.MUL: builder.fmul,
      BinaryOps.DIV: builder.fdiv,
      BinaryOps.POW: lambda x,y: builder.call(llvm_pow, [x,y]),
      BinaryOps.CMPEQ: lambda x,y: builder.uitofp(builder.fcmp_ordered("==", x, y), ir.FloatType()),
      UnaryOps.LOG: lambda x: builder.call(llvm_log, [x]),
      UnaryOps.EXP: lambda x: builder.call(llvm_exp, [x]),
      UnaryOps.NOOP: lambda x: x,
      UnaryOps.NEG: builder.fneg,
      UnaryOps.RECIPROCAL: lambda x: builder.fdiv(ir.Constant(ir.FloatType(), 1), x),
      UnaryOps.RELU: lambda x: builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x), x, ir.Constant(ir.FloatType(), 0)),
      UnaryOps.SIGN: lambda x: builder.select(builder.fcmp_ordered("==", x, ir.Constant(ir.FloatType(), 0)), ir.Constant(ir.FloatType(), 0), builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x), ir.Constant(ir.FloatType(), 1), ir.Constant(ir.FloatType(), -1)))
    }

    reduce_block = func.append_basic_block(name="reduce_loop")
    reduce_builder = ir.IRBuilder(reduce_block)
    store_block = func.append_basic_block(name="store_block")
    store_builder = ir.IRBuilder(store_block)

    if isinstance(op, ReduceOps):
      red = prod([s for s,n in zip(bufs[0].shape, ret.shape) if n == 1])
      red_idx_start = builder.mul(idx, int_const(red))
      # why is this not minus 1?
      red_idx_end = builder.add(red_idx_start, int_const(red-1))
      red_idx = reduce_builder.phi(ir.IntType(64))
      red_idx.add_incoming(red_idx_start, block)

      val = reduce_builder.phi(ir.FloatType())
      tval = idx_deref(reduce_builder, bufs[0], func.args[1], red_idx)
      if op == ReduceOps.SUM:
        new_val = reduce_builder.fadd(tval, val)
        val.add_incoming(ir.Constant(ir.FloatType(), 0), block)
        val.add_incoming(new_val, reduce_block)
      elif op == ReduceOps.MAX:
        new_val = reduce_builder.call(llvm_maxnum, [tval, val])
        val.add_incoming(ir.Constant(ir.FloatType(), -math.inf), block)
        val.add_incoming(new_val, reduce_block)
      else:
        raise Exception(f"unknown op {op}")

      red_idx_p1 = reduce_builder.add(red_idx, int_const(1))
      red_idx.add_incoming(red_idx_p1, reduce_block)
      reduce_builder.cbranch(reduce_builder.icmp_unsigned("==", red_idx, red_idx_end), store_block, reduce_block)
      val = new_val
    elif op in op_lookup:
      values = [idx_deref(builder, buf, ptr, idx) for buf, ptr in zip(bufs, func.args[1:])]
      val = op_lookup[op](*values)
      reduce_builder.branch(store_block)
    else:
      raise NotImplementedError(f"{op} not implemented in LLVM backend")

    builder.branch(reduce_block)
    store_builder.store(val, store_builder.gep(func.args[0], [idx]))
    idx_new = store_builder.add(idx, ir.Constant(ir.IntType(64), 1))
    idx.add_incoming(idx_new, store_block)

    exit_block = func.append_basic_block(name="exit")
    exit_builder = ir.IRBuilder(exit_block)
    exit_builder.ret_void()

    store_builder.cbranch(store_builder.icmp_unsigned("==", idx, end), exit_block, block)
    llvm_ir = str(module)
    if DEBUG >= 1:
      print(llvm_ir)

    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    if DEBUG >= 2:
      print(target_machine.emit_assembly(mod))
    engine.add_module(mod)

    # needed?
    engine.finalize_object()
    engine.run_static_constructors()

    # call function
    bufs = [ret] + bufs
    func_ptr = engine.get_function_address(name)
    argtypes = [ctypes.POINTER(ctypes.c_float) for _ in bufs]
    functype = CFUNCTYPE(ctypes.c_int, *argtypes)
    cfunc = functype(func_ptr)
    cfunc(*[x._buf for x in bufs])

    # we are done
    engine.remove_module(mod)

    return ret
