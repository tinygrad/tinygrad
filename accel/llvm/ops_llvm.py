# https://github.com/numba/llvmlite/blob/main/llvmlite/ir/builder.py
from __future__ import annotations
import os
from typing import Tuple, Union
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker, ZeroView
import ctypes
import numpy as np
from llvmlite import ir
from ctypes import CFUNCTYPE
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps
import llvmlite.binding as llvm

import gc
#gc.set_debug(gc.DEBUG_COLLECTABLE)
#gc.disable()

int_const = lambda x: ir.Constant(ir.IntType(32), x)
def idx_deref(builder, buf, ptr, idx):
  if DEBUG >= 1:
    print(buf.st.expr(), ptr)
  # TODO: unify this with expr in ShapeTracker
  for v in buf.st.views:
    if isinstance(v, ZeroView):
      raise NotImplementedError("no support for zeroview")
    else:
      acc = 1
      ret = int_const(v.offset)
      for i,(d,s) in enumerate(v.shape_strides[::-1]):
        if d != 1 and s != 0:
          lr = builder.urem(builder.udiv(idx, int_const(acc)), int_const(d))
          lr = builder.mul(lr, int_const(s))
          ret = builder.add(ret, lr)
        acc *= d
      idx = ret
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
    raise NotImplementedError("reduceops don't work")
    return type(x)(new_shape) #._processing_op([("A", x)], code="acc", earlycode=GPUBuffer.code_for_op[op], earlybufs=set("A"), op=op)

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

    typ = ir.PointerType(ir.FloatType())
    fnty = ir.FunctionType(ir.VoidType(), [typ]*(1+len(bufs)))
    func = ir.Function(module, fnty, name=name)

    start_block = func.append_basic_block(name="entry")
    block = func.append_basic_block(name="inner_loop")
    exit_block = func.append_basic_block(name="exit")

    builder = ir.IRBuilder(block)
    start_builder = ir.IRBuilder(start_block)
    start_builder.branch(block)
    exit_builder = ir.IRBuilder(exit_block)
    exit_builder.ret_void()

    start = ir.Constant(ir.IntType(32), 0)
    end = ir.Constant(ir.IntType(32), prod(ret.shape)-1)
    idx = builder.phi(ir.IntType(32))
    idx.add_incoming(start, start_block)

    values = [idx_deref(builder, buf, ptr, idx) for buf, ptr in zip(bufs, func.args[1:])]
    op_lookup = {
      BinaryOps.ADD: builder.fadd,
      BinaryOps.SUB: builder.fsub,
      BinaryOps.MUL: builder.fmul,
      BinaryOps.DIV: builder.fdiv,
      BinaryOps.POW: lambda x,y: builder.call(llvm_pow, [x,y]),
      UnaryOps.LOG: lambda x: builder.call(llvm_log, [x]),
      UnaryOps.EXP: lambda x: builder.call(llvm_exp, [x]),
      UnaryOps.NOOP: lambda x: x,
      UnaryOps.NEG: builder.fneg,
      UnaryOps.RECIPROCAL: lambda x: builder.fdiv(ir.Constant(ir.FloatType(), 1), x),
      UnaryOps.RELU: lambda x: builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x), x, ir.Constant(ir.FloatType(), 0)),
      UnaryOps.SIGN: lambda x: builder.select(builder.fcmp_ordered("==", x, ir.Constant(ir.FloatType(), 0)), ir.Constant(ir.FloatType(), 0), builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x), ir.Constant(ir.FloatType(), 1), ir.Constant(ir.FloatType(), -1)))
    }
    if op in op_lookup:
      val = op_lookup[op](*values)
      builder.store(val, builder.gep(func.args[0], [idx]))
    else:
      raise NotImplementedError(f"{op} not implemented in LLVM backend")

    idx_new = builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_new, block)

    builder.cbranch(builder.icmp_unsigned("==", idx, end), exit_block, block)
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
