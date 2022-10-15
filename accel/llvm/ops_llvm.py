# https://github.com/numba/llvmlite/blob/main/llvmlite/ir/builder.py
from __future__ import annotations
from typing import Tuple, Union
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker
import ctypes
import numpy as np
from llvmlite import ir
from ctypes import CFUNCTYPE, c_double
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps

import llvmlite.binding as llvm

class LLVMProgram:
  def __init__(self): 
    pass

# TODO: write this
# TODO: Refactor LLVMBuffer and GPUBuffer into ShapeTrackedBuffer
class LLVMBuffer:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self._buf = (ctypes.c_float * prod(self.shape))() if hostbuf is None else hostbuf._buf

  # copied from GPUBuffer
  def movement_op(x, op:MovementOps, arg): return type(x)(ShapeTracker(x.st).movement_op(op, arg), x)
  def contiguous_op(x): return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)

  def unary_op(x, op:UnaryOps): return type(x)(x.shape)._dumb_processing_op([x], op)
  def binary_op(x, op:BinaryOps, y): return type(x)(x.shape)._dumb_processing_op([x, y], op)
  #def reduce_op(x, op:ReduceOps, new_shape:Tuple[int, ...]): return type(x)(new_shape)._processing_op([("A", x)], code="acc", earlycode=GPUBuffer.code_for_op[op], earlybufs=set("A"), op=op)

  @staticmethod
  def fromCPU(x):
    ret = LLVMBuffer(x.shape)
    ctypes.memmove(ret._buf, x.ctypes.data, prod(ret.shape)*4)
    return ret
  
  def toCPU(x): return np.ctypeslib.as_array(x.contiguous_op()._buf).reshape(x.shape)

  def _dumb_processing_op(ret, bufs, op):
    typ = ir.PointerType(ir.FloatType())

    fnty = ir.FunctionType(ir.VoidType(), [typ]*(1+len(bufs)))
    module = ir.Module(name=__file__)
    func = ir.Function(module, fnty, name="fpadd")

    start_block = func.append_basic_block(name="entry")
    block = func.append_basic_block(name="inner_loop")
    exit_block = func.append_basic_block(name="exit")
    builder = ir.IRBuilder(block)
    start_builder = ir.IRBuilder(start_block)
    start_builder.branch(block)
    exit_builder = ir.IRBuilder(exit_block)
    exit_builder.ret_void()

    start = ir.Constant(ir.IntType(32), 0)
    end = ir.Constant(ir.IntType(32), prod(ret.shape))
    idx = builder.phi(ir.IntType(32))
    idx.add_incoming(start, start_block)

    int_const = lambda x: ir.Constant(ir.IntType(32), x)
    def idx_deref(buf, ptr, idx):
      if DEBUG >= 1:
        print(buf.st.expr(), ptr)
      # TODO: unify this with expr in ShapeTracker
      for v in buf.st.views:
        acc = 1
        ret = int_const(0)
        for i,(d,s) in enumerate(v.shape_strides[::-1]):
          if d != 1 and s != 0:
            lr = builder.urem(builder.udiv(idx, int_const(acc)), int_const(d))
            lr = builder.mul(lr, int_const(s))
            ret = builder.add(ret, lr)
          acc *= d
        idx = ret
      return builder.load(builder.gep(ptr, [idx]))

    values = [idx_deref(buf, ptr, idx) for buf, ptr in zip(bufs, func.args[1:])]
    llvm_pow = module.declare_intrinsic('llvm.pow', [ir.FloatType()])
    llvm_log = module.declare_intrinsic('llvm.log', [ir.FloatType()])
    op_lookup = {
      BinaryOps.ADD: builder.fadd,
      BinaryOps.SUB: builder.fsub,
      BinaryOps.MUL: builder.fmul,
      BinaryOps.DIV: builder.fdiv,
      BinaryOps.POW: lambda x,y: builder.call(llvm_pow, [x,y]),
      UnaryOps.LOG: lambda x: builder.call(llvm_log, [x]),
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

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()  # yes, even this one

    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    target_machine = llvm.Target.from_default_triple().create_target_machine()
    if DEBUG >= 2:
      print(target_machine.emit_assembly(mod))

    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), target_machine)
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    bufs = [ret] + bufs
    func_ptr = engine.get_function_address("fpadd")
    argtypes = [ctypes.POINTER(ctypes.c_float) for _ in bufs]
    cfunc = CFUNCTYPE(ctypes.c_int, *argtypes)(func_ptr)
    cfunc(*[x._buf for x in bufs])

    return ret
