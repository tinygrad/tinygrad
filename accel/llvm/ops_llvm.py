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

    if op == BinaryOps.ADD:
      r, a, b = func.args
      print(bufs[0].st, bufs[1].st)
      ap = builder.load(builder.gep(a, [idx]))
      bp = builder.load(builder.gep(b, [idx]))
      val = builder.fadd(ap, bp)
      builder.store(val, builder.gep(r, [idx]))
    elif op == UnaryOps.NOOP:
      print(bufs[0].st)
      r, a = func.args
      ap = builder.load(builder.gep(a, [idx]))
      val = ap
      builder.store(val, builder.gep(r, [idx]))

    idx_new = builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_new, block)

    builder.cbranch(builder.icmp_unsigned("==", idx, end), exit_block, block)
    llvm_ir = str(module)
    print(llvm_ir)

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()  # yes, even this one

    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    target_machine = llvm.Target.from_default_triple().create_target_machine()
    print(target_machine.emit_assembly(mod))

    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), target_machine)
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    bufs = [ret] + bufs
    func_ptr = engine.get_function_address("fpadd")
    argtypes = [ctypes.POINTER(ctypes.c_float) for _ in bufs]
    cfunc = CFUNCTYPE(*argtypes)(func_ptr)
    cfunc.argtypes = argtypes
    cfunc(*[x._buf for x in bufs])

    return ret
