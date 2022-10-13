# https://github.com/numba/llvmlite/blob/main/llvmlite/ir/builder.py
from typing import Tuple, Union
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker
import ctypes
import numpy as np
from llvmlite import ir
from ctypes import CFUNCTYPE, c_double

import llvmlite.binding as llvm

class LLVMProgram:
  def __init__(self): 
    pass

# TODO: write this
class LLVMBuffer:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]]):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self._buf = (ctypes.c_float * prod(self.shape))()

  @staticmethod
  def fromCPU(x):
    ret = LLVMBuffer(x.shape)
    ctypes.memmove(ret._buf, x.ctypes.data, prod(ret.shape)*4)
    return ret
  
  def toCPU(x): return np.ctypeslib.as_array(x._buf).reshape(x.shape)

  def binary_op(x, op, y):
    ret = LLVMBuffer(x.shape)
    #cnt = prod(x.shape)
    #typ = ir.ArrayType(ir.FloatType(), cnt)
    typ = ir.PointerType(ir.FloatType())
    fnty = ir.FunctionType(ir.VoidType(), (typ, typ, typ))
    module = ir.Module(name=__file__)
    func = ir.Function(module, fnty, name="fpadd")

    start_block = func.append_basic_block(name="entry")
    block = func.append_basic_block(name="inner_loop")
    exit_block = func.append_basic_block(name="exit")
    builder = ir.IRBuilder(block)

    r, a, b = func.args

    #start = builder.alloca(ir.IntType(32))
    start = ir.Constant(ir.IntType(32), 0)
    end = ir.Constant(ir.IntType(32), prod(x.shape))

    idx = builder.phi(ir.IntType(32))
    idx.add_incoming(start, start_block)

    ap = builder.load(builder.gep(a, [idx]))
    bp = builder.load(builder.gep(b, [idx]))
    val = builder.fadd(ap, bp)
    builder.store(val, builder.gep(r, [idx]))

    idx_new = builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_new, block)

    builder.cbranch(builder.icmp_unsigned("==", idx, end), exit_block, block)



    #idx = builder.add(idx, ir.Constant(ir.IntType(32), 1))

    #bp = builder.load(b)
    
    #result = builder.fadd(a, b, name="res")
    #builder.ret(result)

    llvm_ir = str(module)
    print(llvm_ir)

    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()

    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), llvm.Target.from_default_triple().create_target_machine())
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address("fpadd")
    cfunc = CFUNCTYPE(ctypes.c_float * prod(x.shape), ctypes.c_float * prod(x.shape), ctypes.c_float * prod(x.shape))(func_ptr)
    cfunc(ret._buf, x._buf, y._buf)

    return ret
