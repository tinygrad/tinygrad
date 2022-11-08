#!/usr/bin/env python3
import numpy as np
from tinygrad.llops.ops_llvm import LLVM, LLVMBuffer
from llvmlite import ir  # type: ignore

an = np.random.randn(512, 512) - 0.5
bn = np.random.randn(512, 512) - 0.5
cn = an @ bn

a = LLVMBuffer.fromCPU(an)
b = LLVMBuffer.fromCPU(bn)
c = LLVMBuffer.fromCPU(np.zeros((512, 512)))
bufs = [a,b,c]

module = ir.Module(name=__file__)
func = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.FloatType().as_pointer()]*3), name='exec')

builder = ir.IRBuilder(func.append_basic_block(name="entry"))

arg_0 = ir.IntType(64)(1)
arg_1 = ir.IntType(64)(2)

fty = ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(64)])
add = builder.asm(fty, "add $2, $1, $0", "=r,r,r", (arg_0, arg_1), True, name="asm_add")

builder.ret(add)
cfunc = LLVM().exec(module, bufs, 0)
ret = cfunc(*[x._buf for x in bufs])
print(ret)
