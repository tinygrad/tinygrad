#!/usr/bin/env python3
import numpy as np
np.set_printoptions(linewidth=160)
from tinygrad.llops.ops_llvm import LLVM, LLVMBuffer, int_const, AMX
from llvmlite import ir  # type: ignore

N = 16 
an = np.arange(N*N).reshape((N,N)).astype(np.float32)
bn = np.arange(N*N)[::-1].reshape((N,N)).astype(np.float32)
cn = an.T @ bn

a = LLVMBuffer.fromCPU(an)
b = LLVMBuffer.fromCPU(bn)
c = LLVMBuffer.fromCPU(np.zeros((N, N)))
bufs = [c,a,b]

module = ir.Module(name=__file__)
func = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.FloatType().as_pointer()]*3), name='exec')

builder = ir.IRBuilder(func.append_basic_block(name="entry"))

# turn amx on
AMX.set(builder)
zp, xp, yp = [builder.ptrtoint(func.args[i], ir.IntType(64)) for i in range(3)]

# do matmul
for i in range(N):
  # load (reversed for no transpose!)
  AMX.ldy(builder, builder.add(xp, int_const(i*16*4)))
  AMX.ldx(builder, builder.add(yp, int_const(i*16*4)))

  # mul
  AMX.fma32(builder, int_const(0))

# store
for i in range(N):
  AMX.stz(builder, builder.add(zp, int_const((i*4 << 56) | i*16*4)))

# turn amx off
AMX.clr(builder)

builder.ret(int_const(0))
cfunc = LLVM().exec(module, bufs, 0)

print(c.toCPU())
print(cn)

