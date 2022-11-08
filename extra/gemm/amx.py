#!/usr/bin/env python3
import numpy as np
np.set_printoptions(linewidth=160)
from tinygrad.llops.ops_llvm import LLVM, LLVMBuffer, int_const
from llvmlite import ir  # type: ignore

N = 16 
an = np.arange(N*N).reshape((N,N)).astype(np.float32)
bn = np.arange(N*N)[::-1].reshape((N,N)).astype(np.float32)
cn = (an.T @ bn).T

a = LLVMBuffer.fromCPU(an)
b = LLVMBuffer.fromCPU(bn)
c = LLVMBuffer.fromCPU(np.zeros((N, N)))
bufs = [c,a,b]

module = ir.Module(name=__file__)
func = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.FloatType().as_pointer()]*3), name='exec')

builder = ir.IRBuilder(func.append_basic_block(name="entry"))

#arg_0 = ir.IntType(64)(1)
#arg_1 = ir.IntType(64)(2)

def amx_nop_op_imm5(builder, op, imm5): builder.asm(ir.FunctionType(ir.VoidType(), []), f".word (0x201000 + ({op} << 5) + {imm5})", "", tuple(), True)
def amx_op_gpr(builder, op, gpr): builder.asm(ir.FunctionType(ir.VoidType(), [ir.IntType(64)]), f".word (0x201000 + ({op} << 5) + 0$0 - ((0$0 >> 4) * 6))", "r", (gpr,), True)

def amx_set(builder): amx_nop_op_imm5(builder, 17, 0)
def amx_clr(builder): amx_nop_op_imm5(builder, 17, 1)

def amx_ldx(builder, gpr): amx_op_gpr(builder, 0, gpr)
def amx_ldy(builder, gpr): amx_op_gpr(builder, 1, gpr)
def amx_stx(builder, gpr): amx_op_gpr(builder, 2, gpr)
def amx_sty(builder, gpr): amx_op_gpr(builder, 3, gpr)

def amx_stz(builder, gpr): amx_op_gpr(builder, 5, gpr)
def amx_fma32(builder, gpr): amx_op_gpr(builder, 12, gpr)

# turn amx on
amx_set(builder)
zp, xp, yp = [builder.ptrtoint(func.args[i], ir.IntType(64)) for i in range(3)]

# do matmul
for i in range(N):
  # load
  amx_ldx(builder, builder.add(xp, int_const(i*16*4)))
  amx_ldy(builder, builder.add(yp, int_const(i*16*4)))

  # mul
  amx_fma32(builder, int_const(0))

# store
for i in range(N):
  idx = i << 56
  amx_stz(builder, builder.add(zp, int_const((i*4 << 56) | i*16*4)))

# turn amx off
amx_clr(builder)

builder.ret(int_const(0))
cfunc = LLVM().exec(module, bufs, 0)

print(c.toCPU())
print(cn)

