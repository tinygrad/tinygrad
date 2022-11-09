#!/usr/bin/env python3
import numpy as np
#np.set_printoptions(linewidth=160)
np.set_printoptions(linewidth=1000, threshold=10000000000, suppress=False)
from tinygrad.llops.ops_llvm import LLVM, LLVMBuffer, int_const, AMX
from llvmlite import ir  # type: ignore

N = 64
an = np.arange(N*N).reshape(N, N)
bn = np.arange(N*N).reshape(N, N)
#an = np.ones((N, N)).astype(np.float32)
#bn = np.ones((N, N)).astype(np.float32)
an = an.astype(np.float32) - 43*64
bn = bn.astype(np.float32)
cn = (an.T @ bn).T

a = LLVMBuffer.fromCPU(an)
b = LLVMBuffer.fromCPU(bn)
c = LLVMBuffer.fromCPU(np.zeros((N, N)))
bufs = [c,a,b]

module = ir.Module(name=__file__)
func = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.FloatType().as_pointer()]*3), name='exec')

# turn amx on
#AMX.set(builder)

entry = ir.IRBuilder(func.append_basic_block(name="entry"))
loop_1 = ir.IRBuilder(func.append_basic_block(name="loop_y"))
loop_2 = ir.IRBuilder(func.append_basic_block(name="loop_x"))
loop_3 = ir.IRBuilder(func.append_basic_block(name="loop_k"))
loop_3_exit = ir.IRBuilder(func.append_basic_block(name="loop_k_exit"))
loop_2_exit = ir.IRBuilder(func.append_basic_block(name="loop_x_exit"))
loop_1_exit = ir.IRBuilder(func.append_basic_block(name="loop_y_exit"))

zm, xm, ym = [entry.ptrtoint(func.args[i], ir.IntType(64)) for i in range(3)]
exit = ir.IRBuilder(func.append_basic_block(name="exit"))
exit.ret(int_const(0))

y = loop_1.phi(ir.IntType(64), name="y")
x = loop_2.phi(ir.IntType(64), name="x")
k = loop_3.phi(ir.IntType(64), name="k")

AMX.set(loop_2)

# stride
xptr = loop_3_exit.add(x, loop_3_exit.mul(k, int_const(N)))
yptr = loop_3_exit.add(y, loop_3_exit.mul(k, int_const(N)))

# double loads load 32 floats
AMX.ldx(loop_3_exit, loop_3_exit.add(int_const(1<<62), loop_3_exit.add(xm, loop_3_exit.mul(int_const(4), xptr))))
AMX.ldy(loop_3_exit, loop_3_exit.add(int_const(1<<62), loop_3_exit.add(ym, loop_3_exit.mul(int_const(4), yptr))))

# <Z row> <X offset> <Y offset>
AMX.fma32(loop_3_exit, int_const(0<<20 | (0*16*4)<<10 | (0*16*4)))
AMX.fma32(loop_3_exit, int_const(1<<20 | (1*16*4)<<10 | (0*16*4)))
AMX.fma32(loop_3_exit, int_const(2<<20 | (0*16*4)<<10 | (1*16*4)))
AMX.fma32(loop_3_exit, int_const(3<<20 | (1*16*4)<<10 | (1*16*4)))

# store
gptr = loop_2_exit.mul(loop_2_exit.add(loop_2.mul(y, int_const(N)), x), int_const(4))
zmp = loop_2_exit.add(zm, gptr)
for j in range(2):
  for r in range(16):
    z_row = j*2
    ptr = ((j*16)+r)*N
    AMX.stz(loop_2_exit, loop_2_exit.add(zmp, int_const(1 << 62 | ((r*4+z_row) << 56) | ptr*4)))
AMX.clr(loop_2_exit)

yp = loop_1_exit.add(y, int_const(32))
xp = loop_2_exit.add(x, int_const(32))
kp = loop_3_exit.add(k, int_const(1))

y.add_incoming(int_const(0), entry._block)
x.add_incoming(int_const(0), loop_1._block)
k.add_incoming(int_const(0), loop_2._block)
y.add_incoming(yp, loop_1_exit._block)
x.add_incoming(xp, loop_2_exit._block)
k.add_incoming(kp, loop_3_exit._block)

entry.branch(loop_1._block)
loop_1.branch(loop_2._block)
loop_2.branch(loop_3._block)
loop_3.branch(loop_3_exit._block)
loop_3_exit.cbranch(loop_3_exit.icmp_unsigned("==", kp, int_const(N)), loop_2_exit._block, loop_3._block)
loop_2_exit.cbranch(loop_2_exit.icmp_unsigned("==", xp, int_const(N)), loop_1_exit._block, loop_2._block)
loop_1_exit.cbranch(loop_1_exit.icmp_unsigned("==", yp, int_const(N)), exit._block, loop_1._block)

print(str(module))
#exit(0)

# do matmul
#AMX.ldx(builder, builder.add(yp, int_const(i*16*4)))
#AMX.ldy(builder, builder.add(xp, int_const(i*16*4)))

"""
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
"""

# turn amx off
#AMX.clr(builder)

cfunc = LLVM().exec(module, bufs, N**3 * 2)
cfunc = LLVM().exec(module, bufs, N**3 * 2)
cfunc = LLVM().exec(module, bufs, N**3 * 2)

print(c.toCPU().astype(np.int64))
print(cn.astype(np.int64))

