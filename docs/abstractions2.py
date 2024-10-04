# tinygrad is a tensor library, and as a tensor library it has multiple parts
# 1. a "runtime". this allows buffer management, compilation, and running programs
# 2. a "Device" that uses the runtime but specifies compute in an abstract way for all
# 3. a "LazyBuffer" that fuses the compute into kernels, using memory only when needed
# 4. a "Tensor" that provides an easy to use frontend with autograd ".backward()"


print("******** first, the runtime ***********")

from tinygrad.runtime.ops_clang import ClangProgram, ClangCompiler, MallocAllocator

# allocate some buffers
out = MallocAllocator.alloc(4)
a = MallocAllocator.alloc(4)
b = MallocAllocator.alloc(4)

# load in some values (little endian)
MallocAllocator.copyin(a, bytearray([2,0,0,0]))
MallocAllocator.copyin(b, bytearray([3,0,0,0]))

# compile a program to a binary
lib = ClangCompiler().compile("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")

# create a runtime for the program (ctypes.CDLL)
fxn = ClangProgram("add", lib)

# run the program
fxn(out, a, b)

# check the data out
print(val := MallocAllocator.as_buffer(out).cast("I").tolist()[0])
assert val == 5


print("******** second, the Device ***********")

DEVICE = "CLANG"   # NOTE: you can change this!

import struct
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.device import Buffer, Device
from tinygrad.ops import BinaryOps, MetaOps, UOp, UOps
from tinygrad.shape.shapetracker import ShapeTracker

# allocate some buffers + load in values
out = Buffer(DEVICE, 1, dtypes.int32).allocate()
a = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
b = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))
# NOTE: a._buf is the same as the return from MallocAllocator.alloc

# describe the computation
buf_1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int32), (), 1)
buf_2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int32), (), 2)
ld_1 = UOp(UOps.LOAD, dtypes.int32, (buf_1, ShapeTracker.from_shape((1,)).to_uop()))
ld_2 = UOp(UOps.LOAD, dtypes.int32, (buf_2, ShapeTracker.from_shape((1,)).to_uop()))
alu = ld_1 + ld_2
output_buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int32), (), 0)
st_0 = UOp(UOps.STORE, dtypes.void, (output_buf, ShapeTracker.from_shape((1,)).to_uop(), alu))
s = UOp(UOps.SINK, dtypes.void, (st_0,))

# convert the computation to a "linearized" format (print the format)
from tinygrad.engine.realize import get_kernel, CompiledRunner
kernel = get_kernel(Device[DEVICE].renderer, s).linearize()

# compile a program (and print the source)
fxn = CompiledRunner(kernel.to_program())
print(fxn.p.src)
# NOTE: fxn.clprg is the ClangProgram

# run the program
fxn.exec([out, a, b])

# check the data out
assert out.as_buffer().cast('I')[0] == 5


print("******** third, the LazyBuffer ***********")

from tinygrad.engine.lazy import LazyBuffer
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.schedule import create_schedule

# allocate some values + load in values
a = LazyBuffer.metaop(MetaOps.EMPTY, (1,), dtypes.int32, DEVICE)
b = LazyBuffer.metaop(MetaOps.EMPTY, (1,), dtypes.int32, DEVICE)
a.buffer.allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
b.buffer.allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))
del a.srcs
del b.srcs

# describe the computation
out = a.alu(BinaryOps.ADD, b)

# schedule the computation as a list of kernels
sched = create_schedule([out])
for si in sched: print(si.ast.op)  # NOTE: the first two convert it to CLANG

# DEBUGGING: print the compute ast
print(sched[-1].ast)
# NOTE: sched[-1].ast is the same as st_0 above

# run that schedule
run_schedule(sched)

# check the data out
assert out.realized.as_buffer().cast('I')[0] == 5


print("******** fourth, the Tensor ***********")

from tinygrad import Tensor

a = Tensor([2], dtype=dtypes.int32, device=DEVICE)
b = Tensor([3], dtype=dtypes.int32, device=DEVICE)
out = a + b

# check the data out
print(val:=out.item())
assert val == 5
