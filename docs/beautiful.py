# in tinygrad, things are easy
# because if things were hard, tinygrad would not be tiny
# come on a journey where we add 2+3 in CLANG


print("******** first, the runtime ***********")

from tinygrad.runtime.ops_clang import ClangProgram, compile_clang, MallocAllocator

# allocate some buffers
out = MallocAllocator.alloc(4)
a = MallocAllocator.alloc(4)
b = MallocAllocator.alloc(4)

# load in some values
MallocAllocator.copyin(a, bytearray([2,0,0,0]))
MallocAllocator.copyin(b, bytearray([3,0,0,0]))

# compile a program
lib = compile_clang("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")

# create a runtime for the program
fxn = ClangProgram("add", lib)

# run the program
fxn(out, a, b)

# get the data out
outb = bytearray(MallocAllocator.as_buffer(out))
assert outb == bytearray([5,0,0,0])
print(outb)


print("******** second, the Device ***********")

import numpy as np
from tinygrad.helpers import dtypes
from tinygrad.device import Buffer, Device
from tinygrad.ops import LazyOp, BufferOps, MemBuffer, BinaryOps
from tinygrad.shape.shapetracker import ShapeTracker

# allocate some buffers + load in values
out = Buffer("CLANG", 1, dtypes.int32)
a = Buffer("CLANG", 1, dtypes.int32).copyin(np.array([2], np.int32).data)
b = Buffer("CLANG", 1, dtypes.int32).copyin(np.array([3], np.int32).data)

# describe the computation
ld_1 = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.int32, ShapeTracker.from_shape((1,))))
ld_2 = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.int32, ShapeTracker.from_shape((1,))))
alu = LazyOp(BinaryOps.ADD, (ld_1, ld_2))
st_0 = LazyOp(BufferOps.STORE, (alu,), MemBuffer(0, dtypes.int32, ShapeTracker.from_shape((1,))))

# compile a program (and print the code)
fxn = Device["CLANG"].get_runner(st_0)
print(fxn.prg)

# run the program
fxn.exec([out, a, b])
assert out.toCPU().item() == 5
print(out.toCPU())


print("******** third, the LazyBuffer ***********")

from tinygrad.lazy import LazyBuffer
from tinygrad.graph import print_tree
from tinygrad.realize import run_schedule

# allocate some values + load in values
a = LazyBuffer.fromCPU(np.array([2], np.int32)).copy_to_device("CLANG")
b = LazyBuffer.fromCPU(np.array([3], np.int32)).copy_to_device("CLANG")

# describe the computation
out = a.e(BinaryOps.ADD, b)

# schedule the computation (print it)
sched = out.schedule()
print_tree(sched[0].ast)

# run that schedule
run_schedule(sched)
assert out.realized.toCPU().item() == 5
print(out.realized.toCPU())


print("******** fourth, the Tensor ***********")

from tinygrad import Tensor

a = Tensor([2], dtype=dtypes.int32, device="CLANG")
b = Tensor([3], dtype=dtypes.int32, device="CLANG")
out = (a+b).item()
assert out == 5
print(out)
