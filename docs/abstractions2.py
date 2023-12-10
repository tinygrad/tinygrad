# tinygrad is a tensor library, and as a tensor library it has multiple parts
# 1. a "runtime". this allows buffer management, compilation, and running programs
# 2. a "Device" that uses the runtime but specifies compute in an abstract way for all
# 3. a "LazyBuffer" that fuses the compute into kernels, using memory only when needed
# 4. a "Tensor" that provides an easy to use frontend with autograd ".backward()"


print("******** first, the runtime ***********")

from tinygrad.runtime.ops_clang import ClangProgram, compile_clang, MallocAllocator

# allocate some buffers
out = MallocAllocator.alloc(4)
a = MallocAllocator.alloc(4)
b = MallocAllocator.alloc(4)

# load in some values (little endian)
MallocAllocator.copyin(a, bytearray([2,0,0,0]))
MallocAllocator.copyin(b, bytearray([3,0,0,0]))

# compile a program to a binary
lib = compile_clang("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")

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
from tinygrad.helpers import dtypes
from tinygrad.device import Buffer, Device
from tinygrad.ops import LazyOp, BufferOps, MemBuffer, BinaryOps
from tinygrad.shape.shapetracker import ShapeTracker

import numpy as np
def toNumpy(buffer: Buffer) -> np.ndarray:
  # zero copy with as_buffer
  if hasattr(buffer.allocator, 'as_buffer'): return np.frombuffer(buffer.allocator.as_buffer(buffer._buf), dtype=np.dtype(np.int32, metadata={"backing": buffer._buf}))  # type: ignore
  ret = np.empty(buffer.size, np.int32)
  if buffer.size > 0: buffer.allocator.copyout(flat_mv(ret.data), buffer._buf)
  return ret

# allocate some buffers + load in values
out = Buffer(DEVICE, 1, dtypes.int32)
a = Buffer(DEVICE, 1, dtypes.int32).copyin(memoryview(bytearray(struct.pack("I", 2))))
b = Buffer(DEVICE, 1, dtypes.int32).copyin(memoryview(bytearray(struct.pack("I", 3))))
# NOTE: a._buf is the same as the return from MallocAllocator.alloc

# describe the computation
ld_1 = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.int32, ShapeTracker.from_shape((1,))))
ld_2 = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.int32, ShapeTracker.from_shape((1,))))
alu = LazyOp(BinaryOps.ADD, (ld_1, ld_2))
st_0 = LazyOp(BufferOps.STORE, (alu,), MemBuffer(0, dtypes.int32, ShapeTracker.from_shape((1,))))

# convert the computation to a "linearized" format (print the format)
lin = Device[DEVICE].get_linearizer(st_0).linearize()
for u in lin.uops: print(u)

# compile a program (and print the source)
fxn = Device[DEVICE].to_program(lin)
print(fxn.prg)
# NOTE: fxn.clprg is the ClangProgram

# run the program
fxn.exec([out, a, b])

# check the data out
print(val:=toNumpy(out).item())
assert val == 5


print("******** third, the LazyBuffer ***********")

from tinygrad.helpers import prod
from tinygrad.lazy import LazyBuffer, LoadOps
from tinygrad.realize import run_schedule
from tinygrad.tensor import native_shape, native_to_flat_memoryview

# allocate some values + load in values
# TODO: remove numpy here

dtype = dtypes.int32
n_a = [2]
n_b = [3]
a = LazyBuffer(DEVICE, ShapeTracker.from_shape((shape:=native_shape(n_a))), LoadOps, None, dtype, Buffer(DEVICE, prod(shape), dtype).copyin(native_to_flat_memoryview(n_a, dtype, shape=shape)))
b = LazyBuffer(DEVICE, ShapeTracker.from_shape((shape:=native_shape(n_b))), LoadOps, None, dtype, Buffer(DEVICE, prod(shape), dtype).copyin(native_to_flat_memoryview(n_b, dtype, shape=shape)))

# describe the computation
out = a.e(BinaryOps.ADD, b)

# schedule the computation as a list of kernels
sched = out.schedule()
for si in sched: print(si.ast.op)  # NOTE: the first two convert it to CLANG

# DEBUGGING: print the compute ast as a tree
from tinygrad.graph import print_tree
print_tree(sched[-1].ast)
# NOTE: sched[-1].ast is the same as st_0 above

# run that schedule
run_schedule(sched)

# check the data out
print(val:=toNumpy(out.realized).item())
assert val == 5


print("******** fourth, the Tensor ***********")

from tinygrad import Tensor

a = Tensor([2], dtype=dtypes.int32, device=DEVICE)
b = Tensor([3], dtype=dtypes.int32, device=DEVICE)
out = a + b

# check the data out
print(val:=toNumpy(out.realize().lazydata.realized).item())
assert val == 5
