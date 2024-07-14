print("******** third, the LazyBuffer ***********")

DEVICE = "CLANG"   # NOTE: you can change this!
# DEVICE = "INTEL"   # NOTE: you can change this!

import struct
from tinygrad.dtype import dtypes
from tinygrad.lazy import LazyBuffer, LoadOps
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.schedule import create_schedule
from tinygrad.ops import BinaryOps

# allocate some values + load in values
a = LazyBuffer.loadop(LoadOps.EMPTY, (1,), dtypes.int32, DEVICE)
b = LazyBuffer.loadop(LoadOps.EMPTY, (1,), dtypes.int32, DEVICE)
a.buffer.allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
b.buffer.allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))
del a.srcs
del b.srcs

# describe the computation
out = a.e(BinaryOps.ADD, b)

# schedule the computation as a list of kernels
sched = create_schedule([out])
for si in sched: print(si.ast[0].op)  # NOTE: the first two convert it to CLANG

# DEBUGGING: print the compute ast as a tree
from tinygrad.engine.graph import print_tree
print("sched[-1].ast[0]")
print_tree(sched[-1].ast[0])
# NOTE: sched[-1].ast is the same as st_0 above

# run that schedule
run_schedule(sched)

# check the data out
assert out.realized.as_buffer().cast('I')[0] == 5