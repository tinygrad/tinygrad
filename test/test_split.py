from tinygrad.engine.lazy import LazyBuffer
from tinygrad.ops import Ops
from tinygrad import dtypes
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.helpers import Context, KERNEL_SPLIT

def prep():
  shape1 = (64, 64)
  a = LazyBuffer.metaop(Ops.EMPTY, shape1, dtypes.float, "METAL")
  return a

def lazy():
  a = prep()
  shape2 = (64, 16, 4)
  b = a.reshape(shape2).permute((0, 2, 1))
  c = b._reduce_op(Ops.ADD, (1, ))._reduce_op(Ops.ADD, (2,)).reshape((64, 1))
  run_schedule(*create_schedule_with_vars([c]))

def kernel():
  with Context(KERNEL_SPLIT=1):
    a = prep()
    b = a._reduce_op(Ops.ADD, (1,))
    run_schedule(*create_schedule_with_vars([b]))

print("\033[31mTesting kernel version\033[0m")
kernel()
print("\033[31mTesting lazy version\033[0m")
lazy()

