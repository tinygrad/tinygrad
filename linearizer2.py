from typing import Any, List, cast
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.ops import BinaryOps, BufferOps, LazyOp, MemBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.runtime.ops_metal import MetalDevice, MetalRenderer, MetalCompiler, MetalAllocator, MetalProgram

a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
out = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.ADD, src=(a,b)),), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))

lin = Linearizer(out)
lin.linearize()
for u in lin.uops: print(u)
global_size = (3,1,1)
local_size = (1,1,1)

device = cast(MetalDevice, Device[Device.DEFAULT])
code = MetalRenderer("test", lin.uops)
print(code)

lib = MetalCompiler(device).compile(code)
prg = MetalProgram(device, "test", lib)
allocator = MetalAllocator(device)
def to_bytes(x): return bytearray([byte for value in x for byte in value.to_bytes(4, 'little', signed=True)])
def alloc_data(x:List[int]):
  data:Any = to_bytes(x)
  d = allocator._alloc(len(data))
  allocator.copyin(d, data)
  return d

a, b = alloc_data([2,2,3]), alloc_data([4,5,6])
output_bufs = [allocator.alloc(12) for _ in range(1)]
prg(output_bufs[0], a, b, global_size=global_size, local_size=local_size)
for out in output_bufs: print(allocator.as_buffer(out).cast("I").tolist())
