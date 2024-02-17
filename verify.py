from typing import Any, List, cast
from tinygrad.device import Device
from tinygrad.runtime.ops_metal import MetalDevice, MetalRenderer, MetalCompiler, MetalAllocator, MetalProgram

def verify(uops):
  device = cast(MetalDevice, Device[Device.DEFAULT])
  code = MetalRenderer("test", uops)
  lib = MetalCompiler(device).compile(code)
  prg = MetalProgram(device, "test", lib)
  allocator = MetalAllocator(device)
  def to_bytes(x): return bytearray([byte for value in x for byte in value.to_bytes(4, 'little', signed=True)])
  def alloc_data(x:List[int]):
    data:Any = to_bytes(x)
    d = allocator._alloc(len(data))
    allocator.copyin(d, data)
    return d
  init_outputs = lambda szs: [allocator.alloc(sz*4) for sz in szs]
  get_outputs = lambda x: list([allocator.as_buffer(out).cast("I").tolist() for out in x])
  return alloc_data, init_outputs, prg, get_outputs
