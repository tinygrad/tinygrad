import os
os.environ["NV"] = "1"

from tinygrad import Tensor
from tinygrad.runtime.ops_nv import NVDevice, NVProgram
from tinygrad.device import Device
from tinygrad.engine.realize import CompiledRunner

a = Tensor.zeros(3).contiguous().realize()
b = Tensor.ones(3).contiguous().realize()
c = Tensor.ones(3).contiguous().realize()

with open("add.cubin", "rb") as f:
  cubin = f.read()
  
device = Device["NV"]
program = NVProgram(device, "add", cubin)
program(a.lazydata.buffer._buf, b.lazydata.buffer._buf, c.lazydata.buffer._buf, global_size=(1, 1, 1), local_size=(3, 1, 1))
print(a.numpy())

