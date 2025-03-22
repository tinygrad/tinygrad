import os
os.environ["NV"] = "1"
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.runtime.ops_nv import NVDevice, NVProgram
from tinygrad.device import Device, Buffer
from tinygrad.engine.realize import CompiledRunner

a = Buffer("NV", size=3, dtype=dtypes.float)
a.allocate()
print(a._buf)

with open("zero.cubin", "rb") as f:
  cubin = f.read()
  
device = Device["NV"]
program = NVProgram(device, "E_3", cubin)
program(a._buf, global_size=(1, 1, 1), local_size=(3, 1, 1))
np_buf = np.frombuffer(a.as_buffer(), np.float32)
print(np_buf)
