# NV=1 SASS=1 SM=80 python add.py
import os
from tinygrad import Tensor
from tinygrad.runtime.ops_nv import NVDevice, NVProgram
from tinygrad.device import Device
from tinygrad.engine.realize import CompiledRunner

b = Tensor.ones(3).contiguous().realize()
c = Tensor.ones(3).contiguous().realize()

a = b + c
print(a.tolist())
