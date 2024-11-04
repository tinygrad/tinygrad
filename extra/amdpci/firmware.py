import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class Firmware:
  def __init__(self, adev, path, header_t):
    self.adev = adev
    self.path = path
    self.blob = memoryview(bytearray(open(self.path, "rb").read()))
    self.setup()

    self.header = header_t.from_buffer(self.blob[:ctypes.sizeof(header_t)])
