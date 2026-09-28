import unittest, io
from contextlib import redirect_stdout
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import OSX
from tinygrad.codegen import to_program

class TestDisassembly(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "CPU" and OSX, "m series cpus support fp16 arithmetic")
  def test_float16_alu(self):
    c = Tensor([1], dtype=dtypes.float16) + Tensor([1], dtype=dtypes.float16)
    s = c.schedule_linear().src[-1]
    p = to_program(s.src[0], Device[Device.DEFAULT].renderer)
    lib = Device[Device.DEFAULT].compiler.compile(p.src[2].arg)
    out = io.StringIO()
    with redirect_stdout(out): Device[Device.DEFAULT].compiler.disassemble(lib)
    assert "fcvt" not in out.getvalue()

if __name__ == '__main__':
  unittest.main()
