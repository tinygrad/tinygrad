import unittest
import numpy as np
from tinygrad import Device, Tensor
from tinygrad.tensor import _to_np_dtype
from tinygrad.uop.ops import Ops, UOp, buffers
from tinygrad.device import Buffer
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from test.helpers import replace_opts
from test.backend.test_linearizer import helper_realized_ast

def run_program(prg:UOp, bufs:list[Buffer]):
  buf_uops = [UOp.new_buffer(b.device, b.size, b.dtype) for b in bufs]
  for u,b in zip(buf_uops, bufs): buffers[u] = b
  run_linear(UOp(Ops.LINEAR, src=(prg.call(*buf_uops),)))

L = OptOps.LOCAL
class TestWarpLocal(unittest.TestCase):
  # TC + >=3 stacked LOCALs = 4 local-class dims, more than the 3 hw axes: gpudims must group them keeping the WARP whole
  # in threadIdx.x (hw lanes are x-first, WMMA needs lane id == warp idx), folding a LOCAL into it scrambles lane ownership
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores and Device[Device.DEFAULT].renderer.has_local, "test requires tensor cores")
  def test_tc_stacked_locals(self):
    tc = Device[Device.DEFAULT].renderer.tensor_cores[0]
    a, b = (Tensor.randint(256, 256, low=-2, high=3).cast(tc.dtype_in) for _ in range(2))
    ast, bufs = helper_realized_ast(a.matmul(b, dtype=tc.dtype_out))  # integer inputs: any correct kernel is exact
    run_program(replace_opts(ast, []), bufs)
    ref = np.frombuffer(bufs[0].as_memoryview(), _to_np_dtype(bufs[0].dtype)).copy()
    for tail in ([Opt(L,0,2)]*3, [Opt(L,1,2)]*3, [Opt(L,0,2),Opt(L,1,2),Opt(L,0,2)]):
      with self.subTest(tail=tail):
        ast2 = replace_opts(ast, [Opt(OptOps.TC, 0, (-1,2,1))]+tail)
        assert any(u.op is Ops.WMMA for u in to_program(ast2, Device[Device.DEFAULT].renderer).src[1].src), "tensor core not triggered"
        bufs[0].copy_from(Buffer("PYTHON", bufs[0].size, bufs[0].dtype, opaque=memoryview(bytearray(bufs[0].nbytes))))
        run_program(ast2, bufs)
        np.testing.assert_array_equal(np.frombuffer(bufs[0].as_memoryview(), _to_np_dtype(bufs[0].dtype)), ref)

if __name__ == '__main__':
  unittest.main()
