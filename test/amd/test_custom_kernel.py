import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates

from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v

def custom_add_one(A:UOp) -> UOp:
  A = A.flatten()
  assert dtypes.is_float(A.dtype.base), f"buffer dtype must be float32, got {A.dtype}"
  threads = UOp.special(A.size, "lidx0")
  insts = [
    s_load_b64(s[0:1], s[0:1], soffset=NULL),
    s_waitcnt(lgkmcnt=0),
    v_lshlrev_b32_e32(v[0], 2, v[0]), # element offset
    global_load_b32(v[1], v[0], saddr=s[0:1]),
    s_waitcnt(vmcnt=0),
    v_mov_b32_e32(v[2], 1.0),
    v_add_f32_e32(v[1], v[1], v[2]),
    global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads, arg=KernelInfo(f"custom_add_one_{A.size}", estimates=Estimates(ops=A.size, mem=A.size*4*2)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_add_var(A:UOp, B:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert A.dtype.base == dtypes.uint32, f"buffer dtype must be uint32, got {A.dtype}"
  threads = UOp.special(A.size, "lidx0")
  var = UOp.variable("var", 0, 10)
  insts = [
    s_load_b128(s[4:7], s[0:1]),
    s_load_b32(s[8], s[0:1], offset=0x10), # all threads load the same variable
    s_waitcnt(lgkmcnt=0),
    v_lshlrev_b32_e32(v[0], 2, v[0]), # element offset, different per thread
    global_load_b32(v[1], v[0], saddr=s[6:7]),
    s_waitcnt(vmcnt=0),
    v_add_nc_u32_e32(v[1], s[8], v[1]),
    global_store_b32(addr=v[0], data=v[1], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, B.base, var, threads, arg=KernelInfo(f"custom_add_var_{A.size}"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD device")
class TestCustomKernel(unittest.TestCase):
  def test_simple(self):
    a = Tensor.full((16, 16), 1.).contiguous().realize()
    a = Tensor.custom_kernel(a, fxn=custom_add_one)[0]
    ei = a.schedule()[-1].lower()
    self.assertEqual(ei.prg.estimates.ops, a.numel())
    self.assertEqual(ei.prg.estimates.mem, a.nbytes()*2)
    ei.run()
    self.assertTrue((a.numpy() == 2.).all())

  def test_variable(self):
    b = Tensor.full((16, 16), 1, dtype=dtypes.uint32).contiguous().realize()
    a = Tensor.zeros_like(b).contiguous().realize()
    a = Tensor.custom_kernel(a, b, fxn=custom_add_var)[0]
    ei = a.schedule()[-1].lower()
    for i in range(4):
      ei.run({"var":i})
      self.assertTrue((a.numpy() == 1+i).all())

if __name__ == "__main__":
  unittest.main()
