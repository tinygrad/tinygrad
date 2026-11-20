import multiprocessing, os, unittest
from concurrent.futures import ProcessPoolExecutor
from tinygrad import Tensor, UOp, Device, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo
from tinygrad.engine.realize import compile_linear, link_linear, run_linear

def spill_kernel(C:UOp, A:UOp) -> UOp:
  # a dynamically indexed register array lives in scratch
  tmp = UOp.alloc((256,), dtypes.int, addrspace=AddrSpace.REG)
  i = UOp.range(256, 0)
  fill = tmp[i].store(i.cast(dtypes.int)*A[0]).end(i)
  return C[0].store(tmp.after(fill)[A[1]]).sink(arg=KernelInfo(name="spill"))

def compile_then_link(aql:str) -> tuple[bool, bool, int]:
  os.environ["AMD_AQL"] = aql
  a = Tensor([3, 200], dtype=dtypes.int, device="AMD").realize()
  out = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int, device="AMD"), a, fxn=spill_kernel)[0]
  compiled = compile_linear(out.schedule_linear(), input_uops=(inputs:=[]), cache=True)
  after_compile = hasattr(Device["AMD"], "scratch")
  linked = link_linear(compiled, inputs)
  after_link = hasattr(Device["AMD"], "scratch")
  run_linear(linked, input_uops=inputs, jit=True, wait=True)
  return after_compile, after_link, out.item()

@unittest.skipUnless(Device.DEFAULT == "AMD", "AMD scratch")
class TestAMDScratch(unittest.TestCase):
  def test_scratch_is_bound_at_link(self):
    # compiling a batch has no device side effects, the scratch is the linking process's
    for aql in ("0", "1"):
      with self.subTest(aql=aql), ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as p:
        self.assertEqual(p.submit(compile_then_link, aql).result(), (False, True, 600))

if __name__ == "__main__":
  unittest.main()
