import multiprocessing, os, unittest
from concurrent.futures import ProcessPoolExecutor
from tinygrad import Tensor, UOp, Device, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEV
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.uop.ops import KernelInfo
from tinygrad.engine.realize import compile_linear, link_linear, run_linear

def spill_kernel(C:UOp, A:UOp) -> UOp:
  # a dynamically indexed register array lives in scratch
  tmp = UOp.alloc((256,), dtypes.int, addrspace=AddrSpace.REG)
  i = UOp.range(256, 0)
  fill = tmp[i].store(i.cast(dtypes.int)*A[0]).end(i)
  return C[0].store(tmp.after(fill)[A[1] & 255]).sink(arg=KernelInfo(name="spill"))

def check_scratch(aql:str):
  os.environ["AMD_AQL"] = aql
  a = Tensor([3, 200], dtype=dtypes.int, device="AMD").realize()
  out = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int, device="AMD"), a, fxn=spill_kernel)[0]
  compiled = compile_linear(out.schedule_linear(), input_uops=(inputs:=[]), cache=True)
  assert not hasattr(Device["AMD"], "scratch")
  linked = link_linear(compiled, inputs)
  # PM4 always binds the scratch ring. On AQL, LLVM can promote this register array to LDS instead.
  llvm = isinstance(Device["AMD"].renderer, AMDLLVMRenderer)
  if aql == "0" or not llvm: assert hasattr(Device["AMD"], "scratch")
  # The mock PM4 executor assumes the first user SGPRs hold kernargs, not LLVM's private segment/dispatch pointer.
  if not (DEV.interface.startswith("MOCK") and llvm):
    run_linear(linked, input_uops=inputs, jit=True, wait=True)
    assert out.item() == 600

@unittest.skipUnless(Device.DEFAULT == "AMD", "AMD scratch")
class TestAMDScratch(unittest.TestCase):
  def test_scratch_is_bound_at_link(self):
    # Compiling records the scratch requirement; linking allocates the scratch.
    for aql in ("0", "1"):
      with self.subTest(aql=aql), ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as p:
        if aql == "1" and DEV.interface.startswith("MOCK"): self.skipTest("the mock driver does not implement AQL queues")
        p.submit(check_scratch, aql).result()

if __name__ == "__main__":
  unittest.main()
