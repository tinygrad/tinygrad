import unittest, itertools
from tinygrad import Tensor, dtypes, Device
from tinygrad.dtype import AddrSpace
from tinygrad.codegen import to_program
from tinygrad.uop.ops import UOp, KernelInfo

SLOTS = list(itertools.permutations(range(4)))

def muladd_kernel(out_slot:int, inp_slot:int, a_slot:int, b_slot:int) -> UOp:
  out = UOp.param(out_slot, dtypes.int, (4,), name="out")
  inp = UOp.param(inp_slot, dtypes.int, (4,), name="inp")
  a = UOp.param(a_slot, dtypes.int, (), vmin_vmax=(0, 100), name="a", addrspace=AddrSpace.ALU)
  b = UOp.param(b_slot, dtypes.int, (), vmin_vmax=(0, 100), name="b", addrspace=AddrSpace.ALU)
  i = UOp.range(4, 0)
  return out[i].store(inp[i]*a+b).end(i).sink(arg=KernelInfo(name="muladd"))

class TestKernelArgOrder(unittest.TestCase):
  def test_program(self):
    for slots in SLOTS:
      with self.subTest(slots=slots):
        prg = to_program(muladd_kernel(*slots), Device[Device.DEFAULT].renderer)
        self.assertEqual([s[1] for s in prg.to_elf().signature], [0, 1, 2, 3])
        out, inp = Tensor.empty(4, dtype=dtypes.int).realize(), Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
        bufs, vals = {slots[0]: out, slots[1]: inp}, {slots[2]: 13, slots[3]: 37}
        args = prg.arg.merge_args([bufs[s].uop.buffer.ensure_allocated()._buf for s in prg.arg.globals], [vals[s] for s in sorted(vals)])
        Device[Device.DEFAULT].runtime(prg.to_elf())(*args, wait=True)
        self.assertEqual(out.tolist(), [50, 63, 76, 89])

if __name__ == "__main__":
  unittest.main()
