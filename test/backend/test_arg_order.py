import unittest
from tinygrad import Tensor, Device, UOp, TinyJit, Variable
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import KernelInfo
from tinygrad.codegen import to_program

def _alu(slot:int, name:str) -> UOp:
  return UOp.param(slot, dtypes.int, (), name=name, addrspace=AddrSpace.ALU)

def _buf(slot:int, n:int=1) -> UOp:
  return UOp.param(slot, dtypes.int, (n,))

def _launch(sink:UOp, bufs:list[Tensor], vals:tuple[int, ...]=()):
  elf = to_program(sink, Device[Device.DEFAULT].renderer).to_elf()
  Device[Device.DEFAULT].runtime(elf)(*[b.uop.buffer._buf for b in bufs], vals=vals)
  Device[Device.DEFAULT].synchronize()
  return elf

def _abi(elf) -> list[tuple[int, bool, int]]:
  return [(slot, is_buf, idx) for _, slot, _, _, is_buf, idx in elf.signature]

class TestArgOrder(unittest.TestCase):
  def test_scalar_first(self):
    n, buf = _alu(0, "n"), _buf(1)
    sink = buf[0].store(n).sink(arg=KernelInfo(name="scalar_first"), tag=1)
    out = Tensor.zeros(1, dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out], (7,))
    self.assertEqual(_abi(elf), [(0, False, 0), (1, True, 0)])
    self.assertEqual(out.item(), 7)

  def test_buf_first(self):
    buf, n = _buf(0), _alu(1, "n")
    sink = buf[0].store(n).sink(arg=KernelInfo(name="buf_first"), tag=1)
    out = Tensor.zeros(1, dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out], (7,))
    self.assertEqual(_abi(elf), [(0, True, 0), (1, False, 0)])
    self.assertEqual(out.item(), 7)

  def test_interleaved(self):
    out_p, n, inp_p = _buf(0, 4), _alu(1, "n"), _buf(2, 4)
    i = UOp.range(4, 0)
    sink = out_p[i].store(inp_p[i] + n).end(i).sink(arg=KernelInfo(name="interleaved"), tag=1)
    out = Tensor.zeros(4, dtype=dtypes.int).contiguous().realize()
    inp = Tensor([1, 2, 3, 4], dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out, inp], (10,))
    self.assertEqual(_abi(elf), [(0, True, 0), (1, False, 0), (2, True, 1)])
    self.assertEqual(out.tolist(), [11, 12, 13, 14])

  def test_two_scalars_around_buf(self):
    a, buf, b = _alu(0, "a"), _buf(1), _alu(2, "b")
    sink = buf[0].store(a + b).sink(arg=KernelInfo(name="two_scalars"), tag=1)
    out = Tensor.zeros(1, dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out], (3, 4))
    self.assertEqual(_abi(elf), [(0, False, 0), (1, True, 0), (2, False, 1)])
    self.assertEqual(out.item(), 7)

  def test_mixed4_scalar_first(self):
    s0, o_p, s1, inp_p = _alu(0, "s0"), _buf(1, 4), _alu(2, "s1"), _buf(3, 4)
    i = UOp.range(4, 0)
    sink = o_p[i].store(inp_p[i] * s0 + s1).end(i).sink(arg=KernelInfo(name="mixed4"), tag=1)
    out = Tensor.zeros(4, dtype=dtypes.int).contiguous().realize()
    inp = Tensor([1, 2, 3, 4], dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out, inp], (2, 1))
    self.assertEqual(_abi(elf), [(0, False, 0), (1, True, 0), (2, False, 1), (3, True, 1)])
    self.assertEqual(out.tolist(), [3, 5, 7, 9])

  def test_bufs_only(self):
    dst_p, src_p = _buf(0, 4), _buf(1, 4)
    i = UOp.range(4, 0)
    sink = dst_p[i].store(src_p[i]).end(i).sink(arg=KernelInfo(name="bufs_only"), tag=1)
    out = Tensor.zeros(4, dtype=dtypes.int).contiguous().realize()
    inp = Tensor([9, 8, 7, 6], dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out, inp], ())
    self.assertEqual(_abi(elf), [(0, True, 0), (1, True, 1)])
    self.assertEqual(out.tolist(), [9, 8, 7, 6])

  def test_two_scalars_then_buf(self):
    a, b, buf = _alu(0, "a"), _alu(1, "b"), _buf(2)
    sink = buf[0].store(a * b).sink(arg=KernelInfo(name="two_scalars_then_buf"), tag=1)
    out = Tensor.zeros(1, dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out], (6, 7))
    self.assertEqual(_abi(elf), [(0, False, 0), (1, False, 1), (2, True, 0)])
    self.assertEqual(out.item(), 42)

  def test_two_bufs_scalar_last(self):
    o_p, inp_p, n = _buf(0, 4), _buf(1, 4), _alu(2, "n")
    i = UOp.range(4, 0)
    sink = o_p[i].store(inp_p[i] + n).end(i).sink(arg=KernelInfo(name="two_bufs_scalar_last"), tag=1)
    out = Tensor.zeros(4, dtype=dtypes.int).contiguous().realize()
    inp = Tensor([1, 2, 3, 4], dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out, inp], (5,))
    self.assertEqual(_abi(elf), [(0, True, 0), (1, True, 1), (2, False, 0)])
    self.assertEqual(out.tolist(), [6, 7, 8, 9])

  def test_mixed4_buf_first(self):
    o_p, s0, inp_p, s1 = _buf(0, 4), _alu(1, "s0"), _buf(2, 4), _alu(3, "s1")
    i = UOp.range(4, 0)
    sink = o_p[i].store(inp_p[i] * s0 + s1).end(i).sink(arg=KernelInfo(name="mixed4_buf_first"), tag=1)
    out = Tensor.zeros(4, dtype=dtypes.int).contiguous().realize()
    inp = Tensor([1, 2, 3, 4], dtype=dtypes.int).contiguous().realize()
    elf = _launch(sink, [out, inp], (2, 1))
    self.assertEqual(_abi(elf), [(0, True, 0), (1, False, 0), (2, True, 1), (3, False, 1)])
    self.assertEqual(out.tolist(), [3, 5, 7, 9])

  def test_jit_vars_last(self):
    n = Variable("n", 0, 100)
    @TinyJit
    def jit_addn(t, v): return (t + v).contiguous()
    x = Tensor([1, 2, 3, 4], dtype=dtypes.int).contiguous().realize()
    got = [jit_addn(x, n.bind(k)).tolist() for k in (3, 5, 7)]
    self.assertEqual(got, [[4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, 11]])

if __name__ == "__main__":
  unittest.main()
