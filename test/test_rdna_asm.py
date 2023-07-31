import unittest
from tinygrad.helpers import getenv, dtypes
from tinygrad.tensor import Tensor
import numpy as np

@unittest.skipUnless(getenv("RDNA"), "Only tests RDNA asm")
class TestRdnaAsm(unittest.TestCase):
  def test_full_like(self):
    with self.assertRaises(SystemExit) as asm:
      a = Tensor([[1,2,3],[4,5,6]])
      Tensor.full_like(a, 4).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_full(self):
    with self.assertRaises(SystemExit) as asm:
      Tensor.full((45,65), 4).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_zeros(self):
    with self.assertRaises(SystemExit) as asm:
      Tensor.zeros(45,65).realize().numpy()
      self.assertEqual(type(asm.exception.code), bytes)
  def test_zeros_like(self):
    with self.assertRaises(SystemExit) as asm:
      a = Tensor([[1,2,3],[4,5,6]])
      Tensor.zeros_like(a).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_empty_0(self):
    with self.assertRaises(SystemExit) as asm:
      (Tensor.empty(45,65)*0/0).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_ones(self):
    with self.assertRaises(SystemExit) as asm:
      Tensor.ones(45,65).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_ones_like(self):
    with self.assertRaises(SystemExit) as asm:
      a = Tensor([[1,2,3],[4,5,6]])
      Tensor.ones_like(a).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_arange(self):
    with self.assertRaises(SystemExit) as asm:
      Tensor.arange(5, 10, 3).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (assembly_rdna.py, line 114, KeyError: <BinaryOps.MOD: 7>)")
  def test_eye(self):
    with self.assertRaises(SystemExit) as asm:
      Tensor.eye(10).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_floor(self):
    with self.assertRaises(SystemExit) as asm:
      Tensor.floor(Tensor([1.0, 2.1, 0.0, -5.0, -2.5])).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_ceil(self):
    with self.assertRaises(SystemExit) as asm:
      Tensor.ceil(Tensor([1.0, 2.1, 0.0, -5.0, -2.5])).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_tril(self):
    with self.assertRaises(SystemExit) as asm:
      tns = Tensor(((np.random.default_rng(seed=0).random(size=(3,3), dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32)
      tns.tril().realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_triu(self):
    with self.assertRaises(SystemExit) as asm:
      tns = Tensor(((np.random.default_rng(seed=0).random(size=(3,3), dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32)
      tns.triu().realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_maximum(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,65)]]
      tns.maximum(tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_minimum(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,65)]]
      tns.minimum(tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_add(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,68), (45,68)]]
      tns.add(tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_add3(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2,tns3] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,65), (45,65)]]
      (tns + tns2 + tns3).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_add_simple(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(256), (256)]]
      tns.add(tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_broadcast_add(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,1)]]
      tns.add(tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_sub(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,65)]]
      tns.sub(tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_neg(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (-tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_mul(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,65)]]
      (tns*tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_mul_const(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (tns*2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_div(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,65)]]
      (tns/tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_div_const(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (tns/255).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_mul_const_inf(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (tns*float("inf")).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_mul_const_nan(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (tns*float("nan")).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_div_const_inf(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (tns/float("inf")).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_div_const_nan(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (tns/float("nan")).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_pow(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,65)]]
      (tns**tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_pow_const(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      (tns**2.0).realize().numpy()
    self.assertEqual(asm.exception.code, b"\x7fELF\x02\x01\x01@\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\xe0\x00\x01\x00\x00\x00\x10\x15\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x10\x07\x00\x00\x00\x00\x00\x00A\x00\x00\x00@\x008\x00\x08\x00@\x00\r\x00\x0b\x00\x06\x00\x00\x00\x04\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\xc0\x01\x00\x00\x00\x00\x00\x00\xc0\x01\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x05\x00\x00\x00\x00\x00\x00\x10\x05\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00\x10\x05\x00\x00\x00\x00\x00\x00\x10\x15\x00\x00\x00\x00\x00\x00\x10\x15\x00\x00\x00\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x06\x00\x00\x00p\x05\x00\x00\x00\x00\x00\x00p%\x00\x00\x00\x00\x00\x00p%\x00\x00\x00\x00\x00\x00\x90\x00\x00\x00\x00\x00\x00\x00\x90\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x06\x00\x00\x00p\x05\x00\x00\x00\x00\x00\x00p%\x00\x00\x00\x00\x00\x00p%\x00\x00\x00\x00\x00\x00\x90\x00\x00\x00\x00\x00\x00\x00\x90\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00R\xe5td\x04\x00\x00\x00p\x05\x00\x00\x00\x00\x00\x00p%\x00\x00\x00\x00\x00\x00p%\x00\x00\x00\x00\x00\x00\x90\x00\x00\x00\x00\x00\x00\x00\x90\n\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00Q\xe5td\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x88\x02\x00\x00\x00\x00\x00\x00\x88\x02\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00q\x02\x00\x00 \x00\x00\x00AMDGPU\x00\x00\x83\xaeamdhsa.kernels\x91\xde\x00\x10\xa5.args\x93\x86\xae.address_space\xa6global\xa5.name\xa5buf_0\xa7.offset\x00\xa5.size\x08\xaa.type_name\xa6float*\xab.value_kind\xadglobal_buffer\x86\xae.address_space\xa6global\xa5.name\xa5buf_1\xa7.offset\x08\xa5.size\x08\xaa.type_name\xa6float*\xab.value_kind\xadglobal_buffer\x83\xa7.offset\x10\xa5.size\x08\xab.value_kind\xb3hidden_group_size_x\xb9.group_segment_fixed_size\x00\xb6.kernarg_segment_align\x08\xb5.kernarg_segment_size\x18\xa9.language\xa8OpenCL C\xb1.language_version\x92\x01\x02\xb8.max_flat_workgroup_size\xcd\x01\x00\xa5.name\xa4code\xbb.private_segment_fixed_size\x00\xab.sgpr_count\n\xb1.sgpr_spill_count\x00\xa7.symbol\xa7code.kd\xb3.uses_dynamic_stack\xc2\xab.vgpr_count\x07\xb1.vgpr_spill_count\x00\xaf.wavefront_size \xadamdhsa.target\xbaamdgcn-amd-amdhsa--gfx1100\xaeamdhsa.version\x92\x01\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xaf`\x84\x13\x00\x00\x08\x04\x00\x00\x00\x00\x00\x00\x80\x01\x04\xf4\x00\x00\x00\xf8\x00\x02\x04\xf4\x08\x00\x00\xf8\x02\x02\x06~\xff\x00\x006\xff\x03\x00\x00\x80\x00\x00\xf4\x10\x00\x00\xf8\x07\x00\x89\xbf\x03\x00\t\xd5\x03\x05\x00\x00\x00\x07\x06J\x04\x00\t\xd5\x03\t\x01\x00\x00\x00R\xdc\x04\x00\x08\x05\x07\x00\x89\xbf\x05\x0b\x0c\x10\x00\x00j\xdc\x04\x06\x06\x00\x03\x00\xb6\xbf\x00\x00\xb0\xbf\x00\x00\x9f\xbf\xfb\xff\xffo\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x88\x04\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\xcc\x04\x00\x00\x00\x00\x00\x00\n\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xf5\xfe\xffo\x00\x00\x00\x00\xa0\x04\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\xbc\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Linker: LLD 17.0.0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x03\x07\x00\x10\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x02\x08\x00p%\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x10\x00\x07\x00\x10\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x11\x00\x06\x00\xd0\x04\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00.note\x00.dynsym\x00.gnu.hash\x00.hash\x00.dynstr\x00.rodata\x00.text\x00.dynamic\x00.comment\x00.symtab\x00.shstrtab\x00.strtab\x00\x00code\x00_start\x00code.kd\x00_DYNAMIC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x07\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x88\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x0b\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x88\x04\x00\x00\x00\x00\x00\x00\x88\x04\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x01\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\xf6\xff\xffo\x02\x00\x00\x00\x00\x00\x00\x00\xa0\x04\x00\x00\x00\x00\x00\x00\xa0\x04\x00\x00\x00\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x00\x00\x00\x05\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\xbc\x04\x00\x00\x00\x00\x00\x00\xbc\x04\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x1f\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\xcc\x04\x00\x00\x00\x00\x00\x00\xcc\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\xd0\x04\x00\x00\x00\x00\x00\x00\xd0\x04\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00/\x00\x00\x00\x01\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x10\x15\x00\x00\x00\x00\x00\x00\x10\x05\x00\x00\x00\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x005\x00\x00\x00\x06\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00p%\x00\x00\x00\x00\x00\x00p\x05\x00\x00\x00\x00\x00\x00\x90\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00>\x00\x00\x00\x01\x00\x00\x000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\x06\x00\x00\x00\x00\x00\x00x\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\x03\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00O\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x06\x00\x00\x00\x00\x00\x00a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Y\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x06\x00\x00\x00\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
  @unittest.skip("RDNA assembly not correct (assembly_rdna.py, line 114, KeyError: <UnaryOps.SQRT: 6>)")
  def test_sqrt(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.sqrt(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (assembly_rdna.py, line 114, KeyError: <UnaryOps.SQRT: 6>)")
  def test_rsqrt(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.rsqrt(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_sin(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.sin(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_cos(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.cos(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_tan(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.tan(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_relu(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(64,64)]]
      Tensor.relu(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_leakyrelu(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(64,64)]]
      Tensor.leakyrelu(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_celu(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(64,64)]]
      tns.celu(5).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_abs(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(64,64)]]
      Tensor.abs(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_log(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(64,64)]]
      Tensor.log(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_log2(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.log2(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_exp(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.exp(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_sign(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.sign(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_softsign(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32)) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.softsign(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_sigmoid(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.sigmoid(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_softplus(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.softplus(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_gelu(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.gelu(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_quick_gelu(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.quick_gelu(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_elu(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.elu(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_relu6(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.relu6(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_hardswish(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.hardswish(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_mish(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      Tensor.mish(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_dot(self):
    with self.assertRaises(SystemExit) as asm:
      [tns, tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (65,100)]]
      tns.dot(tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_cumsum(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(20,30)]]
      Tensor.cumsum(tns, axis=1).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_sum(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3,4,5,6)]]
      Tensor.sum(tns, axis=3).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_min(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,3)]]
      Tensor.min(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_max(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,3)]]
      Tensor.max(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_mean(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3,4,5,6)]]
      tns.mean().realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_std(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65, 85)]]
      Tensor.std(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_log_softmax(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      tns.log_softmax(1).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_tanh(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      Tensor.tanh(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_hardtanh(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      tns.hardtanh(-10,10).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_topo_sort(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      tns.add(tns).mul(tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_scalar_mul(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      (tns*2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_scalar_rmul(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      (2*tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_scalar_sub(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      (tns-2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_scalar_rsub(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45, 65)]]
      (2-tns).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_broadcast_simple(self):
    with self.assertRaises(SystemExit) as asm:
      [tns,tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65), (45,1)]]
      (tns/tns2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_slice_in_bounds_1dim(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3)]]
      tns[1:3].realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_slice_int_indexing(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3)]]
      tns[1].realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_slice_in_bounds_multidim(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3,3,3)]]
      tns[1:2].realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_slice_stride_gt_one(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(7,5,10)]]
      tns[::2, ::3, ::4].realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_slice_ellipsis(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3,3,3,3)]]
      tns[..., 0].realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_pad2d(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3,3,3,3)]]
      tns.pad2d(padding=(1,2,3,4)).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_transpose(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(3,3,3)]]
      tns.transpose(1,2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_flip(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(4,3,6,6)]]
      tns.flip(axis=(0,)).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_expand(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(4,3,1,6)]]
      tns.expand((4,3,2,6)).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_simple_conv2d(self):
    with self.assertRaises(SystemExit) as asm:
      [tns, tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(1,4,9,9), (4,4,3,3)]]
      Tensor.conv2d(tns,tns2).relu().realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (assembly_rdna.py, line 114, KeyError: <BinaryOps.DIV: 4>)")
  def test_simple_conv_transpose2d(self):
    with self.assertRaises(SystemExit) as asm:
      [tns, tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(2,4,9,9), (4,4,3,3)]]
      Tensor.conv_transpose2d(tns,tns2).relu().realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_padding_add(self):
    with self.assertRaises(SystemExit) as asm:
      [tns, tns2] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(64,64), (60,60)]]
      tns+tns2.pad2d((2,2,2,2)).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("KeyError: dtypes.float2")
  def test_maxpool2d_simple(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(1,1,2,3)]]
      Tensor.max_pool2d(tns, kernel_size=(2,2)).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_avg_pool2d(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(32,2,111,28)]]
      Tensor.avg_pool2d(tns, kernel_size=(111,28)).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  def test_clip(self):
    with self.assertRaises(SystemExit) as asm:
      [tns] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(45,65)]]
      tns.clip(-2.3, 1.2).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)
  @unittest.skip("RDNA assembly not correct (llvm-mc returned non-zero exit status 1)")
  def test_matvec(self):
    with self.assertRaises(SystemExit) as asm:
      [tns, tns2, tns3] = [Tensor(((np.random.default_rng(seed=0).random(size=x, dtype=np.float32) + 0.5) * 3), dtype=dtypes.float32) for x in [(1,128), (128,128), (128,128)]]
      ((tns@tns2).relu()@tns3).realize().numpy()
    self.assertEqual(type(asm.exception.code), bytes)

# @TODO CMP tests, where, chunk, test_flip_eye_crash, skipped matmul to broadcast dot since dot isn't working, skipped conv2d tests, max_pooling 

if __name__ == "__main__":
  unittest.main()