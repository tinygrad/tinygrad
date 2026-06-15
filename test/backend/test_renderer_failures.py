import unittest
import shutil, subprocess
import numpy as np
from tinygrad.device import Device
from tinygrad.dtype import dtypes, ConstType
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import do_to_program, to_program
from tinygrad.helpers import prod, Target
from tinygrad.renderer.cstyle import CStyleLanguage, ClangRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import UOp, Ops, KernelInfo, python_alu
from tinygrad.tensor import Tensor, _to_np_dtype

def _test_uop_result(inputs:list[Tensor], sink:UOp, local_size=None):
  for x in inputs: x.realize()
  sz = 1 if local_size is None else prod(local_size)
  outs = [UOp.new_buffer(Device.DEFAULT, sz, u.src[1].dtype) for u in sink.src if u.op is Ops.STORE]
  for u in outs: u.buffer.allocate().copyin(np.zeros(sz, dtype=_to_np_dtype(u.dtype)).data)
  run_linear(UOp(Ops.LINEAR, src=(sink.call(*outs, *(x.uop.base for x in inputs)),)))
  return [u.buffer.numpy() for u in outs]

def _setup_and_test_alu(alu_op:Ops, input_val:ConstType, *alu_src_uops:UOp):
  dtype = alu_src_uops[0].dtype
  a = UOp.param(0, dtype.ptr(1))
  b = UOp.param(1, dtype.ptr(1))
  idx = UOp.const(dtypes.int, 0)
  ld = b.index(idx, ptr=True).load()
  alu = ld.alu(alu_op, *alu_src_uops)
  store = UOp.store(a.index(idx, ptr=True), alu)
  return _test_uop_result([Tensor([input_val])], UOp(Ops.SINK, dtypes.void, (store,), arg=KernelInfo()))[0]

class _NoopCompiler:
  def compile_cached(self, src:str) -> bytes: return b""

def _clang_src(stores:tuple[UOp, ...]) -> str:
  renderer = ClangRenderer(Target.parse("CPU:CPU:x86_64,core2"))
  renderer.compiler = _NoopCompiler()
  return do_to_program(UOp(Ops.SINK, dtypes.void, stores, arg=KernelInfo()), renderer).src[3].arg

class TestRendererFailures(unittest.TestCase):
  @unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, PythonRenderer)), "test is for ptx or python renderer")
  def test_gated_store_with_alu(self):
    a = UOp.param(0, dtypes.int.ptr(4))
    gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 4),), 'lidx0')).ne(0)
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0.valid(gate_alu), ptr=True), UOp.const(dtypes.int, 1)))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,), arg=KernelInfo())
    ret = _test_uop_result([], sink, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  @unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, PythonRenderer)), "test is for ptx or python renderer")
  def test_gated_store_with_alu_2d(self):
    a = UOp.param(0, dtypes.int.ptr(8))
    gate_alu_0 = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 4),), 'lidx0')).ne(0)
    gate_alu_1 = (lidx1:=UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 2),), 'lidx1')).ne(0)
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index((lidx0+lidx1*4).valid(gate_alu_0&gate_alu_1), ptr=True), UOp.const(dtypes.int, 1)))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,), arg=KernelInfo())
    ret = _test_uop_result([], sink, local_size=[4, 2, 1])[0]
    np.testing.assert_equal(ret, [0, 0, 0, 0, 0, 1, 1, 1])

class TestClangRendererFailures(unittest.TestCase):
  @unittest.skipUnless(shutil.which("clang"), "need clang")
  def test_bfloat16_renders_without_native_bf16(self):
    idx = UOp.const(dtypes.int, 0)
    out_half, out_bf16, inp_bf16 = UOp.param(0, dtypes.half.ptr(1)), UOp.param(0, dtypes.bfloat16.ptr(1)), UOp.param(1, dtypes.bfloat16.ptr(1))
    sources = [
      _clang_src((out_half.index(idx, ptr=True).store(inp_bf16.index(idx, ptr=True).load().cast(dtypes.half)),)),
      _clang_src((out_bf16.index(idx, ptr=True).store(inp_bf16.index(idx, ptr=True).load() + UOp.const(dtypes.bfloat16, 1.0)),)),
      _clang_src((out_bf16.index(idx, ptr=True).store(UOp.const(dtypes.bfloat16, 1.0)),)),
    ]
    cmd = ["clang", "-c", "-x", "c", "--target=x86_64-none-unknown-elf", "-march=core2", "-O2", "-ffreestanding",
           "-nostdlib", "-", "-o", "/tmp/tinygrad_clang_bf16.o"]
    for src in sources:
      self.assertNotIn("__bf16", src)
      self.assertIn("unsigned short", src)
      subprocess.check_output(cmd, input=src.encode())

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "uops are for cstyle")
class TestCStyleFailures(unittest.TestCase):
  def test_inline_const_alu(self):
    # CPU doesn't use the max function
    ret = _setup_and_test_alu(Ops.MAX, 1, UOp.const(dtypes.int, dtypes.int.min+1))
    self.assertEqual(ret[0], 1)

  def _test_src_strip_paren(self, op: Ops, should_strip_paren:bool=True):
    dtype = "bool" if op in (Ops.OR, Ops.XOR, Ops.AND) else None
    ret = Tensor.empty(1, dtype=dtype)
    for _ in range(5): ret = python_alu[op](ret, Tensor.empty(1, dtype=dtype))
    linear = ret.schedule_linear()
    assert len(linear.src) == 1
    src = to_program(linear.src[0].src[0], Device[Device.DEFAULT].renderer).src[3].arg
    self.assertEqual("("*5 not in src, should_strip_paren)

  def test_repeat_add(self): self._test_src_strip_paren(Ops.ADD)
  def test_repeat_mul(self): self._test_src_strip_paren(Ops.MUL)
  def test_repeat_xor(self): self._test_src_strip_paren(Ops.XOR)
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "wgsl ends up with '(' * 5")
  def test_repeat_or(self): self._test_src_strip_paren(Ops.OR)
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "wgsl ends up with '(' * 5")
  def test_repeat_and(self): self._test_src_strip_paren(Ops.AND)
  def test_repeat_sub(self): self._test_src_strip_paren(Ops.SUB, should_strip_paren=False)

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "tests for wgsl renderer")
class TestWGSLFailures(unittest.TestCase):
  def test_multiply_infinity(self):
    # multiplying a positive constant by infinity should return infinity
    # WGSL pipelines do not handle this reliably, some of which return zero, unless infinity always comes from a read on a dynamic buffer
    ret = _setup_and_test_alu(Ops.MUL, 5.0, UOp.const(dtypes.float32, float("inf")))
    self.assertEqual(ret[0], float("inf"))

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "tests for ptx renderer")
class TestPTXFailures(unittest.TestCase):
  @unittest.skip("INDEX can only have a gate ALU parent, not an IF")
  def test_gated_store_with_if(self):
    a = UOp.param(0, dtypes.int.ptr())
    gate_alu = (lidx0:=UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 4),), 'lidx0')).ne(0)
    val = UOp.const(dtypes.int, 1)
    if_uop = UOp(Ops.IF, dtypes.void, (gate_alu,))
    gated_alu_store = UOp(Ops.STORE, dtypes.void, (a.index(lidx0, if_uop), val))
    sink = UOp(Ops.SINK, dtypes.void, (gated_alu_store,), arg=KernelInfo())
    ret = _test_uop_result([], sink, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  @unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
  def test_gated_define_acc_with_half_dtype(self):
    a = Tensor.randn(32, 32, dtype=dtypes.half).realize()
    b = Tensor.randn(34, 32, dtype=dtypes.half).realize()
    result = a.pad((1,1)).matmul(b, dtype=dtypes.half).numpy()
    reference = a.pad((1,1)).matmul(b, dtype=dtypes.float).numpy()
    np.testing.assert_allclose(result, reference, atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
  unittest.main()
