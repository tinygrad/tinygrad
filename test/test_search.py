import unittest

from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.codegen.kernel import Kernel
from tinygrad.ops import UOp, UOps, BinaryOps
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.search import time_linearizer, bufs_from_lin, actions, beam_search
from tinygrad.device import Device, Buffer
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.engine.realize import capturing
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from unittest.mock import Mock, patch
from tinygrad.engine.search import _time_program, _get_test_global_size
from tinygrad.engine.realize import CompiledRunner
from tinygrad.shape.symbolic import Variable
from dataclasses import dataclass, field
from functools import reduce

class TestTimeLinearizer(unittest.TestCase):
  def test_reasonable_time(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast.op is UOps.SINK][0]
    out = Buffer(Device.DEFAULT, si.outputs[0].size, si.outputs[0].dtype).allocate()
    memops = {x.src[0].arg:x.src[-1].arg.real_size() for x in si.ast.parents if x.op is UOps.LOAD}
    rawbufs = [out] + [Buffer(Device.DEFAULT, memops[i], x.dtype).allocate() for i,x in enumerate(si.inputs, start=len(si.outputs))]
    tm = time_linearizer(Kernel(si.ast), rawbufs, allow_test_size=False, cnt=10, disable_cache=True)
    assert tm > 0 and tm != float('inf')

  def test_bufs_from_lin(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast.op is UOps.SINK][0]
    rawbufs = bufs_from_lin(lin:=Kernel(si.ast))
    assert len(rawbufs) == len(lin.membufs) == 2
    assert all(r is not None for r in rawbufs)
    assert all(isinstance(r, Buffer) for r in rawbufs)
    assert all(r.size > 0 for r in rawbufs)

  def test_bufs_from_lin_alt(self):
    a = Tensor.randn(4, 4)
    b = a+a[0]
    si = [si for si in b.schedule() if si.ast.op is UOps.SINK][0]
    rawbufs = bufs_from_lin(k:=Kernel(si.ast))
    assert len(rawbufs) == len(k.membufs) == 2
    assert all(r is not None for r in rawbufs)
    assert all(isinstance(r, Buffer) for r in rawbufs)
    assert all(r.size > 0 for r in rawbufs)

  def test_kernel_count(self):
    """
    Ensure that the kernel count is not incremented by time_linearizer when clearing l2
    """
    # ast of Tensor.zeros(16).contiguous().realize()
    ast = UOp(UOps.SINK, src=(
      UOp(UOps.STORE, src=(
        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0, src=()),
        UOp(UOps.SHAPETRACKER, arg=ShapeTracker(views=(View(shape=(16,), strides=(1,), offset=0, mask=None, contiguous=True),))),
        UOp(UOps.CONST, dtypes.float, arg=0.0, src=(
          UOp(UOps.SHAPETRACKER, arg=ShapeTracker(views=(View(shape=(16,), strides=(0,), offset=0, mask=None, contiguous=False),))),)),)),))
    lin = Kernel(ast)
    bufs = bufs_from_lin(lin)

    kernel_count = GlobalCounters.kernel_count
    time_linearizer(lin, bufs, allow_test_size=False, cnt=2, disable_cache=True, clear_l2=True)
    assert GlobalCounters.kernel_count == kernel_count, "kernel count was incremented by time_linearizer"

class TestBEAM(unittest.TestCase):
  def test_dynamic_beam(self):
    # TODO: make this infra globally usable
    class Capture:
      def __init__(self): self.captured = []
      def add(self, x): self.captured.append(x)

    capturing.append(Capture())
    kernel_count = GlobalCounters.kernel_count
    with Context(BEAM=1): Tensor.zeros(16).contiguous().realize()
    assert GlobalCounters.kernel_count == kernel_count + 1
    k_beam_1 = capturing[0].captured
    capturing.clear()

    capturing.append(Capture())
    kernel_count = GlobalCounters.kernel_count
    with Context(BEAM=0): Tensor.zeros(16).contiguous().realize()
    assert GlobalCounters.kernel_count == kernel_count + 1
    k_beam_0 = capturing[0].captured
    capturing.clear()
    self.assertNotEqual(k_beam_0[-1].prg.p.src, k_beam_1[-1].prg.p.src)

  def test_get_kernel_actions(self):
    from test.test_linearizer import helper_realized_ast
    a = Tensor.rand(4, 3)
    b = Tensor.rand(3)
    realized_ast, _ = helper_realized_ast(a @ b)
    from tinygrad.engine.search import get_kernel_actions
    lins = get_kernel_actions(Kernel(realized_ast), False).values()

    # ensure amt=0 are not duplicated
    if Opt(OptOps.UPCAST, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UPCAST, axis=0, amt=4)]) == 0, "did not de-dup UPCAST"
    if Opt(OptOps.LOCAL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.LOCAL, axis=0, amt=4)]) == 0, "did not de-dup LOCAL"
    if Opt(OptOps.UNROLL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UNROLL, axis=0, amt=3)]) == 0, "did not de-dup UNROLL"
    if Opt(OptOps.GROUP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUP, axis=0, amt=3)]) == 0, "did not de-dup GROUP"
    if Opt(OptOps.GROUPTOP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUPTOP, axis=0, amt=3)]) == 0, "did not de-dup GROUPTOP"

  def test_filter_global_buffer(self):
    # taken from https://github.com/tinygrad/tinygrad/issues/4612
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0, src=()),
        UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(1, 1, 256), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),)), src=()), # noqa: E501
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.MAX, (1,)), src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
            UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
              UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
                UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
                  UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
                    UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
                      UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=1, src=()),
                        UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=0, mask=((0, 64128),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=()),)), # noqa: E501
                      UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=2, src=()),
                        UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-64128, mask=((64128, 128256),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=()),)),)), # noqa: E501
                    UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                      UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=3, src=()),
                      UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-128256, mask=((128256, 192384),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=()),)),)), # noqa: E501
                  UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                    UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=4, src=()),
                    UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-192384, mask=((192384, 256512),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=()),)),)), # noqa: E501
                UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                  UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=5, src=()),
                  UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-256512, mask=((256512, 320640),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=()),)),)), # noqa: E501
              UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=6, src=()),
                UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-320640, mask=((320640, 384768),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=()),)),)), # noqa: E501
            UOp(UOps.CONST, dtypes.float, arg=1.4285714285714286, src=(
              UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(1, 501, 256), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)) # noqa: E501
    lin = Kernel(ast)

    bufs = bufs_from_lin(lin)
    best_lin = beam_search(lin, bufs, 3)
    assert best_lin
    # need disable_cache to trigger.
    tm = time_linearizer(best_lin, bufs, allow_test_size=False, cnt=2, disable_cache=True)
    assert tm

  def test_beam_unnamed_kernels(self):
    a = Tensor.rand(100)
    b = Tensor.rand(100)
    si = (a+b).schedule()[-1]
    lin = Kernel(si.ast)
    bufs = bufs_from_lin(lin)
    # TODO: beam should have better instrumentation so we don't have to check this indirect thing
    kcount = len(Kernel.kernel_cnt)
    beam_search(lin, bufs, 3, disable_cache=True)
    self.assertEqual(kcount, len(Kernel.kernel_cnt))

@dataclass
class MockProgram:
  global_size: list
  dname: str
  uops: list = None
  globals: list = field(default_factory=list)

class MockDevice:
  def __init__(self, has_invalidate=True):
    if has_invalidate:
      self.invalidate_caches = self._invalidate_caches

  def _invalidate_caches(self):
    pass


class TestTimeProgram(unittest.TestCase):
  def setUp(self):
    self.mock_compiled_runner = Mock(spec=CompiledRunner)
    self.mock_compiled_runner.p = MockProgram(
        global_size=None, dname="MockDevice", globals=[0])
    self.mock_device = MockDevice()
    self.mock_tensor = Mock(spec=Tensor)

  def test_time_program_basic(self):
    p = MockProgram(global_size=None, dname="MockDevice", globals=[0])
    lib = b"mock_lib"
    var_vals = {}
    rawbufs = [Mock()]

    with patch('tinygrad.engine.search.CompiledRunner', return_value=self.mock_compiled_runner):
      self.mock_compiled_runner.return_value = 0.1
      result = _time_program(p, lib, var_vals, rawbufs, cnt=3)

    self.assertEqual(len(result), 3)
    self.assertTrue(all(t == 0.1 for t in result))

  def test_time_program_early_stop(self):
    p = MockProgram(global_size=None, dname="MockDevice", globals=[0])
    lib = b"mock_lib"
    var_vals = {}
    rawbufs = [Mock()]

    with patch('tinygrad.engine.search.CompiledRunner', return_value=self.mock_compiled_runner):
      self.mock_compiled_runner.return_value = 0.2
      result = _time_program(p, lib, var_vals, rawbufs, early_stop=0.1, cnt=3)

    self.assertEqual(len(result), 1)
    self.assertEqual(result[0], 0.2)

  def test_time_program_adjust_global_size(self):
    p = MockProgram(global_size=[1000, 1000], dname="MockDevice", globals=[0])
    lib = b"mock_lib"
    var_vals = {Variable("x", 1, 1000): 500}
    rawbufs = [Mock()]

    with patch('tinygrad.engine.search.CompiledRunner', return_value=self.mock_compiled_runner), \
            patch('tinygrad.engine.search._get_test_global_size', return_value=([500, 500], 2)):
      self.mock_compiled_runner.return_value = 0.1
      result = _time_program(p, lib, var_vals, rawbufs,
                             max_global_size=65536, cnt=1)

    self.assertEqual(len(result), 1)
    self.assertEqual(result[0], 0.2)  # 0.1 * 2 (factor)

  def test_time_program_clear_cache_with_invalidate(self):
    self._test_time_program_clear_cache(True, True)

  # probablamatic test
  def test_time_program_clear_cache_without_invalidate(self):
    self._test_time_program_clear_cache(True, False)

  def test_time_program_no_clear_cache(self):
    self._test_time_program_clear_cache(False, True)

  def _test_time_program_clear_cache(self, clear_l2, has_invalidate):
    p = MockProgram(global_size=None, dname="MockDevice", globals=[0])
    lib = b"mock_lib"
    var_vals = {}
    rawbufs = [Mock()]

    mock_device = MockDevice(has_invalidate=has_invalidate)

    with patch('tinygrad.engine.search.CompiledRunner', return_value=self.mock_compiled_runner), \
            patch('tinygrad.engine.search.Device', {p.dname: mock_device}), \
            patch('tinygrad.engine.search.Context') as mock_context:

      self.mock_compiled_runner.return_value = 0.1
      _time_program(p, lib, var_vals, rawbufs, clear_l2=clear_l2, cnt=1)

      if clear_l2:
        if has_invalidate:
          self.assertTrue(hasattr(mock_device, 'invalidate_caches'))
        else:
          mock_context.assert_called_once_with(DEBUG=0, BEAM=0, CAPTURING=0)
      else:
        if has_invalidate:
          self.assertTrue(hasattr(mock_device, 'invalidate_caches'))
        mock_context.assert_not_called()

  def test_time_program_compilation_error(self):
    p = MockProgram(global_size=None, dname="MockDevice", globals=[0])
    lib = b"mock_lib"
    var_vals = {}
    rawbufs = [Mock()]

    with patch('tinygrad.engine.search.CompiledRunner', side_effect=AssertionError):
      result = _time_program(p, lib, var_vals, rawbufs, cnt=3)

    self.assertEqual(len(result), 3)
    self.assertTrue(all(t == float('inf') for t in result))

  def test_get_test_global_size(self):
    global_size = [1000, 1000, 1000]
    max_global_size = 1000000
    var_vals = {Variable("x", 1, 1000): 500}

    result_size, factor = _get_test_global_size(
        global_size, max_global_size, var_vals)

    self.assertEqual(len(result_size), 3)
    self.assertTrue(all(size <= 1000 for size in result_size))
    self.assertGreaterEqual(factor, 1)
    self.assertLessEqual(
        reduce(lambda a, b: a*b, result_size), max_global_size)
