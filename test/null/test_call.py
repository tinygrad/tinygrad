import unittest
from tinygrad import Tensor, UOp, function, Device
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops
from tinygrad.schedule import transform_to_call

def sched_key(t:Tensor): return transform_to_call(UOp.sink(t.uop)).body.key

class TestCall(unittest.TestCase):
  def test_call_scalar_param_shape_mismatch(self):
    scalar_fxn = UOp.param(0, dtypes.float, ()) * 2
    with self.assertRaisesRegex(TypeError, "shape mismatch: expected scalar"):
      Tensor.call(Tensor.ones(2), fxn=scalar_fxn).realize()

class TestCallShape(unittest.TestCase):
  def test_call_shape_int(self):
    # fixed-shape function: shape passes through unchanged
    @function
    def f(x:Tensor) -> Tensor: return x * 2
    self.assertEqual(f(Tensor.empty(4, 8)).shape, (4, 8))

  def test_call_shape_param_substitution(self):
    # symbolic shape dimension is substituted: inner PARAM replaced with the BIND arg
    @function
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    shape = f(Tensor.empty(8)[:sz.bind(5)]).shape
    # the PARAM should be gone, replaced with the BIND from the call arg
    self.assertIsInstance(shape[0], UOp)
    self.assertNotEqual(shape[0].op, Ops.PARAM)
    self.assertEqual(shape[0], sz.bind(5))

  def test_call_shape_expr_substitution(self):
    # expression containing PARAMs in shape gets fully substituted
    @function
    def f(x:Tensor) -> Tensor: return x + 1
    sz = UOp.variable("sz", 1, 10)
    shape = f(Tensor.empty(10, 4)[:sz.bind(3)]).shape
    self.assertIsInstance(shape[0], UOp)
    self.assertNotEqual(shape[0].op, Ops.PARAM)
    self.assertEqual(shape[1], 4)

  def test_call_shape_no_param_passthrough(self):
    # a non-PARAM UOp shape element passes through unchanged
    @function
    def f(x:Tensor) -> Tensor: return x * 3
    sz = UOp.variable("sz", 1, 8)
    shape = f(Tensor.empty(8)[:sz.bind(5)]).shape
    self.assertEqual(shape[0], sz.bind(5))

class TestCallSchedule(unittest.TestCase):
  def test_precompile_schedule_cache_hit(self):
    """two instances of the same @function should produce identical scheduled function keys without aliasing their outputs"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x + Tensor.full(x.shape, -1.0)
    a = Tensor.empty(4, 8)
    b = Tensor.empty(4, 8)
    r0, r1 = f(a), f(b)
    c0 = next(u for u in r0.uop.toposort() if u.op is Ops.CALL and u.arg.precompile)
    c1 = next(u for u in r1.uop.toposort() if u.op is Ops.CALL and u.arg.precompile)
    self.assertTrue(c0.has_unbound_outputs)
    self.assertTrue(c1.has_unbound_outputs)
    # output identities stay unique per call; they canonicalize only when combined into a scheduling scope
    self.assertIsNot(c0.src[-1], c1.src[-1])
    self.assertEqual(sched_key(r0), sched_key(r1))

class TestArgOrder(unittest.TestCase):
  def _dev(self, x): return x.device if isinstance(x.device, str) else (x.device or (Device.DEFAULT,))[0]

  def test_output_pos_symbolic_shape(self):
    # symbolic output shapes resolve against the final arg slots, not the input order (PARAM(2) in the shape, output at 0)
    x = Tensor.empty(8).realize()
    sz = UOp.variable('sz', 1, 8)
    dev = self._dev(x)
    p1, p2 = UOp.param(1, x.dtype, x.shape, dev), sz.param_like(2)
    value = p1.reshape(x.shape).shrink_to((p2,))
    bound = sz.bind(5)
    outs = UOp.call_with_outputs((value,), x.uop, bound, output_pos=(0,))
    # the minted output's shape substituted PARAM(2) with the bind arg from position 2 in the arg list
    shp = outs[0].shape[0]
    self.assertIsInstance(shp, UOp)
    self.assertNotEqual(shp.op, Ops.PARAM)
    self.assertEqual(shp, bound)

  def test_output_pos_must_be_ascending(self):
    x = Tensor.arange(3, dtype=dtypes.int).realize()
    p1 = UOp.param(1, x.dtype, x.shape, self._dev(x))
    with self.assertRaises(AssertionError):
      UOp.call_with_outputs((p1.reshape(x.shape) * 2, p1.reshape(x.shape) + 1), x.uop, output_pos=(1, 0))

class TestCallCodegen(unittest.TestCase):
  def test_call_stack_pointer(self):
    slot = UOp.placeholder((1,), dtypes.uint32, addrspace=AddrSpace.REG)
    call = UOp.custom_function("callback").call(UOp.const(0, dtypes.uint64), slot[0], ret_dtype=dtypes.void)
    prg = to_program(call.sink(arg=KernelInfo("call_stack")), ClangRenderer(Target("CPU", arch="x86_64,x86-64")))
    self.assertIn("(unsigned int*)((buf", prg.src[2].arg)

if __name__ == '__main__':
  unittest.main()
