import unittest
from dataclasses import replace
from tinygrad import Tensor, dtypes, function
from tinygrad.helpers import Context
from tinygrad.schedule import lower_sink_to_linear, resolve_linear_call
from tinygrad.uop.ops import UOp, Ops, ParamArg, KernelInfo
from tinygrad.uop.spec import type_verify, spec_tensor, spec_kernel_graph, spec_program, eval_pyrender

class TestAlloc(unittest.TestCase):
  def alloc(self, size=4, device="NULL"):
    return UOp(Ops.ALLOC, arg=ParamArg(next(UOp.unique_num), dtypes.float, size, device=device))

  def test_properties(self):
    a = self.alloc()
    self.assertEqual((a.dtype, a.shape, a.device), (dtypes.float, (4,), "NULL"))
    self.assertIsNone(a.arg.buffer)
    self.assertIsNone(a.realized)
    self.assertTrue(a.has_buffer_identity())
    self.assertIs(a.reshape((2, 2)).storage_base, a)
    self.assertIs(a.reshape((2, 2)).buf_uop, a)
    with self.assertRaises(AssertionError): _ = a.buffer
    type_verify(a, spec_tensor)
    type_verify(a, spec_kernel_graph)
    with self.assertRaises(RuntimeError): type_verify(a, spec_program)

  def test_scalar_and_multidevice(self):
    self.assertEqual(self.alloc(size=None).shape, ())
    a = self.alloc(device=("NULL:0", "NULL:1"))
    self.assertIsNone(a.realized)
    self.assertIsNone(a.mselect(0).realized)
    self.assertIsNone(UOp.mstack(self.alloc(device="NULL:0"), self.alloc(device="NULL:1")).realized)

  @Context(SPEC=0)
  def test_spec_distinguishes_buffer_and_alloc(self):
    a = self.alloc()
    b = UOp.new_buffer("NULL", 4, dtypes.float)
    for spec in (spec_tensor, spec_kernel_graph):
      type_verify(b, spec)
      with self.assertRaises(RuntimeError): type_verify(a.replace(op=Ops.BUFFER), spec)
      with self.assertRaises(RuntimeError): type_verify(a.replace(arg=replace(a.arg, buffer=b.buffer)), spec)

  def test_pyrender(self):
    a = self.alloc()
    self.assertIs(eval_pyrender(a.pyrender()), a)
    self.assertIs(eval_pyrender(a.reshape((2, 2)).pyrender()).storage_base, a)

  def test_binding_per_invocation(self):
    a, b = self.alloc(device=None), UOp.new_buffer("NULL", 4, dtypes.float)
    p = b.param_like(0)
    kernel = p.index(0).store(p.index(0)).sink(arg=KernelInfo())
    linear = UOp(Ops.LINEAR, src=(kernel.call(a), kernel.call(a), kernel.call(b), kernel.call(p)))
    first, second = (resolve_linear_call(linear.call(b)) for _ in range(2))
    for resolved in (first, second):
      allocated = resolved.src[0].src[1]
      self.assertIs(allocated.op, Ops.BUFFER)
      self.assertEqual(allocated.device, b.device)
      self.assertIsNotNone(allocated.arg.buffer)
      self.assertIs(allocated, resolved.src[1].src[1])
      self.assertIs(resolved.src[2].src[1], b)
      self.assertIs(resolved.src[3].src[1], b)
    self.assertIsNot(first.src[0].src[1], second.src[0].src[1])
    self.assertIs(linear.src[0].src[1], a)
    self.assertIsNone(a.arg.buffer)

  def test_call_output_bufferization(self):
    @function(precompile=True)
    def f(x): return x + 1
    x = Tensor.empty(4, device="NULL")
    out = f(x)
    alloc = out.uop.storage_base
    self.assertIs(alloc.op, Ops.ALLOC)
    out._bufferize_outputs()
    self.assertIs(out.uop.storage_base.op, Ops.BUFFER)
    self.assertIsNone(alloc.arg.buffer)

  @Context(SCACHE=1)
  def test_cached_schedule_keeps_allocs(self):
    for device in ("NULL", ("NULL:0", "NULL:1")):
      with self.subTest(device=device):
        p, out = (UOp.param(i, dtypes.float, (4,), device=device) for i in range(2))
        value = (p + 1).contiguous() + 2 if isinstance(device, str) else p.allreduce(Ops.ADD, device)
        body = out.store(value).sink()
        call = body.call(self.alloc(device=device), self.alloc(device=device), precompile=True)
        first, second = lower_sink_to_linear(call), lower_sink_to_linear(call)
        self.assertIs(first.body, second.body)
        self.assertTrue(any(u.op is Ops.ALLOC for u in first.body.toposort()))
        self.assertFalse(any(u.op is Ops.BUFFER for u in first.body.toposort()))

if __name__ == "__main__": unittest.main()
