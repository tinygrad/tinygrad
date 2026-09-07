import unittest
from tinygrad import Tensor, Context, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo


class TestCallAccess(unittest.TestCase):
  def test_computed_reads_writes_and_unused_arguments(self):
    out, x, unused = (UOp.param(i, dtypes.float, (1,), "CPU") for i in range(3))
    body = out.store(x + 1).sink(arg=KernelInfo())
    self.assertEqual(body.call(out, x, unused).call_access(), ((x,), (out,)))

  def test_computed_read_modify_write(self):
    out, x = (UOp.param(i, dtypes.float, (1,), "CPU") for i in range(2))
    body = out.store(out + x).sink(arg=KernelInfo())
    self.assertEqual(body.call(out, x).call_access(), ((out, x), (out,)))

  def test_computed_empty_effects(self):
    x = UOp.param(0, dtypes.float, (1,), "CPU")
    self.assertEqual(UOp.sink(x, arg=KernelInfo()).call(x).call_access(), ((), ()))

  def test_computed_program_accesses(self):
    out, x = (UOp.param(i, dtypes.float, (1,), "CPU") for i in range(2))
    sink = out.store(x.load()).sink(arg=KernelInfo())
    program = UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo.from_sink(sink))
    self.assertEqual(program.call(out, x).call_access(), ((x,), (out,)))

  def test_nested_linear_parameter_scopes(self):
    a, b, c = (UOp.param(i, dtypes.float, (1,), "CPU") for i in range(3))
    inner = a.store(b + 1).sink(arg=KernelInfo()).call(b, a)
    body = UOp(Ops.LINEAR, src=(inner,))
    self.assertEqual(body.call(a, b, c).call_access(), ((a,), (b,)))

  def test_copy_accesses(self):
    out, x = (UOp.param(i, dtypes.float, (1,), "CPU") for i in range(2))
    self.assertEqual(UOp(Ops.COPY, src=(x,), arg=out.device).call(out, x).call_access(), ((x,), (out,)))

  def test_unknown_opaque_accesses_reject(self):
    x = UOp.param(0, dtypes.float, (1,), "CPU")
    bodies = (UOp(Ops.PROGRAM, src=(UOp.sink(x),)),
              UOp(Ops.CUSTOM, src=(x,), arg=("", dtypes.void)).sink(arg=KernelInfo()))
    for body in bodies:
      with self.assertRaisesRegex(RuntimeError, "cannot compute accesses"): body.call(x).call_access()

  @Context(DEV="CPU")
  def test_unknown_effects_do_not_replace_tensors_on_failure(self):
    def kernel(x): return UOp(Ops.PROGRAM, src=(UOp.sink(x, arg=KernelInfo()),))
    x = Tensor([2.]).realize().custom_kernel(fxn=kernel)[0]
    before = x.uop
    for _ in range(2):
      with self.assertRaisesRegex(RuntimeError, "cannot compute accesses"): x.realize()
      self.assertIs(x.uop, before)

  def test_bad_access_slots(self):
    arg = UOp.param(0, dtypes.float, (1,), "CPU")
    for slot in (-1, 1):
      p = UOp(Ops.PROGRAM, src=(UOp.sink(arg),), arg=ProgramInfo(globals=(0,), ins=(slot,), outs=()))
      with self.assertRaisesRegex(RuntimeError, "invalid CALL access slot"): p.call(arg, arg).call_access()

  def test_compiled_writable_alias_rejects(self):
    a, b = (UOp.param(i, dtypes.float, (1,), "CPU") for i in range(2))
    sink = a.store(b.load()).sink(arg=KernelInfo())
    program = UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo.from_sink(sink))
    with self.assertRaisesRegex(RuntimeError, "aliased opaque"): program.call(a, a).call_access()


if __name__ == "__main__": unittest.main()
