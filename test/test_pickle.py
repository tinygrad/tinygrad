import unittest, pickle
import numpy as np
from test.helpers import assert_equiv_uops
from tinygrad import Tensor, TinyJit, Variable
from tinygrad.codegen.kernel import Kernel
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.engine.schedule import create_schedule
from tinygrad.ops import BinaryOps, TernaryOps, UOp, UOps, UnaryOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

class TestPickle(unittest.TestCase):
  def test_pickle_realized_tensor(self):
    t = Tensor.rand(10, 10).realize()
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_unrealized_tensor(self):
    t = Tensor.ones(10, 10)
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_variable(self):
    v = Variable("i", 1, 20).bind(10)
    t1 = Tensor.ones(10, v).contiguous()
    t2 = Tensor.ones(10, v).contiguous()
    ret = (t1+t2).sum(1)
    st = pickle.dumps(ret)
    del ret
    vt2 = pickle.loads(st)
    np.testing.assert_equal(vt2.numpy(), 20)

  def test_pickle_buffer_view(self):
    t = Tensor.arange(10, device="CLANG").contiguous().realize()
    vt = t[3:5].contiguous().realize()
    assert hasattr(vt.lazydata.buffer, 'base')
    ref_value = vt.tolist()
    st = pickle.dumps(vt)
    del t, vt
    vt2 = pickle.loads(st)
    assert hasattr(vt2.lazydata.buffer, 'base')
    assert ref_value == vt2.tolist()

  def test_pickle_numpy(self):
    t = Tensor(np.array([1,2,3,4.]))
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_jit(self):
    @TinyJit
    def add(a, b): return a.sum()+b+1
    for _ in range(3): add(Tensor.rand(10, 10), Tensor.rand(10, 10))
    st = pickle.dumps(add)
    del add

    add_fxn = pickle.loads(st)
    x = Tensor.ones(10, 10).contiguous().realize()
    y = Tensor.ones(10, 10).contiguous().realize()
    print("post jit")
    out = add_fxn(x, y)
    np.testing.assert_equal(out.numpy(), 102)

  def test_pickle_schedule(self):
    a = Tensor([1,2])
    out = a + 2
    sched = create_schedule([out.lazydata])
    pk = pickle.dumps(sched)
    sched_pk = pickle.loads(pk)
    assert_equiv_uops(sched_pk[-1].ast, sched[-1].ast)

  def test_pickle_define_var(self):
    ast = UOp(UOps.SINK, dtypes.void, arg=None, src=(
      UOp(UOps.STORE, dtypes.void, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0, src=()),
        x2:=UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)), src=()), # noqa: E501
        UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
          UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (0, 1)), src=(
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=1, src=()),
              UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(Variable('i', 1, 10), 3), strides=(3, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)), # noqa: E501
          UOp(UOps.ALU, dtypes.float, arg=UnaryOps.RECIP, src=(
            UOp(UOps.CAST, dtypes.float, arg=None, src=(
              UOp(UOps.ALU, dtypes.int, arg=BinaryOps.MUL, src=(
                UOp(UOps.ALU, dtypes.int, arg=TernaryOps.WHERE, src=(
                  x12:=UOp(UOps.VALID, dtypes.bool, arg=None, src=(
                     x2,)),
                  UOp.define_var("i", dtypes.int, 1, 10),
                  x14:=UOp(UOps.CONST, dtypes.int, arg=0, src=()),)),
                UOp(UOps.ALU, dtypes.int, arg=TernaryOps.WHERE, src=(
                   x12,
                  UOp(UOps.CONST, dtypes.int, arg=3, src=()),
                   x14,)),)),)),)),)),)),))
    p = Kernel(ast).to_program(name_override="test")
    ps = Kernel(pickle.loads(pickle.dumps(ast))).to_program(name_override="test")
    self.assertEqual(ps.src, p.src)

class TestPickleJIT(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    @TinyJit
    def add(a, b): return a.sum()+b+1
    for _ in range(3): add(Tensor.rand(1000, 1000), Tensor.rand(1000, 1000))
    cls.st = pickle.dumps(add)
    del add

  def test_inspect(self):
    import io
    class FakeClass:
      def __init__(self, *args, **kwargs):
        print(self.module, self.name)
    class InspectUnpickler(pickle.Unpickler):
      def find_class(self, module, name): return type("SpecializedFakeClass", (FakeClass,), {"name": name, "module": module})
    InspectUnpickler(io.BytesIO(self.st)).load()

  @unittest.skip("we are still saving intermediate buffers")
  def test_size(self):
    # confirm no intermediate buffers are saved
    self.assertLess(len(self.st), 1_000_000)

if __name__ == '__main__':
  unittest.main()
