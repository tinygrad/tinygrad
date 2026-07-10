import itertools, unittest
from typing import cast
from tinygrad import Device
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite
from tinygrad.renderer.isa.x86 import X86Renderer, X86Ops, x86op
from tinygrad.renderer.isa import IselContext

# INDEX on a register value with a constant index extracts a single element (the old GEP)
def lane(y:UOp, i:int) -> UOp: return y.index(UOp.const(dtypes.int, i), dtype=y.dtype)

def vector(name:str, dtype, count:int) -> UOp:
  return UOp(Ops.NOOP, dtype, (UOp.vectorize(*[UOp.variable(f"{name}{i}", 0, 0, dtype) for i in range(count)]),))

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, X86Renderer), "only x86")
class TestIselX86(unittest.TestCase):
  def isel_rewrite(self, x:UOp):
    ren = cast(X86Renderer, Device[Device.DEFAULT].renderer)
    x = graph_rewrite(x, ren.pre_isel_matcher, itertools.count(-1, -1), bottom_up=True)
    return graph_rewrite(x, ren.isel_matcher, IselContext(x), bottom_up=True)

  def _check_op(self, cases, expr):
    nargs = expr.__code__.co_argcount
    for dt,count,op in cases:
      with self.subTest(dtype=dt, count=count):
        v = [UOp.variable(str(i), 0, 0, dt) if count == 1 else vector(str(i), dt, count) for i in range(nargs)]
        n = self.isel_rewrite(expr(*v))
        self.assertIs(x86op(n), op)
        self.assertIs(n.dtype, dt)
        self.assertEqual(n.shape, () if count == 1 else (count,))

  def test_cmove(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    b = UOp.variable("b", 0, 0, dtypes.int32)
    c = (a < b).where(a, b)
    d = (a != b).where(a, b)
    f = c + d
    n = self.isel_rewrite(f)
    self.assertTrue(x86op(n.src[0]) is X86Ops.CMOVL and x86op(n.src[1]) is X86Ops.CMOVNE)
    # both comparisons become the same instruction
    self.assertTrue(n.src[0].src[2] == n.src[1].src[2] and x86op(n.src[0].src[2]) is X86Ops.CMP)

  def test_vmax(self):
    dt_op = [(dtypes.float32, 1, X86Ops.VMAXSS), (dtypes.float64, 1, X86Ops.VMAXSD),
             (dtypes.float32, 4, X86Ops.VMAXPS), (dtypes.float64, 2, X86Ops.VMAXPD)]
    self._check_op(dt_op, lambda a,b: (a < b).where(b, a))

  def test_vmin(self):
    dt_op = [(dtypes.float32, 1, X86Ops.VMINSS), (dtypes.float64, 1, X86Ops.VMINSD),
             (dtypes.float32, 4, X86Ops.VMINPS), (dtypes.float64, 2, X86Ops.VMINPD)]
    self._check_op(dt_op, lambda a,b: (a < b).where(a, b))

  def test_vfmadd(self):
    dt_op = [(dtypes.float32, 1, X86Ops.VFMADD213SS), (dtypes.float64, 1, X86Ops.VFMADD213SD),
             (dtypes.float32, 4, X86Ops.VFMADD213PS), (dtypes.float64, 2, X86Ops.VFMADD213PD)]
    self._check_op(dt_op, lambda a,b,c: a * b + c)

  # don't use fmadd if op being fused (mul) is used multiple times
  def test_no_vfmadd(self):
    dt_op = [(dtypes.float32, 1, X86Ops.VADDSS), (dtypes.float64, 1, X86Ops.VADDSD),
             (dtypes.float32, 4, X86Ops.VADDPS), (dtypes.float64, 2, X86Ops.VADDPD)]
    self._check_op(dt_op, lambda a,b: a * b + a * b)

  def test_vpbroadcast(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    n = self.isel_rewrite(a.broadcast(4))
    # need to move src from gpr to xmm before broadcasting
    self.assertTrue(x86op(n) is X86Ops.VPBROADCASTD and x86op(n.src[0]) is X86Ops.VMOVD)
    # if we can fuse a load we can skip the move and access memory directly
    load = UOp.param(0, dtypes.int32, (16,)).index(UOp.const(dtypes.int32, 0)).load()
    n = self.isel_rewrite(load.broadcast(4))
    self.assertTrue(x86op(n) is X86Ops.VPBROADCASTD and len(n.src) == 4)

  def test_narrow_load_fold(self):
    load = UOp.param(0, dtypes.uint8, (1,)).index(UOp.const(dtypes.index, 0)).load().cast(dtypes.uint16)
    n = self.isel_rewrite(load)
    self.assertIs(x86op(n), X86Ops.MOVZX)
    self.assertEqual(len(n.src), 4)

  def test_vbroadcastss(self):
    a = UOp.variable("a", 0, 0, dtypes.float32)
    valid = [UOp.vectorize(a, a, a, a), UOp.vectorize(a, a, a, a, a, a, a, a)]
    for shuf in valid: self.assertIs(x86op(self.isel_rewrite(shuf)), X86Ops.VBROADCASTSS)

  def test_vshufps(self):
    a, b = vector("a", dtypes.float32, 8), vector("b", dtypes.float32, 8)
    c = UOp.variable("c", 0, 0, dtypes.float32)
    d = UOp.variable("d", 0, 0, dtypes.float32)

    valid = [UOp.vectorize(c, c, d, d),
             UOp.vectorize(lane(a, 0), lane(a, 1), c, c),
             UOp.vectorize(lane(a, 0), lane(a, 1), lane(b, 2), lane(b, 3)),
             UOp.vectorize(lane(a, 1), lane(a, 2), lane(a, 3), lane(a, 0)),
             UOp.vectorize(lane(a, 3), lane(a, 2), lane(a, 1), lane(a, 0), lane(a, 7), lane(a, 6), lane(a, 5), lane(a, 4)),
             UOp.vectorize(lane(a, 0), lane(a, 0), lane(b, 1), lane(b, 1), lane(a, 4), lane(a, 4), lane(b, 5), lane(b, 5))]
    for shuf in valid: self.assertIs(x86op(self.isel_rewrite(shuf)), X86Ops.VSHUFPS)

    invalid = [UOp.vectorize(lane(a, 0), lane(b, 1), lane(a, 2), lane(b, 3)),
               UOp.vectorize(lane(a, 0), lane(a, 0), lane(a, 0), lane(a, 0), lane(a, 4), lane(a, 4), lane(a, 4), lane(a, 5))]
    for shuf in invalid: self.assertIsNot(x86op(self.isel_rewrite(shuf)), X86Ops.VSHUFPS)

  def test_vshufpd(self):
    a, b = vector("a", dtypes.float64, 4), vector("b", dtypes.float64, 4)
    c = UOp.variable("c", 0, 0, dtypes.float64)
    d = UOp.variable("d", 0, 0, dtypes.float64)

    valid = [UOp.vectorize(c, d),
             UOp.vectorize(lane(a, 0), c),
             UOp.vectorize(lane(a, 1), lane(b, 1)),
             UOp.vectorize(lane(a, 0), lane(b, 1), lane(a, 2), lane(b, 3)),
             UOp.vectorize(lane(a, 1), lane(a, 1), lane(a, 3), lane(a, 3))]
    for shuf in valid: self.assertIs(x86op(self.isel_rewrite(shuf)), X86Ops.VSHUFPD)

    invalid = [UOp.vectorize(c, c, c, c),
               UOp.vectorize(lane(a, 0), lane(a, 1), lane(b, 2), lane(b, 3)),
               UOp.vectorize(lane(a, 0), lane(b, 1), lane(a, 0), lane(b, 1))]
    for shuf in invalid: self.assertIsNot(x86op(self.isel_rewrite(shuf)), X86Ops.VSHUFPD)

  def test_vinsertps(self):
    a, b, c = vector("a", dtypes.float32, 4), vector("b", dtypes.float32, 4), vector("c", dtypes.float32, 4)
    d = UOp.variable("e", 0, 0, dtypes.float32)
    # moving 0th element to position 0 does nothing so only 1 vinsertps is generated
    n = self.isel_rewrite(UOp.vectorize(lane(a, 0), d))
    self.assertIs(x86op(n), X86Ops.VINSERTPS)
    self.assertIsNot(x86op(n.src[0]), X86Ops.VINSERTPS)

    valid = [UOp.vectorize(lane(a, 0), lane(b, 1), lane(a, 2), lane(b, 3)),
             UOp.vectorize(lane(a, 3), lane(b, 2), lane(c, 1), d)]
    for shuf in valid: self.assertIs(x86op(self.isel_rewrite(shuf)), X86Ops.VINSERTPS)

  # complex address is [base + index*scale + displacement]
  def test_complex_address(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    load = UOp.param(0, dtypes.int32, (16,)).index(a + 1).load()
    n = self.isel_rewrite(load)
    # displacement is the constant in "a" scaled to the buffer element size, dtype is int8 when the value fits otherwise int32
    self.assertTrue(n.src[2].op is Ops.CONST and n.src[2].dtype is dtypes.int8 and n.src[2].arg == 4)

  def test_fold_load(self):
    load1 = UOp.param(0, dtypes.int32, (16,)).index(UOp.const(dtypes.int32, 0)).load()
    load2 = UOp.param(0, dtypes.int32, (16,)).index(UOp.const(dtypes.int32, 1)).load()
    n = self.isel_rewrite(load1 + load2)
    self.assertTrue(len(n.src) == 5)

  # don't fold when used multiple times
  def test_dont_fold_load(self):
    load = UOp.param(0, dtypes.int32, (16,)).index(UOp.const(dtypes.int32, 0)).load()
    # used by multiple users
    n = self.isel_rewrite(load + 1 + load)
    self.assertTrue(len(n.src) == 2)
    # used mutiple times by same user
    n = self.isel_rewrite(load * load)
    self.assertTrue(len(n.src) == 2)

if __name__ == "__main__":
  unittest.main()
