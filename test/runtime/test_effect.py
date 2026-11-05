import unittest, functools
from tinygrad import Tensor, UOp, Device, Variable
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError

class TestEffectRetention(unittest.TestCase):
  def test_unused_reduce_dependency(self):
    for value in (UOp.const(2), UOp.const((1, 2, 3, 4)).reshape((2, 2))):
      with self.subTest(shape=value.shape):
        def kernel(C:UOp) -> UOp:
          i = UOp.range(3, 0, AxisType.LOOP)
          cell = C.after(i)[0]
          v = value.after(cell.store(cell + 1))
          return C.after(v.reduce(i, arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
        self.assertEqual(Tensor.zeros(1, dtype=dtypes.int).contiguous().custom_kernel(fxn=kernel)[0].tolist(), [3])

  def test_store_behind_value_after_kept(self):
    def kernel(A:UOp, loop:bool) -> UOp:
      x = UOp.const(1, dtypes.weakint).after(A[0].store(7))
      return (A[(r:=UOp.range(x, 0))+1].store(9).end(r) if loop else A[x].store(9)).sink(arg=KernelInfo(opts_to_apply=()))
    for loop in (False, True):
      out = Tensor.custom_kernel(Tensor.zeros(2, dtype=dtypes.int).contiguous().realize(), fxn=functools.partial(kernel, loop=loop))[0]
      self.assertEqual(out.tolist(), [7, 9])

  def test_store_before_empty_loop(self):
    r = UOp.range(0, 0, AxisType.LOOP)
    self.assertEqual(Tensor.ones(1).contiguous().custom_kernel(Tensor.empty(1),
      fxn=lambda p,q: q.after(p[0].store(8.), r)[0].store(64.).end(r).sink(arg=KernelInfo(opts_to_apply=())))[0].tolist(), [8.])

  def test_deleted_store_keeps_deps(self):
    for f in (lambda a, s: s[1].store(s[1].load()), lambda a, s: s[UOp.invalid()].store(5),
              lambda a, s: s[1].store(UOp.invalid()), lambda a, s: a[UOp.invalid()].store(s[1].load())):
      out = Tensor.custom_kernel(Tensor.zeros(3, dtype=dtypes.int).contiguous(),
                                 fxn=lambda a: f(a, a.after(a[0].store(7))).sink(arg=KernelInfo(opts_to_apply=())))[0]
      self.assertEqual(out.tolist(), [7, 0, 0])

  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage) and Device.DEFAULT != "WEBGPU", "CUSTOM renders C")
  def test_group_keeps_custom(self):
    r = UOp.range(4, 0, AxisType.LOOP)
    def kernel(C:UOp, D:UOp) -> UOp:
      inc = UOp.custom_function("{0}[0] += 1;", fmt=True).call(C.index(r))
      return UOp.group(inc, D[r].store(1)).end(r).sink(arg=KernelInfo(opts_to_apply=()))
    c, d = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).contiguous(), Tensor.zeros(4, dtype=dtypes.int).contiguous(), fxn=kernel)[:2]
    Tensor.realize(c, d)
    self.assertEqual((c.tolist(), d.tolist()), ([1]*4, [1]*4))

class TestEffectOrder(unittest.TestCase):
  def test_snapshot_before_overwrite(self):
    def kernel(A:UOp) -> UOp:
      old, r = A[0].load(), UOp.range(2, 0, AxisType.LOOP)
      return A.after(A.after(old, r)[0].store(9).end(r))[1].store(old).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor([5, 0], dtype=dtypes.int).contiguous().realize().custom_kernel(fxn=kernel)[0].tolist(), [9, 5])

  def test_snapshot_before_loop_bound(self):
    def kernel(A:UOp) -> UOp:
      old = A[0].load()
      l = UOp(Ops.RANGE, src=(UOp(Ops.NOOP).after(old),), arg=(AxisType.WEAK, 0))
      done = (w:=A.after(l)[0].store(9)).backedge(l, A.after(w)[0].load().ne(9))
      return A.after(done)[1].store(old).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor([5, 0], dtype=dtypes.int).contiguous().realize().custom_kernel(fxn=kernel)[0].tolist(), [9, 5])

  def test_store_value_behind_coalesced_store(self):
    def c(v): return UOp.const(v, dtypes.int)
    for stores, want in ((lambda A: (A[0].store(7), A[1].store(A.after(A[0].store(7))[0]+5)), [7, 12, 5, 0]),
                         (lambda A: (A[1].store(c(1).after(A[0].store(7)) + A[2].load()),), [7, 6, 5, 0]),
                         (lambda A: (A[1].store(c(1).after(A[2].store(2))), A[3].store(c(3).after(A[0].store(10)))), [10, 1, 2, 3])):
      out = Tensor([0, 0, 5, 0], dtype=dtypes.int).contiguous().custom_kernel(fxn=lambda A: UOp.sink(*stores(A), arg=KernelInfo(opts_to_apply=())))[0]
      self.assertEqual(out.tolist(), want)

  def test_range_merge_keeps_reduce_order(self):
    # the reduce body stores, so it loops a, k, b in order, and the digits of S are the A[a*2+b] it visits
    def kernel(O:UOp, S:UOp, A:UOp) -> UOp:
      a, k, b = [UOp.range(2, n, AxisType.LOOP) for n in range(3)]
      v = A[a*2+b].after((cell:=S.after(a, k, b)[0]).store(cell*10+A[a*2+b]))
      return O[0].store(v.reduce(a, k, b, arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
    o, s = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), Tensor.zeros(1, dtype=dtypes.int).contiguous(),
                                Tensor([1, 2, 3, 4], dtype=dtypes.int), fxn=kernel)[:2]
    Tensor.realize(o, s)
    self.assertEqual(s.item(), 12123434)

  def test_range_merge_keeps_loop_order(self):
    # loops nest by axis id whatever order END lists them, so x is outer and the digits of S are A[0], A[2], A[1], A[3]
    def kernel(S:UOp, A:UOp) -> UOp:
      x, y = UOp.range(2, 0, AxisType.LOOP), UOp.range(2, 1, AxisType.LOOP)
      return (cell:=S.after(x, y)[0]).store(cell*10+A[y*2+x]).end(y, x).sink(arg=KernelInfo(opts_to_apply=()))
    s = Tensor.custom_kernel(Tensor.zeros(1, dtype=dtypes.int).contiguous(), Tensor([1, 2, 3, 4], dtype=dtypes.int), fxn=kernel)[0]
    self.assertEqual(s.item(), 1324)

class TestEffectFrequency(unittest.TestCase):
  def test_range_merge_keeps_after_domain(self):
    # C is read once per i or once per o, so it counts to 4, and merging o and i must not make that 16
    o, i = UOp.range(4, 0, AxisType.LOOP), UOp.range(4, 1, AxisType.LOOP)
    stores = {"after_i": lambda C,A: (c:=C.after(i)[0]).store(c+1),
              "after_o": lambda C,A: UOp.group((c:=C.after(o)[0]).store(c+1), A.after(o, i)[0].store(1)),
              "load_after_o": lambda C,A: C.after(o, i)[0].store(C.after(o)[0].load()+1)}
    for name, store in stores.items():
      with self.subTest(name):
        def kernel(C:UOp, A:UOp, store=store) -> UOp: return store(C, A).end(o, i).sink(arg=KernelInfo(opts_to_apply=()))
        self.assertEqual(Tensor.zeros(1).contiguous().custom_kernel(Tensor.empty(1), fxn=kernel)[0].item(), 4)

  def test_range_merge_keeps_index_domain(self):
    # C[c] is indexed by o alone, so each cell counts to 2 while the B store merges o and i
    o, i = UOp.range(4, 0, AxisType.LOOP), UOp.range(4, 1, AxisType.LOOP)
    for name, c in {"div": o//2, "cmp": (o<2).cast(dtypes.int)}.items():
      with self.subTest(name):
        def kernel(C:UOp, B:UOp, c=c) -> UOp:
          return UOp.group(C[c].store(C[c]+1), B[o*4+i].store(1)).end(o, i).sink(arg=KernelInfo(opts_to_apply=()))
        self.assertEqual(Tensor.custom_kernel(Tensor.zeros(2).contiguous(), Tensor.zeros(16).contiguous(), fxn=kernel)[0].tolist(), [2, 2])

  def test_range_merge_then_shrink(self):
    # r0<2 shrinks the merged range of 8 to 4, which is the same UOp as r0, so it isn't shrunk again to 2
    def kernel(C:UOp) -> UOp:
      r0, r1 = UOp.range(4, 0, AxisType.LOOP), UOp.range(2, 1, AxisType.LOOP)
      return (cell:=C[(r0*2+r1).valid(r0<2)]).store(cell+1).end(r0, r1).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.zeros(8, dtype=dtypes.int).contiguous().custom_kernel(fxn=kernel)[0].tolist(), [1, 1, 1, 1, 0, 0, 0, 0])

  def test_shrink_range_keeps_ungated_index(self):
    # q is ungated in C[q], so q keeps its 4 iterations after r shrinks to its gate r<2
    q, r = UOp.range(4, 0, AxisType.LOOP), UOp.range(4, 1, AxisType.LOOP)
    def kernel(C:UOp, D:UOp) -> UOp:
      return UOp.group(C[q].store(1), D[(q*4+r).valid((q<2)&(r<2))].store(1)).end(q, r).sink(arg=KernelInfo(opts_to_apply=()))
    c, d = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).contiguous(), Tensor.zeros(16, dtype=dtypes.int).contiguous(), fxn=kernel)[:2]
    Tensor.realize(c, d)
    self.assertEqual(c.tolist(), [1, 1, 1, 1])
    self.assertEqual(d.tolist(), [1, 1, 0, 0, 1, 1, 0, 0] + [0]*8)

  def test_shrink_range_keeps_ungated_store(self):
    # C[0] stores r every iteration, so r keeps its 4 iterations though D gates it to r<2
    r = UOp.range(4, 0, AxisType.LOOP)
    def kernel(C:UOp, D:UOp) -> UOp:
      return UOp.group(C[0].store(r.cast(dtypes.int)), D[r.valid(r<2)].store(1)).end(r).sink(arg=KernelInfo(opts_to_apply=()))
    c, d = Tensor.custom_kernel(Tensor.zeros(1, dtype=dtypes.int).contiguous(), Tensor.zeros(4, dtype=dtypes.int).contiguous(), fxn=kernel)[:2]
    Tensor.realize(c, d)
    self.assertEqual((c.tolist(), d.tolist()), ([3], [1, 1, 0, 0]))

  def test_shrink_range_keeps_store_gated_otherwise(self):
    # C[0] is gated by r>=2, not r<c, so r keeps its 4 iterations though D gates it to r<2
    r = UOp.range(4, 0, AxisType.LOOP)
    def kernel(C:UOp, D:UOp) -> UOp:
      return UOp.group(C[UOp.const(0).valid(r>=2)].store(r.cast(dtypes.int)), D[r.valid(r<2)].store(1)).end(r).sink(arg=KernelInfo(opts_to_apply=()))
    c, d = Tensor.custom_kernel(Tensor.zeros(1, dtype=dtypes.int).contiguous(), Tensor.zeros(4, dtype=dtypes.int).contiguous(), fxn=kernel)[:2]
    Tensor.realize(c, d)
    self.assertEqual((c.tolist(), d.tolist()), ([3], [1, 1, 0, 0]))

  def test_shrink_range_keeps_symbolic_end(self):
    # D gates r to r<4, but r only runs to n=2
    def kernel(D:UOp, N:UOp) -> UOp:
      r = UOp.range(N, 0, AxisType.LOOP)
      return D[r.valid(r<4)].store(1).end(r).sink(arg=KernelInfo(opts_to_apply=()))
    d = Tensor.custom_kernel(Tensor.zeros(8, dtype=dtypes.int).contiguous(), Tensor(Variable("n", 1, 8, dtypes.int).bind(2)), fxn=kernel)[0]
    self.assertEqual(d.tolist(), [1, 1, 0, 0, 0, 0, 0, 0])

  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage) and Device.DEFAULT != "WEBGPU", "CUSTOM renders C")
  def test_shrink_range_keeps_custom(self):
    # the CUSTOM sets C[0] to r every iteration, so r keeps its 4 iterations though the D store gates it to r<2
    r = UOp.range(4, 0, AxisType.LOOP)
    def kernel(C:UOp, D:UOp) -> UOp:
      val = UOp.custom_function("({0}[0] = {1})", shape=C[0].shape, dtype=dtypes.int, fmt=True).call(C[0], r.cast(dtypes.int))
      return D[r.valid(r<2)].store(val).end(r).sink(arg=KernelInfo(opts_to_apply=()))
    c, d = Tensor.custom_kernel(Tensor.zeros(1, dtype=dtypes.int).contiguous(), Tensor.zeros(4, dtype=dtypes.int).contiguous(), fxn=kernel)[:2]
    Tensor.realize(c, d)
    self.assertEqual((c.tolist(), d.tolist()), ([3], [0, 1, 0, 0]))

  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage) and Device.DEFAULT != "WEBGPU", "CUSTOM renders C")
  def test_reduce_collapse_keeps_custom(self):
    # the CUSTOM increments C[r] every iteration, though the sum only picks r==2
    r = UOp.range(4, 0, AxisType.LOOP)
    def kernel(C:UOp, O:UOp) -> UOp:
      inc = UOp.custom_function("({0}[0] += 1)", shape=C.index(r).shape, dtype=dtypes.int, fmt=True).call(C.index(r))
      return O[0].store(UOp.const(2, dtypes.int).eq(r.cast(dtypes.int)).where(inc, 0).reduce(r, arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
    c, o = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).contiguous(), Tensor.zeros(1, dtype=dtypes.int).contiguous(), fxn=kernel)[:2]
    Tensor.realize(c, o)
    self.assertEqual((c.tolist(), o.tolist()), ([1]*4, [1]))

  def test_padto_gates_store(self):
    # C[0] is 3 after the 3 iterations of r, and padding r to 4 adds a no-op iteration
    r, l = UOp.range(3, 0, AxisType.LOOP), UOp.loop(1)
    def loop_until(C:UOp) -> UOp:  # a padded iteration exits the loop: its gated reg never reaches 3 again
      reg = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG).after(r)[0].set(0)
      nxt = reg.after(l)[0]+1
      return C[0].store(reg.after(reg[0].store(nxt).backedge(l, nxt.ne(3)))[0])
    stores = {"after": lambda C: (c:=C.after(r)[0]).store(c+1), "shared_load": lambda C: (c:=C[0]).store(c.load()+r.cast(dtypes.int)+1),
              "backedge": loop_until}
    for name, store in stores.items():
      with self.subTest(name):
        def kernel(C:UOp, store=store) -> UOp: return store(C).end(r).sink(arg=KernelInfo(opts_to_apply=(Opt(OptOps.PADTO, 0, 4),)))
        self.assertEqual(Tensor.zeros(1, dtype=dtypes.int).contiguous().custom_kernel(fxn=kernel)[0].tolist(), [3])

  def test_padto_refuses_ungatable(self):
    # a CUSTOM and a STORE to a whole buffer can't be no-ops, so padding r is refused
    r = UOp.range(3, 0, AxisType.LOOP)
    effects = {"custom": lambda C: UOp.custom_function("{0}[0] += {1}+1;", fmt=True).call(C.index(0), r.cast(dtypes.int)).end(r),
               "whole_store": lambda C: C.store((r.cast(dtypes.int)+1).reshape((1,))).end(r)}
    for name, effect in effects.items():
      with self.subTest(name):
        def kernel(C:UOp, effect=effect) -> UOp: return effect(C).end(r).sink(arg=KernelInfo(opts_to_apply=(Opt(OptOps.PADTO, 0, 4),)))
        with self.assertRaises(KernelOptError): Tensor.zeros(1, dtype=dtypes.int).contiguous().custom_kernel(fxn=kernel)[0].realize()

if __name__ == '__main__':
  unittest.main()
