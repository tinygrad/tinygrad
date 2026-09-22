import unittest
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Context, Target
from tinygrad.uop.ops import KernelInfo, Ops, UOp, graph_rewrite
from tinygrad.uop.spec import spec_shared, spec_tensor, spec_program, spec_hcq, spec_full, eval_pyrender
from tinygrad.uop.render import pyrender
from tinygrad.uop.symbolic import sym
from tinygrad.codegen.simplify import pm_flatten_range
from tinygrad.codegen.late.linearizer import pm_split_ends
from tinygrad.renderer import Estimates
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.renderer.ptx import PTXRenderer
from test.helpers import get_uops


class TestBackedge(unittest.TestCase):
  def setUp(self):
    self.loop = UOp.loop(0)
    self.outer = UOp.range(4, 1, dtype=dtypes.int)
    self.body = UOp.param(0, dtypes.int, 1).index(0).store(UOp.const(1, dtypes.int))
    self.cond = UOp.cconst(False, dtypes.bool)
    self.specs = (spec_shared, spec_tensor, spec_program, spec_hcq, spec_full)

  def test_spec(self):
    # Both statement and value-producing bodies are legal. BACKEDGE always discards the body's result.
    call = UOp.custom_function("poll", UOp.const(0, dtypes.uint64)).call(ret_dtype=dtypes.int)
    for body in (self.body, UOp.const(1, dtypes.int), self.body.src[0].load(), call, UOp(Ops.NOOP)):
      end = body.backedge(self.loop, self.cond)
      self.assertEqual(end.dtype, dtypes.void)
      self.assertIsNone(end._shape)
      self.assertIsNone(end.arg)
      self.assertEqual(end.ended_ranges, (self.loop,))
      for spec in self.specs: self.assertIs(spec.rewrite(end), True)

  def test_condition_scope(self):
    end = self.body.backedge(self.loop, self.outer < 2)
    self.assertEqual(set(end.ranges), {self.outer})
    self.assertEqual(end.ended_ranges, (self.loop,))
    self.assertEqual(set((self.outer+1).after(end).ranges), {self.outer})
    self.assertFalse(end.end(self.outer).ranges)

  def test_symbolic_preserves_effect(self):
    for cond in (UOp.const(True), UOp.const(False)):
      end = self.body.backedge(self.loop, cond)
      result = graph_rewrite(self.body.src[0].after(end), sym)
      self.assertIn(end, result.toposort())
      self.assertIs(graph_rewrite(end, pm_flatten_range+pm_split_ends), end)

  def test_noop_body_keeps_condition(self):
    cond = UOp.param(0, dtypes.int, 1, volatile=True).after(self.loop).index(0).load() < 1
    end = UOp(Ops.NOOP).backedge(self.loop, cond)
    self.assertIs(graph_rewrite(end, sym), end)
    self.assertIn(cond, end.toposort())

  def test_render_roundtrip(self):
    end = self.body.backedge(self.loop, self.outer < 2).end(self.outer)
    self.assertIn(".backedge(", pyrender(end))
    self.assertIs(eval_pyrender(pyrender(end)), end)

  def test_invalid_forms(self):
    with Context(SPEC=0):
      invalid = [
        UOp(Ops.BACKEDGE, src=()),
        UOp(Ops.BACKEDGE, src=(self.body, self.loop)),
        UOp(Ops.BACKEDGE, src=(self.body, self.loop, self.cond, self.outer)),
        UOp(Ops.BACKEDGE, src=(self.body, self.loop, self.cond), arg="loop"),
        self.body.backedge(self.outer, self.cond),
        self.body.backedge(self.loop, UOp.const(1, dtypes.int)),
        self.body.backedge(self.loop, UOp.stack(self.cond, self.cond)),
        self.body.backedge(self.loop, UOp.invalid()),
        self.body.end(self.loop, self.cond),  # old conditional END spelling is no longer accepted
        self.body.end(self.loop),
        self.body.end(self.outer, self.cond),
      ]
      for end in invalid:
        for spec in self.specs: self.assertIsNot(spec.rewrite(end), True)

  def test_bounded_end_unchanged(self):
    end = self.body.end(self.outer)
    for spec in self.specs: self.assertIs(spec.rewrite(end), True)
    self.assertEqual(end.dtype, dtypes.void)
    self.assertEqual(end.ended_ranges, (self.outer,))

  def test_renderers(self):
    out = UOp.param(0, dtypes.int, 1)
    counter = UOp.placeholder((1,), dtypes.int, 0, AddrSpace.REG)
    counter = counter.after(counter.index(0).store(0))
    inc = counter.after(self.loop).index(0).load()+1
    done = counter.index(0).store(inc).backedge(self.loop, inc < 3)
    sink = out.index(0).store(counter.after(done).index(0).load()).sink(arg=KernelInfo(opts_to_apply=()))
    renderers = ((ClangRenderer(Target("CPU", "CLANG")), "break;"),
                 (LLVMRenderer(Target("CPU", "LLVM")), "br i1"),
                 (PTXRenderer(Target("CUDA", "PTX", "sm_80")), "bra WAITLOOP_"))
    for renderer, branch in renderers:
      with self.subTest(renderer=type(renderer).__name__):
        uops = get_uops(sink, renderer)
        self.assertEqual(sum(u.op is Ops.BACKEDGE for u in uops), 1)
        self.assertTrue(all(len(u.src) == 2 for u in uops if u.op is Ops.END))
        self.assertIn(branch, renderer.render(uops))
        Estimates.from_uops(tuple(uops))  # BACKEDGE must balance its RANGE's estimate stack too


if __name__ == '__main__':
  unittest.main()
