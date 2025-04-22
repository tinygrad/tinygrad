import unittest, random
from tinygrad.dtype import dtypes
from tinygrad.ops import print_uops, UOp, Ops
from tinygrad.codegen.linearize import block_reorder
#from tinygrad.renderer.cstyle import ClangRenderer

def is_toposorted(lst:list[UOp]):
  seen = set()
  for u in lst:
    if any(p not in seen for p in u.src): return False
    seen.add(u)
  return True

class TestBlockReorder(unittest.TestCase):
  def test_loads_randomize(self):
    a = UOp(Ops.DEFINE_GLOBAL, dtype=dtypes.float.ptr(), arg=0)
    b = UOp(Ops.DEFINE_GLOBAL, dtype=dtypes.float.ptr(), arg=1)
    c = UOp(Ops.DEFINE_GLOBAL, dtype=dtypes.float.ptr(), arg=2)
    v1 = UOp(Ops.DEFINE_VAR, dtype=dtypes.int, arg=("a",))
    v2 = UOp(Ops.DEFINE_VAR, dtype=dtypes.int, arg=("b",))
    sink = c.store(sum([
      a.index(v1).load(dtype=dtypes.float),
      a.index(v1+1).load(dtype=dtypes.float),
      a.index(v1+2).load(dtype=dtypes.float),
      b.index(v2).load(dtype=dtypes.float),
      b.index(v2+1).load(dtype=dtypes.float),
      b.index(v2+2).load(dtype=dtypes.float),
    ])).sink()

    golden = block_reorder(sink.toposort)

    # test random order is always same
    for _ in range(50):
      # shuffle and form a valid toposort
      lst = golden[:]
      random.shuffle(lst)
      topolst = []
      for u in lst:
        for p in u.toposort:
          if p not in topolst: topolst.append(p)
      assert is_toposorted(topolst)

      for x,y in zip(golden, this_order:=block_reorder(topolst)):
        if x is not y:
          print_uops(golden)
          print_uops(this_order)
        self.assertIs(x, y)

if __name__ == '__main__':
  unittest.main()
