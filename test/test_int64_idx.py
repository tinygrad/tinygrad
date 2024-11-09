import unittest
from tinygrad import dtypes
from tinygrad.codegen.uopgraph import load_store_indexing, graph_rewrite
from tinygrad.ops import UOp, Ops

class TestInt64Indexing(unittest.TestCase):
  def test_int64_indexing(self):
    idx = UOp(Ops.INDEX, dtypes.char.ptr(), arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.char.ptr(), arg=0, src=()),
            UOp(Ops.ADD, dtypes.int, arg=None, src=(
              UOp(Ops.MUL, dtypes.int, arg=None, src=(
                UOp(Ops.RANGE, dtypes.int, arg=(0, False), src=(
                  UOp(Ops.CONST, dtypes.int, arg=0, src=()),
                  UOp(Ops.CONST, dtypes.int, arg=715827884, src=()),)),
                UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
              UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),))

    int64_idx = graph_rewrite(idx, load_store_indexing)

    target = UOp(Ops.INDEX, dtypes.char.ptr(), arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.char.ptr(), arg=0, src=()),
              UOp(Ops.ADD, dtypes.long, arg=None, src=(
                UOp(Ops.CAST, dtypes.long, arg=None, src=(
                  UOp(Ops.MUL, dtypes.long, arg=None, src=(
                    UOp(Ops.CAST, dtypes.long, arg=None, src=(
                      UOp(Ops.RANGE, dtypes.int, arg=(0, False), src=(
                        UOp(Ops.CONST, dtypes.int, arg=0, src=()),
                        UOp(Ops.CONST, dtypes.int, arg=715827884, src=()),)),)),
                    UOp(Ops.CAST, dtypes.long, arg=None, src=(
                      UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),)),)),
                UOp(Ops.CAST, dtypes.long, arg=None, src=(
                  UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),)),))

    assert int64_idx == target

if __name__ == '__main__':
  unittest.main()