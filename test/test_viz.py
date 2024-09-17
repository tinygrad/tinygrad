# ** setup
from typing import List
import unittest
import os

from viz.serve import replace_uop
prev_val = os.getenv("TRACK_MATCH_STATS")
os.environ["TRACK_MATCH_STATS"] = "2"
os.environ["PRINT_MATCH_STATS"] = "0"
os.environ["FORWARD_ONLY"] = "1"
from tinygrad.helpers import DEBUG
from tinygrad.ops import UOp, contexts
from tinygrad import Tensor
from test.external.process_replay.helpers import print_diff

class TestViz(unittest.TestCase):
  def tearDown(self) -> None:
    contexts.clear()
    if prev_val: os.environ["TRACK_MATCH_STATS"]
    else: del os.environ["TRACK_MATCH_STATS"]

  def test_ctx_diff(self):
    a = Tensor.ones(4, 1).contiguous().realize()
    contexts.clear()
    out = a + a.reshape(1, 4)
    out.realize()
    for ctx in contexts:
      uops = [ctx.sink]
      for i, (first, rewritten, pat) in enumerate(ctx.rewrites):
        start = uops[-1]
        found = [x for x in start.sparents if x.key == first.key]
        assert found, f"can't find UOp for rewrite_num={i} pattern={pat}"
        changed: List[UOp] = []
        new = replace_uop(start, first, rewritten, cache={})
        if DEBUG >= 4: print_diff(start, new)
        changed = [x for x in new.sparents if x not in start.sparents]
        assert len(changed) == len(found), f"{len(changed)} != {len(found)}"
        assert tuple(changed) == tuple(found), f"{changed} != {found}"

if __name__ == "__main__":
  unittest.main()
