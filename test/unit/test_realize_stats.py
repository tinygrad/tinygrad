import io, unittest, contextlib
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.engine.realize import update_stats
from tinygrad.renderer import Estimates

class TestUpdateStats(unittest.TestCase):
  def _stats(self, mem, lds=0):
    GlobalCounters.reset()
    out = io.StringIO()
    with Context(DEBUG=2, NO_COLOR=1), contextlib.redirect_stdout(out):
      update_stats("copy", "TEST", Estimates(mem=mem, lds=lds), {}, 1.0, 2)
    return out.getvalue()

  def test_bandwidth_units(self):
    for mem, lds, expected in [
      (0, 0, "0|0      GB/s"),
      (200_000_000, 0, "200|0      MB/s"),
      (10_000_000_000_000, 0, "TB/s"),
    ]: assert expected in self._stats(mem, lds)

if __name__ == '__main__':
  unittest.main()
