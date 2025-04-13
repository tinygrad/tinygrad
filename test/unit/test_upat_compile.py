import unittest
from tinygrad.helpers import DEBUG
from tinygrad.dtype import dtypes
from tinygrad.ops import UPat, track_rewrites, GroupOp, Ops
import dis

@track_rewrites()
def do_compile(up):
  print("\n***** COMPILE", up)
  up.compile()
  print(up.match_code)
  if DEBUG >= 2: dis.dis(up.match)

class TestUPatCompile(unittest.TestCase):
  def test_double(self):
    up = UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1")
    do_compile(up)

  def test_single(self):
    up = UPat.var("x") + UPat.var("y")
    do_compile(up)

  def test_xpx(self):
    up = UPat.var("x") + UPat.var("x")
    do_compile(up)

  def test_xp0(self):
    up = UPat.var("x") + 0
    do_compile(up)

  def test_bool(self):
    up = UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool)
    do_compile(up)

  def test_single_c(self):
    up = (UPat.var("x") + UPat.var("y")) * UPat.var("c")
    do_compile(up)

  def test_const_folding(self):
    up = UPat(GroupOp.ALU, name="a", src=UPat((Ops.VCONST, Ops.CONST)))
    do_compile(up)

if __name__ == "__main__":
  unittest.main()
