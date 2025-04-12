import unittest
from tinygrad.ops import UPat, track_rewrites

@track_rewrites()
def do_compile(up):
  up.compile()

class TestUPatCompile(unittest.TestCase):
  def test_double(self):
    up = UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1")
    do_compile(up)

  def test_single(self):
    up = UPat.var("x") + UPat.var("y")
    do_compile(up)

  def test_single_c(self):
    up = (UPat.var("x") + UPat.var("y")) * UPat.var("c")
    do_compile(up)

if __name__ == "__main__":
  unittest.main()
