import pickle
import unittest
from viz.serve import create_graph

class TestViz(unittest.TestCase):
  def test_get_uop(self):
    with open("/tmp/rewrites.pkl", "rb") as f: uops = pickle.load(f)
    for loc, start, matches in uops[0]:
      print(loc)

if __name__ == "__main__":
  unittest.main()
