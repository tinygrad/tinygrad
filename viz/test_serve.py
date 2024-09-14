import pickle
import unittest
from viz.serve import create_graph

class TestViz(unittest.TestCase):
  def test_get_uop(self):
    with open("/tmp/rewrites.pkl", "rb") as f: uops = pickle.load(f)
    ret = create_graph(uops[4])
    for g in ret.graphs:
      print(len(g))
      print("---------")

if __name__ == "__main__":
  unittest.main()
