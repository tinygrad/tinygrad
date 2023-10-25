from typing import Tuple, Any, List
import unittest
from tinygrad.ops import ASTRunner, GraphBatchExecutor

class TestGraph(GraphBatchExecutor):
  def __init__(self, jit_cache: List[Tuple[Any, Any, Any]]):
    super().__init__(jit_cache)

    self.next_jit = 0
    self.update_called = 0
    self.jcid_to_instid = {}
    self.jc_info = []
    self.exec_set = set()
    self.split_into_graphs(jit_cache)
    assert len(self.jc_info) == len(jit_cache), f"each jit cache entry should be captured into nodes. {len(self.jc_info)} != {len(jit_cache)}"

    target_size = [4, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for i, inst in enumerate(self.graphs):
      assert len(inst) == target_size[i] or (i == len(self.graphs) - 1 and len(inst) > 0), "unexpected graph size"

  def create_graph(self, jit_cache: List[Tuple[Any, Any, Any]]):
    for prg, pargs, variables in jit_cache:
      self.jcid_to_instid[len(self.jc_info)] = len(self.graphs)
      assert pargs == self.next_jit, "prog is written 2+ times in the graph or some of them are skipped"
      self.next_jit += 1
      self.jc_info.append((prg, pargs, variables))
    self.graphs.append(jit_cache)

  def update_node(self, instid, jcid, prg, pargs, variables, updated_args=None):
    self.update_called += 1
    assert instid == self.jcid_to_instid[jcid], "jit cache entry does not belong to the given instance"

  def exec_instance(self, instid):
    assert 0 <= instid < len(self.graphs), "called unknown instance"
    self.exec_set.add(instid)

class TestBatchExec(unittest.TestCase):
  def test_graph_batch_exec_partition(self):

    def _helper(jit_cache_size, updates_count):
      fake_jit_cache = [(ASTRunner("test", "", [1], [1]), i, i) for i in range(jit_cache_size)]
      updatable_entries = {i:0 for i in range(updates_count)}
      gr = TestGraph(fake_jit_cache)
      gr.exec(fake_jit_cache, updatable_entries)

      assert gr.update_called == updates_count, "not all updates are called"
      assert len(gr.exec_set) == len(gr.graphs), "every instance should be executed"

    _helper(512, 512)
    _helper(334, 13)
    _helper(812, 111)
    _helper(2, 0)
    _helper(1, 1)
    _helper(4, 3)
    _helper(7, 2)
    _helper(8, 8)

if __name__ == '__main__':
  unittest.main()