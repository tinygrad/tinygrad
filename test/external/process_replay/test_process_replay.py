import unittest
import contextlib, sqlite3
from test.external.process_replay.helpers import ProcessReplayContext
from test.external.process_replay.process_replay import TABLE_NAME, diff_kernel

from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import to_function_name, db_connection, diskcache_put, VERSION
from tinygrad.ops import UOp
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.tensor import Tensor

def helper_append_replay(ast:UOp, name:str, src:str) -> int:
  name = f"kernel_{TABLE_NAME}"
  diskcache_put(name.replace(f"_{VERSION}", ""), "test_1", (ast, ClangRenderer(), [], to_function_name(name), src, ProcessReplayContext({})))
  conn = db_connection()
  row_count = conn.execute(f"select count(*) from '{name}'").fetchone()[0]
  return row_count

class TestProcessReplay(unittest.TestCase):
  def tearDown(self):
    conn = db_connection()
    cur = conn.cursor()
    with contextlib.suppress(sqlite3.OperationalError): cur.execute(f"DELETE FROM 'kernel_{TABLE_NAME}' WHERE key LIKE 'test_%'")
    conn.commit()
    cur.close()

  def test_simple_diff(self):
    out = Tensor([1, 2, 3])+1
    ast = out.schedule()[-1].ast
    test_src = """
void test(int* restrict a, const int* restrict b) {
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    int val0 = b[ridx0];
    a[ridx0] = (val0+1);
  }
}
    """
    offset = helper_append_replay(ast, "test", test_src)
    assert diff_kernel(offset-1) == (5, 4)

  def test_identical_run(self):
    out = Tensor([1, 2, 3])+1
    ast = out.schedule()[-1].ast
    test_prg = Kernel(ast, ClangRenderer()).to_program()
    offset = helper_append_replay(ast, test_prg.name, test_prg.src)
    assert diff_kernel(offset) == (0, 0)

if __name__ == "__main__":
  unittest.main()
