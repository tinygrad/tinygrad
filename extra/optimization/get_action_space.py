import random
from tqdm import tqdm
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import actions
from tinygrad.codegen.linearizer import Linearizer

tactions = set()
def test_rebuild(lin):
  linr = Linearizer(*lin.ast)
  for o in lin.applied_opts:
    assert o in actions, f"{o} is not in actions"
    tactions.add(o)
    linr.apply_opt(o)

  assert len(lin.sts) == len(linr.sts)
  for st1,st2 in zip(lin.sts, linr.sts):
    assert st1 == st2, f"{st1} != {st2}"

if __name__ == "__main__":
  ast_strs = load_worlds(False, False, False)
  random.shuffle(ast_strs)
  ast_strs = ast_strs[:2000]
  for ast_str in tqdm(ast_strs):
    lin = ast_str_to_lin(ast_str)
    #if not lin.apply_tensor_cores():
    lin.hand_coded_optimizations()
    test_rebuild(lin)
    # confirm linearize can be called twice
    uops1 = lin.linearize().uops
    uops2 = lin.linearize().uops
    assert tuple(uops1) == tuple(uops2), f"uops mismatch {lin.colored_shape()}"

  print(len(tactions), len(actions))
  print(sorted(list(tactions)))
