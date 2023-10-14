from tqdm import tqdm
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.search import actions
from tinygrad.codegen.linearizer import Linearizer

if __name__ == "__main__":
  ast_strs = load_worlds(False, False, False)
  tactions = set()
  for ast_str in tqdm(ast_strs):
    lin = ast_str_to_lin(ast_str)
    if lin.apply_tensor_cores_old():
      linr = ast_str_to_lin(ast_str)
      linr.apply_tensor_cores()
    else:
      lin.hand_coded_optimizations()
      continue
      #lin.hand_coded_optimizations_old()
      #linr = ast_str_to_lin(ast_str)

    """
    if True or not lin.apply_tensor_cores():
      lin.hand_coded_optimizations()
    linr = Linearizer(lin.ast)
    for o in lin.applied_opts:
      assert o in actions
      tactions.add(o)
      linr.apply_opt(o)
    """

    assert len(lin.sts) == len(linr.sts)
    for st1,st2 in zip(lin.sts, linr.sts):
      assert st1 == st2, f"{st1} != {st2}"

    #lin.linearize()
    #linr.linearize()
    #assert lin.uops == linr.uops

  print(len(tactions), len(actions))
  print(sorted(list(tactions)))
