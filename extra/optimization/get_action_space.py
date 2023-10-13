from tqdm import tqdm
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.search import actions

if __name__ == "__main__":
  ast_strs = load_worlds(False, False, False)
  tactions = set()
  for ast_str in tqdm(ast_strs):
    lin = ast_str_to_lin(ast_str)
    lin.hand_coded_optimizations()
    for o in lin.applied_opts:
      assert o in actions
      tactions.add(o)
  print(len(tactions))
  print(sorted(list(tactions)))
