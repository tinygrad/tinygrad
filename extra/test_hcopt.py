from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.lowerer import pm_lowerer, get_index
from tinygrad.uop.ops import graph_rewrite
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations

if __name__ == "__main__":
  ast_strs = load_worlds()
  for i, ast_str in enumerate(ast_strs):
    lin = ast_str_to_lin(ast_str)
    opt1 = hand_coded_optimizations(lin)

    lowered = graph_rewrite(lin.ast, pm_lowerer, ctx=get_index(lin.ast), bottom_up=True)
    sch = Scheduler(lowered, lin.opts)
    opt2 = hand_coded_optimizations(sch)

    if opt1 != opt2:
      print("*******")
      print(opt1)
      print(opt2)





