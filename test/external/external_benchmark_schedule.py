from typing import List
from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.helpers import Profiling, Timing, getenv
from tinygrad.ops import UOps
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.lowerer import ast_to_uop
from tinygrad.codegen.uopgraph import linearize_uop, full_graph_rewrite

if __name__ == "__main__":
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)

  PROFILE = getenv("PROFILE", 0)
  FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
  SCHEDULE_ONLY = getenv("SCHEDULE_ONLY", 0)

  with Timing("all "):
    with Timing("***** model tensor in    "):
      out = mdl(img)

    if not FORWARD_ONLY:
      with Timing("***** model schedule in  "):
        sched = out.schedule()

      if not SCHEDULE_ONLY:
        asts = {x.ast.key:x.ast for x in sched if x.ast.op is UOps.SINK}.values()
        kernels: List[Kernel] = []
        with Timing("***** model opts in      "):
          for ast in asts:
            k = Kernel(ast)
            k.hand_coded_optimizations()
            kernels.append(k)

        with Timing("***** model lower in     "): uops = [ast_to_uop(k.get_optimized_ast(), k.opts) for k in kernels]
        with Profiling(PROFILE, fn="/tmp/rewrite.prof"):
          with Timing("***** model rewrite in   "): uops = [full_graph_rewrite(u, k.opts) for u in uops]
        if getenv("LINEARIZE", 1):
          with Timing("***** model linearize in "): uops = [linearize_uop(u, skip_check=False) for u in uops]
          print(sum(len(u) for u in uops))
          if getenv("GRAPHUOPS", 0):
            for u in uops:
              from tinygrad.engine.graph import graph_uops
              graph_uops(u)
