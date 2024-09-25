from typing import List
from extra.models.resnet import ResNet50
from tinygrad import Tensor, Device
from tinygrad.helpers import Profiling, Timing, getenv, BEAM, NOOPT, DEBUG, Context, ansilen, _CURRENT_KERNEL
from tinygrad.ops import UOps
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.lowerer import ast_to_uop
from tinygrad.codegen.uopgraph import linearize_uop, full_graph_rewrite
from tinygrad.engine.search import beam_search, bufs_from_lin

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
        asts = list({x.ast.key:x.ast for x in sched if x.ast.op is UOps.SINK}.values())
        if (restrict_kernel := getenv("RESTRICT_KERNEL", -1)) != -1: asts = asts[restrict_kernel:restrict_kernel+1]
        kernels: List[Kernel] = []
        with Timing(f"***** model opts({len(asts):2d}) in  "):
          for ast in asts:
            k = Kernel(ast)
            if BEAM:
              with Context(DEBUG=max(2, DEBUG.value)): k = beam_search(k, bufs_from_lin(k), BEAM.value)
            elif NOOPT: pass
            else: k.hand_coded_optimizations()
            kernels.append(k)

        with Timing("***** model lower in     "): uops = [ast_to_uop(k.get_optimized_ast(), k.opts) for k in kernels]
        with Profiling(PROFILE, fn="/tmp/rewrite.prof"):
          with Timing("***** model rewrite in   "):
            rewritten_uops = []
            for i,(k,u) in enumerate(zip(kernels, uops)):
              with Timing(f"rewrite {i:2d} {k.name}{' '*(50-ansilen(k.name))}", enabled=getenv("VERBOSE", 0)):
                if getenv("VIZ"): _CURRENT_KERNEL.set(k.name)
                rewritten_uops.append(full_graph_rewrite(u, k.opts))
            uops = rewritten_uops
        if getenv("LINEARIZE", 1):
          with Timing("***** model linearize in "): uops = [linearize_uop(u) for u in uops]
          print(sum(len(u) for u in uops))
          if getenv("GRAPHUOPS", 0):
            for u in uops:
              from tinygrad.engine.graph import graph_uops
              graph_uops(u)
          if getenv("SRC", 0):
            renderer = Device[Device.DEFAULT].renderer
            for k,u in zip(kernels, uops): print(renderer.render(k.name, u))
