import time
from typing import Dict, List, Tuple
from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context
from tinygrad.engine.schedule import _get_output_groups, _lower_lazybuffer, reduceop_fusor
from tinygrad.ops import graph_rewrite
from tinygrad.ops import UOp, UOps

if __name__ == "__main__":
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)
  out = mdl(img)
  output_groups, realizes, _ = _get_output_groups(out.lazydata.lbs, set())

  raw_sinks: List[UOp] = []
  for k,v in output_groups.items():
    with Context(AST_REWRITE=1, DO_AST_REWRITE=0):
      lsi = _lower_lazybuffer(v, realizes)
      if lsi.ast.op is UOps.EXT: continue
      raw_sinks.append(lsi.ast)

  rewrite_tms: Dict[bytes, Tuple[UOp, float]] = {}
  for sink in raw_sinks:
    st = time.perf_counter_ns()
    sink = graph_rewrite(sink, reduceop_fusor)
    et = time.perf_counter_ns()-st
    rewrite_tms[sink.key] = (sink, (et*1e-6))
  rewrite_tms = dict(sorted(rewrite_tms.items(), reverse=True, key=lambda x:x[1][1]))

  for sink,tm in list(rewrite_tms.values())[:5]:
    p = Kernel(sink).to_program()
    print(f"{p.name} {tm:4.2f} ms")
    print(p.src)
