import time, os, logging
from typing import Dict, List, Tuple
from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import DEBUG, Context
from tinygrad.engine.schedule import _get_output_groups, _lower_lazybuffer, reduceop_fusor
from tinygrad.ops import graph_rewrite
from tinygrad.ops import UOp, UOps
from test.external.process_replay.utils import print_diff
logging.basicConfig(level=logging.INFO, format='%(message)s')
os.environ["UPAT_LOC"] = "schedule"

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

  rewrite_tms: Dict[bytes, Tuple[UOp, UOp, int, float]] = {}
  for i,rsink in enumerate(raw_sinks):
    if (num:=os.getenv("NUM")) is not None and int(num) != i: continue
    st = time.perf_counter_ns()
    sink = graph_rewrite(rsink, reduceop_fusor)
    et = time.perf_counter_ns()-st
    rewrite_tms[sink.key] = (sink, rsink, i, (et*1e-6))
  rewrite_tms = dict(sorted(rewrite_tms.items(), reverse=True, key=lambda x:x[1][3]))

  for sink,rsink,i,tm in list(rewrite_tms.values())[:10]:
    with Context(DEBUG=0): p = Kernel(sink).to_program()
    print(f"{i} {p.name} {tm:4.2f} ms")
    if DEBUG >= 3: print_diff(rsink, sink)
    if DEBUG >= 4: print(p.src)
