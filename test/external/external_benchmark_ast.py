import time
import plotly.graph_objects as go
from typing import Dict, List, Tuple
from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, getenv, to_function_name
from tinygrad.engine.schedule import _get_output_groups, _lower_lazybuffer
from tinygrad.ops import UOps

if __name__ == "__main__":
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)
  out = mdl(img)
  output_groups, realizes, _ = _get_output_groups(out.lazydata.lbs, set())

  kernels: List[str] = []
  no_rewrite: List[float] = []
  for k,v in output_groups.items():
    st = time.perf_counter_ns()
    lsi = _lower_lazybuffer(v, realizes)
    et = time.perf_counter_ns() - st
    if lsi.ast.op is UOps.EXT: continue
    no_rewrite.append(et*1e-6)
    kernels.append(Kernel(lsi.ast).name)

  rewrite: List[float] = []
  with Context(AST_REWRITE=1):
    for k,v in output_groups.items():
      st = time.perf_counter_ns()
      lsi = _lower_lazybuffer(v, realizes)
      et = time.perf_counter_ns() - st
      if lsi.ast.op is UOps.EXT: continue
      rewrite.append(et*1e-6)

  assert len(rewrite) == len(no_rewrite) == len(kernels)

  data: Dict[str, Tuple[float, float]] = {k:(rewrite[i], no_rewrite[i]) for i,k in enumerate(kernels)}
  slowest_kernels = list(sorted(data.items(), reverse=False, key=lambda x:((x[1][1]-x[1][0])/x[1][1])*100))
  print("slowest ast rewrites:")
  for k,tms in slowest_kernels[:10]:
    print(f"{k:10s}   {tms[1]:4.2f} ms -> {tms[0]:4.2f} ms")

  if getenv("GRAPH_TIMING"):
    sample = slowest_kernels[:20]
    x = [to_function_name(x) for x,_ in sample]
    y1, y2 = [x[1][0] for x in sample], [x[1][1] for x in sample]
    fig = go.Figure(data=[go.Bar(name="no graph_rewrite", x=x, y=y2, marker=dict(color="#524eed", line=dict(color='rgba(0,0,0,0)'))),
                          go.Bar(name="graph_rewrite", x=x, y=y1, marker=dict(color="#6fcf97", line=dict(color='rgba(0,0,0,0)')))])
    fig.update_layout(barmode="group", paper_bgcolor="black", plot_bgcolor="black",
                      font={"color":"white"}, yaxis={"gridcolor":"rgba(255, 255, 255, 0.3)"})
    fig.show()
