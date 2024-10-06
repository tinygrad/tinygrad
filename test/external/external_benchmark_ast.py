import time, pickle
import plotly.graph_objects as go
from typing import Dict, List, Tuple
from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, getenv, to_function_name
from tinygrad.engine.schedule import _get_output_groups, _lower_lazybuffer
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.ops import UOp, UOps

if __name__ == "__main__":
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)
  out = mdl(img)
  output_groups, realizes, _ = _get_output_groups(out.lazydata.lbs, set())

  asts: List[UOp] = []
  no_rewrite: List[float] = []
  for k,v in output_groups.items():
    st = time.perf_counter_ns()
    lsi = _lower_lazybuffer(v, realizes)[0]
    et = time.perf_counter_ns() - st
    if lsi.ast.op is UOps.EXT: continue
    no_rewrite.append(et*1e-6)
    asts.append(lsi.ast)

  rewrite: List[float] = []
  bufs: List[List[LazyBuffer]] = []
  with Context(AST_REWRITE=1):
    for k,v in output_groups.items():
      st = time.perf_counter_ns()
      lsi = _lower_lazybuffer(v, realizes)[0]
      bufs.append(v)
      et = time.perf_counter_ns() - st
      if lsi.ast.op is UOps.EXT: continue
      rewrite.append(et*1e-6)

  assert len(rewrite) == len(no_rewrite) == len(asts)

  kernel_tms: Dict[bytes, Tuple[UOp, float, float, List[LazyBuffer]]] = {k.key:(k, no_rewrite[i], rewrite[i], bufs[i]) for i,k in enumerate(asts)}
  pct_change: Dict[bytes, float] = {k:((x-y)/x)*100 for k,(_,x,y,_) in kernel_tms.items()}
  slowest_kernels = list(sorted(pct_change.items(), key=lambda x:x[1]))
  names = {ast.key:Kernel(ast).name for ast,_,_,_ in kernel_tms.values()}
  print("slowest ast rewrites:")
  for k,pct in slowest_kernels[:10]:
    _, no_rw, rw, outs = kernel_tms[k]
    print(f"{names[k]:10s}   {no_rw:4.2f} ms -> {rw:4.2f} ms {pct:4.2f}%")
  with open("/tmp/kernel_tms", "wb") as f: pickle.dump(kernel_tms, f)

  if getenv("GRAPH_TIMING"):
    sample = slowest_kernels[:20]
    x: List[str] = [to_function_name(names[k]) for k,_ in sample]
    y1, y2 = [kernel_tms[k][1] for k,_ in sample], [kernel_tms[k][2] for k,_ in sample]
    fig = go.Figure(data=[go.Bar(name="no graph_rewrite", x=x, y=y1, marker=dict(color="#524eed", line=dict(color='rgba(0,0,0,0)'))),
                          go.Bar(name="graph_rewrite", x=x, y=y2, marker=dict(color="#6fcf97", line=dict(color='rgba(0,0,0,0)')))])
    fig.update_layout(barmode="group", paper_bgcolor="black", plot_bgcolor="black",
                      font={"color":"white"}, yaxis={"gridcolor":"rgba(255, 255, 255, 0.3)"})
    fig.show()
