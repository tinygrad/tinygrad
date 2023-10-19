#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.codegen.optimizer import Opt, OptOps
from tinygrad.renderer.opencl import OpenCLRenderer

def graph_uops(uops):
  import os
  import networkx as nx
  colors = {UOps.ALU: "#ffffc0", UOps.LOAD: "#ffc0c0", UOps.STORE: "#c0ffc0", UOps.SPECIAL: "#c0c0ff", UOps.CONST: "#e0e0e0",
            UOps.DEFINE_GLOBAL: "#ffe0b0", UOps.DEFINE_LOCAL: "#ffe0d0", UOps.DEFINE_ACC: "#f0ffe0",
            UOps.LOOP: "#c8a0e0", UOps.PHI: "#e0ffc0"}
  G = nx.DiGraph()
  for u in uops:
    G.add_node(u.num, label=f"{str(u.uop)[5:]}{(' '+str(u.arg)) if u.arg is not None else ''}\n{str(u.dtype)}", style="filled", fillcolor=colors.get(u.uop, "#ffffff"))
    for v in u.vin: G.add_edge(v.num, u.num)
  GRAPHPATH = "/tmp/uops"
  nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.dot')
  os.system(f'dot -Grankdir=LR -Tsvg {GRAPHPATH}.dot -o {GRAPHPATH}.svg')

class TestUopsFlopCounter(unittest.TestCase):
  def test_flops_matmul(self):
    N = 1024
    a = Tensor.rand(N,N)
    b = Tensor.rand(N,N)
    si = (a@b).lazydata.schedule()[-1]
    lin = Linearizer(si.ast)
    #lin.apply_opt(Opt(OptOps.UPCAST, 0, 2))
    #lin.apply_opt(Opt(OptOps.UPCAST, 1, 2))
    #lin.apply_opt(Opt(OptOps.UNROLL, 0, 2))
    lin.hand_coded_optimizations()
    print(lin.colored_shape())
    uops = lin.linearize().uops
    graph_uops(uops)
    print(OpenCLRenderer("matmul", uops)[0])

  def test_flops_reduce(self):
    a = Tensor.rand(1024*1024)
    si = a.sum().lazydata.schedule()[-1]
    lin = Linearizer(si.ast)
    lin.hand_coded_optimizations()
    uops = lin.linearize().uops
    graph_uops(uops)
    #print(OpenCLRenderer("reduce", uops)[0])

if __name__ == '__main__':
  unittest.main()
