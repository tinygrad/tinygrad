#!/usr/bin/env python3
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.realize import create_schedule
from tinygrad.helpers import to_function_name
from tinygrad.device import CompiledASTRunner, Buffer
from graphlib import TopologicalSorter

if __name__ == "__main__":
  a = Tensor.rand(128, 128).realize()
  b = Tensor.rand(128, 128).realize()
  c = a@b
  si = create_schedule([c.lazydata])[0]
  dev = Device[c.device]
  k = dev.get_linearizer(si.ast)
  k.linearize()

  graph = {u:u.vin for u in k.uops}
  ts = TopologicalSorter(graph)
  new_uops = list(ts.static_order())
  print(len(new_uops))


  src = dev.compiler.render(name:=to_function_name(k.name), new_uops)
  for u in new_uops: print(u)
  print(src)
  c.lazydata.realized = Buffer(si.out.device, si.out.size, si.out.dtype)
  prg = CompiledASTRunner(k.name, src, dev, k.global_size, k.local_size)
  prg([c.lazydata.realized, a.lazydata.realized, b.lazydata.realized], {})
  np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy())


