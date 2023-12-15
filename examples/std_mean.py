from tinygrad.tensor import Tensor
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.ops import LoadOps, ReduceOps
from tinygrad.runtime.ops_cuda import CUDAProgram, CUDADevice, CUDARenderer

if __name__ == '__main__':
  def merge(reduceops):
    k = ""
    uops = []
    for i,x in enumerate(reduceops):
      print(x[2].arg)
      lin = Linearizer(x[2], LinearizerOptions(device="CUDA")) 
      lin.hand_coded_optimizations()
      lin.linearize()
      #for _ in lin.uops:
      #  print(_)
      renderer = CUDARenderer(lin.name, lin.uops)
      #print(renderer)
    #print(k)
  a = Tensor.rand([5, 5])
  b = Tensor.rand([5, 5])
  c = (a.std() + b.mean())
  sche = c.lazydata.schedule()
  ops = [x.ast for x in sche if x.ast.op not in LoadOps]
  reduceops = []    


  print(c.numpy())
  for o in ops:
    for i, x in enumerate(o.get_lazyops()):
      if len(x.src) >= 1 and x.src[0].op is ReduceOps.SUM: 
        reduceops.append((i, x, o))
      #if i == 2: print(o.get_lazyops()[1])
    lin = Linearizer(o, LinearizerOptions(device="CUDA"))
    lin.required_optimizations()
    lin.hand_coded_optimizations()
    lin.linearize()

  group_reduces = [[x] for x in reduceops]
  for i,a in enumerate(group_reduces): 
    for b in reduceops: 
        if a[-1][0] + 1 == b[0]: group_reduces[i].append(b)
  for x in group_reduces: 
      #print(len(x))
    if len(x) >= 2: 
      merge(x)
