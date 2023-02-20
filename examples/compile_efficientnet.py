from models.efficientnet import EfficientNet
from tinygrad.tensor import Tensor

if __name__ == "__main__":
  model = EfficientNet(0)
  model.load_from_pretrained()

  from extra.jit import TinyJit
  @TinyJit
  def run(x): return model.forward(x).realize()

  # twice to run the JIT
  run(Tensor.randn(1,3,224,224))
  run(Tensor.randn(1,3,224,224))

  bufs = {}
  bufnum = 0
  helpers = []
  statements = []
  for fxn,args in run.jit_cache:
    helpers.append(fxn.clprg.prg)
    cargs = []
    for i,arg in enumerate(args):
      if i in fxn.bufs_to_delete: continue
      key = id(arg.cl)
      if key not in bufs:
        bufs[key] = (f"buf_{bufnum}", len(arg.cl))
        bufnum += 1
      cargs.append(bufs[key][0])
    statements.append(f"{fxn.clprg.name}({', '.join(cargs)});")
  cprog = ["#include <math.h>","#define max(x,y) fmax(x,y)"] + helpers + [f"float {x[0]}[{x[1]}];" for x in bufs.values()] + ["int main() {"] + statements + ["}"]
  print('\n'.join(cprog))
