from models.resnet import ResNet50
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps
from extra.utils import print_tree
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.runtime.ops_metal import renderer, MetalProgram, RawMetalBuffer
from tinygrad.helpers import ansilen

if __name__ == "__main__":
  mdl = ResNet50()
  seen = set()
  mdl(Tensor.empty(64, 3, 224, 224)).lazydata.schedule(seen) # init the weights

  # run again to get only what changes
  x = Tensor.empty(64, 3, 224, 224)
  out = mdl(x)
  sched = out.lazydata.schedule(seen)
  sched = [x for x in sched if x[0].op not in LoadOps]

  # work with the schedule
  total_tm = 0
  for i,(op,out,inp) in enumerate(sched):
    choices = []
    for tc in [0,1]:
      # "linearize" the op into uops
      lin = Linearizer(op, LinearizerOptions(device="METAL"))
      lin.hand_coded_optimizations(use_tensor_cores=tc)
      lin.linearize()

      # render the code and create the program
      code = renderer(lin.function_name, lin.uops)
      prg = MetalProgram(lin.function_name, code)

      # create output/input buffers
      rout = RawMetalBuffer(out.st.size(), out.dtype)
      rin = [RawMetalBuffer(x.st.size(), x.dtype) for x in inp]

      # benchmark it by running 10 times
      tm = min([prg(lin.global_size, lin.local_size, rout, *rin, wait=True) for _ in range(10)])
      choices.append((tm, lin))
    tm, lin = sorted(choices, key=lambda x: x[0])[0]
    print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))}  takes {tm*1000:7.2f} ms, {lin.info.flops*1e-9/tm:6.0f} GFLOPS")
    total_tm += tm
  print(f"******* total {total_tm*1000:.2f} ms")
