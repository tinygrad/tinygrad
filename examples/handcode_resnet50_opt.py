from typing import List
from models.resnet import ResNet50
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import ansilen, DEBUG
from tinygrad.graph import print_tree

if __name__ == "__main__":
  mdl = ResNet50()
  seen = set()

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  print(f"optimizing for {Device.DEFAULT}")

  # first model run to init the weights, they are saved in seen
  mdl(Tensor.empty(64, 3, 224, 224)).lazydata.schedule(seen)

  # run model again to get only what changes, these are the kernels of the model
  x = Tensor.empty(64, 3, 224, 224)
  out = mdl(x)
  sched = out.lazydata.schedule(seen)
  sched = [x for x in sched if x.ast.op not in LoadOps]

  # work with the schedule
  total_tm = 0
  for i,si in enumerate(sched):
    if DEBUG >= 2: print_tree(si.ast)

    # enable only one kernel to focus on it
    #if i != 1: continue

    # "linearize" the op into uops in different ways
    lins:List[Linearizer] = []

    if Device.DEFAULT == "METAL" and i == 1:
      # through careful work, we discovered 1,8,0
      for big_chomp in [1,2]: #[1,2,4,8,16]:
        for lil_chomp in [2,4,7,8,14]:
          for upcasted in [0,1,2]:
            lin = Linearizer(si.ast, device.linearizer_opts)
            lin.reshape_and_permute(lambda x: (4096//big_chomp,big_chomp,56//lil_chomp,lil_chomp,56//lil_chomp,lil_chomp)+x[-2:], [0,2,4,1,3,5,6,7])
            lin.upcasted += upcasted
            lin.local_dims += 3
            lins.append(lin)
    else:
      # try with and without tensor cores
      for tc in [0,1]:
        lin = Linearizer(si.ast, device.linearizer_opts)
        lin.hand_coded_optimizations(use_tensor_cores=tc)
        lins.append(lin)

    # create output/input buffers
    rawbufs = [device.buffer(si.out.st.size(), si.out.dtype)] + [device.buffer(x.st.size(), x.dtype) for x in si.inputs]

    # benchmark the programs
    choices = []
    for lin in lins:
      prg = device.to_program(lin)

      # benchmark it by running 10 times
      try:
        tm = min([prg(rawbufs, force_wait=True) for _ in range(10)])
        choices.append((tm, lin))
      except AssertionError:
        tm = float('inf')

      # print all kernels
      if DEBUG >= 1: print(f"                 kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {lin.info.flops*1e-9/tm:6.0f} GFLOPS")
    tm, lin = sorted(choices, key=lambda x: x[0])[0]
    print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {lin.info.flops*1e-9/tm:6.0f} GFLOPS")
    total_tm += tm
  print(f"******* total {total_tm*1000:.2f} ms")
