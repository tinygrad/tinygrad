from typing import List
from extra.models.resnet import ResNet50
from tinygrad import Tensor, nn
from tinygrad.ops import LoadOps, get_lazyop_info
from tinygrad.device import Device, Compiled
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.engine.search import time_linearizer, beam_search, bufs_from_lin
from tinygrad.helpers import ansilen, DEBUG, getenv
from tinygrad.shape.symbolic import sym_infer
from tinygrad.dtype import dtypes
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.graph import print_tree

if __name__ == "__main__":
  if getenv("HALF"):
    dtypes.default_float = dtypes.half

  mdl = ResNet50()
  seen = set()

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  if getenv("BACKWARD"):
    Tensor.training = True
    optim = (nn.optim.LARS if getenv("LARS") else nn.optim.SGD)(nn.state.get_parameters(mdl))
  print(f"optimizing for {Device.DEFAULT}")

  # run model twice to get only what changes, these are the kernels of the model
  for i in range(2):
    out = mdl(Tensor.empty(64, 3, 224, 224))
    targets = [out.lazydata]
    if getenv("BACKWARD"):
      optim.zero_grad()
      out.sparse_categorical_crossentropy(Tensor.empty(64, dtype=dtypes.int)).backward()
      targets += [x.lazydata for x in optim.schedule_step()]
    sched = create_schedule(targets, seen)
    print(f"schedule length {len(sched)}")
  sched = [x for x in sched if x.ast[0].op not in LoadOps]

  # focus on one kernel
  if getenv("KERNEL", -1) >= 0: sched = sched[getenv("KERNEL", -1):getenv("KERNEL", -1)+1]

  # work with the schedule
  total_tm = 0
  running_gflops = 0
  for i,si in enumerate(sched):
    ops = sum(get_lazyop_info(ast).flops for ast in si.ast)

    if DEBUG >= 2:
      for ast in si.ast: print_tree(ast)

    rawbufs = bufs_from_lin(Linearizer(*si.ast))

    # "linearize" the op into uops in different ways
    lins:List[Linearizer] = []

    # always try hand coded opt
    lin = Linearizer(*si.ast, renderer=device.renderer)
    lin.hand_coded_optimizations()
    lins.append(lin)

    # maybe try tensor cores
    lin = Linearizer(*si.ast, renderer=device.renderer)
    if lin.apply_tensor_cores():
      lins.append(lin)

    # try a beam search
    if beam:=getenv("BEAM"):
      lin = Linearizer(*si.ast, renderer=device.renderer)
      lin = beam_search(lin, rawbufs, beam, bool(getenv("BEAM_ESTIMATE", 1)))
      lins.append(lin)

    # benchmark the programs
    choices = []
    for lin in lins:
      tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10)
      gflops = sym_infer(ops, {k:k.min for k in lin.ast[0].vars()})*1e-9/tm
      choices.append((tm, gflops, lin.linearize()))

      # print all kernels
      if DEBUG >= 1: print(f"                 kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
    tm, gflops, lin = sorted(choices, key=lambda x: x[0])[0]
    total_tm += tm
    running_gflops += gflops * tm
    print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
  print(f"******* total {total_tm*1000:.2f} ms, {running_gflops/total_tm:6.0f} GFLOPS")
