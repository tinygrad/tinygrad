"""
61: op Conv shape [(1, 256, 32, 64), (64, 256, 1, 1), (64,)] opt {'dilations': (1, 1), 'group': 1, 'kernel_shape': (1, 1), 'pads': (0, 0, 0, 0), 'strides': (1, 1)}
62: op Mul shape [(1, 64, 32, 64), (64, 1, 1)] opt {}
63: op Add shape [(1, 64, 32, 64), (1, 64, 32, 64)] opt {}
64: op Conv shape [(1, 64, 32, 64), (64, 1, 3, 3), (64,)] opt {'dilations': (1, 1), 'group': 64, 'kernel_shape': (3, 3), 'pads': (1, 1, 1, 1), 'strides': (1, 1)}
65: op Conv shape [(1, 64, 32, 64), (64, 1, 7, 7), (64,)] opt {'dilations': (1, 1), 'group': 64, 'kernel_shape': (7, 7), 'pads': (3, 3, 3, 3), 'strides': (1, 1)}
66: op Conv shape [(1, 64, 32, 64), (256, 64, 1, 1), (256,)] opt {'dilations': (1, 1), 'group': 1, 'kernel_shape': (1, 1), 'pads': (0, 0, 0, 0), 'strides': (1, 1)}
"""

import pathlib
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
from tinygrad.realize import run_schedule
from tinygrad.helpers import partition, GlobalCounters, Context, getenv, prod, dtypes
from tinygrad.runtime.ops_gpu import CLBuffer, CLProgram
from tinygrad.ops import LoadOps, ReduceOps

def single_kernel():
  # single kernel
  sz1, sz2, sz3 = (32, 1024, 4), (32, 4096, 4), (16, 256, 4)
  out = CLBuffer(prod(sz1), dtypes.imageh(sz1))
  x = CLBuffer(prod(sz2), dtypes.imageh(sz2))
  w = CLBuffer(prod(sz3), dtypes.imageh(sz3))

  old = CLProgram("r_32_16_16_64_4_4_4", open(pathlib.Path(__file__).parent / "conv1_reorder.cl").read())
  old_tms = [old([1,1,32], [16,16,1], out, x, w, wait=True)*1e6 for _ in range(5)]
  print(old_tms, 67.107/min(old_tms)*1e3)
  exit(0)

# CONV=0 PYTHONPATH="." LATEDEBUG=5 OPT=99 IMAGE=2 FLOAT16=1 NOLOCALS=1 python3 extra/fastvits/fastvits_speed.py
if __name__ == "__main__":
  #single_kernel()

  # this is stage 1 in fastvits
  c1 = Conv2d(256, 64, (1,1), bias=False)
  c2 = Conv2d(64, 64, (3,3), groups=64, padding=1, bias=False)
  c3 = Conv2d(64, 64, (7,7), groups=64, padding=3, bias=False)
  c4 = Conv2d(64, 256, (1,1), bias=False)
  c5 = Conv2d(256, 64, (1,1), bias=False)

  # TODO: the elementwise ops shouldn't rerun with normal realize
  x = Tensor.randn(1, 256, 32, 64)
  out = x.sequential([c1,c2,c3,c4,c5])
  schedule = out.lazydata.schedule()

  schedule, schedule_input = partition(schedule, lambda x: x.ast.op not in LoadOps and any(y.op in ReduceOps for y in x.ast.get_lazyops()))
  run_schedule(schedule_input)
  run_schedule(schedule[:getenv("CONV")])
  print("*** init done ***")

  GlobalCounters.reset()
  with Context(DEBUG=getenv("LATEDEBUG", 2), BEAM=getenv("LATEBEAM")):
    run_schedule(schedule[getenv("CONV"):getenv("CONV")+1])
