from extra import dist
from tinygrad.jit import TinyJit
if __name__ == "__main__":
  dist.preinit()

from extra.dist import collectives
from tinygrad.helpers import CI, getenv
from tinygrad.tensor import Tensor
import numpy as np

@TinyJit
def allreduce_jit(t:Tensor, cache_id=None) -> Tensor:
  return collectives.allreduce(t, cache_id=cache_id).realize()

SIZE = 2048 if not CI else 2
SIZE_2 = 255 if not CI else 3

def run():
  # set a deterministic seed so that both ranks generate the same random tensor
  Tensor.manual_seed(42)

  rank = getenv("RANK")

  # loop 3 times to make sure it works with the jit
  for _ in range(3):
    # create a tensor to send
    t = Tensor.zeros(SIZE, SIZE) if rank != 0 else Tensor.ones(SIZE, SIZE)
    t2 = allreduce_jit(t.contiguous().realize(), cache_id="test")
    assert np.allclose(np.ones((SIZE, SIZE)), t2.numpy()), f"{t2.numpy()} wasn't ones"

  # reset jit
  allreduce_jit.cnt = 0
  allreduce_jit.input_replace = {}

  # test uneven chunk sizes
  for _ in range(3):
    # create a tensor to send
    t = Tensor.ones(SIZE_2, SIZE_2, SIZE_2) if rank == 0 else Tensor.zeros(SIZE_2, SIZE_2, SIZE_2)
    t2 = allreduce_jit(t.contiguous().realize(), cache_id="test2")
    assert np.allclose(np.ones((SIZE_2, SIZE_2, SIZE_2)), t2.numpy()), f"{t2.numpy()} wasn't ones"

  print(f"rank {rank} passed")

if __name__ == "__main__":
  if getenv("HIP"):
    from tinygrad.runtime.ops_hip import HIP
    devices = [f"hip:{i}" for i in range(HIP.device_count)]
  else:
    from tinygrad.runtime.ops_gpu import CL
    devices = [f"gpu:{i}" for i in range(len(CL.devices))] if not CI else ["gpu:0", "gpu:0"]
  world_size = len(devices)

  dist.init_oob(world_size)

  processes = []
  for rank, device in enumerate(devices):
    processes.append(dist.spawn(rank, device, fn=run, args=()))
  for p in processes: p.join()

  # exit with error code if any of the processes failed
  for p in processes:
    if p.exitcode != 0: exit(p.exitcode)
