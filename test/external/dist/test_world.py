from extra import dist
from tinygrad.jit import TinyJit
if __name__ == "__main__":
  dist.preinit()

from extra.dist import world
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
import numpy as np

@TinyJit
def send_jit(t, target_rank, cache_id=None) -> Tensor:
  return world.send(t, target_rank, cache_id=cache_id).realize()

@TinyJit
def recv_jit(t, target_rank, cache_id=None) -> Tensor:
  (t + 1).realize() # this does nothing but make sure that jit doesn't error
  return world.recv(t, target_rank, cache_id=cache_id).realize()

def run():
  # set a deterministic seed so that both ranks generate the same random tensor
  Tensor.manual_seed(42)

  rank = getenv("RANK")

  # loop 3 times to make sure it works with the jit
  for _ in range(3):
    # create a tensor to send
    t = Tensor.randn(2048, 2048)

    # send to rank 1
    if rank == 0:
      send_jit(t, 1, cache_id="test")
    elif rank == 1:
      t2 = recv_jit(t, 0, cache_id="test")

    # recv from rank 1
    if rank == 0:
      t2 = recv_jit(t, 1, cache_id="test2")
    elif rank == 1:
      send_jit(t2, 0, cache_id="test2")

    # check that the received tensor is the same as the sent tensor
    if rank == 0:
      assert np.allclose(t.numpy(), t2.numpy())

  print(f"rank {rank} passed")

if __name__ == "__main__":
  devices = ["gpu:0", "gpu:1"]
  world_size = len(devices)

  dist.init_oob(world_size)

  processes = []
  for rank, device in enumerate(devices):
    processes.append(dist.spawn(rank, device, fn=run, args=()))
  for p in processes: p.join()
