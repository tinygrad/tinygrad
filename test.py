import time
from extra import dist
from tinygrad.helpers import getenv
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
if __name__ == "__main__":
  dist.preinit()

from extra.dist import collectives, world

@TinyJit
def jitted_allreduce(t:Tensor, cache_id=None) -> Tensor:
  t += 1
  t -= 1
  return collectives.allreduce(t, cache_id=cache_id).realize()

def run():
  rank, world_size = getenv("RANK"), getenv("WORLD_SIZE")

  t = Tensor.zeros(2688, 2688) if rank != 0 else Tensor.ones(2688, 2688)

  times = []
  for _ in range(20):
    st = time.perf_counter()
    a = jitted_allreduce(t, cache_id="test").realize()
    times.append(time.perf_counter() - st)
    assert a.shape == (2688, 2688)
    assert a.sum().numpy().item() == 2688 * 2688
  best = min(times)
  print(f"rank {rank} of {world_size} best {best} avg {sum(times)/len(times)}")

  for N in [1, 2, 4, 8, 16, 32]:
    t = Tensor.zeros(2048, 2048, N)
    next = (rank + 1) % world_size
    prev = (rank - 1) % world_size

    send_times, recv_times = [], []
    for _ in range(20):
      st = time.perf_counter()
      world.send(t, target_rank=next, cache_id=f"testing{N}")
      rt = time.perf_counter()
      world.recv(t, target_rank=prev)
      et = time.perf_counter()
      send_times.append(rt - st)
      recv_times.append(et - rt)
    best_send, best_recv = min(send_times), min(recv_times)
    print(f"{rank} send {best_send:.5f} s {t.nbytes() / best_send / 1e9:.5f} GB/s, recv {best_recv:.5f} s {t.nbytes() / best_recv / 1e9:.5f} GB/s, {2048}x{2048}x{N} {t.nbytes() / 1e3:.5f} KB")

  for i, N in enumerate([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]):
    t = Tensor.zeros(N*world_size, N*world_size)

    times = []
    for _ in range(20):
      st = time.perf_counter()
      collectives.allreduce(t, cache_id=i).realize()
      et = time.perf_counter()
      times.append(et - st)
    best_time = min(times)
    bandwidth = (2 * t.nbytes() * (world_size - 1) / world_size) / best_time / 1e9
    print(f"{1 / best_time:.5f} steps/s, size {N*6}x{N*6}, bandwidth {bandwidth:.5f} GB/s")

if __name__ == "__main__":
  devices = ["gpu:0", "gpu:1", "gpu:2", "gpu:3", "gpu:4", "gpu:5"]
  world_size = len(devices)

  # startup our manager
  dist.init_oob(world_size)

  processes = []
  for rank, device in enumerate(devices):
    processes.append(dist.spawn(rank, device, fn=run, args=()))
  for p in processes: p.join()
