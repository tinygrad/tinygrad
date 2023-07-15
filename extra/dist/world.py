from extra import dist
from math import prod
from multiprocessing import shared_memory
from tinygrad.lazy import Device, LazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.tensor import Tensor, Function

# sends a lazybuffer from our rank to the target rank
# transfer method differs depending on if this is a cross or intra node transfer
def send_lb(x:LazyBuffer, target_rank, **kwargs) -> None:
  # assuming intra node transfer so we just use shared memory
  # cache the shared memory so we don't have to create it every time
  cache_id = kwargs.get("cache_id", None)
  if cache_id is not None and cache_id in send_lb.shared_memory_cache:
    shm_name = send_lb.shared_memory_cache[cache_id]
  else:
    shm_name = (s := shared_memory.SharedMemory(create=True, size=prod(x.shape) * x.dtype.itemsize)).name
    s.close()
    if cache_id is not None: send_lb.shared_memory_cache[cache_id] = shm_name
  # we instantly realize here to force the copy into shared memory
  device = f"shm:{shm_name},{cache_id}" if cache_id is not None else f"shm:{shm_name}"
  LazyBuffer.loadop(LoadOps.FROM, x.shape, x.dtype, device, src=x).realize()
  dist.OOB.send((x.shape, x.dtype, (shm_name, cache_id)), target_rank)
setattr(send_lb, "shared_memory_cache", {})

def recv_lb(target_rank) -> LazyBuffer:
  shape, dtype, extra = dist.OOB.recv(target_rank)
  # intra node transfer so we just use shared memory
  device = f"shm:{extra[0]},{extra[1]}" if extra[1] is not None else f"shm:{extra[0]}"
  lb = LazyBuffer.loadop(LoadOps.FROM, shape, dtype, Device.DEFAULT, src=LazyBuffer.loadop(LoadOps.EMPTY, shape, dtype, device)).realize()
  # delete the shared memory if we're not caching it
  if extra[1] is None:
    (s := shared_memory.SharedMemory(name=extra[0])).close()
    s.unlink()
  return lb

# these aren't true lazyop adding functions
class Send(Function):
  def forward(self, x:LazyBuffer, target_rank, **kwargs) -> LazyBuffer:
    self.target_rank, self.kwargs = target_rank, kwargs
    send_lb(x, target_rank, **kwargs)
    return x
  def backward(self, _:LazyBuffer) -> LazyBuffer:
    return recv_lb(self.target_rank)

class Recv(Function):
  def forward(self, _:LazyBuffer, target_rank, **kwargs) -> LazyBuffer:
    self.target_rank, self.kwargs = target_rank, kwargs
    return recv_lb(target_rank)
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    send_lb(grad_output, self.target_rank, **self.kwargs)
    return grad_output

def send(x:Tensor, target_rank, **kwargs) -> Tensor:
  return Send.apply(x, target_rank=target_rank, **kwargs)

def recv(x:Tensor, target_rank, **kwargs) -> Tensor:
  return Recv.apply(x, target_rank=target_rank, **kwargs)
