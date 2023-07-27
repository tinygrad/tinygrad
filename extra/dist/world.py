from typing import Any, Optional, Tuple
from extra import dist
from multiprocessing import shared_memory
from tinygrad.helpers import GlobalCounters
from tinygrad.lazy import Device, LazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.runtime.lib import RawBuffer
from tinygrad.runtime.ops_shm import RawShmBuffer
from tinygrad.tensor import Tensor, Function
import numpy as np
import time

# fake the function signature of ASTRunner so we can put it in the cache
def _send_rb(args:Tuple[RawBuffer, RawShmBuffer, int, Any], jit=False, force_wait=False):
  # we only support copyout buffers right now
  st = time.perf_counter()
  args[0]._copyout(np.frombuffer(args[1]._buffer(), dtype=args[0].dtype.np))
  dist.OOB.send(st, args[2])

def _recv_rb(args:Tuple[RawBuffer, RawShmBuffer, int], jit=False, force_wait=False):
  st = dist.OOB.recv(args[2])
  args[0]._copyin(args[1].toCPU())
  print(f"recv {time.perf_counter() - st}, bandwidth {args[0].size * args[0].dtype.itemsize / (time.perf_counter() - st) / 1e9} GB/s")

# send a rawbuffer from out rank to the target rank
def send_rb(x:RawBuffer, target_rank:int, cache_id:Optional[str]=None):
  # cache the shared memory so we don't have to create it every time
  if cache_id is not None and cache_id in send_rb.shared_memory_cache:
    shm_name = send_rb.shared_memory_cache[cache_id]
  else:
    shm_name = (s := shared_memory.SharedMemory(create=True, size=x.size * x.dtype.itemsize)).name
    s.close()
    if cache_id is not None: send_rb.shared_memory_cache[cache_id] = shm_name
  # copy the buffer into shared memory
  device = f"{shm_name},{cache_id}" if cache_id is not None else shm_name
  rb = RawShmBuffer(x.size, x.dtype, device=device)
  x._copyout(np.frombuffer(rb._buffer(), dtype=x.dtype.np))
  dist.OOB.send((shm_name, cache_id), target_rank)
  # _send_rb((x, rb, target_rank, (shm_name, cache_id)))

  # jit support
  if GlobalCounters.cache is not None: GlobalCounters.cache.append((_send_rb, [x, rb, target_rank, None]))
setattr(send_rb, "shared_memory_cache", {})

# receive a rawbuffer from the target rank
def recv_rb(x:RawBuffer, target_rank:int):
  extra = dist.OOB.recv(target_rank)
  device = f"{extra[0]},{extra[1]}" if extra[1] is not None else f"{extra[0]}"
  rb = RawShmBuffer(x.size, x.dtype, device=device)
  x._copyin(rb.toCPU())

  if extra[1] is None:
    (s := shared_memory.SharedMemory(name=extra[0])).close()
    s.unlink()

  # jit support
  if GlobalCounters.cache is not None: GlobalCounters.cache.append((_recv_rb, [x, rb, target_rank]))

# sends a lazybuffer from our rank to the target rank
def send_lb(x:LazyBuffer, target_rank:int, cache_id:Optional[str]=None) -> None: send_rb(x.contiguous().realize().realized, target_rank, cache_id=cache_id)

# receive a lazybuffer from the target rank
def recv_lb(shape, dtype, target_rank:int) -> LazyBuffer:
  lb = LazyBuffer.loadop(LoadOps.EMPTY, shape, dtype, Device.DEFAULT).realize()
  recv_rb(lb.realized, target_rank)
  return lb

class Send(Function):
  def forward(self, x:LazyBuffer, target_rank:int, cache_id:Optional[str]=None) -> LazyBuffer:
    self.target_rank, self.shape, self.dtype = target_rank, x.shape, x.dtype
    send_lb(x, target_rank, cache_id=cache_id)
    return x
  def backward(self, _:LazyBuffer) -> LazyBuffer:
    return recv_lb(self.shape, self.dtype, self.target_rank)

class Recv(Function):
  def forward(self, x:LazyBuffer, target_rank:int, cache_id:Optional[str]=None) -> LazyBuffer:
    self.target_rank, self.cache_id = target_rank, cache_id
    return recv_lb(x.shape, x.dtype, target_rank)
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    send_lb(grad_output, self.target_rank, cache_id=self.cache_id)
    return grad_output

def send(x:Tensor, target_rank:int, cache_id:Optional[str]=None) -> Tensor: return Send.apply(x, target_rank=target_rank, cache_id=cache_id)
def recv(x:Tensor, target_rank:int, cache_id:Optional[str]=None) -> Tensor: return Recv.apply(x, target_rank=target_rank, cache_id=cache_id)
