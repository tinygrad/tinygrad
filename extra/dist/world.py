from typing import Any, Optional, Tuple
from extra import dist
from multiprocessing import shared_memory
from tinygrad.helpers import DEBUG, colored
from tinygrad.lazy import LazyBuffer
from tinygrad.runtime.lib import RawBufferCopyIn, RawBufferCopyInOut
from tinygrad.runtime.ops_shm import RawShmBuffer
from tinygrad.jit import CacheCollector
from tinygrad.tensor import Tensor, Function
import numpy as np

# fake the function signature of ASTRunner so we can put it in the cache
def __send_rb(args:Tuple[RawBufferCopyInOut, RawShmBuffer, int, Any], variables=None, jit=False, force_wait=False):
  args[0]._copyout(np.frombuffer(args[1]._buffer(), dtype=args[0].dtype.np))
  dist.OOB.send(args[3], args[2])
  if DEBUG >= 2: print(f"{colored('****', 'magenta' if jit else None)}   sent {args[0]} to rank {args[2]}")

def __recv_rb(args:Tuple[RawBufferCopyIn, RawShmBuffer, int], variables=None, jit=False, force_wait=False):
  dist.OOB.recv(args[2])
  args[0]._copyin(args[1].toCPU())
  if DEBUG >= 2: print(f"{colored('****', 'magenta' if jit else None)}   recv {args[0]} from rank {args[2]}")

# send a rawbuffer from out rank to the target rank
def _send_rb(x:RawBufferCopyInOut, target_rank:int, cache_id:Optional[str]=None):
  assert isinstance(x, RawBufferCopyInOut), "we only support RawBufferCopyInOut for now"
  # cache the shared memory so we don't have to create it every time
  if cache_id is not None and cache_id in _send_rb.shared_memory_cache:
    shm_name = _send_rb.shared_memory_cache[cache_id]
  else:
    shm_name = (s := shared_memory.SharedMemory(create=True, size=x.size * x.dtype.itemsize)).name
    s.close()
    if cache_id is not None: _send_rb.shared_memory_cache[cache_id] = shm_name
  # copy the buffer into shared memory
  device = f"{shm_name},{cache_id}" if cache_id is not None else shm_name
  rb = RawShmBuffer(x.size, x.dtype, device=device)
  __send_rb((x, rb, target_rank, (shm_name, cache_id)))

  # jit support
  CacheCollector.add(__send_rb, [x, rb, target_rank, None], {})
setattr(_send_rb, "shared_memory_cache", {})

# receive a rawbuffer from the target rank
def _recv_rb(x:RawBufferCopyIn, target_rank:int):
  assert isinstance(x, RawBufferCopyIn), "we only support RawBufferCopyIn for now"
  extra = dist.OOB.recv(target_rank)
  device = f"{extra[0]},{extra[1]}" if extra[1] is not None else f"{extra[0]}"
  rb = RawShmBuffer(x.size, x.dtype, device=device)
  x._copyin(rb.toCPU())
  if DEBUG >= 2: print(f"****   got {x} from rank {target_rank}")

  if extra[1] is None:
    (s := shared_memory.SharedMemory(name=extra[0])).close()
    s.unlink()

  # jit support
  CacheCollector.add(__recv_rb, [x, rb, target_rank], {})

# sends a lazybuffer from our rank to the target rank
def _send_lb(x:LazyBuffer, target_rank:int, cache_id:Optional[str]=None) -> None:
  assert x.st.contiguous and x.realized, "sending buffer must be contiguous and realized"
  _send_rb(x.realized, target_rank, cache_id=cache_id)

# receive a lazybuffer from the target rank
def _recv_lb(x:LazyBuffer, target_rank:int) -> LazyBuffer:
  assert x.st.contiguous and x.realized, "receiving buffer must be contiguous and realized"
  _recv_rb(x.realized, target_rank)
  return x

class Send(Function):
  def forward(self, x:LazyBuffer, target_rank:int, cache_id:Optional[str]=None) -> LazyBuffer:
    self.target_rank, self.shape, self.dtype = target_rank, x.shape, x.dtype
    _send_lb(x, target_rank, cache_id=cache_id)
    return x

class Recv(Function):
  def forward(self, x:LazyBuffer, target_rank:int, cache_id:Optional[str]=None) -> LazyBuffer:
    self.target_rank, self.cache_id = target_rank, cache_id
    return _recv_lb(x, target_rank)

def send(x:Tensor, target_rank:int, cache_id:Optional[str]=None) -> Tensor: return Send.apply(x.contiguous().realize(), target_rank=target_rank, cache_id=cache_id)
def recv(x:Tensor, target_rank:int, cache_id:Optional[str]=None) -> Tensor: return Recv.apply(x.contiguous().realize(), target_rank=target_rank, cache_id=cache_id)
