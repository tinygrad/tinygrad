import ctypes
from extra import dist
from multiprocessing import shared_memory
from tinygrad.helpers import DEBUG, colored, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.runtime.lib import RawBuffer, RawBufferCopyInOut
try:
  import tinygrad.runtime.autogen.hip as hip
  from tinygrad.runtime.ops_hip import RawHIPBuffer, check
except: RawHIPBuffer = None
from tinygrad.runtime.ops_disk import RawDiskBuffer
from tinygrad.features.jit import CacheCollector
from tinygrad.tensor import Tensor, Function
import numpy as np

# match the function signature of JITRunner so we can put it in the cache
def __send_rb(args, variables=None, wait=False, jit=False):
  x, target_rank, y = args[:3]
  if RawHIPBuffer and x.__class__ is RawHIPBuffer:
    check(hip.hipSetDevice(x._device))
    check(hip.hipDeviceSynchronize())
  else:
    if isinstance(x, RawBufferCopyInOut): x._copyout(np.frombuffer(y._buffer(), dtype=x.dtype.np))
    else: y.fromCPU(x.toCPU())
  dist.OOB.send(None, target_rank)
  if DEBUG >= 2: print(f"{colored('****', 'magenta' if jit else None)}  rank {getenv('RANK')} sent {x} to rank {target_rank}")

def __recv_rb(args, variables=None, wait=False, jit=False):
  x, target_rank, y = args[:3]
  dist.OOB.recv(target_rank)
  if RawHIPBuffer and x.__class__ is RawHIPBuffer:
    x._transfer(y)
  elif isinstance(x, RawBuffer): x._copyin(y.toCPU())
  else: x.fromCPU(y.toCPU())
  if DEBUG >= 2: print(f"{colored('****', 'magenta' if jit else None)}  rank {getenv('RANK')} recv {x} from rank {target_rank}")

# send a rawbuffer from out rank to the target rank
def _send_rb(x:RawBuffer, target_rank:int):
  if RawHIPBuffer and x.__class__ is RawHIPBuffer:
    # send ipc handle
    check(hip.hipSetDevice(x._device))
    check(hip.hipDeviceSynchronize())
    check(hip.hipIpcGetMemHandle(ctypes.byval(handle := hip.hipIpcMemHandle_t()), x._buf))
    dist.OOB.send((handle, x._device), target_rank)

    # jit support
    x._allocator = None # need to disconnect allocator for sent buffers
    CacheCollector.add(__send_rb, [x, target_rank, None], {})
  else:
    # create shared memory
    shm_name = (s := shared_memory.SharedMemory(create=True, size=x.size * x.dtype.itemsize)).name
    s.close()

    # copy the buffer into shared memory
    y = RawDiskBuffer(x.size, x.dtype, device="disk:shm:"+shm_name)
    # fast path when we can directly copyout
    if isinstance(x, RawBufferCopyInOut): x._copyout(np.frombuffer(y._buffer(), dtype=x.dtype.np))
    else: y.fromCPU(x.toCPU())

    dist.OOB.send(shm_name, target_rank)

    # jit support
    CacheCollector.add(__send_rb, [x, target_rank, y], {})
  if DEBUG >= 2: print(f"****  rank {getenv('RANK')} sent {x} to rank {target_rank}")

# receive a rawbuffer from the target rank
def _recv_rb(x:RawBuffer, target_rank:int):
  if RawHIPBuffer and isinstance(x, RawHIPBuffer):
    # open ipc handle
    handle, y_device = dist.OOB.recv(target_rank)
    check(hip.hipSetDevice(y_device))
    check(hip.hipIpcOpenMemHandle(ctypes.byval(ptr := ctypes.c_void_p()), handle, 0))

    # build a new buffer
    y = RawHIPBuffer(x.size, x.dtype, device=str(y_device), buf=ptr, allocator=None)
    x._transfer(y)

    CacheCollector.add(__recv_rb, [x, target_rank, y], {})
  else:
    shm_name = dist.OOB.recv(target_rank)
    y = RawDiskBuffer(x.size, x.dtype, device="disk:shm:"+shm_name)

    # fast path when we can directly copyin
    if isinstance(x, RawBuffer): x._copyin(y.toCPU())
    else: x.fromCPU(y.toCPU())

    # jit support
    CacheCollector.add(__recv_rb, [x, target_rank, y], {})
  if DEBUG >= 2: print(f"****  rank {getenv('RANK')} got {x} from rank {target_rank}")

# sends a lazybuffer from our rank to the target rank
def _send_lb(x:LazyBuffer, target_rank:int) -> None:
  assert x.st.contiguous and x.realized, "sending buffer must be contiguous and realized"
  _send_rb(x.realized, target_rank)

# receive a lazybuffer from the target rank
def _recv_lb(x:LazyBuffer, target_rank:int) -> LazyBuffer:
  assert x.st.contiguous and x.realized, "receiving buffer must be contiguous and realized"
  _recv_rb(x.realized, target_rank)
  return x

class Send(Function):
  def forward(self, x:LazyBuffer, target_rank:int) -> LazyBuffer:
    self.target_rank, self.shape, self.dtype = target_rank, x.shape, x.dtype
    _send_lb(x, target_rank)
    return x

class Recv(Function):
  def forward(self, x:LazyBuffer, target_rank:int) -> LazyBuffer:
    self.target_rank = target_rank
    return _recv_lb(x, target_rank)

def send(x:Tensor, target_rank:int) -> Tensor: return Send.apply(x.contiguous().realize(), target_rank=target_rank)
def recv(x:Tensor, target_rank:int) -> Tensor: return Recv.apply(x.contiguous().realize(), target_rank=target_rank)
