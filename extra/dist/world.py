from extra import dist
from tinygrad.tensor import Tensor
from tinygrad.lazy import Device
from multiprocessing import shared_memory

# sends a tensor from our rank to the target rank
# transfer method differs depending on if this is a cross or intra node transfer
def send(tensor:Tensor, target_rank, **kwargs):
  # assuming intra node transfer so we just use shared memory
  # cache the shared memory so we don't have to create it every time
  cache_id = kwargs.get("cache_id", None)
  if cache_id in send.shared_memory_cache:
    shm_name = send.shared_memory_cache[cache_id]
  else:
    shm_name = (s := shared_memory.SharedMemory(create=True, size=tensor.nbytes())).name
    s.close()
    if cache_id is not None: send.shared_memory_cache[cache_id] = shm_name
  tensor.to(f"shm:{shm_name}").realize()
  dist.OOB.send((tensor.shape, tensor.dtype, (shm_name, cache_id is not None)), target_rank)
setattr(send, "shared_memory_cache", {})

def recv(into=None):
  shape, dtype, extra = dist.OOB.recv()
  # intra node transfer so we just use shared memory
  tensor = Tensor.empty(*shape, dtype=dtype, device=f"shm:{extra[0]}").to(Device.DEFAULT).realize()
  if into is not None: into.assign(tensor)
  # delete the shared memory if we're not caching it
  if not extra[1]:
    (s := shared_memory.SharedMemory(name=extra[0])).close()
    s.unlink()
  return tensor
