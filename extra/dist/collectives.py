from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

from extra.dist import world

def allreduce(t:Tensor) -> Tensor:
  RANK, WORLD_SIZE = getenv("RANK"), getenv("WORLD_SIZE")

  # flatten
  flattened = t.flatten()

  # pad to evenly divide
  if flattened.shape[0] % WORLD_SIZE != 0:
    flattened = Tensor.cat(flattened, Tensor.empty(WORLD_SIZE - (flattened.shape[0] % WORLD_SIZE)))

  # chunk
  chunks = flattened.chunk(WORLD_SIZE, dim=0)

  next_rank = (RANK + 1) % WORLD_SIZE
  prev_rank = ((RANK - 1) + WORLD_SIZE) % WORLD_SIZE

  # scatter reduce
  current_chunk_index = RANK
  for _ in range(WORLD_SIZE - 1):
    world.send(chunks[current_chunk_index], next_rank)
    current_chunk_index = ((current_chunk_index - 1) + WORLD_SIZE) % WORLD_SIZE
    recv_buf = Tensor.empty(*chunks[current_chunk_index].shape)
    world.recv(recv_buf, prev_rank)
    chunks[current_chunk_index] += recv_buf

  # gather
  current_chunk_index = (RANK + 1) % WORLD_SIZE
  for _ in range(WORLD_SIZE - 1):
    world.send(chunks[current_chunk_index], next_rank)
    current_chunk_index = ((current_chunk_index - 1) + WORLD_SIZE) % WORLD_SIZE
    recv_buf = Tensor.empty(*chunks[current_chunk_index].shape)
    world.recv(recv_buf, prev_rank)
    chunks[current_chunk_index].assign(recv_buf)

  return Tensor.cat(*chunks, dim=0).shrink(((0, t.numel()),)).reshape(*t.shape)
