from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

from extra.dist import world

def allreduce(t:Tensor, cache_id=None) -> Tensor:
  RANK, WORLD_SIZE = getenv("RANK"), getenv("WORLD_SIZE")
  cache_id = f"{RANK}-{cache_id}" if cache_id is not None else None

  chunks = t.chunk(WORLD_SIZE, dim=0)
  reduced = chunks[RANK]

  next_rank = (RANK + 1) % WORLD_SIZE
  prev_rank = ((RANK - 1) + WORLD_SIZE) % WORLD_SIZE

  current_chunk_index = RANK
  for i in range(WORLD_SIZE - 1):
    world.send(reduced, next_rank, cache_id=f"{cache_id}-{i}" if cache_id is not None else None)
    current_chunk_index = ((current_chunk_index - 1) + WORLD_SIZE) % WORLD_SIZE
    reduced = world.recv(chunks[current_chunk_index], prev_rank) + chunks[current_chunk_index]

  chunks[current_chunk_index] = reduced
  current_chunk_index = (RANK + 1) % WORLD_SIZE
  for i in range(WORLD_SIZE - 1):
    world.send(reduced, next_rank, cache_id=f"{cache_id}-{i}" if cache_id is not None else None)
    current_chunk_index = ((current_chunk_index - 1) + WORLD_SIZE) % WORLD_SIZE
    chunks[current_chunk_index] = reduced = world.recv(chunks[current_chunk_index], prev_rank)

  return Tensor.cat(*chunks, dim=0)
