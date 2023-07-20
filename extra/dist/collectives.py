from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

from extra.dist import world

def allreduce(t:Tensor) -> Tensor:
  RANK, WORLD_SIZE = getenv("RANK"), getenv("WORLD_SIZE")

  chunks = t.chunk(WORLD_SIZE, dim=0)
  reduced = chunks[RANK]

  next_rank = (RANK + 1) % WORLD_SIZE
  prev_rank = ((RANK - 1) + WORLD_SIZE) % WORLD_SIZE

  current_chunk_index = RANK
  for _ in range(WORLD_SIZE - 1):
    world.send(reduced, next_rank)
    current_chunk_index = ((current_chunk_index - 1) + WORLD_SIZE) % WORLD_SIZE
    got = world.recv(chunks[current_chunk_index], prev_rank)
    reduced = got + chunks[current_chunk_index]

  chunks[current_chunk_index] = reduced
  current_chunk_index = (RANK + 1) % WORLD_SIZE
  for _ in range(WORLD_SIZE - 1):
    world.send(reduced, next_rank)
    current_chunk_index = ((current_chunk_index - 1) + WORLD_SIZE) % WORLD_SIZE
    chunks[current_chunk_index] = reduced = world.recv(chunks[current_chunk_index], prev_rank)

  return Tensor.cat(*chunks, dim=0)

