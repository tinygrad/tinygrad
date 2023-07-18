from tinygrad.tensor import Tensor

from extra.dist import WORLD_SIZE, RANK
from extra.dist import world

def allreduce(t:Tensor) -> Tensor:
  chunks = t.chunk(WORLD_SIZE, dim=0)
  bucket = chunks[RANK]

  next_rank = (RANK + 1) % WORLD_SIZE
  prev_rank = (RANK - 1) % WORLD_SIZE

  world.send(bucket, next_rank)

  for i in range(WORLD_SIZE - 1):
    bucket = bucket + world.recv(bucket, prev_rank)
    world.send(bucket, next_rank)
