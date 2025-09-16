from __future__ import annotations

import os
from typing import Optional

from .backend import resolve_backend, set_backend, get_backend, reset_backend

_rank: Optional[int] = None
_world_size: Optional[int] = None


def init_process_group(backend: str = "mpi", *, rank: Optional[int] = None, world_size: Optional[int] = None, **kwargs) -> None:
  """Initialize the global process group and backend."""
  global _rank, _world_size
  if _rank is not None:
    raise RuntimeError("process group already initialized")

  if rank is None:
    rank = int(os.getenv("RANK", "0"))
  if world_size is None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))

  if world_size < 1:
    raise ValueError(f"WORLD_SIZE must be at least 1, got {world_size}")
  if rank < 0 or rank >= world_size:
    raise ValueError(f"rank {rank} outside valid range for world size {world_size}")

  backend_impl = resolve_backend(backend)
  backend_impl.init(rank=rank, world_size=world_size, **kwargs)
  set_backend(backend_impl)

  _rank = rank
  _world_size = world_size


def destroy_process_group() -> None:
  """Tear down the global process group."""
  global _rank, _world_size
  if _rank is None:
    return

  backend = get_backend(optional=True)
  if backend is not None:
    backend.finalize()

  reset_backend()
  _rank = None
  _world_size = None


def is_initialized() -> bool:
  return _rank is not None


def get_rank() -> int:
  if _rank is None:
    raise RuntimeError("process group is not initialized")
  return _rank


def get_world_size() -> int:
  if _world_size is None:
    raise RuntimeError("process group is not initialized")
  return _world_size
