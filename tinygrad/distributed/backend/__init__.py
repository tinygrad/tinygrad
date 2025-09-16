from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
  import numpy as np
except ImportError:  # pragma: no cover - numpy is available in normal setups
  np = None  # type: ignore

from tinygrad.tensor import Tensor


class Backend:
  name: str

  def init(self, *, rank: int, world_size: int, **kwargs: Any) -> None:  # pragma: no cover - interface
    raise NotImplementedError

  def finalize(self) -> None:  # pragma: no cover - interface
    raise NotImplementedError

  def all_reduce(self, value: Any, *, op: str = "sum") -> Any:  # pragma: no cover - interface
    raise NotImplementedError

  def all_gather(self, value: Any, *, dim: int = 0) -> Any:  # pragma: no cover - interface
    raise NotImplementedError

  def broadcast(self, value: Any, *, src_rank: int = 0) -> Any:  # pragma: no cover - interface
    raise NotImplementedError

  def send(self, value: Any, *, dst_rank: int) -> None:  # pragma: no cover - interface
    raise NotImplementedError

  def recv(self, value: Any, *, src_rank: int) -> Any:  # pragma: no cover - interface
    raise NotImplementedError


@dataclass
class DummyBackend(Backend):
  """Single-process backend used for tests and development."""

  name: str = "dummy"
  rank: int = 0
  world_size: int = 1

  def init(self, *, rank: int, world_size: int, **kwargs: Any) -> None:
    self.rank = rank
    self.world_size = world_size
    if world_size != 1:
      raise RuntimeError("dummy backend only supports a single rank")

  def finalize(self) -> None:
    return

  def all_reduce(self, value: Any, *, op: str = "sum") -> Any:
    return self._clone_value(value)

  def all_gather(self, value: Any, *, dim: int = 0) -> Any:
    return self._clone_value(value)

  def broadcast(self, value: Any, *, src_rank: int = 0) -> Any:
    return self._clone_value(value)

  def send(self, value: Any, *, dst_rank: int) -> None:
    if dst_rank != self.rank:
      raise RuntimeError("dummy backend cannot communicate across ranks")

  def recv(self, value: Any, *, src_rank: int) -> Any:
    if src_rank != self.rank:
      raise RuntimeError("dummy backend cannot communicate across ranks")
    return self._clone_value(value)

  @staticmethod
  def _clone_value(value: Any) -> Any:
    if isinstance(value, Tensor):
      return value
    if np is not None and isinstance(value, np.ndarray):
      return value.copy()
    return value


_current_backend: Optional[Backend] = None


def resolve_backend(name: str) -> Backend:
  lowered = name.lower()
  if lowered == "dummy":
    return DummyBackend()
  if lowered == "mpi":
    from .mpi import MPIBackend
    return MPIBackend()
  if lowered == "ucx":
    from .ucx import UCXBackend
    return UCXBackend()
  raise ValueError(f"unsupported backend: {name}")


def set_backend(backend: Backend) -> None:
  global _current_backend
  _current_backend = backend


def get_backend(*, optional: bool = False) -> Backend:
  if _current_backend is None:
    if optional:
      return None  # type: ignore
    raise RuntimeError("process group not initialized")
  return _current_backend


def reset_backend() -> None:
  global _current_backend
  _current_backend = None


__all__ = [
  "Backend",
  "DummyBackend",
  "resolve_backend",
  "set_backend",
  "get_backend",
  "reset_backend",
]
