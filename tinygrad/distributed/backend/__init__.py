from __future__ import annotations

from dataclasses import dataclass, field
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
  _mailboxes: dict[tuple[int, int], Any] = field(default_factory=dict)

  def init(self, *, rank: int, world_size: int, **kwargs: Any) -> None:
    self.rank = rank
    self.world_size = world_size
    if world_size != 1:
      raise RuntimeError("dummy backend only supports a single rank")
    self._mailboxes.clear()

  def finalize(self) -> None:
    return

  def all_reduce(self, value: Any, *, op: str = "sum") -> Any:
    array = self._to_numpy(value)
    if op != "sum":
      raise NotImplementedError(f"dummy backend only supports sum reduction, got {op}")
    return self._from_numpy(array, template=value)

  def all_gather(self, value: Any, *, dim: int = 0) -> Any:
    array = self._to_numpy(value)
    # In single rank setups this is a no-op copy.
    return self._from_numpy(array, template=value)

  def broadcast(self, value: Any, *, src_rank: int = 0) -> Any:
    return self._from_numpy(self._to_numpy(value), template=value)

  def send(self, value: Any, *, dst_rank: int) -> None:
    if dst_rank != self.rank:
      raise RuntimeError("dummy backend cannot communicate across ranks")
    self._mailboxes[(self.rank, dst_rank)] = self._to_numpy(value).copy()

  def recv(self, value: Any, *, src_rank: int) -> Any:
    if src_rank != self.rank:
      raise RuntimeError("dummy backend cannot communicate across ranks")
    key = (src_rank, self.rank)
    if key not in self._mailboxes:
      raise RuntimeError("no message to receive for given ranks")
    array = self._mailboxes.pop(key)
    return self._from_numpy(array, template=value)

  @staticmethod
  def _to_numpy(value: Any) -> 'np.ndarray':
    if isinstance(value, Tensor):
      return value.numpy()
    if np is not None and isinstance(value, np.ndarray):
      return value
    raise TypeError(f"unsupported value type {type(value)!r} for dummy backend")

  @staticmethod
  def _from_numpy(array: 'np.ndarray', *, template: Any) -> Any:
    if isinstance(template, Tensor):
      return Tensor(array.copy(), device="CPU")
    if np is not None:
      return array.copy()
    raise TypeError("numpy is required for dummy backend conversions")


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
