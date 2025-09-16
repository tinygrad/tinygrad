from __future__ import annotations

from typing import Any

from .backend import get_backend

__all__ = [
  "all_reduce",
  "all_gather",
  "broadcast",
  "send",
  "recv",
]


def all_reduce(value: Any, op: str = "sum") -> Any:
  """Reduce ``value`` across all ranks using the selected backend."""
  return get_backend().all_reduce(value, op=op)


def all_gather(value: Any, dim: int = 0) -> Any:
  """Gather ``value`` from every rank along ``dim``."""
  return get_backend().all_gather(value, dim=dim)


def broadcast(value: Any, src_rank: int = 0) -> Any:
  """Broadcast ``value`` from ``src_rank`` to all ranks."""
  return get_backend().broadcast(value, src_rank=src_rank)


def send(value: Any, dst_rank: int) -> None:
  """Send ``value`` to ``dst_rank``."""
  get_backend().send(value, dst_rank=dst_rank)


def recv(value: Any, src_rank: int) -> Any:
  """Receive ``value`` from ``src_rank``."""
  return get_backend().recv(value, src_rank=src_rank)
