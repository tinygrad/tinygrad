from __future__ import annotations

from typing import Any

from . import Backend


class UCXBackend(Backend):
  name = "ucx"

  def __init__(self) -> None:
    raise RuntimeError("UCX backend is not implemented in this build")

  def init(self, *, rank: int, world_size: int, **kwargs: Any) -> None:  # pragma: no cover - interface stub
    raise RuntimeError("UCX backend is not implemented in this build")

  def finalize(self) -> None:  # pragma: no cover - interface stub
    raise RuntimeError("UCX backend is not implemented in this build")

  def all_reduce(self, value: Any, *, op: str = "sum") -> Any:  # pragma: no cover - interface stub
    raise RuntimeError("UCX backend is not implemented in this build")

  def all_gather(self, value: Any, *, dim: int = 0) -> Any:  # pragma: no cover - interface stub
    raise RuntimeError("UCX backend is not implemented in this build")

  def broadcast(self, value: Any, *, src_rank: int = 0) -> Any:  # pragma: no cover - interface stub
    raise RuntimeError("UCX backend is not implemented in this build")

  def send(self, value: Any, *, dst_rank: int) -> None:  # pragma: no cover - interface stub
    raise RuntimeError("UCX backend is not implemented in this build")

  def recv(self, value: Any, *, src_rank: int) -> Any:  # pragma: no cover - interface stub
    raise RuntimeError("UCX backend is not implemented in this build")


__all__ = ["UCXBackend"]
