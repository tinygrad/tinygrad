from __future__ import annotations

from typing import Any

import numpy as np

from tinygrad.tensor import Tensor

from . import Backend


class MPIBackend(Backend):
  name = "mpi"

  def __init__(self) -> None:
    try:
      from mpi4py import MPI  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
      raise RuntimeError("mpi4py is required for the MPI backend") from exc
    self._mpi = MPI
    self._comm = MPI.COMM_WORLD
    self.rank = self._comm.Get_rank()
    self.world_size = self._comm.Get_size()

  def init(self, *, rank: int, world_size: int, **kwargs: Any) -> None:
    if self.rank != rank or self.world_size != world_size:
      raise RuntimeError(
        f"MPI world mismatch: expected rank {rank}/{world_size}, got {self.rank}/{self.world_size}")

  def finalize(self) -> None:
    return

  def all_reduce(self, value: Any, *, op: str = "sum") -> Any:
    array = self._to_numpy(value)
    result = np.empty_like(array)
    mpi_op = self._resolve_op(op)
    self._comm.Allreduce(array, result, op=mpi_op)
    return self._from_numpy(result, template=value)

  def all_gather(self, value: Any, *, dim: int = 0) -> Any:
    array = self._to_numpy(value)
    gathered = self._comm.allgather(array)
    stacked = np.stack(gathered, axis=dim)
    return self._from_numpy(stacked, template=value)

  def broadcast(self, value: Any, *, src_rank: int = 0) -> Any:
    is_src = self.rank == src_rank
    array = self._to_numpy(value) if is_src else self._allocate_like(value)
    self._comm.Bcast(array, root=src_rank)
    return self._from_numpy(array, template=value)

  def send(self, value: Any, *, dst_rank: int) -> None:
    array = self._to_numpy(value)
    self._comm.Send(array, dest=dst_rank)

  def recv(self, value: Any, *, src_rank: int) -> Any:
    array = self._allocate_like(value)
    self._comm.Recv(array, source=src_rank)
    return self._from_numpy(array, template=value)

  def _resolve_op(self, op: str):
    if op != "sum":
      raise NotImplementedError(f"MPI backend only supports sum reduction, got {op}")
    return self._mpi.SUM

  @staticmethod
  def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
      return value.numpy()
    if isinstance(value, np.ndarray):
      return value
    raise TypeError(f"unsupported value type {type(value)!r} for MPI backend")

  @staticmethod
  def _allocate_like(value: Any) -> np.ndarray:
    array = MPIBackend._to_numpy(value)
    return np.empty_like(array)

  @staticmethod
  def _from_numpy(array: np.ndarray, *, template: Any) -> Any:
    if isinstance(template, Tensor):
      return Tensor(array)
    return array


__all__ = ["MPIBackend"]
