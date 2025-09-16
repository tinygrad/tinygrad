from __future__ import annotations

from typing import Iterable, List

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Optimizer

from . import communication as dist
from .process_group import get_world_size


class ZeroOptimizer:
  """Minimal ZeRO-2 style optimizer wrapper."""

  def __init__(self, optimizer: Optimizer, parameters: Iterable[Tensor] | None = None) -> None:
    self.optimizer = optimizer
    self.parameters: List[Tensor] = []
    if parameters is None:
      self.parameters = list(getattr(self.optimizer, "params", []))
    else:
      self.parameters = [p for p in parameters if p.requires_grad]

    if not self.parameters:
      raise ValueError("ZeroOptimizer requires at least one parameter")

    self.params = self.parameters
    self.world_size = get_world_size()

  def zero_grad(self) -> None:
    self.optimizer.zero_grad()

  def step(self) -> None:
    self._sync_gradients()
    self.optimizer.step()

  def state_dict(self):
    return getattr(self.optimizer, "state", {})

  def load_state_dict(self, state_dict) -> None:
    if hasattr(self.optimizer, "state"):
      self.optimizer.state = state_dict

  def _sync_gradients(self) -> None:
    if self.world_size <= 1:
      return
    for param in self.parameters:
      if param.grad is None:
        continue
      reduced = dist.all_reduce(param.grad, op="sum")
      param.grad.assign((reduced if isinstance(reduced, Tensor) else Tensor(reduced)).to(param.device) * (1.0 / self.world_size))
