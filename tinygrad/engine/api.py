# TODO: temp
from dataclasses import dataclass
from typing import Tuple
from tinygrad.device import Buffer
from tinygrad.ops import LazyOp

@dataclass(frozen=True)
class ScheduleItem:
  ast: Tuple[LazyOp, ...]
  bufs: Tuple[Buffer, ...]
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return self.bufs[:len(self.ast)]
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return self.bufs[len(self.ast):]
