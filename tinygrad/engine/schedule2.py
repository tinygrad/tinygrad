from typing import List, Tuple, Optional
from dataclasses import dataclass
from tinygrad.device import Buffer
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.ops import UOp, UOps
from tinygrad.helpers import Metadata

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Optional[Tuple[Metadata, ...]]
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return self.bufs[:len(self.ast.src)] if self.ast.op is UOps.SINK else self.bufs[0:1]
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return self.bufs[len(self.ast.src):] if self.ast.op is UOps.SINK else self.bufs[1:]

def _graph(outs:List[LazyBuffer]) -> List[ScheduleItem]:
  ret: List[ScheduleItem] = []
  return ret
