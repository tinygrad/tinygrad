import functools
from typing import Dict, List, Tuple
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata
from tinygrad.ops import Ops, UOp, Variable
from dataclasses import dataclass

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Tuple[Metadata, ...]
  @property
  def outputs(self) -> Tuple[Buffer, ...]: return tuple(b for i,b in enumerate(self.bufs) if i in self._output_idxs)
  @property
  def inputs(self) -> Tuple[Buffer, ...]: return tuple(b for i,b in enumerate(self.bufs) if i not in self._output_idxs)
  @functools.cached_property
  def _output_idxs(self) -> Tuple[int, ...]: return tuple(x.src[0].arg for x in self.ast.src) if self.ast.op is Ops.SINK else (0,)

def create_schedule_with_vars(outs:List[UOp]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  return [], {}

def create_schedule(outs:List[UOp]) -> List[ScheduleItem]:
  return []
