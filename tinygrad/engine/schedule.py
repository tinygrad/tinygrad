from dataclasses import dataclass
from tinygrad.ops import UOp, Ops, Variable
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  becomes_map: dict[UOp, UOp] = {}
  return schedule, var_vals, becomes_map
