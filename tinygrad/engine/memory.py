from typing import List, Dict, DefaultDict, Tuple, Union
from collections import defaultdict
from tinygrad.dtype import DType
from tinygrad.device import Buffer
from tinygrad.helpers import getenv, DEBUG, dedup
from tinygrad.engine.schedule import ScheduleItem

def _internal_memory_planner(buffers:List[Union[List[Buffer], Tuple[Buffer, ...]]], debug_prefix="") -> Dict[Buffer, Buffer]:
  if getenv("NO_MEMORY_PLANNER"): return {}
  last_appearance = {}
  for i,u in enumerate(buffers):
    for buf in u: last_appearance[buf] = i

  # LRU algorithm
  assigned: Dict[Buffer, Buffer] = {}
  local_cache: DefaultDict[Tuple[str, int, DType], List[Buffer]] = defaultdict(list)

  def handle_buffer(buf):
    key = (buf.device, buf.size, buf.dtype)
    if buf not in assigned:
      if len(ll:=local_cache[key]): assigned[buf] = ll.pop()
      else: assigned[buf] = Buffer(*key)
    if i == last_appearance[buf]:
      if assigned[buf] not in local_cache[key]: local_cache[key].append(assigned[buf])

  for i,u in enumerate(buffers):
    for buf in u:
      # all unallocated unparented buffers are fair game to replace
      if buf.is_allocated() or buf.lb_refcount > 0: continue
      # handle view buffers
      if buf._base is not None:
        assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=assigned.get(buf._base, buf._base), offset=buf.offset)
      else:
        handle_buffer(buf)

  if DEBUG >= 1 and len(ak:=dedup(assigned.keys())) != len(av:=dedup(assigned.values())):
    print(debug_prefix+f"memory reduced from {sum([x.nbytes for x in ak])/1e6:.2f} MB -> {sum([x.nbytes for x in av])/1e6:.2f} MB,",
          f"{len(ak)} -> {len(av)} bufs")
  return assigned

def memory_planner(schedule:List[ScheduleItem]) -> List[ScheduleItem]:
  assigned = _internal_memory_planner([si.bufs for si in schedule])
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs)) for si in schedule]
