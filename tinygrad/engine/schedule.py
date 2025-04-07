import atexit, pickle
from dataclasses import dataclass
from collections import deque
from tinygrad.ops import UOp, Variable, Ops, buffers
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata, CAPTURE_PROCESS_REPLAY, DEBUG, Context, ContextVar, diskcache_put
from tinygrad.engine.grouper import get_becomes_map

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()

PROCESS_REPLAY_CAPTURE:dict[str, bytes] = {}
if CAPTURE_PROCESS_REPLAY:
  @atexit.register
  def save_process_replay():
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

# **** schedule linearizer

def create_schedule_with_vars(big_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  becomes_map, var_vals = get_becomes_map(big_sink)
  sched_sink = becomes_map.pop(big_sink)

  # bfs toposort
  children: dict[UOp, list[UOp]] = {}
  in_degree: dict[UOp, int] = {}
  for u in sched_sink.toposort:
    if u.op is not Ops.ASSIGN: continue
    in_degree[u] = 0
    for s in u.src[1].src:
      if s.op is not Ops.ASSIGN: continue
      children.setdefault(s, []).append(u)
      in_degree[u] += 1

  queue = deque(k for k,v in in_degree.items() if v == 0)
  schedule: list[ScheduleItem] = []
  while queue:
    u = queue.popleft()
    # map the BUFFER UOp to a subbuffer if it's a BUFFER_VIEW
    if (k:=u.src[1]).arg.ast.op is Ops.BUFFER_VIEW:
      buffers[k.src[0]] = (base:=k.src[1].buf_uop.buffer).view(k.size, k.arg.ast.dtype, k.arg.ast.arg[1]*base.dtype.itemsize)
    schedule.append(ScheduleItem(k.arg.ast, tuple(s.buf_uop.buffer for s in k.src), k.arg.metadata))
    for x in children.get(u, []):
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if len(schedule) != len(in_degree): raise RuntimeError(f"created {len(in_degree)} kernels but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")

  # capture process replay
  if CAPTURE_PROCESS_REPLAY:
    with Context(PICKLE_BUFFERS=0): PROCESS_REPLAY_CAPTURE[str(big_sink.key)] = pickle.dumps((big_sink, ContextVar._cache, [x.ast for x in schedule]))

  return schedule, var_vals, becomes_map
