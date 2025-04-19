from dataclasses import dataclass
from collections import deque
from tinygrad.ops import UOp, Variable, Ops, UPat, PatternMatcher, graph_rewrite, buffers
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata, DEBUG, unwrap

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()

# **** unbind Variables

def unbind_view(ctx:dict[Variable, int], x:UOp):
  st = unwrap(x.st).simplify()
  if any(x.op is Ops.BIND for x in st.vars()):
    st, var_vals = st.unbind()
    ctx.update(var_vals)
  return x.replace(arg=st) if st != x.st else None

def unbind_bind(ctx:dict[Variable, int], x:UOp):
  var, val = x.unbind()
  ctx[var.replace(src=())] = val
  return var

pm_unbind = PatternMatcher([
  (UPat(Ops.VIEW, name="x"), unbind_view),
  (UPat(Ops.BIND, name="x"), unbind_bind),
])

# **** schedule linearizer

def create_schedule_with_vars(sched_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  # bfs toposort
  children: dict[UOp, list[UOp]] = {}
  in_degree: dict[UOp, int] = {}
  for u in (toposort:=sched_sink.toposort):
    if u.op is not Ops.ASSIGN: continue
    in_degree[u] = 0
    for s in u.src[1].src:
      if s.op is not Ops.ASSIGN: continue
      children.setdefault(s, []).append(u)
      in_degree[u] += 1

  queue = deque(k for k,v in in_degree.items() if v == 0)
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  while queue:
    u = queue.popleft()
    # map the BUFFER UOp to a subbuffer if it's a BUFFER_VIEW
    if (k:=u.src[1]).arg.ast.op is Ops.BUFFER_VIEW:
      buffers[k.src[0]] = (base:=k.src[1].buf_uop.buffer).view(k.size, k.arg.ast.dtype, k.arg.ast.arg[1]*base.dtype.itemsize)
    schedule.append(ScheduleItem(graph_rewrite(k.arg.ast, pm_unbind, ctx=var_vals), tuple(s.buf_uop.buffer for s in k.src), k.arg.metadata))
    for x in children.get(u, []):
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if len(schedule) != len(in_degree): raise RuntimeError(f"created {len(in_degree)} kernels but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")

  # map ASSIGN to BUFFER after ScheduleItems are constructed
  becomes_map = {u:u.buf_uop for u in toposort if u.op is Ops.ASSIGN}
  assert all(u.op in {Ops.BUFFER, Ops.BUFFER_VIEW} for u in becomes_map.values()), f"Schedule didn't end with BUFFER {becomes_map.values()}"

  return schedule, var_vals, becomes_map
