from typing import cast
from dataclasses import dataclass, field
from collections import deque, defaultdict
from tinygrad.uop.ops import UOp, Variable, Ops, UPat, PatternMatcher, graph_rewrite, buffers
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.helpers import Metadata, unwrap, merge_dicts

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[Variable, int] = field(default_factory=dict)

# **** unbind Variables

def unbind_view(ctx:list[dict[Variable, int]], x:UOp):
  st = unwrap(x.st).simplify()
  if any(x.op is Ops.BIND for x in st.vars()):
    st, var_vals = st.unbind()
    ctx.append(var_vals)
    return x.replace(arg=st)
  return None

def unbind_bind(ctx:list[dict[Variable, int]], x:UOp):
  var, val = x.unbind()
  ctx.append({var.replace(src=()):val})
  return var

pm_unbind = PatternMatcher([
  (UPat(Ops.VIEW, name="x"), unbind_view),
  (UPat(Ops.BIND, name="x"), unbind_bind),
])

# **** schedule linearizer

def create_schedule_with_vars(sched_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  # construct the KERNEL children graph based on assigns
  children: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree: dict[UOp, int] = {}
  for u in sched_sink.toposort():
    if u.op is not Ops.ASSIGN: continue  # anything that's not an ASSIGN doesn't write a kernel, so we can skip
    k = u.src[1]
    in_degree.setdefault(k, 0)
    for s in k.src:
      if s.op is Ops.ASSIGN:
        children[s.src[1]].append(k)
        in_degree[k] += 1
      elif s.op in {Ops.MSELECT, Ops.MSTACK}:
        for ss in s.src:
          if ss.op is Ops.MSELECT: ss = ss.src[0]
          if ss.op is not Ops.BUFFER:
            assert ss.op is Ops.ASSIGN
            children[ss.src[1]].append(k)
            in_degree[k] += 1
      elif s.op is Ops.BUFFER:
        pass  # a BUFFER is already realized, nothing to do here
      else:
        raise RuntimeError(f"input to kernel must be ASSIGN or BUFFER, not {s.op}")

  # linearize KERNEL UOps into ScheduleItems in BFS order
  queue = deque(k for k,v in in_degree.items() if v == 0)
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  while queue:
    k = queue.popleft()
    # unbind var_vals from the kernel
    local_var_vals: list[dict[Variable, int]] = []
    ast = graph_rewrite(k.arg.ast, pm_unbind, ctx=local_var_vals, name="unbind vars")
    var_vals = merge_dicts([var_vals, *local_var_vals])
    # create subbuffers if needed
    if ast.op is Ops.BUFFER_VIEW:
      base = k.src[1].buf_uop.buffer
      assert isinstance(base, Buffer), "base can't be MultiBuffer"
      buffers[k.src[0]] = base.view(k.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
    ubufs = tuple(s.buf_uop.buffer for s in k.src)
    if any(isinstance(x, MultiBuffer) for x in ubufs):
      assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
      dnums = [x for x in ast.variables() if x.arg[0] == '_device_num']
      for i,bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
        schedule.append(ScheduleItem(ast, bufs, k.arg.metadata, {dnums[0]:i} if len(dnums) else {}))
    else:
      # ONE -> ONE
      schedule.append(ScheduleItem(ast, cast(tuple[Buffer, ...], ubufs), k.arg.metadata))
    for x in children[k]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  return schedule, var_vals
