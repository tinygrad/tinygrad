# create a diff of two schedule graphs
import shutil, importlib, uuid, os, logging, contextlib
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple
from test.external.process_replay.utils import print_diff
from tinygrad.engine.schedule import LBScheduleItem, ScheduleItem
from tinygrad.helpers import CI, DEBUG, Context, colored, diskcache_put, fetch, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.realize import CompiledRunner, lower_schedule_item
from tinygrad.ops import UOp

def process_replay(outs:List[LazyBuffer], graph:DefaultDict[LBScheduleItem, List[LBScheduleItem]], in_degree:DefaultDict[LBScheduleItem, int]):
  # copy the reference module
  ref_schedule = getenv("REF_COMMIT_HASH", "master")
  fp = __file__.replace("diff_schedule", "master_schedule")
  if not os.path.isfile(fp):
    shutil.copyfile(fetch(f"https://raw.githubusercontent.com/tinygrad/tinygrad/{ref_schedule}/tinygrad/engine/schedule.py", allow_caching=False), fp)
  # create the reference graph
  ref_graph, ref_in_degree = importlib.import_module("test.external.process_replay.master_schedule")._graph_schedule(outs, set())
  # compare
  diff_schedule([(ref_graph, ref_in_degree), (graph, in_degree)])

def diff_schedule(s:List[Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]], DefaultDict[LBScheduleItem, int]]]) -> int:
  si_for_buf: DefaultDict[LazyBuffer, List[ScheduleItem]] = defaultdict(list)
  for _,in_degree in s:
    for lsi in in_degree:
      for buf in lsi.outputs:
        si_for_buf[buf].append(ScheduleItem(lsi.ast, tuple(x.buffer for x in lsi.outputs+lsi.inputs if x.size != 0), lsi.metadata))
  changed = 0
  seen_diffs: Set[Tuple[bytes, ...]] = set()
  for buf, si in si_for_buf.items():
    asts: Dict[bytes, UOp] = {x.ast.key:x.ast for x in si}
    # no new kernel for buf
    if len(asts) == 1: continue
    if (cache_key:=tuple(asts)) in seen_diffs: continue
    seen_diffs.add(cache_key)
    changed += 1
    if getenv("RUN_PROCESS_REPLAY"): diskcache_put("schedule_diff", str(uuid.uuid4()), (str(buf), list(asts.values())))
    if not CI: print_si_diff(si[0], si[1])
  if DEBUG >= 1: print(f"*** process replay: {changed} unique kernel{'s' if changed>1 else ''} changed")
  return changed

def print_si_diff(si0:ScheduleItem, si1:ScheduleItem):
  logging.basicConfig(level=logging.INFO)
  print_diff(si0.ast, si1.ast)
  # skip lowering/runtime error
  with contextlib.suppress(Exception):
    ei0 = lower_schedule_item(si0)
    ei1 = lower_schedule_item(si1)
    assert isinstance(ei0.prg, CompiledRunner) and isinstance(ei1.prg, CompiledRunner)
    print_diff(ei0.prg.p.src, ei1.prg.p.src)
    # TODO: create new Buffers for process replay to test correctness
    if getenv("TIMING"):
      with Context(DEBUG=2):
        tm0 = ei0.run(wait=True)
        tm1 = ei1.run(wait=True)
      assert tm0 is not None and tm1 is not None
      tm_diff = ((tm0 - tm1) / tm0) * 100
      if tm_diff > 0: print(colored(f"{tm_diff:.2f}% faster", "green"))
      else: print(colored(f"{tm_diff:,.2f}% slower", "red"))
