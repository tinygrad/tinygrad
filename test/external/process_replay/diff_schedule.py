# create a diff of two schedule graphs
import shutil, importlib, uuid, os, logging, contextlib
from collections import defaultdict
from typing import DefaultDict, List, Set, Tuple
from test.external.process_replay.helpers import print_diff
from tinygrad.engine.schedule import LBScheduleItem, ScheduleItem
from tinygrad.helpers import CI, DEBUG, Context, ContextVar, colored, diskcache_put, fetch, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.realize import CompiledRunner, lower_schedule_item

CAPTURING_PROCESS_REPLAY = ContextVar("CAPTURING_PROCESS_REPLAY", getenv("RUN_PROCESS_REPLAY"))

def process_replay(outs:List[LazyBuffer], graph:DefaultDict[LBScheduleItem, List[LBScheduleItem]], in_degree:DefaultDict[LBScheduleItem, int]):
  # copy the reference module
  ref_schedule = getenv("REF_COMMIT_HASH", "master")
  fp = __file__.replace("diff_schedule", "master_schedule")
  if not os.path.isfile(fp):
    shutil.copyfile(fetch(f"https://raw.githubusercontent.com/tinygrad/tinygrad/{ref_schedule}/tinygrad/engine/schedule.py", allow_caching=False), fp)
  # create the reference graph
  ref_graph, ref_in_degree, _ = importlib.import_module("test.external.process_replay.master_schedule")._graph_schedule(outs)
  # compare
  diff_schedule([(ref_graph, ref_in_degree), (graph, in_degree)])

def diff_schedule(s:List[Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]], DefaultDict[LBScheduleItem, int]]]) -> int:
  si_for_buf: DefaultDict[LazyBuffer, List[ScheduleItem]] = defaultdict(list)
  for _,in_degree in s:
    for lsi in in_degree:
      for buf in lsi.outputs:
        si_for_buf[buf].append(ScheduleItem(lsi.ast, tuple(x.buffer for x in lsi.outputs+lsi.inputs if x.size != 0), tuple(lsi.metadata)))
  changed = 0
  seen_diffs: Set[bytes] = set()
  for buf,si in si_for_buf.items():
    si = list({x.ast.key:x for x in si}.values())
    if len(si) == 1: continue
    assert len(si) == 2, f"must have a ref and a compare schedule {len(si)}"
    ref, compare = si
    # no new kernel for buf
    if ref.ast.key == compare.ast.key: continue
    if (cache_key:=ref.ast.key+compare.ast.key) in seen_diffs: continue
    seen_diffs.add(cache_key)
    changed += 1
    if CAPTURING_PROCESS_REPLAY: diskcache_put("schedule_diff", str(uuid.uuid4()), (str(buf), [ref.ast.key, compare.ast.key]))
    if not CI: print_si_diff(ref, compare)
  if DEBUG >= 1: print(f"*** process replay: {changed} unique kernel{'s' if changed>1 else ''} changed")
  return changed

def print_si_diff(ref:ScheduleItem, compare:ScheduleItem) -> None:
  logging.basicConfig(level=logging.INFO)
  print_diff(ref.ast, compare.ast)
  # skip lowering/runtime error
  with contextlib.suppress(Exception): lower_si_diff(ref, compare)

def lower_si_diff(ref:ScheduleItem, compare:ScheduleItem) -> None:
  if DEBUG >= 4:
    ref_ei = lower_schedule_item(ref)
    compare_ei = lower_schedule_item(compare)
    assert isinstance(ref_ei.prg, CompiledRunner) and isinstance(compare_ei.prg, CompiledRunner)
    print_diff(ref_ei.prg.p.src, compare_ei.prg.p.src)
    # TODO: create new Buffers for process replay to test correctness
    if getenv("TIMING"):
      with Context(DEBUG=2):
        tm0 = ref_ei.run(wait=True)
        tm1 = compare_ei.run(wait=True)
      assert tm0 is not None and tm1 is not None
      tm_diff = ((tm0 - tm1) / tm0) * 100
      if tm_diff > 0: print(colored(f"{tm_diff:.2f}% faster", "green"))
      else: print(colored(f"{tm_diff:,.2f}% slower", "red"))
