# ** setup
import os, sys, time, pickle
from typing import List
prev_val = os.getenv("TRACK_MATCH_STATS")
os.environ["TRACK_MATCH_STATS"] = "2"
#os.environ["VIZ"] = "1"
from tinygrad.helpers import tqdm
from tinygrad.engine.realize import lower_schedule
from tinygrad.tensor import Tensor
from tinygrad.ops import contexts
from viz.serve import create_graph
from extra.models.resnet import ResNet50

if __name__ == "__main__":
  # ** simple test
  a = Tensor.randn(32, 32)
  out = a+2
  sched = out.schedule()
  list(lower_schedule(sched))
  uret = create_graph(contexts[0])
  assert uret.loc.split(":")[0] == "schedule.py"
  assert len(uret.graphs) == len(uret.extra) == 1
  assert len(uret.diffs) == 0
  contexts.clear()

  # ** fuzz
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)
  out = mdl(img)
  sched = out.schedule()
  list(lower_schedule(sched))
  tms: List[float] = []
  for ctx in tqdm(contexts):
    st = time.perf_counter()
    ret = create_graph(ctx)
    assert len(ret.graphs) == len(ret.extra)
    assert len(ret.diffs) == len(ret.graphs)-1, f"found {len(ret.diffs)} diffs but only {len(ret.graphs)-1} graphs"
    tms.append(time.perf_counter()-st)

  timings = list(zip(contexts, tms))
  for ctx,tm in timings[:10]:
    print(f"{len(ctx.sink.sparents)} uops {len(ctx.rewrites):5d} rw {tm:5.2f}s")

  # ** teardown
  with open("/tmp/timings.pkl", "wb") as f: pickle.dump(timings, f)
  sys.stdout = open("/tmp/match_stats.log", "w")
  if prev_val: os.environ["TRACK_MATCH_STATS"]
  else: del os.environ["TRACK_MATCH_STATS"]
  print("saved serialization timings in /tmp/timings.pkl and match stats in /tmp/match_stats.log")
