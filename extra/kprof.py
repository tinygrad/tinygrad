"""GPU kernel times from the profiler (PROFILE=1): kernel name -> (count, total us) of the events since the last reset()"""
import re, collections
from tinygrad import Device
from tinygrad.device import Compiled, ProfileGraphEvent
from tinygrad.helpers import ProfileRangeEvent

def _name(n) -> str: return re.sub(r"\x1b\[[0-9;]*m", "", str(getattr(n, "display_name", n)))
def _collect(devs:tuple[str, ...]):
  for d in devs:
    Device[d].synchronize()
    Device[d].collect_prof()
def reset(*devs:str):
  _collect(devs)
  Compiled.profile_events.clear()
def times(*devs:str) -> dict[str, tuple[int, float]]:
  _collect(devs)
  c:collections.Counter = collections.Counter()
  t:dict[str, float] = collections.defaultdict(float)
  for e in Compiled.profile_events:
    if isinstance(e, ProfileGraphEvent):
      for ent in e.ents:
        if ent.device.startswith(("AMD", "NV", "CUDA")):
          n = _name(ent.name)
          c[n] += 1
          t[n] += float(e.sigs[ent.en_id] - e.sigs[ent.st_id])
    elif isinstance(e, ProfileRangeEvent) and e.device.startswith("AMD") and e.en is not None:
      n = _name(e.name)
      c[n] += 1
      t[n] += float(e.en - e.st)
  return {n: (c[n], t[n]) for n in c}

def sequence_timed(*devs:str) -> list[tuple[str, str, float, float]]:
  """(device, name, start us, duration us) of the profiled kernels in order"""
  _collect(devs)
  out: list[tuple[float, str, str, float]] = []
  for e in Compiled.profile_events:
    if isinstance(e, ProfileGraphEvent):
      for ent in e.ents:
        if ent.device.startswith("AMD"):
          out.append((float(e.sigs[ent.st_id]), ent.device, _name(ent.name), float(e.sigs[ent.en_id] - e.sigs[ent.st_id])))
  return [(device, name, start, duration) for start, device, name, duration in sorted(out)]

def sequence(*devs:str) -> list[tuple[str, str, float]]:
  """(device, name, us) of the profiled kernels in order"""
  return [(device, name, duration) for device, name, _, duration in sequence_timed(*devs)]

def critical_path(seq:list[tuple[str, str, float, float]], eps:float=2.0) -> dict:
  """
  the critical path of a timeline (sequence_timed): from the last event back, an event's predecessor is the one before it on its lane (device,
  kernels or copies) if that one ended within eps us of its start (the lane was busy), else the event of any lane that ended last before
  its start (what it waited for). returns the path's time by kind/name, its waits (gaps) and lane switches
  """
  import bisect
  ev = sorted(((st, st + du, d, n, n.startswith("copy")) for d, n, st, du in seq), key=lambda e: e[1])
  ends = [e[1] for e in ev]
  lanes: dict = collections.defaultdict(list)
  for e in sorted(ev): lanes[(e[2], e[4])].append(e)
  lane_st = {k: [e[0] for e in v] for k, v in lanes.items()}
  t0 = min(e[0] for e in ev)
  cur, out = ev[-1], {"wall": ev[-1][1] - t0, "kernels": 0.0, "copies": 0.0, "wait": 0.0, "switches": 0, "names": collections.Counter(),
                      "devs": collections.Counter(), "waits": collections.Counter()}
  while True:
    out["copies" if cur[4] else "kernels"] += cur[1] - cur[0]
    out["names"][cur[3] if not cur[4] else " ".join(cur[3].split()[:3])] += cur[1] - cur[0]
    out["devs"][(cur[2], cur[4])] += cur[1] - cur[0]
    k = (cur[2], cur[4])
    i = bisect.bisect_left(lane_st[k], cur[0]) - 1
    prev = lanes[k][i] if i >= 0 else None
    if prev is not None and prev[1] >= cur[0] - eps: nxt = prev
    else:
      j = bisect.bisect_right(ends, cur[0]) - 1
      while j >= 0 and ev[j] is cur: j -= 1
      if j < 0: break
      nxt = ev[j]
      out["wait"] += max(0.0, cur[0] - nxt[1])
      out["waits"][("copy" if nxt[4] else "kern") + "->" + ("copy" if cur[4] else "kern")] += max(0.0, cur[0] - nxt[1])
      if (nxt[2], nxt[4]) != k: out["switches"] += 1
    cur = nxt
  out["lead"] = cur[0] - t0
  return out

def busy_all(seq:list[tuple[str, str, float, float]]) -> tuple[float, float]:
  """(span, the union of every device's events): span - union is the time no device ran anything (host, launch)"""
  iv = sorted((st, st + du) for _, _, st, du in seq)
  tot, end = 0.0, float("-inf")
  for a, b in iv:
    if b > end: tot, end = tot + b - max(a, end), b
  return (max(b for _, b in iv) - iv[0][0]) if iv else 0.0, tot
