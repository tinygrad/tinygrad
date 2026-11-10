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
