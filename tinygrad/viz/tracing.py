import time, decimal, contextlib
from typing import Generator
from dataclasses import dataclass
from tinygrad.helpers import PROFILE

@dataclass(frozen=True)
class TracingKey:
  display_name:str                       # display name of this trace event
  keys:tuple[str, ...]=()                # optional keys to search for related traces
  fmt:str|None=None                      # optional detailed formatting
  cat:str|None=None                      # optional category to color this by

class ProfileEvent: pass

@dataclass
class ProfileRangeEvent(ProfileEvent): device:str; name:str|TracingKey; st:decimal.Decimal; en:decimal.Decimal|None=None; is_copy:bool=False # noqa: E702

cpu_events:list[ProfileEvent] = []
@contextlib.contextmanager
def cpu_profile(name:str|TracingKey, device="CPU", is_copy=False, display=True, dst:list|None=None) -> Generator[ProfileRangeEvent, None, None]:
  res = ProfileRangeEvent(device, name, decimal.Decimal(time.perf_counter_ns()) / 1000, is_copy=is_copy)
  try: yield res
  finally:
    res.en = decimal.Decimal(time.perf_counter_ns()) / 1000
    if PROFILE and display: (cpu_events if dst is None else dst).append(res)
