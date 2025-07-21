# xctrace parser
import subprocess, xml.etree.ElementTree as ET
from typing import Generator, Callable
from decimal import Decimal
from tinygrad.helpers import unwrap

# sample_time: relative ns passed since start
# mach time: monotonic clock
time_info:dict = {}
def to_cpu_time(sample_time:float) -> float:
  perf_ns_base = Decimal(time_info["mabs-epoch"])*Decimal(time_info["timebase-info"])
  return float((perf_ns_base+Decimal(sample_time))/Decimal("1_000"))

tags:dict[str, Callable] = {
  "boolean":lambda x:x!="0","uint32":int,"uint64":int,"event-time":int,"sample-time":float,"fixed-decimal":float, "time-since-epoch":float,
  "mach-absolute-time":float, "mach-continuous-time":float, "mach-timebase-info":lambda _:125/3, "string":str, "gpu-counter-name":str}

schema:dict[str, list[str]] = {"time-info":["update-time", "mabs-epoch", "mct-epoch", "timebase-info", "trace-start-time"],
  "gpu-counter-value":["timestamp", "counter-id", "value", "accelerator-id", "sample-index", "ring-buffer-index"],
  "gpu-counter-info": ["timestamp", "counter-id", "name", "max-value", "accelerator-id", "description", "group-index", "type", "ring-buffer-count",
                      "require-weighted-accumulation", "sample-interval"]}

def xctrace_export(fp:str, schemas:list[str]) -> Generator[dict, None, None]:
  xp = '/trace-toc/run[@number="1"]/data/table[' +' or '.join(f'@schema="{s}"' for s in schemas)+']'
  proc = subprocess.Popen(["xctrace", "export", "--input", fp, "--xpath", xp], stdout=subprocess.PIPE)
  id_cache:dict[str, str] = {}
  curr_schema = ""
  for event,e in ET.iterparse(unwrap(proc.stdout), events=("start", "end")):
    if event == "start" and e.tag == "node": curr_schema = schemas.pop(0)
    elif event == "end" and e.tag == "row":
      row = time_info if curr_schema == "time-info" else {"schema":curr_schema}
      for col,v in zip(schema[curr_schema], e):
        value = tags[v.tag](id_cache[ref] if (ref:=v.attrib.get("ref")) else (v.text or ""))
        if col == "timestamp" and time_info: value = to_cpu_time(value)
        if (eid:=v.attrib.get("id")): id_cache[eid] = value
        row[col] = value
      yield row
