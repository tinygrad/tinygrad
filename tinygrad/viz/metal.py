import sys, subprocess, xml.etree.ElementTree as ET
from typing import Generator
from decimal import Decimal

schema = {"time-info":["update-time", "mabs-epoch", "mct-epoch", "timebase-info", "trace-start-time"],
  "gpu-counter-value":["timestamp", "counter-id", "value", "accelerator-id", "sample-index", "ring-buffer-index"],
  "gpu-counter-info": ["timestamp", "counter-id", "name", "max-value", "accelerator-id", "description", "group-index", "type", "ring-buffer-count",
                      "require-weighted-accumulation", "sample-interval"]}

tags = {"boolean":lambda x:x!="0","uint32":int,"uint64":int,"event-time":int,"sample-time":Decimal,"fixed-decimal":float,"time-since-epoch":Decimal,
        "mach-absolute-time":Decimal, "mach-continuous-time":Decimal, "mach-timebase-info":lambda _:125/3, "string":str, "gpu-counter-name":str}

# sample_time: Time relative to start in ns
# trace_start: Wallclock time
# mach hw time (diff is in behavior if device sleeps, does it matter here?)
def to_cpu_time(sample_time:Decimal, time_info:dict) -> float:
  perf_ns_ref = time_info["mabs-epoch"]*Decimal(time_info["timebase-info"])
  perf_ns = perf_ns_ref+sample_time
  return float(perf_ns/Decimal("1_000"))

def parse_metal_trace(fp:str) -> Generator[dict, None, None]:
  #xp = '/trace-toc/run[@number="1"]/data/table[' +' or '.join(f'@schema="{s}"' for s in schema)+']'
  #proc = subprocess.Popen(["xctrace", "export", "--input", fp, "--xpath", xp, "--output", "/tmp/tempfile.xml"], stdout=subprocess.PIPE)
  id_cache:dict[str, str] = {}
  time_info = {}
  schemas, curr_schema = list(schema), ""
  for event,e in ET.iterparse("/tmp/tempfile.xml", events=("start", "end")):
    if event == "start" and e.tag == "node": curr_schema = schemas.pop(0)
    elif event == "end" and e.tag == "row":
      row = time_info if curr_schema == "time-info" else {}
      for col,v in zip(schema[curr_schema], e):
        row[col] = value = tags[v.tag](id_cache[ref] if (ref:=v.attrib.get("ref")) else (v.text or ""))
        if (eid:=v.attrib.get("id")): id_cache[eid] = to_cpu_time(value, time_info) if v.tag == "event-time" else value
      yield row

if __name__ == "__main__":
  print(list(parse_metal_trace(sys.argv[1])))
