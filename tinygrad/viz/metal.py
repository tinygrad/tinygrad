import sys, subprocess, os
from typing import Generator, Callable
from decimal import Decimal
import xml.etree.ElementTree as ET

def xctrace_export(fp:str, query:str) -> str:
  #out_path = f"{temp(query)}.xml"
  out = f"/tmp/{query}.xml"
  if os.path.exists(out): os.system(f"rm {out}")
  print(f"exporting to {out}")
  subprocess.run(["xctrace","export","--input",fp,"--output",out,"--xpath",f'/trace-toc/run[@number="1"]/data/table[@schema="{query}"]'], check=True)
  return out

# TODO: parse mach-timebase-info
TAGS:dict[str, Callable] = {"boolean":lambda x:x!="0", "uint32":int, "uint64":int, "event-time":int, "sample-time":Decimal, "string":str,
                             "mach-absolute-time":Decimal, "mach-continuous-time":Decimal, "mach-timebase-info":lambda _:125/3, "fixed-decimal":float,
                             "time-since-epoch":Decimal, "gpu-counter-name":str}

def parse_xml(xml:str) -> Generator[dict, None, None]:
  id_cache:dict[str, str] = {}
  columns:list[str] = []
  for _,e in ET.iterparse(xml, events=("end",)):
    if e.tag == "col" and (mnemonic:=e.find("mnemonic")) is not None: columns.append(str(mnemonic.text))
    if e.tag != "row": continue
    rec:dict[str, bool|int|str] = {}
    for col,v in zip(columns, e):
      rec[col] = value = TAGS[v.tag](id_cache[ref] if (ref:=v.attrib.get("ref")) else (v.text or ""))
      if (eid:=v.attrib.get("id")): id_cache[eid] = value
    yield rec

# sample_time: Time relative to start in ns
# trace_start: Wallclock time
# mach hw time (diff is in behavior if device sleeps, does it matter here?)
def to_cpu_time(sample_time:Decimal, time_info:dict) -> float:
  perf_ns_ref = time_info["mabs-epoch"]*Decimal(time_info["timebase-info"])
  perf_ns = perf_ns_ref+sample_time
  return float(perf_ns/Decimal("1_000"))

def parse_metal_trace(fp:str) -> list[dict]:
  ret:dict[int, dict] = {}
  time_info = next(parse_xml(xctrace_export(fp, "time-info")))
  for v in parse_xml(xctrace_export(fp, "gpu-counter-info")): ret[v.pop("counter-id")] = {**v, "data":[]}
  for v in parse_xml(xctrace_export(fp, "gpu-counter-value")):
    ret[v.pop("counter-id")]["data"].append({"x":to_cpu_time(v["timestamp"], time_info), "y":v["value"]})
  return list(ret.values())

if __name__ == "__main__":
  ret = parse_metal_trace(sys.argv[1])
  print(ret)
