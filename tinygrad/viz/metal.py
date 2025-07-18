import sys, subprocess, os
from typing import Generator
from decimal import Decimal
import xml.etree.ElementTree as ET
from tinygrad.helpers import temp, tqdm, getenv

def xctrace_export(fp:str, query:str) -> str:
  #out_path = f"{temp(query)}.xml"
  out = f"/tmp/{query}.xml"
  if os.path.exists(out):
    return out
    os.system(f"rm {out}")
  print(f"exporting to {out}")
  subprocess.run(["xctrace","export","--input",fp,"--output",out,"--xpath",f'/trace-toc/run[@number="1"]/data/table[@schema="{query}"]'], check=True)
  return out

TAGS = {"boolean":lambda x:x!="0", "uint32":int, "uint64":int, "event-time":int, "sample-time":Decimal, "mach-absolute-time":int, "string":str,
        "mach-continuous-time":int, "mach-timebase-info":str, "time-since-epoch":Decimal, "gpu-counter-name":str, "fixed-decimal":float}

def parse_xml(xml:str) -> Generator[dict, None, None]:
  id_cache:dict[str, str] = {}
  columns:list[str] = []
  for _,e in ET.iterparse(xml, events=("end",)):
    if getenv("XML_DEBUG"): print(ET.tostring(e, encoding="unicode"))
    if e.tag == "col": columns.append(e.find("mnemonic").text)
    if e.tag != "row": continue
    rec:dict[str, bool|int|str] = {}
    for col,v in zip(columns, e):
      rec[col] = value = TAGS[v.tag](id_cache[ref] if (ref:=v.attrib.get("ref")) else (v.text or ""))
      if (eid:=v.attrib.get("id")): id_cache[eid] = value
    yield rec

def to_cpu_time(ts:Decimal, trace_start:Decimal) -> float:
  ts_us = ts/Decimal("1_000")
  return float(trace_start*Decimal("1_000_000")+ts_us)

def parse_metal_trace(fp:str) -> list[dict]:
  ret:dict[int, dict] = {}
  trace_start = next(parse_xml(xctrace_export(fp, "time-info")))["trace-start-time"]
  for v in parse_xml(xctrace_export(fp, "gpu-counter-info")): ret[v.pop("counter-id")] = {**v, "data":[]}
  for v in parse_xml(xctrace_export(fp, "gpu-counter-value")):
    ret[v.pop("counter-id")]["data"].append({"x":to_cpu_time(v["timestamp"], trace_start), "y":v["value"]})
  return list(ret.values())

if __name__ == "__main__":
  ret = parse_metal_trace(sys.argv[1])
  print(ret)
