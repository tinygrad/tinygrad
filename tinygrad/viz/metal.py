import sys, subprocess, os
from typing import Generator
import xml.etree.ElementTree as ET
from tinygrad.helpers import temp

COLS = {"gpu-counter-info":["timestamp", "counter_id", "name", "max_value", "accelerator_id", "description", "group_index", "type",
                            "ring_buffer_count", "require_weighted_accumulation", "sample_interval"],
        "gpu-counter-value":["timestamp", "counter_id", "value", "accelerator_id", "sample_index", "ring_buffer_index"]}
TAGS = {"boolean":lambda x:x!="0", "uint32":int, "uint64":int, "event-time": int, "gpu-counter-name":str, "string":str, "fixed-decimal":float}

def xctrace_export(fp:str, query:str) -> str:
  try:
    with open(out_path:=f"{temp(query)}.xml") as f: return f.read()
  except FileNotFoundError:
    print(f"exporting to {out_path}")
    subprocess.check_output(["xctrace", "export", "--input", fp, "--output", out_path, "--xpath",
                             f'/trace-toc/run[@number="1"]/data/table[@schema="{query}"]'], text=True)
    return xctrace_export(fp, query)

def parse_xml(fp:str, query:str) -> Generator[dict, None, None]:
  id_cache:dict[str, str] = {}
  for row in ET.fromstring(xctrace_export(fp, query)).findall(f".//schema[@name='{query}']/..//row"):
    rec:dict[str, bool|int|str] = {}
    for col,v in zip(COLS[query], row):
      rec[col] = value = TAGS[v.tag](id_cache[ref] if (ref:=v.attrib.get("ref")) else (v.text or ""))
      if (eid:=v.attrib.get("id")): id_cache[eid] = value
    yield rec

def parse_metal_trace(fp:str) -> list[dict]:
  ret:dict[int, dict] = {}
  for v in parse_xml(fp, "gpu-counter-info"): ret[v.pop("counter_id")] = {**v, "data":[]}
  for v in parse_xml(fp, "gpu-counter-value"): ret[v.pop("counter_id")]["data"].append({"x":v["timestamp"], "y":v["value"]})
  return list(ret.values())

if __name__ == "__main__":
  ret = parse_metal_trace(sys.argv[1])
  print(ret)
