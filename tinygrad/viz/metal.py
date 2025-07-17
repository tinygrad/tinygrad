import sys, subprocess, os
from typing import Generator
import xml.etree.ElementTree as ET

COLS = {"gpu-counter-info":["timestamp", "counter_id", "name", "max_value", "accelerator_id", "description", "group_index", "type",
                            "ring_buffer_count", "require_weighted_accumulation", "sample_interval"],
        "gpu-counter-value":["timestamp", "counter_id", "value", "accelerator_id", "sample_index", "ring_buffer_index"]}
TAGS = {"boolean":lambda x:x!="0", "uint32":int, "uint64":int, "event-time": int, "gpu-counter-name":str, "string":str, "fixed-decimal":float}
def parse(query:str) -> Generator[dict, None, None]:
  xml_txt = subprocess.check_output(["xctrace", "export", "--input", sys.argv[1], "--xpath",
                                 f'/trace-toc/run[@number="1"]/data/table[@schema="{query}"]'], text=True)
  id_cache:dict[str, str] = {}
  for row in ET.fromstring(xml_txt).findall(f".//schema[@name='{query}']/..//row"):
    rec:dict[str, bool|int|str] = {}
    for col,v in zip(COLS[query], row):
      rec[col] = value = TAGS[v.tag](id_cache[ref] if (ref:=v.attrib.get("ref")) else (v.text or ""))
      if (eid:=v.attrib.get("id")): id_cache[eid] = value
    yield rec

if __name__ == "__main__":
  counters = {v.pop("counter_id"):v for v in parse("gpu-counter-info")}
  ret:dict[int, list[dict]] = {}
  for v in parse("gpu-counter-value"): ret.setdefault(v.pop("counter_id"), []).append(v)
  from pprint import pprint
  for k,v in ret.items():
    print(f"{counters[k]['name']}: {len(v)}")
