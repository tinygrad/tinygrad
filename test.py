# PROFILE=2 VIZ=1 python ./extra/gemm/amd_uop_matmul.py
import pickle, statistics
from tinygrad.viz.serve import get_metal_counters
from tinygrad.helpers import ProfileRangeEvent,temp

from tinygrad.helpers import diskcache
@diskcache
def get(s):
  return list(parse_xml(xctrace_export(s).stdout))

with open(temp("profile.pkl", append_user=True), "rb") as f: profile = pickle.load(f)
for e in profile:
  if isinstance(e, ProfileRangeEvent) and e.device == "METAL":
    if not (ret:=get_metal_counters(e.st, e.en)): continue
    print("\n"+e.name)
    for group, vals in ret.items():
      print(f"** {group}")
      long = max(len(k) for k in vals)
      for k, v in vals.items(): print(f"{k:<{long}} {v:>6.2f}%")
