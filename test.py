# PROFILE=2 VIZ=1 python ./extra/gemm/amd_uop_matmul.py
import pickle
from tinygrad.viz.serve import xctrace_export, parse_xml, xctrace_to_cpu_time
from tinygrad.helpers import ProfileRangeEvent,temp

from tinygrad.helpers import diskcache
@diskcache
def get(s):
  return list(parse_xml(xctrace_export(s).stdout))

with open(temp("profile.pkl", append_user=True), "rb") as f: w = pickle.load(f)
commands = [e for e in w if isinstance(e, ProfileRangeEvent) and e.device == "METAL"]
tinygemm = next(iter([c for c in commands if c.name == "tinygemm"]))

time_info = get("time-info")[0]
num, denom = [int(field.text) for field in time_info["timebase-info"].findall("mach-timebase-info-field")]
mabs_epoch, timebase = int(time_info["mabs-epoch"]), num/denom

for row in get("gpu-counter-value"):
  sample_ts = xctrace_to_cpu_time(int(row["timestamp"]), mabs_epoch, timebase)
  if sample_ts < tinygemm.st: continue
  if sample_ts > tinygemm.en: break
  print(row)
