# PROFILE=2 VIZ=1 python ./extra/gemm/amd_uop_matmul.py
import pickle, statistics
from tinygrad.viz.serve import xctrace_export, parse_xml, xctrace_to_cpu_time
from tinygrad.helpers import ProfileRangeEvent,temp

from tinygrad.helpers import diskcache
@diskcache
def get(s):
  return list(parse_xml(xctrace_export(s).stdout))

time_info = get("time-info")[0]
num, denom = [int(field.text) for field in time_info["timebase-info"].findall("mach-timebase-info-field")]
mabs_epoch, timebase = int(time_info["mabs-epoch"]), num/denom

counter_info = {row.pop("counter-id"):row for row in get("gpu-counter-info")}
COUNTER_GROUPS = {"ALU":[11, 13, 15, 17, 19, 21, 23], "DRAM":[62, 64], "SRAM":[25]}

def get_stats(st:float, en:float):
  counter_values:dict[int, float] = {}
  for row in get("gpu-counter-value"):
    sample_ts = xctrace_to_cpu_time(int(row["timestamp"]), mabs_epoch, timebase)
    if sample_ts < st: continue
    if sample_ts > en: break
    counter_values.setdefault(row.pop("counter-id"), []).append(float(row["value"]))

  ret = {g:{} for g in COUNTER_GROUPS}
  for k,v in counter_values.items():
    if (group:=next((g for g,lst in COUNTER_GROUPS.items() if int(k) in lst), None)) is None: continue
    ret[group][counter_info[k]["name"]] = statistics.mean(v)
  return ret

with open(temp("profile.pkl", append_user=True), "rb") as f: profile = pickle.load(f)
for e in profile:
  if isinstance(e, ProfileRangeEvent) and e.device == "METAL":
    if not (ret:=get_stats(e.st, e.en)): continue
    print(e.name)
    for group, vals in ret.items():
      print(f"** {group}")
      long = max((len(k) for k in vals), default=0)
      for k, v in vals.items(): print(f"{k:<{long}} {v:>6.2f}%")
