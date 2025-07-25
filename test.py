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

cntr_info = {}
for row in get("gpu-counter-info"): cntr_info[row.pop("counter-id")] = row

metric_groups = {}
for row in get("gpu-counter-value"):
  sample_ts = xctrace_to_cpu_time(int(row["timestamp"]), mabs_epoch, timebase)
  if sample_ts < tinygemm.st: continue
  if sample_ts > tinygemm.en: break
  metric_groups.setdefault(row.pop("counter-id"), []).append({"ts":sample_ts, "value":row["value"]})


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Number of plots
n = len(metric_groups)

# Create subplot layout with one row per metric
fig = make_subplots(rows=n, cols=1, shared_xaxes=True)

for i, (k, v) in enumerate(metric_groups.items(), start=1):
  info = cntr_info[k]
  df = pd.DataFrame(v)
  fig.add_trace(
    go.Scatter(x=df["ts"], y=df["value"], mode="lines", name=info["name"]),
    row=i, col=1
  )
  fig.update_yaxes(title_text=info["name"], row=i, col=1)

fig.update_layout(height=250*n, showlegend=False, title_text="GPU Metrics")
fig.update_xaxes(title_text="Timestamp", row=n, col=1)
fig.show()
