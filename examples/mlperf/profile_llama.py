#!/usr/bin/env python3
"""
Profile LLaMA3 training and categorize kernel time by component.

Usage:
  # First run training with profiling enabled:
  VIZ=-1 PROFILE=1 FAKEDATA=1 BENCHMARK=2 LLAMA3_SIZE=1B SEQLEN=512 BS=1 MODEL=llama3 \
    python examples/mlperf/model_train.py

  # Then analyze the profile:
  python extra/thunder/tiny/profile_llama.py
"""

import pickle
import struct
import json
import tempfile
import getpass
import re
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

# Category patterns for kernel classification based on kernel name
# Order matters - first match wins
KERNEL_CATEGORIES = [
  # Flash attention kernels
  ("flash_attention", [r"^custom_fa_forward", r"^custom_fa_backward"]),
  # GEMM/matmul operations
  ("gemm", [r"^gemm_", r"^hk_fp8_gemm_"]),
  # Embedding backward (forward is just a table lookup, appears as E_ kernel)
  ("embedding", [r"^embedding_bwd"]),
  # Memory copies between devices
  ("copy", [r"^AMD.*->", r"^CPU.*->", r"^TINY.*->"]),
]

def get_profile_path() -> Path:
  """Get the profile pickle file path."""
  return Path(tempfile.gettempdir()) / f"profile.pkl.{getpass.getuser()}"

class TinyUnpacker:
  """Simple struct unpacker with offset tracking."""
  def __init__(self, buf: bytes):
    self.buf = buf
    self.offset = 0

  def __call__(self, fmt: str) -> tuple:
    ret = struct.unpack_from(fmt, self.buf, self.offset)
    self.offset += struct.calcsize(fmt)
    return ret

def option(i: int) -> int | None:
  """Convert 0 to None, otherwise i-1."""
  return None if i == 0 else i - 1

def load_raw_profile(path: Path) -> list:
  """Load raw profile events from pickle file."""
  with open(path, "rb") as f:
    return pickle.load(f)

def load_profile(profile_events: list) -> dict:
  """Parse the binary profile format."""
  from tinygrad.viz.serve import get_profile

  ret = get_profile(profile_events)
  if ret is None:
    return {"dur": 0, "peak": 0, "layout": {}, "markers": []}

  u = TinyUnpacker(ret)
  total_dur, global_peak, index_len, layout_len = u("<IQII")
  index_data = json.loads(ret[u.offset:u.offset + index_len])
  strings = index_data.get("strings", [])
  markers = index_data.get("markers", [])
  u.offset += index_len

  layout: dict[str, dict] = {}
  for _ in range(layout_len):
    klen = u("<B")[0]
    k = ret[u.offset:u.offset + klen].decode()
    u.offset += klen
    layout[k] = v = {"events": []}
    event_type, event_count = u("<BI")
    if event_type == 0:
      for _ in range(event_count):
        name_idx, ref, key, st, dur, fmt_idx = u("<IIIIfI")
        v["events"].append({
          "name": strings[name_idx] if name_idx < len(strings) else f"unknown_{name_idx}",
          "ref": option(ref),
          "key": option(key),
          "st": st,
          "dur": dur,
          "fmt": strings[fmt_idx] if fmt_idx < len(strings) else ""
        })
    else:
      flag = u("<B")[0]
      v["peak"] = u("<Q")[0]
      if flag == 1:  # LINE graph: each entry is (ts, value) as <IQ
        for _ in range(event_count):
          ts, val = u("<IQ")
          v["events"].append({"event": "line", "ts": ts, "val": val})
      else:  # Memory graph: alloc/free events
        for _ in range(event_count):
          alloc, ts, key = u("<BII")
          if alloc:
            dtype_idx = u("<I")[0]
            sz = u("<Q")[0]
            v["events"].append({
              "event": "alloc", "ts": ts, "key": key,
              "arg": {"dtype": strings[dtype_idx] if dtype_idx < len(strings) else "unknown", "sz": sz}
            })
          else:
            count = u("<I")[0]
            users = [u("<IIIB") for _ in range(count)]
            v["events"].append({"event": "free", "ts": ts, "key": key, "arg": {"users": users}})

  # Convert markers list to dict by name
  marker_ts = {m['name']: m['ts'] for m in markers}
  return {"dur": total_dur, "peak": global_peak, "layout": layout, "markers": markers, "marker_ts": marker_ts}

def is_gpu_kernel(name: str) -> bool:
  """Check if this is an actual GPU kernel (not CPU-side event)."""
  # GPU kernels start with: E_, r_, gemm_, fa_, embedding_bwd
  # CPU events: realize, get_program, Apply, memory planner, etc.
  return bool(re.match(r"^(E_|r_|gemm_|hk_fp8_gemm_|custom_|embedding_)", name))

def categorize_kernel(name: str) -> str:
  """Categorize a kernel based on its name."""
  for category, patterns in KERNEL_CATEGORIES:
    for pattern in patterns:
      if re.search(pattern, name):
        return category
  return "other"

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
  """Merge overlapping intervals. Returns sorted, non-overlapping intervals."""
  if not intervals:
    return []
  sorted_ivs = sorted(intervals)
  merged = [sorted_ivs[0]]
  for start, end in sorted_ivs[1:]:
    if start <= merged[-1][1]:
      merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
      merged.append((start, end))
  return merged

def compute_overlap(start: float, end: float, merged: list[tuple[float, float]]) -> float:
  """Compute how much of [start, end) overlaps with merged intervals."""
  overlap = 0.0
  for ms, me in merged:
    if me <= start: continue
    if ms >= end: break
    overlap += min(end, me) - max(start, ms)
  return overlap

def analyze_profile(profile_path: Path, skip_steps: int = 0, device: str | None = None) -> dict[str, dict]:
  """Analyze the profile and categorize kernels."""
  raw_profile = load_raw_profile(profile_path)
  profile = load_profile(raw_profile)

  # Get the timestamp range for training steps (excluding warmup and eval)
  marker_ts = profile.get("marker_ts", {})
  start_after_ts = 0
  end_before_ts = float('inf')
  num_steps = 0

  # Find train and eval markers
  train_markers = sorted([k for k in marker_ts.keys() if k.startswith("train @")])
  eval_markers = sorted([k for k in marker_ts.keys() if k.startswith("eval @")])

  # Set end timestamp to first eval marker
  if eval_markers:
    end_before_ts = marker_ts[eval_markers[0]]

  if skip_steps > 0:
    skip_marker = f"train @ {skip_steps}"
    if skip_marker in marker_ts:
      start_after_ts = marker_ts[skip_marker]
      num_steps = len([m for m in train_markers if marker_ts[m] >= start_after_ts])
      print(f"Skipping first {skip_steps} steps, analyzing {num_steps} train steps (excluding eval)")
    else:
      print(f"Warning: marker '{skip_marker}' not found, not skipping warmup")
      num_steps = len(train_markers)
  else:
    num_steps = len(train_markers)

  # Find available GPU devices
  gpu_devices = [d for d in profile["layout"].keys()
                 if not any(x in d for x in ["Memory", "Graph", "USER", "TINY", "CPU", "PYTHON"])]

  if device is None and gpu_devices:
    device = gpu_devices[0]  # Default to first GPU
    print(f"Using device: {device} (available: {', '.join(gpu_devices)})")
  elif device and device not in profile["layout"]:
    print(f"Warning: device '{device}' not found, available: {', '.join(gpu_devices)}")
    device = gpu_devices[0] if gpu_devices else None

  categories: dict[str, dict] = defaultdict(lambda: {"time_us": 0.0, "count": 0, "kernels": []})
  total_time_us = 0.0
  compute_intervals: list[tuple[float, float]] = []

  for device_name, device_data in profile["layout"].items():
    if "Memory" in device_name or "Graph" in device_name:
      continue
    # Only count specified device
    if device and device_name != device:
      continue

    for event in device_data.get("events", []):
      if "name" not in event:
        continue
      name = event["name"]
      dur = event.get("dur", 0)
      fmt = event.get("fmt", "")
      st = event.get("st", 0)
      if dur <= 0:
        continue
      # Skip CPU-side events, only count actual GPU kernels
      if not is_gpu_kernel(name):
        continue
      # Skip warmup and eval events
      if st < start_after_ts or st >= end_before_ts:
        continue

      category = categorize_kernel(name)
      categories[category]["time_us"] += dur
      categories[category]["count"] += 1
      categories[category]["kernels"].append({"name": name, "dur": dur, "fmt": fmt})
      total_time_us += dur
      compute_intervals.append((st, st + dur))

  # Second pass: find copy events across all devices
  # Copies on different SDMA queues run in parallel, so merge all copy intervals
  # to get actual wall-clock time, then subtract overlap with compute
  merged_compute = merge_intervals(compute_intervals)
  copy_intervals: list[tuple[float, float]] = []
  copy_count = 0
  copy_kernels = []
  for device_name, device_data in profile["layout"].items():
    if "Memory" in device_name or "Graph" in device_name:
      continue
    for event in device_data.get("events", []):
      if "name" not in event:
        continue
      name = event["name"]
      dur = event.get("dur", 0)
      st = event.get("st", 0)
      if dur <= 0 or st < start_after_ts or st >= end_before_ts:
        continue
      if categorize_kernel(name) != "copy":
        continue
      copy_intervals.append((st, st + dur))
      copy_count += 1
      copy_kernels.append({"name": name, "dur": dur, "fmt": event.get("fmt", "")})

  # Merge concurrent copies, then compute wall-clock time not overlapped by compute
  merged_copies = merge_intervals(copy_intervals)
  exposed_copy_us = 0.0
  for cs, ce in merged_copies:
    wall = ce - cs
    overlap = compute_overlap(cs, ce, merged_compute)
    exposed_copy_us += wall - overlap

  if copy_count > 0:
    categories["copy"]["count"] = copy_count
    categories["copy"]["kernels"] = copy_kernels
    categories["copy"]["time_us"] = exposed_copy_us
    total_time_us += exposed_copy_us

  # Compute wall-clock step times from marker timestamps
  wall_clock_steps_us: list[float] = []
  analyzed_markers = sorted([m for m in train_markers if marker_ts[m] >= start_after_ts and marker_ts[m] < end_before_ts], key=lambda m: marker_ts[m])
  for i_m in range(len(analyzed_markers)):
    step_start = marker_ts[analyzed_markers[i_m]]
    if i_m + 1 < len(analyzed_markers):
      step_end = marker_ts[analyzed_markers[i_m + 1]]
    elif end_before_ts < float('inf'):
      step_end = end_before_ts
    else:
      continue  # skip last step if no end boundary
    wall_clock_steps_us.append(step_end - step_start)

  return {"categories": dict(categories), "total_time_us": total_time_us, "profile_dur_us": profile["dur"],
          "num_steps": num_steps, "wall_clock_steps_us": wall_clock_steps_us}

def print_analysis(analysis: dict):
  """Print the analysis results."""
  categories = analysis["categories"]
  total_time = analysis["total_time_us"]
  num_steps = analysis.get("num_steps", 1) or 1

  if total_time == 0:
    print("No kernel execution time recorded.")
    return

  avg_step_time = total_time / num_steps

  # Wall-clock step time stats
  wall_steps = analysis.get("wall_clock_steps_us", [])
  avg_wall_ms = (sum(wall_steps) / len(wall_steps) / 1e3) if wall_steps else 0
  median_wall_ms = sorted(wall_steps)[len(wall_steps) // 2] / 1e3 if wall_steps else 0

  print("\n" + "=" * 80)
  print("LLaMA3 Training Profile Analysis (per-step average)")
  print("=" * 80)
  print(f"\nSteps analyzed: {num_steps}")
  print(f"Average step GPU time: {avg_step_time / 1e3:.2f} ms")
  if wall_steps:
    print(f"Wall-clock step time:  {avg_wall_ms:.2f} ms avg, {median_wall_ms:.2f} ms median")
    print(f"Overhead (idle/dispatch): {avg_wall_ms - avg_step_time / 1e3:.2f} ms ({(avg_wall_ms - avg_step_time / 1e3) / avg_wall_ms * 100:.1f}%)")
  print()

  sorted_cats = sorted(categories.items(), key=lambda x: x[1]["time_us"], reverse=True)

  print(f"{'Category':<20} {'Time/step (ms)':<15} {'Pct':<8} {'Count/step':<12}")
  print("-" * 60)

  for cat_name, cat_data in sorted_cats:
    time_per_step_ms = cat_data["time_us"] / num_steps / 1000
    pct = (cat_data["time_us"] / total_time) * 100
    count_per_step = cat_data["count"] / num_steps
    print(f"{cat_name:<20} {time_per_step_ms:<15.2f} {pct:<8.1f}% {count_per_step:<12.1f}")

  print("\nTop kernels per category:")
  print("-" * 80)

  for cat_name, cat_data in sorted_cats:
    if cat_data["count"] == 0:
      continue
    time_per_step = cat_data['time_us'] / num_steps / 1000
    print(f"\n{cat_name.upper()} ({time_per_step:.2f} ms/step):")
    # Aggregate kernels by name and show average duration
    kernel_agg: dict[str, list[float]] = defaultdict(list)
    for k in cat_data["kernels"]:
      kernel_agg[k["name"]].append(k["dur"])
    # Sort by total time
    sorted_kernels = sorted(kernel_agg.items(), key=lambda x: sum(x[1]), reverse=True)[:5]
    for name, durs in sorted_kernels:
      avg_dur_ms = sum(durs) / len(durs) / 1000
      count_per_step = len(durs) / num_steps
      print(f"  {name[:50]:<52} {avg_dur_ms:>8.2f} ms x {count_per_step:.1f}/step")

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Profile LLaMA3 training kernel breakdown")
  parser.add_argument("--profile", type=str, default=None, help="Path to profile.pkl file")
  parser.add_argument("--skip", type=int, default=3, help="Number of warmup steps to skip (default: 3)")
  parser.add_argument("--device", type=str, default=None, help="Device to analyze (default: first GPU)")
  parser.add_argument("--verbose", "-v", action="store_true", help="Print all kernels")
  parser.add_argument("--dump-names", action="store_true", help="Just dump all unique kernel names")
  args = parser.parse_args()

  profile_path = Path(args.profile) if args.profile else get_profile_path()

  if not profile_path.exists():
    print(f"Profile file not found: {profile_path}")
    print("\nTo generate a profile, run:")
    print("  VIZ=-1 PROFILE=1 ... python examples/mlperf/model_train.py")
    return 1

  print(f"Loading profile from: {profile_path}")

  if args.dump_names:
    # Just dump unique kernel names for inspection
    raw_profile = load_raw_profile(profile_path)
    profile = load_profile(raw_profile)
    names = set()
    for device_name, device_data in profile["layout"].items():
      if "Memory" in device_name or "Graph" in device_name:
        continue
      for event in device_data.get("events", []):
        if "name" in event:
          names.add(event["name"])
    for name in sorted(names):
      print(name)
    return 0

  analysis = analyze_profile(profile_path, skip_steps=args.skip, device=args.device)
  print_analysis(analysis)

  if args.verbose:
    print("\n" + "=" * 80)
    print("All Kernels")
    print("=" * 80)
    all_kernels = []
    for cat_name, cat_data in analysis["categories"].items():
      for k in cat_data["kernels"]:
        all_kernels.append((k["name"], k["dur"], cat_name, k["fmt"]))
    all_kernels.sort(key=lambda x: x[1], reverse=True)
    for name, dur, cat, fmt in all_kernels:
      print(f"[{cat:<15}] {name:<60} {dur / 1000:>10.2f} ms  {fmt[:30]}")

  return 0

if __name__ == "__main__":
  exit(main())
