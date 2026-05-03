#!/usr/bin/env python3
import argparse, json, sys
from collections import defaultdict

def main() -> int:
  parser = argparse.ArgumentParser(description="Detect overlapping profiler events from tinygrad.viz.cli --json output.")
  parser.add_argument("jsonl", nargs="?", help="JSONL file to read. Defaults to stdin.")
  parser.add_argument("-d", "--device", action="append", help="Only check this device/track. Can be passed more than once.")
  parser.add_argument("--limit", type=int, default=20, help="Maximum overlaps to print.")
  parser.add_argument("--eps-ms", type=float, default=1e-9, help="Tolerance in milliseconds.")
  args = parser.parse_args()

  events:dict[str, list[dict]] = defaultdict(list)
  f = open(args.jsonl) if args.jsonl else sys.stdin
  with f:
    for line_no,line in enumerate(f, 1):
      line = line.strip()
      if not line: continue
      row = json.loads(line)
      if row.get("device") in {"SOURCE", "MARKER"} or "dur_ms" not in row or "st_ms" not in row: continue
      if args.device is not None and row["device"] not in args.device: continue
      row["line"] = line_no
      row["en_ms"] = row["st_ms"] + row["dur_ms"]
      events[row["device"]].append(row)

  overlaps = []
  for device,dev_events in events.items():
    prev = None
    for ev in sorted(dev_events, key=lambda x: (x["st_ms"], x["en_ms"], x.get("name", ""))):
      if prev is not None and ev["st_ms"] < prev["en_ms"] - args.eps_ms:
        overlaps.append((device, prev, ev, prev["en_ms"] - ev["st_ms"]))
        if ev["en_ms"] > prev["en_ms"]: prev = ev
      elif prev is None or ev["en_ms"] > prev["en_ms"]:
        prev = ev

  print(f"tracks={len(events)} events={sum(len(v) for v in events.values())} overlaps={len(overlaps)}")
  for device,prev,ev,overlap in overlaps[:args.limit]:
    print(f"{device}: {overlap:.9f} ms overlap")
    print(f"  prev line {prev['line']}: {prev['name']} [{prev['st_ms']:.9f}, {prev['en_ms']:.9f})")
    print(f"  next line {ev['line']}: {ev['name']} [{ev['st_ms']:.9f}, {ev['en_ms']:.9f})")
  return 1 if overlaps else 0

if __name__ == "__main__":
  raise SystemExit(main())
